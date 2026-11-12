"""
Shared plumbing: the torch backend, the example registry, logging and checkpoints.

DEVICE lives here. Modules read it as ``utils.DEVICE``, an attribute lookup at call time rather
than ``from .utils import DEVICE``, so that a set_device call made before a run or an
evaluation retargets the package. This lets the checkpoints of a GPU run be evaluated on a CPU.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import logging
import os
import sys
import types

import numpy as np
import torch


# --------------------------------------------------------------------------
# torch backend


torch.set_default_dtype(torch.float64)

DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_device(dev):
    """Retarget the package (e.g. set_device('cpu') to render checkpoints on a machine
       without a GPU, or to force deterministic CPU evaluation)."""
    global DEVICE
    DEVICE = str(dev)
    return DEVICE


# --------------------------------------------------------------------------
# example registry


# The examples live in pfte.problems, which imports geometry and solver, and both of these ask
# the registry for the running example. Importing it lazily here, inside get_problem, avoids an
# import cycle.

def _registry():
    from . import problems
    from . import problems_thermal          # noqa: F401  (registers the thermo examples)
    return problems.REGISTRY


def available():
    """Names of the registered examples."""
    return tuple(_registry())


def get_problem(name):
    """Return the (singleton) Problem instance registered under this name."""
    reg = _registry()
    key = str(name)
    if key not in reg:
        raise KeyError(f"unknown problem {name!r}; available: {', '.join(reg)}")
    return reg[key]


def problem_of(cfg):
    """The Problem behind a cfg. It also accepts configs rebuilt from JSON, where cfg.problem is
       still the plain name, and configs without the field, which are taken as the
       single-edge-notched plate."""
    p = getattr(cfg, "problem", None)
    if p is None or isinstance(p, str):
        p = get_problem(p or "sen")
        try:
            cfg.problem = p
        except AttributeError:              # frozen/duck-typed config: resolve every call
            pass
    return p


# --------------------------------------------------------------------------
# checkpoints and logging


def save_ckpt(path, net, prev_state, step, ud, FD, cfg, netT=None, prevT_state=None, t_now=None,
              opt=None):
    """Snapshot a run. A thermal run also stores its temperature network and the frozen previous
       one, so a resume picks the transient up where it left off rather than restarting it.
       `opt` is the optimizer carried across steps (--adam_persist). Its moments are stored so a
       resume continues with them instead of starting a fresh Adam. The CUDA generators are
       stored too (the thermal boundary sampler draws on them), so a resume replays exactly."""
    name, keys, pos, has_g, cached = np.random.get_state()
    d = {"model": net.state_dict(), "prev": prev_state,
         "step": step, "ud": ud, "FD": FD, "cfg": cfg.to_dict(),
         "torch_rng": torch.get_rng_state(),
         # plain Python values: an ndarray pickled under numpy 2 refers to numpy._core, which
         # numpy 1.x cannot unpickle
         "numpy_rng": (str(name), [int(v) for v in keys], int(pos), int(has_g), float(cached))}
    if torch.cuda.is_available():
        d["cuda_rng"] = torch.cuda.get_rng_state_all()
    if opt is not None:
        d["adam"] = opt.state_dict()
    if netT is not None:
        d["modelT"] = netT.state_dict()
        d["prevT"] = prevT_state
        d["t_now"] = t_now
    torch.save(d, path)

def load_ckpt(path, net, netT=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ck["model"])
    if netT is not None and ck.get("modelT") is not None:
        netT.load_state_dict(ck["modelT"])
    torch.set_rng_state(ck["torch_rng"].to("cpu"))
    np.random.set_state(ck["numpy_rng"])
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        states = [s.to("cpu") for s in ck["cuda_rng"]]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)
        else:                                        # written on a machine with another GPU count
            torch.cuda.set_rng_state(states[0])
    return ck

def get_logger(save_dir):
    logger = logging.getLogger("pfte")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(save_dir, "train.log")); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------
# module device alias


class _LegacyModule(types.ModuleType):
    """Module whose DEVICE attribute is an alias of pfte.utils.DEVICE.

       Reading it returns the package's current device, and assigning to it retargets the whole
       package (for example ``mod.DEVICE = "cpu"`` before rendering checkpoints of a GPU run on a
       machine without one)."""

    def __getattr__(self, name):                     # only called when normal lookup fails
        if name == "DEVICE":
            return DEVICE
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "DEVICE":
            set_device(value)                        # retarget the package, not just this module
            return
        object.__setattr__(self, name, value)


def install_device_proxy(module_name):
    """Turn the named (already imported) module into one whose DEVICE attribute follows the
    package device.

    The module must not bind a plain ``DEVICE`` name of its own, or reads would find that copy
    instead of reaching the alias."""
    mod = sys.modules[module_name]
    mod.__dict__.pop("DEVICE", None)
    mod.__class__ = _LegacyModule
    return mod
