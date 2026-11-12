"""
pfte: a mesh-free deep energy method for phase-field modeling of thermally driven brittle
fracture.

One network on a multiresolution B-spline feature encoding represents the displacement and phase
fields. Where the temperature is computed, a second network represents it, and each load step
solves the heat problem first and then the mechanics at that temperature (staggered coupling:
the crack enters the heat problem only through the conductivity of the previous step, when it is
degraded). Where it is prescribed in closed form, the mechanics reads it directly. The energy is
integrated by Monte Carlo quadrature over strata that follow the crack, with the integration
points redrawn at every iteration, and is minimized directly, without a mesh or crack tracking.
All derivatives come from automatic differentiation. The geometry is a single B-spline or NURBS
patch, so curved domains are exact.

Modules:

    utils             device and dtype, the example registry, logging, checkpoints
    geometry          the B-spline/NURBS patch, crack seeds, corner treatment
    config            Cfg and the shared command line
    plots             run-time figures
    solver            encoding, network, energy, quadrature, reaction force, load stepping
    thermal           the temperature network, the heat functional and the thermal strain
    problems          the Problem interface and the single-edge-notched plate
    problems_thermal  the quenched ceramic slab, the quenched glass strip, the quenched annulus
    stepcheck         an optional step-end acceptance check of the mechanical solve

Runs are launched from the entry scripts in examples/, see README.md.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from . import utils
from .config import Cfg, build_parser, cfg_from_json
from .geometry import (DEFAULT_CRACKS, GRIP_CORNERS, SEN_NOTCH, GeoMap, build_geomap,
                       build_ring_patch, crack_profile, dist_to_notch, gc_bump, grip_corners,
                       parse_cracks, phi0_field, phi_gate, to_phys, update_grip_release)
from .plots import (plot_FD_compare, plot_FD_solo, plot_fields, plot_geometry, plot_geometry_bc,
                    plot_linecut, plot_parametric, plot_phi)
from .solver import (DirField, MultiResEncoding, NetMS, Sampler, _fields, _frac_energy,
                     drive_field, energy_terms, estimate_Pi, kinematics, phi_forward,
                     phi_squash, phi_value, reaction_force, run, run_cli, solve_step, validate)
from .utils import (DTYPE, available, get_logger, get_problem, load_ckpt, problem_of, save_ckpt,
                    set_device)
from .problems import Problem          # last: it imports geometry and solver

__version__ = "1.0.0"


def __getattr__(name):
    """DEVICE is re-read from utils on every access, so pfte.set_device('cpu') is visible
       through this name too (a plain import would have frozen a copy at import time)."""
    if name == "DEVICE":
        return utils.DEVICE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "utils", "set_device", "DEVICE", "DTYPE",
    "Cfg", "cfg_from_json", "build_parser", "Problem", "get_problem", "problem_of", "available",
    "GeoMap", "build_geomap", "build_ring_patch",
    "MultiResEncoding", "NetMS", "kinematics", "phi_squash", "phi_forward", "phi_value",
    "drive_field", "_fields",
    "DirField", "energy_terms", "_frac_energy",
    "Sampler", "estimate_Pi", "reaction_force",
    "to_phys", "parse_cracks", "dist_to_notch", "crack_profile", "phi0_field", "phi_gate",
    "grip_corners", "update_grip_release", "gc_bump", "SEN_NOTCH", "DEFAULT_CRACKS",
    "GRIP_CORNERS",
    "save_ckpt", "load_ckpt", "get_logger",
    "plot_phi", "plot_fields", "plot_linecut", "plot_FD_compare", "plot_FD_solo",
    "plot_geometry", "plot_geometry_bc", "plot_parametric",
    "solve_step", "run", "run_cli", "validate",
]
