"""
Run configuration: the Cfg object every other module reads, and the command line that fills it.

Cfg holds everything shared by the examples (material, phase-field model, encoding, sampler,
load stepping). Anything specific to one example, such as crack seeds, domain size or annulus
radii, is attached by that example's Problem in ``problem.configure(cfg, args)``.

Cfg accepts any object with the right attributes, not just an argparse.Namespace, for example
one rebuilt from a saved config.json (see cfg_from_json).

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import json
import os

import numpy as np

from .utils import get_problem


class _Namespace:
    """argparse.Namespace stand-in, for configs rebuilt from a run directory."""


def _infer_problem(js):
    """Which example a saved config came from: the recorded name, or the single-edge-notched
       plate when no name is recorded."""
    if js.get("problem"):
        return js["problem"]
    return "sen"


def cfg_from_json(source, **overrides):
    """Rebuild a Cfg from a finished run: pass its config.json (as a dict, a path to the file,
       or a path to the run directory holding it).

       The saved config names the example it came from, so the right Problem is restored with
       it and the fields evaluate as they did during training. Keyword overrides are applied
       last. The size of the temperature network (enc_ch_T, width_T, depth_T) is not saved,
       so a run that changed it needs those values passed as overrides.

       The geometry map is not rebuilt here (it costs a patch validation). Set cfg.geomap with
       problem.build_geomap(cfg) when physical coordinates are needed."""
    if isinstance(source, str):
        path = source
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            js = json.load(f)
    else:
        js = dict(source)

    a = _Namespace()
    for k, v in js.items():
        setattr(a, k, v)
    a.problem = _infer_problem(js)
    # Cfg parses these two from comma-separated strings, but they are saved as lists
    a.levels = ",".join(str(int(s)) for s in js["levels"])
    a.enc_start = ",".join(str(int(s)) for s in js["enc_start"])
    # names that differ between the config file and the command line
    a.reg = js.get("lambda_reg", 0.0)
    a.resample = js.get("resample_every", 1)
    # fallbacks for fields the config file does not contain
    for k, dflt in (("plateau_tol", 2e-4), ("plateau_win", 400), ("grip_release", 0.0),
                    ("gamma_star", 0.0), ("geo", "identity"), ("rho_new", 0.0),
                    ("eta_new", 0.3), ("iters0", 0), ("log_every", 300), ("pf_order", 2),
                    ("fixed_notch", False), ("loading", "shear"), ("gc_grip", 0.0),
                    ("ckpt_every", 0), ("plot_every", 0), ("seed", 0), ("grid_aspect", 1.0),
                    ("stress_state", "plane_strain"), ("alpha_mode", "2d")):
        if not hasattr(a, k):
            setattr(a, k, dflt)
    # the example's own flags, which config.json keeps only under their derived names
    for k, v in get_problem(a.problem).config_args(js).items():
        setattr(a, k, v)
    for k, v in overrides.items():
        setattr(a, k, v)
    cfg = Cfg(a)
    # Cfg derives rho_c from the command line's rho_si * cp_si, which config.json does not keep,
    # so the saved heat capacity is restored here
    if "rho_c" in js and not {"rho_si", "cp_si"} & set(overrides):
        cfg.rho_c = js["rho_c"]
    return cfg


# --------------------------------------------------------------------------
# configuration


class Cfg:
    def __init__(self, a, problem=None):
        # which example this run is (the Problem supplies crack seeds, BCs, extra physics)
        if problem is None:
            problem = getattr(a, "problem", None) or "sen"
        self.problem = get_problem(problem) if isinstance(problem, str) else problem
        self.problem_name = self.problem.name
        # material (MPa, mm, N)
        self.E = getattr(a, "E", 340e3)
        self.nu = getattr(a, "nu", 0.22)
        # 2-D reduction: plane_strain (the default) or plane_stress for a thin sheet. It sets
        # lambda, the bulk modulus and whether the out-of-plane elastic strain carries thermal
        # driving (see solver.elastic_parts).
        self.stress_state = getattr(a, "stress_state", None) or "plane_strain"
        # how the out-of-plane elastic strain is treated under plane strain when a temperature
        # field is present: "ps3d" keeps ezz_e = -a_th, "2d" drops it as in a purely 2-D model.
        # Not used under plane stress, where ezz_e is eliminated.
        self.alpha_mode = getattr(a, "alpha_mode", "2d")
        if self.plane_stress:
            self.alpha_mode = "2d"
        self.g_floor = 1e-6
        self.Gc = getattr(a, "Gc", 0.04247)       # mJ/mm^2 (= 42.47 J/m^2)
        self.l = a.l
        # ---- temperature. Every field here is inert unless the example declares heat_on. The
        # thermal part of the solver (a second network, the heat functional, the thermal strain)
        # switches off with it, so an isothermal run carries no thermal cost.
        self.heat_on = bool(getattr(a, "heat_on", None) is True or self.problem.heat_on)
        # optional hook for a steady temperature problem, solved once at full load and scaled by
        # the step's load fraction (see pfte.problems.Problem.T_steady / pfte.thermal.T_effective).
        # The shipped examples do not use it.
        self.T_steady = bool(getattr(self.problem, "T_steady", False))
        # the temperature is a prescribed closed-form field (the example's T_lift alone): no
        # temperature network and no thermal solves
        self.T_analytic = bool(getattr(self.problem, "T_analytic", False))
        self.alpha = getattr(a, "alpha", 0.0)          # thermal expansion (1/K)
        self.T0 = getattr(a, "T0", 300.0)              # initial = stress-free reference (K)
        self.T_env = getattr(a, "T_env", 293.0)        # coolant temperature for a convective wall
        self.hconv = getattr(a, "hconv", 0.0)          # convection coefficient, mJ/(s mm^2 K)
        self.kcond = getattr(a, "kcond", 0.0)          # conductivity, mJ/(s mm K)
        # rho*cp given in SI (kg/m^3, J/kg K) -> mJ/(mm^3 K)
        self.rho_c = getattr(a, "rho_si", 0.0) * getattr(a, "cp_si", 0.0) * 1e-6
        self.dT_final = getattr(a, "dT_final", 0.0)    # the example's total temperature swing
        self.DTref = max(1.0, abs(self.dT_final))      # temperature scale of the T network
        self.gc_tdep = bool(getattr(a, "gc_tdep", False))   # temperature-dependent toughness
        self.gc_b1 = getattr(a, "gc_b1", 1.80)
        self.gc_b2 = getattr(a, "gc_b2", 1.10)
        self.gc_Tref = getattr(a, "gc_Tref", 300.0)
        self.gc_Tmax = getattr(a, "gc_Tmax", 1000.0)
        self.kdeg = int(getattr(a, "kdeg", 0))         # degrade k by (1-phi)^2: an adiabatic crack
        self.T_head_bias = getattr(a, "T_head_bias", 0.0)
        self.T_zero_init = bool(getattr(a, "T_zero_init", True))
        self.t_final = getattr(a, "t_final", 1.0)      # physical end time of the transient (s)
        self.t_start = getattr(a, "t_start", 0.0)      # first time in the schedule
        self.time_power = getattr(a, "time_power", 1.0)   # >1 clusters early steps (fast fronts)
        # the temperature network: its own encoding, width, depth and learning rates
        # (read from the raw flag, because self.levels is parsed further down)
        _lv_T = getattr(a, "levels_T", None) or a.levels
        if isinstance(_lv_T, (list, tuple)):
            _lv_T = ",".join(str(int(s)) for s in _lv_T)
        self.levels_T = tuple(int(s) for s in str(_lv_T).split(","))
        self.enc_ch_T = getattr(a, "enc_ch_T", 2)
        self.width_T, self.depth_T = getattr(a, "width_T", 64), getattr(a, "depth_T", 3)
        self.lr_T, self.lr_enc_T = getattr(a, "lr_T", 5e-4), getattr(a, "lr_enc_T", 2e-3)
        self.enc_reg_T = getattr(a, "enc_reg_T", 1e-8)
        self.iters_T = getattr(a, "iters_T", 400)      # thermal iters per step
        # the three thermal budgets: iters_T0 for the step-0 alignment fit (or the one-off
        # pre-solve of the Problem.T_steady hook), iters_T1 for the first transient solve (the largest dt of the run, which
        # builds the initial boundary layer), iters_T for every later step
        self.iters_T0 = getattr(a, "iters_T0", None) or 1500
        self.iters_T1 = getattr(a, "iters_T1", None) or getattr(a, "iters_T", 400)
        # Adam's epsilon. Where the energy scale is small, the default 1e-8 can dominate the
        # per-parameter gradients, which stalls the localized crack-tip features while the bulk
        # keeps moving.
        self.adam_eps = getattr(a, "adam_eps", 1e-8)
        # weight of the rigid-mode pinning term a body with no displacement boundary condition
        # needs (a quenched slab is loaded only by its own thermal contraction)
        self.c_pin = getattr(a, "c_pin", 1e6)
        # displacement scale: the load is non-dimensionalized by U_ref
        self.U_ref, self.u_delta_nd = getattr(a, "U_ref", 1e-3), 1.0
        # domain / notch
        self.x0 = self.y0 = 0.0
        self.x1 = self.y1 = 1.0
        self.loading = getattr(a, "loading", "shear")   # "shear" (mode II, u-lift) | "tension"
                                                        # (mode I, v-lift) | example-specific
        # single-notch fields, read by the plotting and fixed-notch sampler code. The Problem
        # overwrites them with the first crack's bounding box or its own notch line.
        self.nx0, self.nx1, self.ny = 0.0, 0.5, 0.5
        # phi ansatz
        self.phi_bias = a.phi_bias                # sigmoid bias init (<0 => bulk phi~0)
        self.gamma_ir = a.gamma_ir                # irreversibility penalty weight
        self.ir_tol = a.ir_tol                    # irreversibility dead-band
        # crack-set irreversibility: the penalty acts only where phi_prev >= ir_thresh
        # (0 = everywhere). ir_lock is an alias of the same setting.
        self.ir_thresh = float(getattr(a, "ir_thresh", 0.0) or getattr(a, "ir_lock", 0.0) or 0.0)
        if not 0.0 <= self.ir_thresh < 1.0:
            raise SystemExit("--ir_thresh must lie in [0, 1)")
        self.lr_final = float(getattr(a, "lr_final", 0.0) or 0.0)   # 0 = constant lr in a step
        self.phi_map = a.phi_map                  # "sigmoid" | "leaky" (non-saturating, Manav
                                                  # et al., CMAME 2024, eq. 23)
        self.at = a.at                            # 2 = AT2 (w = phi^2, no threshold) | 1 = AT1
                                                  # (w = phi, c_w = 8/3, elastic threshold
                                                  # psi_th = 3 Gc / (16 l), no damage below it)
        self.pf_order = getattr(a, "pf_order", 2)  # 2 = second order (AT1/AT2) | 4 = fourth order
                                                  # (Borden et al., AT2-type plus a (Lap phi)^2
                                                  # term, identity or no geometry map only)
        self.grip_l = a.grip_l                    # corner treatment radius (units of l), 0 = off
        self.grip_release = a.grip_release        # crack-proximity gate release (0 = never)
        self.grip_released = (False, False, False, False)
        self.gc_grip = a.gc_grip                  # corner toughening beta (0 = off)
        self.gamma_star = a.gamma_star            # star-convex split parameter (0 = Amor)
        self.split = a.split                      # "hybrid" | "amor" | "spectral" |
                                                  # "directional" | "iso"
        self.dirfield = None                      # per-step frozen orientation field (directional)
        self.geo = a.geo                          # "none" | "identity" | "distort" | "strip:W,H" |
                                                  # "square:a" | "ring" | "ring_quarter" | path
                                                  # to a patch JSON
        self.geomap = None                        # GeoMap (set in run(), None for --geo none)
        # physical width/height of the body's bounding box. 1.0 (the default, for the unit square)
        # gives square encoding grids and square figures. An elongated example sets it in
        # configure(), and the encoding then gets proportionally more cells along the long axis,
        # so the physical resolution stays isotropic and the figures match the shape.
        self.grid_aspect = float(getattr(a, "grid_aspect", 1.0) or 1.0)
        # multiresolution encoding
        self.levels = tuple(int(s) for s in a.levels.split(","))
        self.enc_ch = a.enc_ch                    # feature channels per level
        self.lr_enc = a.lr_enc
        self.enc_reg = a.enc_reg                  # small L2 penalty on the grid features
        self.enc_start = tuple(int(s) for s in a.enc_start.split(","))
        self.enc_ramp = a.enc_ramp                # step-0 progressive-activation ramp (iters)
        # network
        self.width, self.depth = a.width, a.depth
        # sampling: a three-component mixture (uniform, notch, damage)
        self.N = a.N
        self.w_unif, self.w_notch, self.w_damage = a.w_unif, a.w_notch, a.w_damage
        self.sigma_band_factor = 1.5
        self.eta_damage = a.eta_damage            # band indicator ~ phi(1-phi)+eta*phi0
        self.beta_drive = a.beta_drive            # + beta * normalized g(phi)*psi_plus (prev)
        self.grid_n = a.grid_n                    # density grid cells per side
        # optional added points: N stays, and points are added on the propagated band
        # (phi_prev > 0.5 where phi0 < 0.4), n_add = rho_new x new-band area
        self.rho_new = a.rho_new                  # areal density of added pts (pts per unit^2)
        self.eta_new = a.eta_new                  # new-band indicator ~ phi(1-phi)+eta_new*phi
        self.fixed_notch = getattr(a, "fixed_notch", False)   # fixed symmetric notch stratum
        # load stepping / training
        self.nsteps, self.delta_max = a.nsteps, a.delta_max
        self.iters = a.iters                      # inner Adam iters per load step
        self.iters0 = a.iters0 if a.iters0 > 0 else a.iters  # step-0 (cold) gets more
        # in the notched-plate, slab and annulus schedules step 1 is the actual cold start (step 0
        # is a zero-load anchor): the temperature first departs from its initial state and the
        # mechanics first sees it. In the strip step 0 is already a full solve. None = --iters.
        self.iters1 = int(a.iters1) if getattr(a, "iters1", None) is not None else self.iters
        self.resample_every = a.resample
        self.lr = a.lr
        self.lambda_reg = a.reg
        self.plateau_tol, self.plateau_win = a.plateau_tol, a.plateau_win
        # step-end acceptance (pfte/stepcheck.py), off by default
        self.accept = bool(int(getattr(a, "accept", 0)))
        self.accept_pi = float(getattr(a, "accept_pi", 0.01))
        self.accept_r = float(getattr(a, "accept_r", 2.0))
        self.accept_dmax = float(getattr(a, "accept_dmax", 1e-2))
        self.accept_chunk = int(getattr(a, "accept_chunk", 300))
        self.accept_extra = int(getattr(a, "accept_extra", 1200))
        self.accept_h = float(getattr(a, "accept_h", 0.02))
        # annealed end of step: after the step's iterations, polish_iters more on polish_N x N
        # points, redrawn every iteration, with the learning rate decayed to zero. 0 (default) = off
        self.polish_iters = int(getattr(a, "polish_iters", 0))
        self.polish_N = float(getattr(a, "polish_N", 4.0))
        # learning-rate warm-up at the start of every step. The optimizer is rebuilt each step,
        # and a fresh Adam moves every parameter by about lr per iteration whatever its gradient,
        # which perturbs the field at the step start. 0 (default) = off
        self.lr_warmup = int(getattr(a, "lr_warmup", 0))
        # keep the Adam state across steps instead of a fresh optimizer per step. 0 (default) = off
        self.adam_persist = int(getattr(a, "adam_persist", 0))
        self.seed = a.seed
        # bookkeeping
        self.N_meter = 80000
        self.log_every = a.log_every
        self.ckpt_every = a.ckpt_every            # >0: also save a checkpoint every k steps
        self.plot_every = a.plot_every            # >0: plot every k steps (0 = about 12 plots)
        self.ud_prev = 0.0                        # prev step's converged u_delta (drive eval)
        # example-specific fields (cracks, domain size, annulus radii, ...)
        self.problem.configure(self, a)

    @property
    def plane_stress(self):
        return getattr(self, "stress_state", "plane_strain") == "plane_stress"

    @property
    def lam(self):
        if self.plane_stress:
            return self.E * self.nu / (1.0 - self.nu ** 2)          # reduced plane-stress lambda'
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def K(self):
        # Amor's volumetric/deviatoric split has to close with the reduction it is written in:
        # under plane stress the deviator is in-plane (tr/2) and the matching 2-D bulk modulus is
        # lam' + mu, not the 3-D lam + 2mu/3.
        if self.plane_stress:
            return self.lam + self.mu
        return self.lam + 2.0 * self.mu / 3.0
    @property
    def sigma_band(self): return self.sigma_band_factor * self.l

    def schedule(self):
        s = np.linspace(0.0, 1.0, self.nsteps + 1)[1:]      # skip delta=0 (trivial)
        ud = (self.delta_max / self.U_ref) * s
        return ud.tolist()

    def time_schedule(self):
        """The physical time of each step of a transient thermal run, one per load step, so the
           two schedules advance together. time_power > 1 clusters the steps early, where a
           thermal front moves fastest and needs the resolution."""
        s = np.linspace(0.0, 1.0, self.nsteps) ** self.time_power
        return (self.t_start + (self.t_final - self.t_start) * s).tolist()

    def to_dict(self):
        d = {k: getattr(self, k) for k in
             ("E", "nu", "g_floor", "Gc", "l", "U_ref", "phi_bias", "gamma_ir", "ir_tol",
              "ir_thresh", "lr_final", "at", "pf_order", "phi_map", "grip_l", "grip_release", "gc_grip", "gamma_star", "split",
              "enc_ch", "lr_enc", "enc_reg", "enc_ramp", "width", "depth", "N",
              "w_unif", "w_notch", "w_damage", "eta_damage", "beta_drive", "grid_n",
              "rho_new", "eta_new",
              "nsteps", "delta_max", "iters", "iters0", "iters1", "resample_every", "lr",
              "lambda_reg", "seed", "ckpt_every", "plot_every", "geo", "grid_aspect", "loading",
              "fixed_notch", "stress_state", "alpha_mode", "heat_on", "alpha", "T0",
              "T_env", "hconv", "kcond", "rho_c", "dT_final", "kdeg", "c_pin",
              "t_start", "t_final", "time_power", "iters_T", "iters_T0", "iters_T1", "adam_eps",
              "gc_tdep", "gc_b1", "gc_b2", "gc_Tref", "gc_Tmax",
              "accept", "accept_pi", "accept_r", "accept_dmax", "accept_chunk", "accept_extra",
              "accept_h", "polish_iters", "polish_N", "lr_warmup", "adam_persist")}
        d["levels_T"] = list(self.levels_T)
        d["levels"] = list(self.levels); d["enc_start"] = list(self.enc_start)
        d["grip_released"] = list(self.grip_released)
        d["problem"] = self.problem_name
        for k in self.problem.config_keys():
            v = getattr(self, k, None)
            d[k] = [list(t) for t in v] if k == "cracks" and v is not None else v
        return d


# --------------------------------------------------------------------------
# command line


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Option defaults appended to the help, and the example's docstring kept as written."""


def build_parser(problem, description=None):
    """Shared flags + this example's own flags, with the example's default overrides applied."""
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=_HelpFormatter)

    # ---- material ---------------------------------------------------------------
    ap.add_argument("--E", type=float, default=340e3, help="Young's modulus (MPa)")
    ap.add_argument("--nu", type=float, default=0.22, help="Poisson ratio")
    ap.add_argument("--Gc", type=float, default=0.04247,
                    help="critical energy release rate (mJ/mm^2)")
    ap.add_argument("--U_ref", type=float, default=1e-3,
                    help="displacement scale used to non-dimensionalize the load (mm)")
    ap.add_argument("--l", type=float, default=0.01, help="phase-field length scale (mm)")
    ap.add_argument("--stress_state", type=str, default=None,
                    choices=["plane_strain", "plane_stress"],
                    help="2-D reduction (plane strain when unset). plane_stress (a thin sheet) "
                         "changes lambda, the bulk modulus and the deviator, and removes the "
                         "out-of-plane thermal driving.")
    ap.add_argument("--alpha_mode", type=str, default="2d", choices=["ps3d", "2d"],
                    help="out-of-plane elastic strain under plane strain with a temperature "
                         "field: ps3d = keep ezz_e = -alpha dT (cooling drives tension through "
                         "it) | 2d = drop it. Ignored under plane stress.")

    # ---- temperature ---------------------------------------------------------------
    # Every flag here is inert unless the running example carries a temperature field.
    ap.add_argument("--alpha", type=float, default=0.0, help="thermal expansion (1/K)")
    ap.add_argument("--T0", type=float, default=300.0,
                    help="initial temperature = the stress-free reference (K)")
    ap.add_argument("--T_env", type=float, default=293.0,
                    help="bath temperature (K): the prescribed face or rim temperature (--bc_T "
                         "dirichlet, dirichlet_net) or the coolant of the convective condition "
                         "(robin). The glass strip uses --Tc instead")
    ap.add_argument("--hconv", type=float, default=0.0,
                    help="convection coefficient, mJ/(s mm^2 K) = 1e-3 * W/(m^2 K)")
    ap.add_argument("--kcond", type=float, default=0.0,
                    help="conductivity, mJ/(s mm K) = W/(m K)")
    ap.add_argument("--rho_si", type=float, default=0.0, help="density (kg/m^3)")
    ap.add_argument("--cp_si", type=float, default=0.0, help="specific heat (J/kg K)")
    ap.add_argument("--dT_final", type=float, default=0.0,
                    help="temperature scale of the temperature network (K). For the notched "
                         "plates the final change of the top edge. For the slab and annulus the "
                         "bath is --T_env, so set dT_final = T_env - T0")
    ap.add_argument("--kdeg", type=int, default=0, choices=[0, 1],
                    help="1 = degrade the conductivity by (1-phi)^2, making the crack adiabatic")
    ap.add_argument("--t_start", type=float, default=0.0, help="first time in the schedule (s)")
    ap.add_argument("--t_final", type=float, default=1.0, help="end of the transient (s)")
    ap.add_argument("--time_power", type=float, default=1.0,
                    help=">1 clusters the load steps early, where a thermal front moves fastest")
    ap.add_argument("--levels_T", type=str, default=None,
                    help="feature-grid sides of the temperature encoding, the same as --levels "
                         "if not given. It must resolve the transient front and, with --kdeg 1, "
                         "the steep change of T across the damage band that the adiabatic "
                         "crack allows.")
    ap.add_argument("--enc_ch_T", type=int, default=2,
                    help="feature channels per level of the temperature encoding")
    ap.add_argument("--width_T", type=int, default=64, help="width of the temperature network")
    ap.add_argument("--depth_T", type=int, default=3,
                    help="number of hidden layers of the temperature network")
    ap.add_argument("--lr_T", type=float, default=5e-4,
                    help="learning rate of the temperature network weights")
    ap.add_argument("--lr_enc_T", type=float, default=2e-3,
                    help="learning rate of the temperature feature grids")
    ap.add_argument("--enc_reg_T", type=float, default=1e-8,
                    help="L2 penalty on the temperature grid features")
    ap.add_argument("--iters_T", type=int, default=400, help="thermal iterations per load step")
    ap.add_argument("--iters_T0", type=int, default=None,
                    help="iterations of the step-0 alignment fit, or of the one-off pre-solve "
                         "of the Problem.T_steady hook. None = 1500")
    ap.add_argument("--iters_T1", type=int, default=None,
                    help="thermal iterations of the first transient solve (the largest dt, which "
                         "builds the initial boundary layer). None = --iters_T")
    ap.add_argument("--gc_tdep", type=int, default=0, choices=[0, 1],
                    help="1 = Gc falls with temperature, Gc0 [1 - b1 th + b2 th^2] with "
                         "th = (T - Tref)/Tmax")
    ap.add_argument("--gc_b1", type=float, default=1.80, help="coefficient b1 of --gc_tdep")
    ap.add_argument("--gc_b2", type=float, default=1.10, help="coefficient b2 of --gc_tdep")
    ap.add_argument("--gc_Tref", type=float, default=300.0,
                    help="reference temperature Tref of --gc_tdep (K)")
    ap.add_argument("--gc_Tmax", type=float, default=1000.0,
                    help="temperature scale Tmax of --gc_tdep (K)")
    ap.add_argument("--T_head_bias", type=float, default=0.0,
                    help="initial bias of the output layer of the temperature network")
    ap.add_argument("--c_pin", type=float, default=1e6,
                    help="rigid-mode pinning weight, for a body with no displacement boundary "
                         "condition (a quenched slab)")
    ap.add_argument("--adam_eps", type=float, default=1e-8,
                    help="Adam's epsilon. Where the energy is small the default can dominate "
                         "the per-parameter gradients and stall localized crack-tip features. "
                         "A smaller value such as 1e-16 avoids this.")

    # ---- load stepping / training -----------------------------------------------
    ap.add_argument("--nsteps", type=int, default=18, help="number of load steps")
    ap.add_argument("--delta_max", type=float, default=2.7e-3,
                    help="final imposed displacement (mm)")
    ap.add_argument("--iters", type=int, default=2000, help="Adam iterations per load step")
    ap.add_argument("--iters0", type=int, default=4000, help="step-0 iterations, 0 = --iters")
    ap.add_argument("--iters1", type=int, default=None,
                    help="iterations of step 1. In the heated notched-plate, slab and annulus "
                         "schedules step 0 is a zero-load anchor and step 1 the actual cold start. "
                         "In the strip step 0 is already a full solve. None = --iters")
    ap.add_argument("--resample", type=int, default=1,
                    help="redraw the integration points every k iterations. With 1 every "
                         "iteration draws a fresh point set, so the network cannot overfit one "
                         "fixed set.")
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="learning rate of the network weights")
    ap.add_argument("--reg", type=float, default=0.0, help="L2 penalty on the network weights")

    # ---- phase-field model -------------------------------------------------------
    ap.add_argument("--gamma_ir", type=float, default=1e3, help="irreversibility penalty weight")
    ap.add_argument("--ir_tol", type=float, default=0.0,
                    help="irreversibility dead-band: phi may fall up to this far below phi_prev "
                         "without penalty, which keeps far-field sampling noise from accumulating "
                         "across steps. 0 = exact")
    ap.add_argument("--ir_thresh", "--ir_lock", dest="ir_thresh", type=float, default=0.0,
                    help="crack-set irreversibility (after Bourdin et al., discussed by "
                         "Gerasimov and De Lorenzis, CMAME 2019): the penalty acts only where "
                         "phi_prev >= this value, and below it damage may heal. 0 = everywhere. "
                         "--ir_lock is an alias")
    ap.add_argument("--lr_final", type=float, default=0.0,
                    help="0 = constant learning rate within a step. >0 decays both mechanical "
                         "learning rates exponentially over each load step, to the fraction "
                         "lr_final/lr of their initial values")
    ap.add_argument("--at", type=int, default=2, choices=[2, 1],
                    help="AT model: 2 = AT2 (no elastic threshold) | 1 = AT1 (w = phi, "
                         "c_w = 8/3, elastic threshold psi_th, no damage below it)")
    ap.add_argument("--pf_order", type=int, default=2, choices=[2, 4],
                    help="phase-field order: 2 (AT1/AT2) | 4 (Borden et al., CMAME 2014, "
                         "AT2-type: halved gradient coefficient plus a (Laplacian phi)^2 term "
                         "from one extra autodiff pass, which needs a C1 field). Order 4 "
                         "requires --at 2 and identity or none geometry, and uses the "
                         "fourth-order profile phi0 = exp(-2d/l)(1+2d/l).")
    ap.add_argument("--phi_map", type=str, default="sigmoid", choices=["sigmoid", "leaky"],
                    help="phi logit map: sigmoid | leaky (Manav et al., CMAME 2024, eq. 23, "
                         "non-saturating, required for --at 1, use with --phi_bias -2)")
    ap.add_argument("--phi_bias", type=float, default=-4.0, help="phi-channel sigmoid bias init")
    ap.add_argument("--split", type=str, default="hybrid",
                    choices=["amor", "spectral", "directional", "hybrid", "iso"],
                    help="energy split: hybrid (Ambati et al. 2015: isotropic degradation in "
                         "the mechanics, spectral tension in the crack driving force) | amor "
                         "(volumetric-deviatoric) | spectral (tension only) | directional "
                         "(crack-frame split on the band, spectral in the bulk, requires --geo "
                         "none) | iso (no split, "
                         "the full isotropic energy is degraded)")

    # ---- corner treatment ---------------------------------------------------------
    ap.add_argument("--grip_l", type=float, default=3.0,
                    help="radius of the corner treatment in units of l: of the phi gate when "
                         "--gc_grip is 0, of the toughened discs otherwise. 0 = off. The gate "
                         "prevents damage at the corners but also holds the final ligament, see "
                         "--grip_release")
    ap.add_argument("--grip_release", type=float, default=0.0,
                    help="crack-proximity release of the corner grip: when phi_prev exceeds this "
                         "threshold in the annulus [1.3, 2.5] x grip_l x l around a corner disc, "
                         "that corner's grip switches off permanently so the final ligament can "
                         "break. 0 = the grip stays on. It releases the phi gate only (--gc_grip "
                         "0) and requires --geo none")
    ap.add_argument("--gc_grip", type=float, default=1.0,
                    help="corner toughening beta: Gc -> Gc (1 + beta bump) inside corner discs "
                         "of radius about grip_l x l. A variational alternative to the grip, "
                         "which it replaces when > 0: the weak driving of the corner singularity "
                         "cannot nucleate damage there, while an arriving crack still cuts "
                         "through. 0 = off")
    ap.add_argument("--gamma_star", type=float, default=0.0,
                    help="star-convex split parameter, used only with --split amor (0 = Amor)")

    # ---- geometry -----------------------------------------------------------------
    ap.add_argument("--geo", type=str, default="identity",
                    help="geometry map of the notched plate: identity (the unit square as a "
                         "degree-2 B-spline patch, with the full mapping pipeline) | distort "
                         "(unit square with a distorted interior and invariant boundary and "
                         "notch midline, a test of the pullback) | none (unit square without the "
                         "geometry map). The slab, strip and annulus build their own patch and "
                         "ignore this option.")

    # ---- multiresolution encoding --------------------------------------------------
    ap.add_argument("--levels", type=str, default="32,128,384",
                    help="feature-grid sides, coarse to fine, counted along the second "
                         "parametric direction (the short side of the slab and strip, the arc of "
                         "the annulus). The other direction gets cells in proportion to its "
                         "length, so the physical spacing is about isotropic. The "
                         "finest spacing, about that side over (finest level - 2) for the "
                         "quadratic B-spline encoding, limits the resolution of the "
                         "representation. Raise it together with --N")
    ap.add_argument("--enc_ch", type=int, default=2, help="feature channels per level")
    ap.add_argument("--lr_enc", type=float, default=2e-3, help="learning rate of the feature grids")
    ap.add_argument("--enc_reg", type=float, default=1e-8, help="L2 penalty on the grid features")
    ap.add_argument("--enc_start", type=str, default="0,0,0",
                    help="step-0 activation start iteration per level. With 0 for every level "
                         "the zero-initialized features alone make the activation progressive")
    ap.add_argument("--enc_ramp", type=int, default=1, help="activation ramp length (iterations)")

    # ---- sampler -------------------------------------------------------------------
    ap.add_argument("--N", type=int, default=12000, help="integration points per iteration")
    ap.add_argument("--w_unif", type=float, default=0.4,
                    help="mixture weight of the uniform stratum of the sampler")
    ap.add_argument("--w_notch", type=float, default=0.3,
                    help="mixture weight of the crack stratum (notch and propagated crack)")
    ap.add_argument("--w_damage", type=float, default=0.3,
                    help="mixture weight of the damage-band stratum")
    ap.add_argument("--eta_damage", type=float, default=0.3,
                    help="weight of phi0 in the damage-band indicator phi (1 - phi) + eta phi0")
    ap.add_argument("--beta_drive", type=float, default=0.5,
                    help="driving-force weight in the density indicator (0 = plain indicator)")
    ap.add_argument("--grid_n", type=int, default=256, help="sampling density grid cells per side")
    ap.add_argument("--rho_new", type=float, default=0.0,
                    help="optional extra points on the propagated crack. 0 = off: the crack "
                         "stratum (max(phi0, phi_prev) |detJ|) already covers the whole crack, "
                         "notch and propagated part, at one physical density. >0 = add points on "
                         "the propagated crack at this areal density (points per unit area)")
    ap.add_argument("--eta_new", type=float, default=0.3, help="new-band indicator phi weight")
    ap.add_argument("--fixed_notch", action="store_true",
                    help="middle (notch) stratum = a fixed analytic band symmetric about the "
                         "notch line (Gaussian in y about ny, uniform in x over the notch) "
                         "instead of the adaptive crack grid. It does not depend on phi.")

    # ---- network -------------------------------------------------------------------
    ap.add_argument("--width", type=int, default=128, help="width of the network")
    ap.add_argument("--depth", type=int, default=4, help="number of hidden layers of the network")
    ap.add_argument("--seed", type=int, default=0, help="seed of the network initialization")

    # ---- misc ----------------------------------------------------------------------
    ap.add_argument("--plateau_tol", type=float, default=2e-4,
                    help="a step stops early when the relative spread of the energy over the "
                         "last plateau_win iterations falls below plateau_tol")
    ap.add_argument("--plateau_win", type=int, default=400,
                    help="window of the plateau test (iterations), see --plateau_tol")
    ap.add_argument("--accept", type=int, default=0, choices=[0, 1],
                    help="1 = check each step's end state against its start state under the "
                         "step's conditions (pfte/stepcheck.py) and continue a rejected step, "
                         "rectangular strip geometries and the second-order AT2 model with the "
                         "hybrid split only. The check applies the irreversibility penalty "
                         "everywhere, as with --ir_thresh 0. 0 = the step ends on its last "
                         "iterate")
    ap.add_argument("--accept_pi", type=float, default=0.01,
                    help="reject when the grid functional rises above (1 + this) x that of the "
                         "start state and the band residual above accept_r x that of the start "
                         "state")
    ap.add_argument("--accept_r", type=float, default=2.0,
                    help="band-residual factor of the rejection test, see --accept_pi")
    ap.add_argument("--accept_dmax", type=float, default=1e-2,
                    help="reject when max (phi_prev - phi)_+ on the check grid exceeds this")
    ap.add_argument("--accept_chunk", type=int, default=300,
                    help="iterations of each continuation of a rejected step")
    ap.add_argument("--accept_extra", type=int, default=1200,
                    help="maximum number of extra iterations a step may spend on continuations")
    ap.add_argument("--accept_h", type=float, default=0.02,
                    help="spacing (mm) of the regular grid of the check")
    ap.add_argument("--polish_iters", type=int, default=0,
                    help="> 0: end every step with this many iterations on polish_N x N points, "
                         "still redrawn every iteration, with the learning rate decayed to zero on "
                         "a cosine, which averages out the sampling noise. 0 = the step ends on "
                         "its last iterate")
    ap.add_argument("--polish_N", type=float, default=4.0,
                    help="size of the polishing point set, in units of N")
    ap.add_argument("--lr_warmup", type=int, default=0,
                    help="> 0: ramp both learning rates linearly from 0 over the first lr_warmup "
                         "iterations of every step, which avoids the large first updates of the "
                         "optimizer rebuilt at each step. 0 = off")
    ap.add_argument("--adam_persist", type=int, default=0, choices=[0, 1],
                    help="1: keep one Adam optimizer (its moments) across load steps instead of a "
                         "fresh one per step, and store its state in the checkpoints. 0 = a fresh "
                         "optimizer per step")
    ap.add_argument("--out", type=str, default="runs/out", help="output folder")
    ap.add_argument("--fem", type=str, default="NONE",
                    help="optional finite element force-displacement file to overlay on the "
                         "comparison plot, NONE = no overlay. Reference curves come from the "
                         "scripts in fem/.")
    ap.add_argument("--log_every", type=int, default=300,
                    help="write a log line every k iterations")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help=">0: also save a checkpoint into step_NNN/ every k steps (1 = every step)")
    ap.add_argument("--plot_every", type=int, default=0,
                    help=">0: plot the fields and the force-displacement curve into step_NNN/ "
                         "every k steps. 0 = about 12 plots per run")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest checkpoint in the output folder")

    problem.add_arguments(ap)
    ap.set_defaults(geo=problem.default_geo)      # the patch this example lives on
    defaults = problem.defaults()
    if defaults:
        ap.set_defaults(**defaults)               # ... which defaults() may still override
    return ap
