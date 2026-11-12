"""
The quench examples: the quenched ceramic slab of Tang et al. (2016), the directionally quenched
glass strip of Yuse and Sano (1993) and the quenched ceramic annulus.

They use the same solver as the mechanically loaded examples. The temperature field enters only
through the Problem hooks: where heat enters (a convective or a prescribed surface temperature,
or a closed-form imposed field), what the material is (uniform, or drawn cell by cell from a
Weibull distribution), and how the body is supported.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import math

import numpy as np
import torch

from . import utils
from .geometry import build_geomap, crack_profile, dist_to_notch, to_phys
from .problems import REGISTRY, Problem
from .solver import phi_squash
from .geometry import phi_gate
from .utils import DTYPE


# --------------------------------------------------------------------------
# shared Weibull meso-structure machinery (Tang et al. 2016)


def _weibull_fields(E_mean, S_mean, m, seed, ny, nx, mirror=False):
    """Independent Weibull draws of modulus and strength per meso-element.

       The draw uses numpy's rng.weibull stream, E first, then strength, unnormalized (the mean
       is Gamma(1+1/m) times the nominal value), so a seed reproduces the same field.
       mirror=True reflects the drawn half about its right edge, which makes the full field
       symmetric about the mid-length."""
    rng = np.random.default_rng(seed)
    E = E_mean * rng.weibull(m, (ny, nx))
    S = S_mean * rng.weibull(m, (ny, nx))
    if mirror:
        E = np.concatenate([E, E[:, ::-1]], axis=1)
        S = np.concatenate([S, S[:, ::-1]], axis=1)
    return E, S


def _calibrate_gc(cfg, Efield, Sfield):
    """Convert per-cell strength to toughness for the model in use.

       AT2 uses the homogeneous peak stress sqrt(27 E Gc / 256 l), AT1 the elastic limit
       sqrt(3 E Gc / 8 l). The field is then rescaled by one constant so that its mean is
       cfg.gc_mean (0 = no rescale). Sets cfg.gc_scale and cfg.strength_implied (logged at
       start). Returns (Gc field, calibration factor)."""
    cal = (256.0 / 27.0) if cfg.at == 2 else (8.0 / 3.0)
    Gfield = cal * cfg.l * Sfield ** 2 / Efield
    cfg.gc_scale = 1.0
    if cfg.gc_mean > 0:
        cfg.gc_scale = cfg.gc_mean / float(Gfield.mean())
        Gfield = Gfield * cfg.gc_scale
    cfg.strength_implied = float(Sfield.mean() * math.sqrt(cfg.gc_scale))
    return Gfield, cal


def _assert_ring_radii(cfg):
    """Check that the built annulus patch has the configured radii.

       The patch is built from radii passed explicitly, and build_geomap has its own defaults.
       This check catches a Problem that does not override build_geomap, which would otherwise
       run on an annulus of a different size."""
    mid = torch.full((1, 1), 0.5, dtype=DTYPE, device=utils.DEVICE)
    for xi_v, want, name in ((0.0, cfg.ring_ri, "inner"), (1.0, cfg.ring_ro, "outer")):
        xp, yp = to_phys(torch.full_like(mid, xi_v), mid, cfg)
        r = float((xp ** 2 + yp ** 2).sqrt())
        assert abs(r - want) < 1e-6 * want, \
            f"{name} radius of the built patch is {r:g}, not the configured {want:g} " \
            "(does this Problem override build_geomap and pass its radii?)"


def _robin_times(cfg):
    """Time stepping of the quench: a zero-load state at t = 0, then nsteps states on the
       power-law clock from t_start to t_final. Step 1 spans the whole first window and gets
       the --iters_T1 budget."""
    n = cfg.nsteps
    if n == 1:
        return [0.0, cfg.t_final]
    return [0.0] + [cfg.t_start + (cfg.t_final - cfg.t_start)
                    * ((k - 1) / (n - 1)) ** cfg.time_power for k in range(1, n + 1)]


# --------------------------------------------------------------------------
# quenched slab  (Tang et al., Int. J. Solids Struct. 80 (2016) 520-531)


class QuenchProblem(Problem):
    """A hot ceramic slab dropped into water. There is no mechanical load: the slab cracks
       from the thermal contraction of the surfaces, which cool faster than the core.

       The cooled surfaces either exchange heat with the bath by convection (Robin condition)
       or are held at the bath temperature (Dirichlet condition), see --bc_T. There is no
       initial crack, so the cracks nucleate from the material, which is drawn cell by cell
       from a Weibull distribution as in Tang et al. (2016). The heterogeneous modulus turns
       the smooth thermal strain into a spatially varying stress. Without it the symmetric
       solution is a uniform damage band.

       The quarter model (--quarter) keeps one corner of the slab and puts symmetry planes on
       the two cuts. The heterogeneity mirror is then switched off, since the cuts already act
       as mirrors."""

    name = "quench"
    default_geo = "strip:50,10"
    heat_on = True

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--strip", type=str, default="50,10", help="slab W,H in mm")
        ap.add_argument("--quarter", type=int, default=1, choices=[0, 1],
                        help="1 = model one quarter of the slab, with symmetry planes on the cuts. "
                             "0 = the full slab")
        ap.add_argument("--sigma_t", type=float, default=180.0,
                        help="nominal tensile strength of the Weibull field (MPa). The draw is "
                             "sigma_t * Weibull(m, scale 1), whose mean is Gamma(1+1/m) times "
                             "the nominal value (about 0.966 at m = 15), as in Tang et al. (2016)")
        ap.add_argument("--het_m", type=float, default=15.0, help="Weibull modulus (homogeneity)")
        ap.add_argument("--het_cell", type=float, default=0.05, help="meso-element size (mm)")
        ap.add_argument("--het_seed", type=int, default=1,
                        help="seed of the Weibull draw")
        ap.add_argument("--het_mirror", type=int, default=None, choices=[0, 1],
                        help="draw the field on one half of the slab and reflect it about the "
                             "mid-length. Default: on for the full slab, off for the quarter, "
                             "whose symmetry planes act as mirrors")
        ap.add_argument("--gc_mean", type=float, default=0.0,
                        help="rescale the calibrated Gc field to this mean (mJ/mm^2). 0 = keep "
                             "the strength calibration")
        ap.add_argument("--N_boundary", type=int, default=8000,
                        help="points per iteration on the cooled surfaces, split between them. "
                             "The convective term is a boundary integral with its own "
                             "Monte Carlo sampling")
        ap.add_argument("--w_env", type=float, default=0.15,
                        help="half the standard deviation of the cooled-surface sampling band "
                             "(mm)")
        ap.add_argument("--bc_T", default="robin", choices=["robin", "dirichlet", "dirichlet_net"],
                        help="robin = convection with hconv on the cooled surfaces (Tang et al.), "
                             "solved by the temperature network. dirichlet = the bath temperature "
                             "prescribed on them (Bourdin et al. 2014, Mandal et al. 2021), with "
                             "the closed-form quarter-slab solution and no temperature network. "
                             "dirichlet_net = the same prescribed surface temperature imposed "
                             "exactly through the lift, with the field solved by the temperature "
                             "network")
        ap.add_argument("--w_T", type=float, default=0.005,
                        help="width (mm) of the lift envelope tanh(d / w_T) of --bc_T "
                             "dirichlet_net. It should be well below the boundary layer of the "
                             "first state, 2 sqrt(kappa t_start), so that it does not smear the "
                             "gradient")

    def defaults(self):
        # problem-level defaults: the ceramic of Tang et al. (2016) with the toughness of Bourdin
        # et al. (2014), in N and mm, and the cubic clock from 1e-4 to 0.2 s. examples/slab.py
        # sets the paper case.
        return dict(E=370e3, nu=0.3, Gc=0.04247, l=0.05, at=2, split="hybrid",
                    stress_state="plane_stress", alpha=7.5e-6, kcond=31.0,
                    rho_si=3980.0, cp_si=880.0, hconv=50000.0 * 1e-3,   # W/(m^2 K) -> mJ/(s mm^2 K)
                    T0=873.0, T_env=293.0, dT_final=-580.0, kdeg=0,
                    t_start=1e-4, t_final=0.2, time_power=3.0,
                    grip_l=0.0, gc_grip=0.0, U_ref=1e-3, delta_max=0.0,
                    w_unif=0.3, w_notch=0.3, w_damage=0.4)

    def configure(self, cfg, a):
        w, h = (float(v) for v in str(getattr(a, "strip", "50,10")).split(","))
        cfg.quarter = bool(int(getattr(a, "quarter", 1)))
        if cfg.quarter:
            w, h = 0.5 * w, 0.5 * h          # one corner, and the cuts become symmetry planes
        cfg.strip_w, cfg.strip_h = w, h
        cfg.geo = f"strip:{w:g},{h:g}"
        cfg.grid_aspect = w / h
        cfg.w_env = getattr(a, "w_env", 0.15)
        cfg.N_boundary = getattr(a, "N_boundary", 8000)
        cfg.cracks = []                       # nothing pre-existing: every crack nucleates
        cfg.nx0 = cfg.nx1 = cfg.ny = 0.0
        cfg.sigma_t = getattr(a, "sigma_t", 180.0)
        cfg.het_m = getattr(a, "het_m", 15.0)
        cfg.het_cell = getattr(a, "het_cell", 0.05)
        cfg.het_seed = getattr(a, "het_seed", 1)
        mir = getattr(a, "het_mirror", None)
        cfg.het_mirror = bool(int(mir)) if mir is not None else (not cfg.quarter)
        cfg.gc_mean = getattr(a, "gc_mean", 0.0)
        cfg.bc_T = getattr(a, "bc_T", "robin") or "robin"
        cfg.w_T = float(getattr(a, "w_T", 0.005))
        if cfg.bc_T in ("dirichlet", "dirichlet_net"):
            assert cfg.quarter, "--bc_T dirichlet and dirichlet_net need the quarter model"
        if cfg.bc_T == "dirichlet":
            cfg.T_analytic = True             # the field is T_lift alone (pfte.thermal.T_field)
        self._build_material(cfg)

    @staticmethod
    def _theta_1d(z, L, s):
        """Dimensionless temperature (T - T_env)/(T0 - T_env) of a layer 0 <= z <= L held at the
           bath at z = 0, insulated at z = L (the symmetry plane), uniform at t = 0: images of
           the semi-infinite erf solution, theta = 1 - sum_m (-1)^m [erfc((2mL + z)/2s) +
           erfc((2(m+1)L - z)/2s)], s = sqrt(kappa t). Four images are exact to round-off for
           s <= L/2 (t <= 0.2 s gives s = 1.33 mm against L = 5)."""
        out = torch.ones_like(z)
        for m in range(4):
            sg = 1.0 if m % 2 == 0 else -1.0
            out = out - sg * (torch.special.erfc((2 * m * L + z) / (2.0 * s))
                              + torch.special.erfc((2 * (m + 1) * L - z) / (2.0 * s)))
        return out

    def T_lift(self, x, y, cfg, t):
        """robin: the base lift (T0, 1), and the network carries the field. dirichlet: the exact
           quarter-slab field T_env + (T0 - T_env) theta(x; W) theta(y; H) with a zero envelope,
           so the lift is the whole field. With the bath on x = 0 and y = 0 and the cuts
           insulated, the 2D problem separates into the product of the two 1D layers."""
        bc = getattr(cfg, "bc_T", "robin")
        if bc == "dirichlet_net":
            # hard bath on x = 0 and y = 0: the envelope vanishes there, so T = T_env exactly. The
            # lift is T0 elsewhere, so the zero-initialized head starts the body uniform at T0
            # (step 0 needs no fit) and the network carries the cooling as a deviation from it
            xp, yp = to_phys(x, y, cfg)
            env = torch.tanh(xp.clamp(min=0.0) / cfg.w_T) * torch.tanh(yp.clamp(min=0.0) / cfg.w_T)
            return cfg.T_env + (cfg.T0 - cfg.T_env) * env, env
        if bc != "dirichlet":
            return super().T_lift(x, y, cfg, t)
        if t <= 0.0:
            return cfg.T0 * torch.ones_like(y), torch.zeros_like(y)
        xp, yp = to_phys(x, y, cfg)
        s = math.sqrt(cfg.kcond / cfg.rho_c * t)
        th = (self._theta_1d(xp.clamp(min=0.0), cfg.strip_w, s)
              * self._theta_1d(yp.clamp(min=0.0), cfg.strip_h, s))
        return cfg.T_env + (cfg.T0 - cfg.T_env) * th, torch.zeros_like(th)

    def T_reference(self, x, y, cfg, t):
        """The closed-form prescribed-surface field (the --bc_T dirichlet lift), for both Dirichlet
           variants. The solved field of dirichlet_net is compared against it. Robin: the lift."""
        if getattr(cfg, "bc_T", "robin") == "dirichlet_net":
            saved = cfg.bc_T
            cfg.bc_T = "dirichlet"
            try:
                return self.T_lift(x, y, cfg, t)[0]
            finally:
                cfg.bc_T = saved
        return self.T_lift(x, y, cfg, t)[0]

    def config_keys(self):
        return ("strip_w", "strip_h", "quarter", "sigma_t", "het_m", "het_cell", "het_seed",
                "het_mirror", "gc_mean", "w_env", "N_boundary", "gc_scale", "strength_implied",
                "bc_T", "w_T")

    def config_args(self, js):
        # strip_w / strip_h are the modeled part: one quarter of --strip when quarter is on
        if "strip_w" not in js:
            return {}
        f = 2.0 if js.get("quarter", True) else 1.0
        return dict(strip="%r,%r" % (f * js["strip_w"], f * js["strip_h"]))

    def schedules(self, cfg):
        """Step 0 is the zero-load state at t = 0 (the body still uniform at T0), followed by
           nsteps transient states on the power-law clock, all at zero mechanical load. Step 1
           spans the whole first window 0 -> t_start and builds the initial boundary layer,
           which is why it gets the --iters_T1 budget."""
        times = _robin_times(cfg)
        return [0.0] * len(times), times

    # ---------------------------------------------------------------- the Weibull material
    def _build_material(self, cfg):
        """One independent Weibull draw of the modulus and of the strength per meso-element:
           two scalar fields on the same grid, not one bivariate distribution. Lookup is by
           nearest cell, never interpolated, since smoothing the field would narrow the
           distribution it was drawn from.

           The draw is numpy's rng.weibull stream, E first then strength, unnormalized, so the
           same seed reproduces the same field bit for bit. With --het_mirror (the default on
           the full slab) one half-field is drawn and reflected about the mid-length."""
        ny = max(1, int(round(cfg.strip_h / cfg.het_cell)))
        if cfg.het_mirror:
            nx_half = max(1, int(round(0.5 * cfg.strip_w / cfg.het_cell)))
            Efield, Sfield = _weibull_fields(cfg.E, cfg.sigma_t, cfg.het_m, cfg.het_seed,
                                             ny, nx_half, mirror=True)
        else:
            nx = max(1, int(round(cfg.strip_w / cfg.het_cell)))
            Efield, Sfield = _weibull_fields(cfg.E, cfg.sigma_t, cfg.het_m, cfg.het_seed,
                                             ny, nx, mirror=False)
        nx = Efield.shape[1]
        m = cfg.het_m
        Gfield, cal = _calibrate_gc(cfg, Efield, Sfield)
        self._stats = (f"quench Weibull fields: shape=({ny},{nx}) "
                       f"cell~({cfg.strip_w/nx:.4f},{cfg.strip_h/ny:.4f})mm m={m:g} "
                       f"mirror={int(cfg.het_mirror)} seed={cfg.het_seed} | "
                       f"E mean/CV={Efield.mean()/1e3:.2f}GPa/{Efield.std()/Efield.mean():.3f} "
                       f"sigma_t mean/CV={Sfield.mean():.2f}MPa/{Sfield.std()/Sfield.mean():.3f} | "
                       f"AT{cfg.at} calib (cal={cal:.3f}) Gc mean={Gfield.mean():.5f}mJ/mm^2 "
                       f"gc_scale={cfg.gc_scale:.4f} implied strength~{cfg.strength_implied:.1f}MPa")
        dev = utils.DEVICE
        self._E = torch.as_tensor(Efield, dtype=DTYPE, device=dev)
        self._G = torch.as_tensor(Gfield, dtype=DTYPE, device=dev)
        self._nx, self._ny = nx, ny

    def validate(self, cfg, logger):
        logger.info(self._stats)
        assert cfg.at == 2 or cfg.at == 1
        if cfg.quarter:
            assert not cfg.het_mirror, "the quarter's symmetry planes already act as mirrors"
        gm = cfg.geomap
        if gm is not None and not gm.is_identity:
            import torch as _t
            _, _, _, _, det = gm.jac(_t.full((1, 1), 0.5, dtype=DTYPE, device=utils.DEVICE),
                                     _t.full((1, 1), 0.5, dtype=DTYPE, device=utils.DEVICE))
            want = cfg.strip_w * cfg.strip_h
            assert abs(float(det) - want) < 1e-6 * want, \
                f"strip map detJ {float(det)} != W*H {want}"

    def material(self, x, y, cfg):
        if x is None:
            return cfg.lam, cfg.mu, cfg.K, cfg.Gc
        ix = torch.clamp((x * self._nx).long(), 0, self._nx - 1).squeeze(-1)
        iy = torch.clamp((y * self._ny).long(), 0, self._ny - 1).squeeze(-1)
        E = self._E[iy, ix].unsqueeze(-1)
        Gc = self._G[iy, ix].unsqueeze(-1)
        nu = cfg.nu
        if cfg.plane_stress:
            lam = E * nu / (1.0 - nu ** 2)
            mu = E / (2.0 * (1.0 + nu))
            return lam, mu, lam + mu, Gc
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return lam, mu, lam + 2.0 * mu / 3.0, Gc

    # ---------------------------------------------------------------- fields / BCs
    def phi0(self, x, y, cfg):
        return torch.zeros_like(x)            # no pre-existing crack anywhere

    def fields(self, net, x, y, cfg, u_delta):
        """The slab carries no mechanical load. The full model floats free, and the rigid modes
           are removed by the pinning term in extra_loss. On the quarter model the two cuts are
           symmetry planes, which hard envelopes impose exactly and which also pin it."""
        out = net(x, y)
        if cfg.quarter:
            u = cfg.U_ref * ((1.0 - x) * out[:, 0:1])      # x=1 is the mid-length plane
            v = cfg.U_ref * ((1.0 - y) * out[:, 1:2])      # y=1 is the mid-height plane
        else:
            u = cfg.U_ref * out[:, 0:1]
            v = cfg.U_ref * out[:, 1:2]
        phi = torch.zeros_like(x) + phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, torch.zeros_like(x)

    def extra_loss(self, net, cfg, u_delta):
        """Pin the rigid modes of the free-floating full slab. The quarter model does not need it
           (its symmetry planes already remove them)."""
        if cfg.quarter or cfg.c_pin <= 0:
            return None
        g = torch.Generator(device=utils.DEVICE); g.manual_seed(1234)
        p = torch.rand(4096, 2, generator=g, device=utils.DEVICE, dtype=DTYPE)
        x = p[:, 0:1].requires_grad_(True); y = p[:, 1:2].requires_grad_(True)
        u, v, _, _ = self.fields(net, x, y, cfg, u_delta)
        ou = torch.ones_like(u)
        uy = torch.autograd.grad(u, y, ou, create_graph=True)[0]
        vx = torch.autograd.grad(v, x, ou, create_graph=True)[0]
        rot = 0.5 * (uy - vx)
        return cfg.c_pin * (u.mean() ** 2 + v.mean() ** 2 + rot.mean() ** 2)

    # ---------------------------------------------------------------- heat
    def _cooled_edges(self, cfg):
        """(axis, at_zero) for each cooled surface. The quarter model cools only the two real
           outer surfaces. The two cuts are symmetry planes and insulated."""
        if cfg.quarter:
            return [("y", True), ("x", True)]            # bottom and left
        return [("y", True), ("y", False), ("x", True), ("x", False)]

    def sample_boundary(self, sampler):
        """Points on each cooled surface for the convective term, with the physical length of
           the edge each set represents."""
        cfg = sampler.cfg
        edges = self._cooled_edges(cfg)
        n = max(64, cfg.N_boundary // len(edges))
        out = []
        for axis, at_zero in edges:
            s = torch.rand(n, 1, device=utils.DEVICE, dtype=DTYPE)
            c = torch.zeros_like(s) if at_zero else torch.ones_like(s)
            if axis == "y":
                out.append((s, c, cfg.strip_w))
            else:
                out.append((c, s, cfg.strip_h))
        return out

    def crack_stratum(self, crack, gx, gy, cfg):
        """With no pre-existing crack this stratum is the fixed cooled-edge band where the cracks
           start: per edge a Gaussian in the surface-normal coordinate (std 2 w_env / side),
           uniform along the edge, with the edges averaged. Each edge therefore gets the same
           share of points, and the corners get the sum of two bands. The growing cracks are
           sampled by the damage stratum, so `crack` (phi of the previous step) is not used
           here, which avoids counting the crack twice."""
        band = torch.zeros_like(gx)
        sig_y = 2.0 * cfg.w_env / cfg.strip_h            # parametric std, y-edges (bottom/top)
        sig_x = 2.0 * cfg.w_env / cfg.strip_w            # parametric std, x-edges (left/right)
        edges = self._cooled_edges(cfg)
        for axis, at_zero in edges:
            d = (gy if axis == "y" else gx)
            d = d if at_zero else (1.0 - d)
            s = sig_y if axis == "y" else sig_x
            band = band + torch.exp(-0.5 * (d / s) ** 2) / s
        return band / len(edges)


# --------------------------------------------------------------------------
# directional quenching of a glass strip (A. Yuse and M. Sano, "Transition between crack
# patterns in quenched glass plates", Nature 362 (1993) 329-331), set up as in Zeng and Dolbow
# (2026), "A complete phase-field fracture model for brittle materials subjected to thermal
# shocks", arXiv:2602.09031v2


class YuseSanoProblem(Problem):
    """A hot glass strip pulled at constant speed from an oven into a cold bath (Yuse and Sano
       1993). The crack follows the immersion front, and as the temperature jump grows its path
       goes from straight to a regular oscillation, to branching and to a chaotic pattern. The
       tip moves at the pulling speed (mm/s), far below the elastic wave speeds, so the problem
       is quasi-static.

       The temperature is the closed-form advection-diffusion profile of the moving front of
       Zeng and Dolbow (2026), imposed one-way: T = TH - dT exp(-(V/D)(x - Vt)) ahead of the
       front, TC behind it. There is no temperature network and there are no thermal solves
       (Problem.T_analytic).

       The control parameters are the Peclet number P = bV/D and the thermal-shock number
       R = alpha E dT sqrt(b)/Kc with Kc = sqrt(E Gc) (plane stress). At V = 0.4 mm/s Zeng and
       Dolbow (2026) report a straight crack for dT = 67-127 K and oscillation for
       dT = 147-167 K. Their model takes the strength sigma_ts = 80 MPa as an input. In AT2 the
       strength follows from sigma_c = sqrt(27 E Gc/(256 l)) ~ l^{-1/2}, so l also sets the
       strength: 80 MPa needs l ~ 0.0095 mm, and a larger l models a weaker glass."""

    name = "yusesano"
    default_geo = "strip:30,16"
    heat_on = True
    T_analytic = True

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--strip", type=str, default="30,16",
                        help="plate length,height in mm (height = 2b, with b = 8 in Zeng and "
                             "Dolbow 2026)")
        ap.add_argument("--dT", type=float, default=157.0,
                        help="temperature jump TH - TC across the front (K)")
        ap.add_argument("--Tc", type=float, default=23.0,
                        help="bath temperature (deg C, 23 deg C = 296 K). Only temperature "
                             "differences enter this example, so the offset from kelvin has no "
                             "effect")
        ap.add_argument("--Vpull", type=float, default=0.4, help="pulling speed (mm/s)")
        ap.add_argument("--Dtherm", type=float, default=0.47,
                        help="thermal diffusivity of the glass (mm^2/s)")
        ap.add_argument("--crack_len", type=float, default=5.0,
                        help="initial edge crack length at the leading (cold) edge (mm)")
        ap.add_argument("--crack_off", type=float, default=0.25,
                        help="y-offset of the initial crack above the midline (mm), the "
                             "symmetry-breaking perturbation of Zeng and Dolbow (2026)")
        ap.add_argument("--travel", type=float, default=24.0,
                        help="distance the immersion front travels over the run (mm), "
                             "and t_final = travel / Vpull")
        ap.add_argument("--gc_edge_sig", type=float, default=0.0,
                        help="strength (MPa) of a toughened band along the leading edge x = 0 "
                             "(0 = off). At t = 0 the full dT acts on this free edge at "
                             "once, and the AT2 strength at l = 0.1 (24.7 MPa) lets cracks "
                             "nucleate there. The band raises Gc so that "
                             "sqrt(27 E Gc / 256 l) equals the given value, for example 80 for "
                             "the glass of Zeng and Dolbow (2026)")
        ap.add_argument("--gc_edge_w", type=float, default=0.0,
                        help="width of that band (mm). 0 = the thermal boundary layer "
                             "D/Vpull (1.175 mm at Vpull 0.4 and Dtherm 0.47), the length over which the "
                             "start-up transient acts. Ignored unless --gc_edge_sig > 0")

    def defaults(self):
        # the glass of Zeng and Dolbow (2026), in N and mm. T0 (= TH) and dT_final are set in
        # configure from --dT
        return dict(E=70e3, nu=0.22, Gc=8.25e-3, l=0.1, at=2, split="hybrid",
                    stress_state="plane_stress", alpha=7.7e-6,
                    kcond=1.08, rho_si=2500.0, cp_si=921.0,
                    kdeg=0, grip_l=0.0, gc_grip=0.0, U_ref=1e-3, delta_max=0.0)

    def configure(self, cfg, a):
        w, h = (float(v) for v in str(getattr(a, "strip", "30,16")).split(","))
        cfg.strip_w, cfg.strip_h = w, h
        cfg.geo = f"strip:{w:g},{h:g}"
        cfg.grid_aspect = w / h
        cfg.dT_jump = float(getattr(a, "dT", 157.0))
        cfg.Tc = float(getattr(a, "Tc", 23.0))
        cfg.Vpull = float(getattr(a, "Vpull", 0.4))
        cfg.Dtherm = float(getattr(a, "Dtherm", 0.47))
        cfg.travel = float(getattr(a, "travel", 24.0))
        # the stress-free state is the uniform oven temperature, and the swing is a cooling by dT
        cfg.T0 = cfg.Tc + cfg.dT_jump
        cfg.dT_final = -cfg.dT_jump
        cfg.t_final = cfg.travel / cfg.Vpull
        yc = 0.5 * h + float(getattr(a, "crack_off", 0.25))
        Ln = float(getattr(a, "crack_len", 5.0))
        cfg.cracks = [(0.0, yc, Ln, yc)]
        c0 = cfg.cracks[0]
        cfg.nx0, cfg.nx1, cfg.ny = c0[0], c0[2], c0[1]
        # leading-edge band (see --gc_edge_sig). The toughening factor is derived from the
        # strength asked for and l. Off by default, which leaves cfg.gc_grip at 0 so that the
        # solver skips the toughening entirely.
        cfg.gc_edge_sig = float(getattr(a, "gc_edge_sig", 0.0))
        if cfg.gc_edge_sig > 0.0:
            sig_c = math.sqrt(27.0 * cfg.E * cfg.Gc / (256.0 * cfg.l))
            cfg.gc_grip = max(0.0, (cfg.gc_edge_sig / sig_c) ** 2 - 1.0)
            w = float(getattr(a, "gc_edge_w", 0.0))
            cfg.gc_edge_w = w if w > 0.0 else cfg.Dtherm / cfg.Vpull
        else:
            cfg.gc_edge_w = 0.0

    def grip_edges(self, xp, yp, cfg):
        """A toughened strip along the leading edge x = 0, folded into the --gc_grip bump.

           The edge x = 0 is traction-free (the displacement condition is on the right edge).
           The strip addresses the initial condition: T_lift places a fully developed steady
           front at x = 0 at t = 0, which drops the entire dT onto a free surface in one step.
           This start-up transient is not part of the physics under study. Off unless
           --gc_edge_sig is given, and the whole gc_grip branch is skipped when it is not.

           grip_l is 0 for this problem, so the corner discs of gc_bump have zero measure and
           the bump reduces to 1 - tanh(x/w): 1 on the edge, decaying over w."""
        w = getattr(cfg, "gc_edge_w", 0.0)
        if w <= 0.0:
            return 1.0
        return torch.tanh(torch.clamp(xp, min=0.0) / w)

    def config_keys(self):
        return ("strip_w", "strip_h", "dT_jump", "Tc", "Vpull", "Dtherm", "travel",
                "t_final", "cracks", "gc_edge_sig", "gc_edge_w")

    def config_args(self, js):
        if "dT_jump" not in js:
            return {}
        w, h = js["strip_w"], js["strip_h"]
        x0, yc, x1, _ = js["cracks"][0]
        # dT_final is reset to its command-line value 0. configure sets it from --dT, and Cfg
        # derives DTref = max(1, |dT_final|) before configure runs, so the rebuilt run gets the
        # same DTref = 1. DTref only scales a temperature network, which this example does not
        # have.
        return dict(strip="%r,%r" % (w, h), dT=js["dT_jump"], crack_len=x1 - x0,
                    crack_off=yc - 0.5 * h, dT_final=0.0)

    def schedules(self, cfg):
        """There is no mechanical load. The load parameter is the front position x_C = V t,
           advanced linearly since the front moves at constant speed. At step 0 (t = 0) the
           developed profile already sits at the leading edge, so step 0 is a full solve with
           the budget --iters0, not a zero-load start."""
        times = np.linspace(0.0, cfg.t_final, cfg.nsteps + 1)
        return [0.0] * (cfg.nsteps + 1), times.tolist()

    def phi0(self, x, y, cfg):
        return crack_profile(dist_to_notch(x, y, cfg), cfg)

    def fields(self, net, x, y, cfg, u_delta):
        """The strip is supported only enough to remove the rigid modes, as in Zeng and Dolbow
           (2026): u = 0 along the whole right (trailing, still hot) edge, imposed through the
           (1-x) envelope, and v = 0 at the center of that edge, imposed by subtracting the
           network's own value there.
           Everything else is traction-free, and the only load is the thermal strain."""
        out = net(x, y)
        u = cfg.U_ref * ((1.0 - x) * out[:, 0:1])
        pin_x = torch.ones(1, 1, dtype=x.dtype, device=x.device)
        pin_y = torch.full((1, 1), 0.5, dtype=x.dtype, device=x.device)
        v_pin = net(pin_x, pin_y)[:, 1:2]
        v = cfg.U_ref * (out[:, 1:2] - v_pin)
        phi0 = self.phi0(x, y, cfg)
        phi = phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, phi0

    # ---------------------------------------------------------------- the imposed temperature
    def T_lift(self, x, y, cfg, t):
        """The closed-form moving-front profile of Zeng and Dolbow (2026): TC behind the
           immersion front x_C = V t, an exponential relaxation of length D/V ahead of it, TH
           far ahead. The zero envelope makes this the whole field."""
        xp, _ = to_phys(x, y, cfg)
        s = torch.clamp(xp - cfg.Vpull * t, min=0.0)
        Td = (cfg.Tc + cfg.dT_jump) - cfg.dT_jump * torch.exp(-(cfg.Vpull / cfg.Dtherm) * s)
        return Td, torch.zeros_like(Td)

    def T_reference(self, x, y, cfg, t):
        return self.T_lift(x, y, cfg, t)[0]

    def validate(self, cfg, logger):
        assert cfg.plane_stress, "R uses Kc^2 = E Gc, which holds in plane stress"
        assert cfg.Vpull > 0 and cfg.Dtherm > 0 and cfg.dT_jump > 0
        b = 0.5 * cfg.strip_h
        P = b * cfg.Vpull / cfg.Dtherm
        assert P > 1.0, f"advection-dominated regime needs P = bV/D > 1 (got {P:.2f})"
        Kc = math.sqrt(cfg.E * cfg.Gc)                      # MPa sqrt(mm), plane stress
        R = cfg.alpha * cfg.E * cfg.dT_jump * math.sqrt(b) / Kc
        sig_c = math.sqrt(27.0 * cfg.E * cfg.Gc / (256.0 * cfg.l))
        # the dT regimes below are the V = 0.4 mm/s row (P = 6.81) of Zeng and Dolbow (2026).
        # The phase boundaries move with P, so they are quoted only at that speed
        ladder = (f"Zeng and Dolbow (2026) at V=0.4: straight dT 67-127 K | oscillation "
                  f"147-167 K | this run dT={cfg.dT_jump:g} K"
                  if abs(cfg.Vpull - 0.4) < 1e-9 else
                  f"dT={cfg.dT_jump:g} | the dT regimes of Zeng and Dolbow (2026) are for "
                  f"V=0.4 (P=6.81) and do not apply at P={P:.2f}, see their phase diagram")
        logger.info(f"  [validate] yusesano OK: P={P:.2f} R={R:.2f} ({ladder}) | "
                    f"AT2 sigma_c(l={cfg.l:g}) = {sig_c:.1f} MPa vs glass sigma_ts 80 MPa")
        if getattr(cfg, "gc_edge_sig", 0.0) > 0.0:
            got = sig_c * math.sqrt(1.0 + cfg.gc_grip)
            assert abs(got - cfg.gc_edge_sig) < 1e-6 * cfg.gc_edge_sig, \
                f"leading-edge strength came out {got:.3f} MPa, asked for {cfg.gc_edge_sig:.3f}"
            logger.info(f"  [validate] leading-edge band: x < ~{cfg.gc_edge_w:.3f} mm "
                        f"(= {cfg.gc_edge_w / (cfg.Dtherm / cfg.Vpull):.2f} thermal boundary "
                        f"layers D/V, = {cfg.gc_edge_w / cfg.l:.1f} l) toughened Gc x "
                        f"{1.0 + cfg.gc_grip:.3f} => sigma_c {sig_c:.1f} -> {got:.1f} MPa. "
                        f"The bulk is unchanged")


# --------------------------------------------------------------------------
# quenched ceramic annulus on an exact NURBS half or quarter annulus patch


class RingQuenchProblem(Problem):
    """A hot thin annular disk dropped into water. The rims are cooled by the bath, either by
       convection or at a prescribed temperature (--bc_T), the cooled surface layers go into
       tension, and a radial crack array
       nucleates from the Weibull meso-structure and arrests selectively. The physics is that
       of the quenched slab, on a curved domain that the patch represents exactly (rational
       quadratic 90-degree arcs, so the annulus is not faceted).

       The default geometry is the right half of the annulus, whose two radial edges lie on the
       x = 0 symmetry line. With --ring_quarter it is the upper-right quadrant, with the cuts on
       the two axes. Either way the cuts carry hard symmetry conditions and act as mirrors, as
       in the quarter slab. The Weibull field is drawn on the bounding box of the half
       annulus, and the modeled part looks up its cells by physical position."""

    name = "ringquench"
    default_geo = "ring"
    heat_on = True

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--ring_ri", type=float, default=2.5, help="inner radius (mm)")
        ap.add_argument("--ring_ro", type=float, default=10.0, help="outer radius (mm)")
        ap.add_argument("--cool", type=str, default="both", choices=["outer", "both"],
                        help="which rims exchange heat with the bath (immersion cools both)")
        ap.add_argument("--sigma_t", type=float, default=180.0,
                        help="nominal tensile strength of the Weibull field (MPa), drawn as in "
                             "Tang et al. (2016)")
        ap.add_argument("--het_m", type=float, default=15.0, help="Weibull modulus")
        ap.add_argument("--het_cell", type=float, default=0.05, help="meso-element size (mm)")
        ap.add_argument("--het_seed", type=int, default=1,
                        help="seed of the Weibull draw")
        ap.add_argument("--gc_mean", type=float, default=0.0,
                        help="rescale the calibrated Gc field to this mean (mJ/mm^2). 0 = off")
        ap.add_argument("--N_boundary", type=int, default=8000,
                        help="points per iteration on the cooled rims (arc-length weighted)")
        ap.add_argument("--w_env", type=float, default=0.15,
                        help="half the standard deviation of the cooled-rim sampling band (mm)")
        ap.add_argument("--bc_T", default="robin", choices=["robin", "dirichlet_net"],
                        help="robin = convection with hconv on the cooled rims. "
                             "dirichlet_net = the bath temperature prescribed on them, imposed "
                             "exactly through the lift (a tanh envelope in the distance to each "
                             "cooled rim), with the field solved by the temperature network, as "
                             "--bc_T dirichlet_net of the quenched slab")
        ap.add_argument("--w_T", type=float, default=0.005,
                        help="width (mm) of the lift envelope tanh(d / w_T) of --bc_T "
                             "dirichlet_net")
        ap.add_argument("--ring_quarter", action="store_true",
                        help="model the upper-right quadrant instead of the right half. Both "
                             "cuts are symmetry planes either way, so the physics is unchanged. "
                             "The swept arc halves, which halves the angular point count needed "
                             "at a given h/l. The radial count is unchanged, since the wall "
                             "thickness is the same")

    def defaults(self):
        # problem-level defaults: the ceramic and clock of the quenched slab, except l.
        # examples/annulus.py sets the paper case.
        return dict(E=370e3, nu=0.3, Gc=0.04247, l=0.2, at=2, split="hybrid",
                    stress_state="plane_stress", alpha=7.5e-6, kcond=31.0,
                    rho_si=3980.0, cp_si=880.0, hconv=50000.0 * 1e-3,
                    T0=873.0, T_env=293.0, dT_final=-580.0, kdeg=0,
                    t_start=1e-4, t_final=0.2, time_power=3.0,
                    grip_l=0.0, gc_grip=0.0, U_ref=1e-3, delta_max=0.0,
                    w_unif=0.3, w_notch=0.3, w_damage=0.4)

    def configure(self, cfg, a):
        cfg.ring_ri = float(getattr(a, "ring_ri", 2.5))
        cfg.ring_ro = float(getattr(a, "ring_ro", 10.0))
        cfg.cool = getattr(a, "cool", "both")
        cfg.ring_quarter = bool(getattr(a, "ring_quarter", False))
        cfg.geo = "ring_quarter" if cfg.ring_quarter else "ring"
        # parametric xi = radial (short), eta = angular (long): give the eta axis the finer
        # grids by declaring the aspect radial-span / mean-arc (< 1). The quarter sweeps half
        # the angle, so its mean arc halves and the same levels give twice the h/l resolution.
        span = cfg.ring_ro - cfg.ring_ri
        arc = math.pi * (0.25 if cfg.ring_quarter else 0.5) * (cfg.ring_ri + cfg.ring_ro)
        cfg.grid_aspect = span / arc
        cfg.w_env = float(getattr(a, "w_env", 0.15))
        cfg.N_boundary = int(getattr(a, "N_boundary", 8000))
        cfg.cracks = []
        cfg.nx0 = cfg.nx1 = cfg.ny = 0.0
        cfg.sigma_t = float(getattr(a, "sigma_t", 180.0))
        cfg.het_m = float(getattr(a, "het_m", 15.0))
        cfg.het_cell = float(getattr(a, "het_cell", 0.05))
        cfg.het_seed = int(getattr(a, "het_seed", 1))
        cfg.gc_mean = float(getattr(a, "gc_mean", 0.0))
        cfg.bc_T = getattr(a, "bc_T", "robin") or "robin"
        cfg.w_T = float(getattr(a, "w_T", 0.005))
        self._build_material(cfg)

    def config_keys(self):
        return ("ring_ri", "ring_ro", "ring_quarter", "cool", "sigma_t", "het_m", "het_cell",
                "het_seed", "gc_mean", "w_env", "N_boundary", "gc_scale", "strength_implied",
                "bc_T", "w_T")

    def build_geomap(self, cfg, logger=None):
        spec = "ring_quarter" if getattr(cfg, "ring_quarter", False) else "ring"
        return build_geomap(spec, logger, ring=(cfg.ring_ri, cfg.ring_ro))

    def schedules(self, cfg):
        times = _robin_times(cfg)
        return [0.0] * len(times), times

    # ---------------------------------------------------------------- the Weibull material
    def _build_material(self, cfg):
        """The Weibull meso-structure of Tang et al. (2016) on the annulus: cells tile the
           physical bounding box [0, Ro] x [-Ro, Ro] of the half-annulus, with nearest-cell
           lookup by physical position, never interpolated. Cells outside the modeled part are
           never queried."""
        ny = max(1, int(round(2.0 * cfg.ring_ro / cfg.het_cell)))
        nx = max(1, int(round(cfg.ring_ro / cfg.het_cell)))
        Efield, Sfield = _weibull_fields(cfg.E, cfg.sigma_t, cfg.het_m, cfg.het_seed,
                                         ny, nx, mirror=False)
        Gfield, cal = _calibrate_gc(cfg, Efield, Sfield)
        self._stats = (f"ringquench Weibull fields: box [0,{cfg.ring_ro:g}]x"
                       f"[-{cfg.ring_ro:g},{cfg.ring_ro:g}] shape=({ny},{nx}) "
                       f"cell={cfg.het_cell:g}mm m={cfg.het_m:g} seed={cfg.het_seed} | "
                       f"E mean/CV={Efield.mean()/1e3:.2f}GPa/{Efield.std()/Efield.mean():.3f} "
                       f"sigma_t mean={Sfield.mean():.2f}MPa | AT{cfg.at} (cal={cal:.3f}) "
                       f"Gc mean={Gfield.mean():.5f}mJ/mm^2 gc_scale={cfg.gc_scale:.4f} "
                       f"implied strength~{cfg.strength_implied:.1f}MPa")
        dev = utils.DEVICE
        self._E = torch.as_tensor(Efield, dtype=DTYPE, device=dev)
        self._G = torch.as_tensor(Gfield, dtype=DTYPE, device=dev)
        self._nx, self._ny = nx, ny

    def material(self, x, y, cfg):
        if x is None:
            return cfg.lam, cfg.mu, cfg.K, cfg.Gc
        xp, yp = to_phys(x, y, cfg)
        Ro = cfg.ring_ro
        ix = torch.clamp((xp / Ro * self._nx).long(), 0, self._nx - 1).squeeze(-1)
        iy = torch.clamp(((yp + Ro) / (2.0 * Ro) * self._ny).long(), 0, self._ny - 1).squeeze(-1)
        E = self._E[iy, ix].unsqueeze(-1)
        Gc = self._G[iy, ix].unsqueeze(-1)
        nu = cfg.nu
        mu = E / (2.0 * (1.0 + nu))
        if cfg.plane_stress:
            lam = E * nu / (1.0 - nu ** 2)
            return lam, mu, lam + mu, Gc
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu, lam + 2.0 * mu / 3.0, Gc

    def validate(self, cfg, logger):
        logger.info(self._stats)
        # this example builds its own annulus (build_geomap ignores --geo), so refuse a --geo
        # that would be recorded in config.json while a different map was actually used
        quarter = bool(getattr(cfg, "ring_quarter", False))
        assert cfg.geo == ("ring_quarter" if quarter else "ring"), \
            f"ringquench builds its own annulus patch, so --geo {cfg.geo!r} would be recorded " \
            "but not used"
        _assert_ring_radii(cfg)
        gm = cfg.geomap
        span = cfg.ring_ro - cfg.ring_ri
        arc_out = math.pi * (0.5 if quarter else 1.0) * cfg.ring_ro
        n_eta = cfg.levels[-1]
        n_xi = max(3, int(round(cfg.grid_aspect * n_eta)))
        logger.info(f"  [validate] ringquench: {'quarter' if quarter else 'half'}-annulus "
                    f"Ri={cfg.ring_ri:g} Ro={cfg.ring_ro:g}"
                    f" | finest grids (eta {n_eta}, xi {n_xi}) => outer-arc h~"
                    f"{arc_out/(n_eta-2):.4f}mm ({arc_out/(n_eta-2)/cfg.l:.2f}l), radial h~"
                    f"{span/(n_xi-2):.4f}mm ({span/(n_xi-2)/cfg.l:.2f}l) | cool={cfg.cool}")
        assert gm is not None and not gm.is_identity, "ringquench needs the annulus patch"

    # ---------------------------------------------------------------- fields / BCs
    def phi0(self, x, y, cfg):
        return torch.zeros_like(x)

    def fields(self, net, x, y, cfg, u_delta):
        """Symmetry cuts imposed hard. Which cut is where depends on the patch.

           Half (eta = 0 and eta = 1 both lie on the x = 0 line): u = 0 there, via the
           eta(1-eta) envelope. v is free, and the one remaining rigid mode (y-translation) is
           pinned in extra_loss.

           Quarter (eta = 0 is the +x axis, eta = 1 is the +y axis): each cut removes the
           displacement normal to itself, so v = 0 at eta = 0 (envelope eta) and u = 0 at
           eta = 1 (envelope 1-eta). That removes every rigid mode on its own. Translation in x
           violates u = 0 on the x = 0 cut, translation in y violates v = 0 on the y = 0 cut,
           and a rotation (u = -w y, v = w x) violates both. extra_loss therefore does not pin
           the mean of v, which is a genuine degree of freedom here.

           There is no mechanical load in either case."""
        out = net(x, y)
        if getattr(cfg, "ring_quarter", False):
            u = cfg.U_ref * ((1.0 - y) * out[:, 0:1])
            v = cfg.U_ref * (y * out[:, 1:2])
        else:
            u = cfg.U_ref * ((y * (1.0 - y)) * out[:, 0:1])
            v = cfg.U_ref * out[:, 1:2]
        phi = phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, torch.zeros_like(x)

    def extra_loss(self, net, cfg, u_delta):
        # the quarter's two cuts already remove every rigid mode (see fields)
        if cfg.c_pin <= 0 or getattr(cfg, "ring_quarter", False):
            return None
        g = torch.Generator(device=utils.DEVICE); g.manual_seed(1234)
        p = torch.rand(4096, 2, generator=g, device=utils.DEVICE, dtype=DTYPE)
        _, v, _, _ = self.fields(net, p[:, 0:1], p[:, 1:2], cfg, u_delta)
        return cfg.c_pin * v.mean() ** 2

    # ---------------------------------------------------------------- heat
    def T_lift(self, x, y, cfg, t):
        """robin: the base lift (T0, 1), and the network carries the field. dirichlet_net: the
           bath temperature imposed exactly on every cooled rim. The envelope tanh(d / w_T), with
           d the radial distance to the rim, vanishes there, so T = T_env exactly and the
           convective term evaluated on the rims is identically zero. Elsewhere the lift is T0,
           so the zero-initialized head starts the body uniform at T0 and the network carries
           the cooling, as for the slab."""
        if getattr(cfg, "bc_T", "robin") != "dirichlet_net":
            return super().T_lift(x, y, cfg, t)
        xp, yp = to_phys(x, y, cfg)
        r = torch.sqrt(xp * xp + yp * yp)
        env = torch.tanh((cfg.ring_ro - r).clamp(min=0.0) / cfg.w_T)
        if cfg.cool == "both":
            env = env * torch.tanh((r - cfg.ring_ri).clamp(min=0.0) / cfg.w_T)
        return cfg.T_env + (cfg.T0 - cfg.T_env) * env, env

    def _cooled_edges(self, cfg):
        return [1.0] if cfg.cool == "outer" else [0.0, 1.0]     # xi of each cooled rim

    def sample_boundary(self, sampler):
        """Points on the cooled rims with per-point arc-length weights ds/deta, since the
           rational circle parametrization is not arc-uniform."""
        cfg = sampler.cfg
        rims = self._cooled_edges(cfg)
        n = max(64, cfg.N_boundary // len(rims))
        out = []
        for xi_v in rims:
            eta = torch.rand(n, 1, device=utils.DEVICE, dtype=DTYPE)
            xi = torch.full_like(eta, xi_v)
            _, b, _, d, _ = cfg.geomap.jac(xi, eta)
            L = torch.sqrt(b * b + d * d + 1e-30)               # ds/deta on this rim
            out.append((xi, eta, L))
        return out

    def crack_stratum(self, crack, gx, gy, cfg):
        """The fixed nucleation band: a Gaussian in the radial parametric coordinate at each
           cooled rim (std 2 w_env / wall thickness), uniform along the arc. The |detJ| cell
           weight then makes it physically uniform along the non-uniformly parametrized
           circumference. The growing cracks are sampled by the damage stratum."""
        band = torch.zeros_like(gx)
        sig = 2.0 * cfg.w_env / (cfg.ring_ro - cfg.ring_ri)
        for xi_v in self._cooled_edges(cfg):
            d = (gx - xi_v).abs()
            band = band + torch.exp(-0.5 * (d / sig) ** 2) / sig
        return band / len(self._cooled_edges(cfg))


REGISTRY["quench"] = QuenchProblem()
REGISTRY["yusesano"] = YuseSanoProblem()
REGISTRY["ringquench"] = RingQuenchProblem()
