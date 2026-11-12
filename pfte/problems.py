"""
The Problem interface and the single-edge-notched plate.

The solver core is identical for every example in the paper. What changes from one to the next
is a short list: the geometry patch, where the pre-existing cracks are, how the displacement
boundary conditions are imposed, which part of the parametric square is inside the body, how
the temperature enters, and whatever extra reporting that example needs. A Problem packages
exactly those. The quench examples are in pfte.problems_thermal.

To add an example, subclass Problem, override the few hooks that differ, add it to REGISTRY, and
write a short entry script under examples/.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import math

import torch

from .geometry import (SEN_NOTCH, build_geomap, crack_profile, dist_to_notch, parse_cracks,
                       phi_gate)
from .solver import phi_squash


# --------------------------------------------------------------------------
# the interface


class Problem:
    """Base class. The defaults describe a unit-square domain on a B-spline patch with no
       extra loss terms and no example-specific reporting."""

    name = "base"
    #: default value of --geo for this problem
    default_geo = "identity"

    # ------------------------------------------------------------------ CLI / config
    @staticmethod
    def add_arguments(ap):
        """Add this example's own command-line flags to an argparse parser."""

    def defaults(self):
        """Shared flags whose defaults this example shifts (geometry, material, step size)."""
        return {}

    def configure(self, cfg, a):
        """Copy this example's parsed flags onto cfg. Called at the end of Cfg.__init__."""

    def config_keys(self):
        """Names of the problem-specific cfg fields to record in config.json."""
        return ()

    def config_args(self, js):
        """The command-line flags that `configure` reads, recovered from a saved config.json.

           config.json keeps what configure derives (dT_jump, strip_w, cracks), not the flags it
           reads (--dT, --strip, --crack_off). Without this a rebuilt run would get the flags'
           defaults. Return {flag: value}. cfg_from_json sets them before Cfg is built."""
        return {}

    # ------------------------------------------------------------------ geometry
    def build_geomap(self, cfg, logger=None):
        """Return the GeoMap for this example (None = no map, as for --geo none)."""
        return build_geomap(cfg.geo, logger)

    def phi0(self, x, y, cfg):
        """The crack seed: phi0 in [0,1] at the (parametric) query points, 1 on a pre-existing
           crack and decaying over the length scale l. Examples without a pre-existing crack
           (the quenched slab and annulus) return zeros."""
        raise NotImplementedError

    # ------------------------------------------------------------------ fields / BCs
    def fields(self, net, x, y, cfg, u_delta):
        """Map the raw network outputs to (u, v, phi, phi0) with the displacement boundary
           conditions imposed exactly (hard constraints), so no BC loss term is needed."""
        raise NotImplementedError

    # ------------------------------------------------------------------ domain
    def in_domain(self, x, y, cfg):
        """Indicator of the physical body on the parametric square: 1.0 inside, 0.0 outside.
           Only an example whose body does not fill the parametric square overrides this."""
        c = cfg
        return ((x >= c.x0) & (x <= c.x1) & (y >= c.y0) & (y <= c.y1))

    def domain_aspect(self, cfg):
        """Physical width/height of the body's bounding box. Figures are sized to it and the
           render grid gets proportionally more points along the long axis, so an elongated body
           resolves (and prints) isotropically instead of aliasing its long-axis features. The
           default reads cfg.grid_aspect, which an example sets in configure(). 1.0 throughout
           reproduces the square-figure layout exactly."""
        return float(getattr(cfg, "grid_aspect", 1.0) or 1.0)

    def sample_uniform(self, sampler, n):
        """Draw n points for the uniform (bulk) stratum. Override when part of the parametric
           square is outside the body, so that the rejected points are redrawn."""
        return None      # None = use the sampler's own default (Sobol / |det J|-weighted grid)

    def prepare_sampler(self, sampler):
        """Called once per Sampler, after the |det J| cell weights are built and before any
           stratum uses them. This is where an example whose body does not fill the parametric
           square removes cells from every stratum at once (and rescales sampler.dom_area, the
           body's share of the parametric square)."""

    def crack_stratum(self, crack, gx, gy, cfg):
        """Adjust the grid that decides where the crack stratum puts its points. It normally
           follows max(phi0, phi_prev). An example with no pre-existing crack has to say where
           a crack is expected instead, or the stratum would have nothing to track."""
        return crack

    # ------------------------------------------------------------------ boundary-zone toughening
    def grip_points(self, cfg):
        """Physical points the --gc_grip toughening protects, or None (the default) to use the
           patch's own clamped corners. These are the places where a prescribed-displacement edge
           meets a free flank, an idealized-boundary singularity that nucleates damage as an
           artifact of the boundary condition rather than of the specimen. An example whose
           loaded edges meet its flanks somewhere other than the patch corners lists them here.
           A geometric corner of the specimen itself does not belong in this list."""
        return None

    def grip_edges(self, xp, yp, cfg):
        """An extra factor folded into the toughening bump, for an example that needs a toughened
           strip along an edge and not only discs at the corners. 1.0 (the default) leaves the
           bump as the corner discs make it."""
        return 1.0

    def gc_extra(self, x, y, cfg):
        """A second, independent multiplicative toughening factor on Gc, on top of the grip
           bump, for a zone the grip toughening does not cover. Only consulted when the example
           sets cfg.gc_extra_on, so the default run never calls it."""
        return 1.0

    # ------------------------------------------------------------------ material
    def material(self, x, y, cfg):
        """(lambda, mu, K, Gc) at the query points. The default is the homogeneous body: four
           scalars off cfg, which every split then uses as constants. An example whose material
           is graded or randomly heterogeneous returns tensors shaped like x instead. The code
           downstream accepts either."""
        return cfg.lam, cfg.mu, cfg.K, cfg.Gc

    # ------------------------------------------------------------------ temperature
    #: does this example carry a temperature field at all? False leaves every thermal code path
    #: switched off, and the run is the plain isothermal one.
    heat_on = False

    #: True = the temperature problem is steady and linear in its boundary data. It is solved
    #: once at full load (budget --iters_T0) and the thermal strain then scales with the step's
    #: load fraction, with no per-step thermal solves. An optional hook, not used by the shipped
    #: examples.
    T_steady = False

    #: True = the temperature is prescribed in closed form: T_lift(t) is the field (one-way
    #: coupling), so no temperature network is built and no thermal solves run at all. The
    #: mechanics reads the analytic thermal strain directly, step by step.
    T_analytic = False

    def schedules(self, cfg):
        """(u_delta_nd list, times list), one entry per step. The run walks the two lists
           together. The default is the isothermal convention: nsteps displacement increments
           starting at delta/nsteps (no trivial zero step), paired with cfg.time_schedule() when
           there is a temperature field. A thermo-mechanical example that starts from a
           zero-load state at step 0 and then applies the load overrides this."""
        uds = cfg.schedule()
        times = cfg.time_schedule() if cfg.heat_on else [0.0] * len(uds)
        return uds, times

    def validate(self, cfg, logger):
        """Fail-fast checks after the geometry map is built, before any training: assert the
           pieces the example depends on (map, notch, material) are what it thinks they are.
           Raise on a hard failure, so that a misconfigured run stops before training."""
        return None

    def T_lift(self, x, y, cfg, t):
        """(T_D, E): the hard temperature lift and its envelope, at time t. T_D carries whatever
           temperature is prescribed on the boundary and E vanishes there, so that Dirichlet
           condition is satisfied exactly by construction. An example driven only through natural
           conditions (a convective surface) prescribes nothing and returns (T0, 1), which
           starts the body at the uniform temperature T0."""
        return cfg.T0 * torch.ones_like(y), torch.ones_like(y)

    def T_reference(self, x, y, cfg, t):
        """The temperature the step-0 fit aims at: an analytic early-time solution where the
           example has one, otherwise the uniform initial temperature."""
        return cfg.T0 * torch.ones_like(y)

    def thermal_material(self, x, y, cfg):
        """(conductivity, volumetric heat capacity) at the query points. Scalars for a
           homogeneous body, and fields for a graded one."""
        return cfg.kcond, cfg.rho_c

    def heat_source(self, x, y, cfg, t):
        """Volumetric heat source at the query points, or None for no source (the default)."""
        return None

    def sample_boundary(self, sampler):
        """Points for the boundary term of the heat functional, as [(x, y, length), ...]: the
           parametric coordinates of a boundary sample and its length weight, a scalar physical
           edge length for a straight edge, or per-point weights ds/dparam for a curved edge.
           Empty (the default) means every surface is insulated."""
        return []

    # ------------------------------------------------------------------ extra physics
    def reaction_force(self, net, cfg, prev_model, u_delta_nd, seed=999):
        """Override when the load does not ride the displacement ansatz, so that differentiating
           the elastic energy with respect to delta would miss it. Return None (the default) to
           use the energy-consistent reaction in pfte.solver.reaction_force."""
        return None

    def extra_loss(self, net, cfg, u_delta):
        """Optional additional term added to the per-iteration objective, e.g. the term that pins
           the rigid-body modes of a free-floating body (the full quenched slab). Return None
           when the example has none (the default)."""
        return None

    # ------------------------------------------------------------------ reporting
    def render_mask(self, cfg, X, Y):
        """Boolean array marking render pixels outside the body, which are drawn blank. Return
           None (the default) when the body fills the domain."""
        return None

    def overlay(self, cfg, ax=None):
        """Draw the body's own outline (a free surface) onto a field figure."""

    def plot_domain(self, cfg, save_dir):
        """One geometry figure per run, written next to the shared patch/parametric views."""

    def setup(self, net, cfg, save_dir, logger):
        """Called once after the network is built and before load stepping. Return True to end
           the run there. The default returns False."""
        return False

    def plot(self, net, cfg, save_dir, tag, ud):
        """Example-specific figures, called wherever the shared plots are written."""

    def finalize(self, net, cfg, save_dir, logger):
        """Called once after the last load step (summary figures)."""


# --------------------------------------------------------------------------
# single-edge-notched plate


# Graded end values at x = 1, as in the FEniCS code released with Tangella et al. (2022). The
# x = 0 side is the base ceramic set by --E/--Gc/--kcond/--rho_si/--cp_si. nu and alpha are
# constant across the plate. cp at x = 1 is fixed at 0.775, so pass --cp_si 0.775 to keep it
# constant.
FGM_E2 = 450e3                        # MPa
FGM_GC2 = 0.120                       # mJ/mm^2
FGM_K2 = 350.0                        # mJ/(s mm K)
# rho = 4000 kg/m^3 times c = 0.775, with c and the time scale divided by 1e3 as in the SEN
# examples (cp_si = 0.775)
FGM_RHOCP2 = 4000.0 * 0.775 * 1e-6


class SENProblem(Problem):
    name = "sen"
    default_geo = "identity"

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--loading", type=str, default="shear", choices=["shear", "tension"],
                        help="shear = mode II, the imposed displacement is horizontal "
                             "| tension = mode I, the imposed displacement is vertical")
        ap.add_argument("--cracks", type=str, default=SEN_NOTCH,
                        help="physical crack segments 'ax,ay,bx,by;...'. Default = the single "
                             "edge notch (0,0.5)-(0.5,0.5)")
        ap.add_argument("--heat_on", action="store_true",
                        help="also solve a temperature field: the top edge is ramped to "
                             "T0 + dT_final while the bottom is held at T0, and the thermal strain "
                             "loads the plate alongside the imposed displacement (the "
                             "thermo-mechanical single-edge-notched cases of Tangella et al. "
                             "2022)")
        ap.add_argument("--ramp_rate", type=float, default=1e8,
                        help="rate of the top-edge temperature change (K per unit of the clock "
                             "t), held once it reaches T0 + dT_final. The change per load step is "
                             "ramp_rate * time_per_delta * (displacement increment): 2.5 K per "
                             "step at 1e8 and time_per_delta 1e-3 with 2.5e-5 mm steps")
        ap.add_argument("--time_per_delta", type=float, default=1e-3,
                        help="seconds of transient per mm of imposed displacement, "
                             "t = time_per_delta * delta. With the heat capacity "
                             "entered divided by 1e3 (cp_si = 0.775 in the examples), the default "
                             "1e-3 is the clock t = 1 s/mm x delta of the paper")
        ap.add_argument("--fgm", action="store_true",
                        help="functionally graded plate: E, Gc, conductivity and heat capacity "
                             "run linearly in x from the base ceramic at x = 0 to the values of "
                             "Tangella et al. (2022) at x = 1: E = 450 GPa, Gc = 120 J/m^2, "
                             "k = 350 W/(m K), rho = 4000 kg/m^3. nu and alpha stay constant. "
                             "The x = 1 heat capacity uses c = 0.775, so pass --cp_si 0.775 (as "
                             "the example does) to keep c constant")

    def configure(self, cfg, a):
        assert cfg.loading in ("shear", "tension"), \
            f"the SEN plate is loaded in shear or tension, got {cfg.loading!r}"
        raw = getattr(a, "cracks", SEN_NOTCH)
        cfg.cracks = raw if isinstance(raw, list) else parse_cracks(raw)
        # single-notch fields read by the plot and notch code: the extent of the first crack
        c0 = cfg.cracks[0]
        cfg.nx0, cfg.nx1, cfg.ny = min(c0[0], c0[2]), max(c0[0], c0[2]), 0.5 * (c0[1] + c0[3])
        cfg.ramp_rate = getattr(a, "ramp_rate", 1e8)
        cfg.time_per_delta = getattr(a, "time_per_delta", 1e-3)
        cfg.fgm = bool(getattr(a, "fgm", False))

    def schedules(self, cfg):
        """Heated: a zero-load state at step 0, then the increments, with the physical time
           tied to the displacement (t = time_per_delta * delta), so the edge ramp, the
           conduction depth and the load advance together. Isothermal: the plain convention."""
        if not cfg.heat_on:
            return super().schedules(cfg)
        uds = [0.0] + cfg.schedule()
        times = [cfg.time_per_delta * cfg.U_ref * ud for ud in uds]
        return uds, times

    # ---------------------------------------------------------------- temperature (optional)
    def T_lift(self, x, y, cfg, t):
        """Top edge ramped, bottom edge held at T0, both as hard Dirichlet conditions: the lift
           interpolates linearly between them and the envelope y(1-y) vanishes on both, so the
           network only adds the interior deviation from that line."""
        dT = cfg.dT_final
        if dT != 0.0:
            dT = math.copysign(min(cfg.ramp_rate * t, abs(dT)), dT)
        return cfg.T0 + dT * y, y * (1.0 - y)

    def T_reference(self, x, y, cfg, t):
        """the step-0 target: the linear profile between the two held edges"""
        return self.T_lift(x, y, cfg, t)[0]

    def config_keys(self):
        return ("cracks", "ramp_rate", "time_per_delta", "fgm")

    # ---------------------------------------------------------------- FGM material (optional)
    def material(self, x, y, cfg):
        """Homogeneous by default. With --fgm, the moduli and toughness run linearly in the
           physical x (clamped to the plate), so the same stiffness that grades the elastic
           response also grades the thermal-stress source term."""
        if not getattr(cfg, "fgm", False) or x is None:
            return cfg.lam, cfg.mu, cfg.K, cfg.Gc
        from .geometry import to_phys
        xp, _ = to_phys(x, y, cfg)
        w = torch.clamp(xp, 0.0, 1.0)
        E = cfg.E + (FGM_E2 - cfg.E) * w
        nu = cfg.nu
        mu = E / (2.0 * (1.0 + nu))
        Gc = cfg.Gc + (FGM_GC2 - cfg.Gc) * w
        if cfg.plane_stress:
            lam = E * nu / (1.0 - nu ** 2)
            return lam, mu, lam + mu, Gc
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu, lam + 2.0 * mu / 3.0, Gc

    def thermal_material(self, x, y, cfg):
        if not getattr(cfg, "fgm", False):
            return cfg.kcond, cfg.rho_c
        from .geometry import to_phys
        xp, _ = to_phys(x, y, cfg)
        w = torch.clamp(xp, 0.0, 1.0)
        k = cfg.kcond + (FGM_K2 - cfg.kcond) * w
        rho_c = cfg.rho_c + (FGM_RHOCP2 - cfg.rho_c) * w
        return k, rho_c

    def phi0(self, x, y, cfg):
        return crack_profile(dist_to_notch(x, y, cfg), cfg)

    def fields(self, net, x, y, cfg, u_delta):
        """The network sees (x, y) and the encoding. Hard BCs via env = y(1-y), bottom edge
           fixed. Shear (mode II): delta rides u (top u=delta, v=0). Tension (mode I): delta
           rides v (top v=delta, u=0)."""
        out = net(x, y)
        env = y * (1.0 - y)
        if cfg.loading == "shear":
            u = cfg.U_ref * (env * out[:, 0:1] + y * u_delta)
            v = cfg.U_ref * (env * out[:, 1:2])
        else:                                                   # tension (mode I): delta rides v
            u = cfg.U_ref * (env * out[:, 0:1])
            v = cfg.U_ref * (env * out[:, 1:2] + y * u_delta)
        phi0 = self.phi0(x, y, cfg)
        phi = phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, phi0


SEN = SENProblem()


# --------------------------------------------------------------------------
# registry


#: every example the package ships, by the name that appears in --problem and config.json
REGISTRY = {
    "sen": SEN,        # single-edge-notched square in tension or shear
}
