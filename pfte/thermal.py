"""
The temperature field: its network, its variational problem, and the strain it imposes.

Within each step the temperature is solved first, then the displacement and damage at that
temperature. The heat block is the minimization of one functional, and the two blocks are solved
in sequence rather than as one monolithic system. This is why the thermal strain below is
detached: the mechanical energy treats a_th as data and is never differentiated through it.

The temperature has its own multiresolution encoding, separate from the mechanical one, because
it needs the resolution to represent a near-discontinuity. An adiabatic crack (conductivity
degraded to zero across the damage band) makes a steep change of temperature across the band
energetically free, so the minimizer places a near-discontinuity in T there. A plain MLP without
the encoding resolves it poorly. The temperature network receives no crack information directly.
The crack reaches it only through the degraded conductivity, and the network locates the
discontinuity itself. This rationale applies with --kdeg 1 (the heated notched plates). The quench
examples keep the intact conductivity (the slab and annulus set kdeg 0).

The heat functional is the backward-Euler incremental potential

    Pi_T = int [ 1/2 k_d |grad T|^2 + rho_c (T - T_prev)^2 / (2 dt) - gamma T ] dV
         + int_boundary 1/2 h (T - T_env)^2 dS

whose stationarity condition is rho_c dT/dt = div(k_d grad T) + gamma with a convective boundary
condition. The source gamma (Problem.heat_source), the boundary samples (Problem.sample_boundary)
and the hard temperature lift (Problem.T_lift) come from the example, so a convective and a
prescribed-temperature quench share this code.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import time

import numpy as np
import torch
import torch.nn as nn

from . import utils
from .solver import MultiResEncoding, phi_value
from .utils import DTYPE, problem_of


# --------------------------------------------------------------------------
# the temperature network


class NetT(nn.Module):
    """The temperature field on its own multiresolution encoding, with parameters and feature
       grids separate from the mechanical network. There is no phi input: the crack enters only
       through the degraded conductivity in the heat functional."""

    def __init__(self, cfg):
        super().__init__()
        self.enc = MultiResEncoding(cfg, levels=cfg.levels_T, ch=cfg.enc_ch_T)
        L = [nn.Linear(2 + self.enc.n_feat, cfg.width_T), nn.GELU()]
        for _ in range(cfg.depth_T - 1):
            L += [nn.Linear(cfg.width_T, cfg.width_T), nn.GELU()]
        last = nn.Linear(cfg.width_T, 1)
        L += [last]
        self.mlp = nn.Sequential(*L)
        with torch.no_grad():
            last.bias[0] = cfg.T_head_bias
            if cfg.T_zero_init:
                # the head weights start at zero, so with --T_head_bias 0 (the default) T equals
                # the lift T_D (uniform T0 for an example driven only by natural conditions) at
                # the first step, otherwise the first step begins from a spurious temperature
                # field.
                last.weight.zero_()

    def forward(self, x, y):
        return self.mlp(torch.cat([x, y, self.enc(x, y)], dim=1))


# --------------------------------------------------------------------------
# the field, and the strain it imposes


def T_field(netT, x, y, cfg, t):
    """Temperature (K) at the parametric query points at time t.

       T = T_D + E * DTref * netT, with (T_D, E) the example's hard temperature lift and its
       envelope. T_D carries any prescribed boundary temperature and E vanishes there, so the
       Dirichlet condition holds exactly and no boundary loss term is needed. An example driven
       only by natural conditions returns (T0, 1). netT=None gives the lift alone."""
    Td, E = problem_of(cfg).T_lift(x, y, cfg, t)
    if netT is None:
        return Td
    return Td + E * cfg.DTref * netT(x, y)


def T_effective(netT, x, y, cfg, t):
    """The temperature seen by the mechanics at load fraction or time t.

       Transient examples use the field itself, T(x, y; t). For a steady example
       (Problem.T_steady) the field is solved once at full load and is linear in its boundary
       data, so the state at fraction t is T0 + t (T_full - T0). It is scaled rather than
       re-solved, which also avoids per-step solver noise."""
    if getattr(cfg, "T_steady", False):
        return cfg.T0 + t * (T_field(netT, x, y, cfg, 1.0) - cfg.T0)
    return T_field(netT, x, y, cfg, t)


def thermal_a(netT, x, y, cfg, t):
    """The thermal strain a_th = alpha (T - T0), detached, so the mechanical block treats the
       temperature as data. Returns None when the example has no temperature field, which is
       the isothermal path through energy_terms."""
    if not getattr(cfg, "heat_on", False):
        return None
    with torch.no_grad():
        T = T_effective(netT, x, y, cfg, t)
    return cfg.alpha * (T - cfg.T0)


# --------------------------------------------------------------------------
# the heat functional


def heat_estimate(netT, pts, sampler, cfg, prevT_model, prev_model, t_now, t_prev, dt,
                  boundary_samples=None):
    """Monte Carlo estimate of the backward-Euler incremental heat potential at the sampled
       points, returned as its three parts (conduction plus source, capacity, boundary) so each
       can be logged. The temperature gradient is mapped to physical coordinates through the
       geometry map, in the same way as the mechanical one."""
    x = pts[:, 0:1].clone().requires_grad_(True)
    y = pts[:, 1:2].clone().requires_grad_(True)
    T = T_field(netT, x, y, cfg, t_now)
    oT = torch.ones_like(T)
    Tx = torch.autograd.grad(T, x, oT, create_graph=True)[0]
    Ty = torch.autograd.grad(T, y, oT, create_graph=True)[0]
    gm = getattr(cfg, "geomap", None)
    if gm is None or gm.is_identity:
        detJ = torch.ones_like(T)
    else:
        ja, jb, jc, jd, detJ = gm.jac(x, y)
        bad = ~torch.isfinite(detJ) | (detJ.abs() < 1e-12)     # outside the patch: see kinematics
        if bad.any():
            z, o = torch.zeros_like(detJ), torch.ones_like(detJ)
            ja, jb, jc, jd = torch.where(bad, z, ja), torch.where(bad, z, jb), \
                torch.where(bad, z, jc), torch.where(bad, z, jd)
            detJ = torch.where(bad, o, detJ)
        Tx, Ty = (jd * Tx - jc * Ty) / detJ, (-jb * Tx + ja * Ty) / detJ
    with torch.no_grad():
        Tp = T_field(prevT_model, pts[:, 0:1], pts[:, 1:2], cfg, t_prev)
    kcond, rho_c = problem_of(cfg).thermal_material(pts[:, 0:1], pts[:, 1:2], cfg)
    if cfg.kdeg:
        # adiabatic crack: conductivity degraded by the damage of the frozen previous step
        phs = phi_value(prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
        kcond = kcond * ((1.0 - phs) ** 2 + cfg.g_floor)
    e_cond = 0.5 * kcond * (Tx ** 2 + Ty ** 2)
    e_cap = 0.5 * rho_c * (T - Tp) ** 2 / dt
    src = problem_of(cfg).heat_source(pts[:, 0:1], pts[:, 1:2], cfg, t_now)
    if src is not None:
        e_cond = e_cond - src * T     # stationarity: rho_c dT/dt = div(k_d grad T) + gamma
    rho, inom = sampler.rho_and_mask(pts)
    w = (inom * detJ / rho.clamp_min(1e-300)).detach()
    Pi_cond, Pi_cap = (e_cond * w).mean(), (e_cap * w).mean()
    Pi_bnd = torch.zeros((), dtype=DTYPE, device=utils.DEVICE)
    samples = (problem_of(cfg).sample_boundary(sampler) if boundary_samples is None
               else boundary_samples)
    for xb, yb, edge_len in samples:
        Tb = T_field(netT, xb, yb, cfg, t_now)
        term = 0.5 * cfg.hconv * (Tb - cfg.T_env) ** 2
        if torch.is_tensor(edge_len):
            # per-point arc-length weights ds/dxi for a curved edge, whose NURBS
            # parametrization is not arc-length uniform: integral = E[term * ds/dxi]
            Pi_bnd = Pi_bnd + (edge_len * term).mean()
        else:
            Pi_bnd = Pi_bnd + edge_len * term.mean()
    return Pi_cond, Pi_cap, Pi_bnd


def solve_T_step(netT, prevT_model, prev_model, cfg, sampler, t_now, t_prev, logger, step, n_iter):
    """The thermal block, run first in each step: minimize the heat potential over the
       temperature parameters alone, with the damage frozen at the previous step."""
    opt = torch.optim.Adam([{"params": netT.mlp.parameters(), "lr": cfg.lr_T},
                            {"params": netT.enc.parameters(), "lr": cfg.lr_enc_T}],
                           eps=cfg.adam_eps)
    dt = t_now - t_prev
    pts = sampler.sample()
    bpts = problem_of(cfg).sample_boundary(sampler)
    hist = []
    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % cfg.resample_every == 0:
            pts = sampler.sample()
            bpts = problem_of(cfg).sample_boundary(sampler)
        opt.zero_grad()
        Pc, Pk, Pb = heat_estimate(netT, pts, sampler, cfg, prevT_model, prev_model,
                                   t_now, t_prev, dt, boundary_samples=bpts)
        loss = Pc + Pk + Pb
        if cfg.enc_reg_T > 0:
            loss = loss + cfg.enc_reg_T * netT.enc.reg()
        loss.backward(); opt.step()
        hist.append(float(loss.detach()))
        if it % cfg.log_every == 0 or it == n_iter - 1:
            logger.info(f"  [T] step {step:3d} t {t_now:.3e}s dt {dt:.2e}  it {it:4d}  "
                        f"Pi_T {hist[-1]:.6e}  (cond {float(Pc.detach()):.4e} "
                        f"cap {float(Pk.detach()):.4e} bnd {float(Pb.detach()):.4e})  "
                        f"{time.time()-t0:4.0f}s")
        if len(hist) > cfg.plateau_win:
            recent = hist[-cfg.plateau_win:]
            if (max(recent) - min(recent)) / (abs(np.mean(recent)) + 1e-30) < cfg.plateau_tol:
                logger.info(f"  [T] step {step} plateau @ it {it}")
                break
    return hist


def solve_T_steady(netT, cfg, sampler, logger, n_iter):
    """The single steady pre-solve of a Problem.T_steady example: minimize the heat potential
       at full load (t=1). With no heat capacity (rho_c = 0, the steady idealization) the
       functional is pure conduction against the hard lift, and the per-step states are exact
       scalings of this one solution (see T_effective)."""
    assert cfg.rho_c == 0.0, \
        "a steady temperature example must have rho_c = 0 (its transient would otherwise matter)"
    opt = torch.optim.Adam([{"params": netT.mlp.parameters(), "lr": cfg.lr_T},
                            {"params": netT.enc.parameters(), "lr": cfg.lr_enc_T}],
                           eps=cfg.adam_eps)
    pts = sampler.sample()
    bpts = problem_of(cfg).sample_boundary(sampler)
    hist = []
    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % cfg.resample_every == 0:
            pts = sampler.sample()
            bpts = problem_of(cfg).sample_boundary(sampler)
        opt.zero_grad()
        Pc, Pk, Pb = heat_estimate(netT, pts, sampler, cfg, None, None,
                                   1.0, 0.0, 1.0, boundary_samples=bpts)
        loss = Pc + Pk + Pb
        if cfg.enc_reg_T > 0:
            loss = loss + cfg.enc_reg_T * netT.enc.reg()
        loss.backward(); opt.step()
        hist.append(float(loss.detach()))
        if it % cfg.log_every == 0 or it == n_iter - 1:
            logger.info(f"  [Tsteady] it {it:4d}  Pi_T {hist[-1]:.6e}  {time.time()-t0:4.0f}s")
        if len(hist) > cfg.plateau_win:
            recent = hist[-cfg.plateau_win:]
            if (max(recent) - min(recent)) / (abs(np.mean(recent)) + 1e-30) < cfg.plateau_tol:
                logger.info(f"  [Tsteady] plateau @ it {it}")
                break
    return hist


def fit_T_step0(netT, cfg, sampler, t1, logger, n_iter):
    """Step-0 initialization: fit the temperature network to the example's reference state at
       the first time (an analytic early-time solution, or the uniform initial temperature), so
       the first step starts from a temperature consistent with its own problem."""
    ref = problem_of(cfg).T_reference
    opt = torch.optim.Adam([{"params": netT.mlp.parameters(), "lr": cfg.lr_T},
                            {"params": netT.enc.parameters(), "lr": cfg.lr_enc_T}],
                           eps=cfg.adam_eps)
    pts = sampler.sample()
    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % cfg.resample_every == 0:
            pts = sampler.sample()
        x, y = pts[:, 0:1], pts[:, 1:2]
        opt.zero_grad()
        T = T_field(netT, x, y, cfg, t1)
        with torch.no_grad():
            Tref = ref(x, y, cfg, t1)
        # importance-weighted L2 (w = inom/rho): the fit error is measured uniformly over the
        # parametric square, not over the sampling mixture, which oversamples the crack and
        # boundary strata
        rho, inom = sampler.rho_and_mask(pts)
        w = (inom / rho.clamp_min(1e-300)).detach()
        loss = ((T - Tref) ** 2 * w).mean() / cfg.DTref ** 2
        loss.backward(); opt.step()
        if it % cfg.log_every == 0 or it == n_iter - 1:
            logger.info(f"  [T0] it {it:4d}  fit {float(loss.detach()):.4e}  "
                        f"{time.time()-t0:4.0f}s")
    return float(loss.detach())


# --------------------------------------------------------------------------
# reporting


@torch.no_grad()
def audit_T(netT, cfg, t, ng=200):
    """Minimum, maximum and mean of the effective temperature field on a render grid, logged
       once per step as a check that the field stays within its driving range."""
    from .plots import _render_grid
    GX, GY, nx, ny = _render_grid(cfg, ng)
    T = T_effective(netT, GX.reshape(-1, 1), GY.reshape(-1, 1), cfg, t)
    return float(T.min()), float(T.max()), float(T.mean())
