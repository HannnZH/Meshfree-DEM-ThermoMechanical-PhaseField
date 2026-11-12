"""
The solver: feature encoding, the network ansatz, the energy, the mesh-free quadrature and the
load-stepping loop.

The method is here in the order it runs. A multiresolution B-spline encoding feeds one network
that outputs displacement and damage. Hard boundary conditions are imposed on its raw outputs.
Strains, grad phi and, for the fourth-order model, the Laplacian of phi come from automatic
differentiation. The energy is integrated by resampled Monte Carlo quadrature over strata that
follow the crack, and each load step minimizes that energy directly with Adam, warm-started
from the previous step (with the hybrid split, the mechanics follows the gradients of the
staggered hybrid scheme instead of those of one functional).

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import copy
import math
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from . import utils
from .config import Cfg, build_parser
from .geometry import gc_bump, phi0_field, phi_gate, update_grip_release
from .plots import (plot_FD_compare, plot_FD_solo, plot_fields, plot_geometry, plot_geometry_bc,
                    plot_linecut, plot_parametric, plot_phi, plot_T)
from .utils import DTYPE, get_logger, load_ckpt, problem_of, save_ckpt


# --------------------------------------------------------------------------
# multiresolution feature encoding


def _qbspline_w(t):
    """Uniform quadratic B-spline weights over 3 consecutive control points, t in [0,1].
       C1 across segments, and the weights sum to 1. Returns (w0, w1, w2), each shaped like t."""
    return 0.5 * (1.0 - t) ** 2, (-t * t + t + 0.5), 0.5 * t * t

class MultiResEncoding(nn.Module):
    """Dense multiresolution feature grids with quadratic B-spline (C1) interpolation.
       This is a positional encoding only, not a finite element mesh and not an integration
       rule. All levels cover the whole domain, with no crack tracking. The grids are
       initialized to zero, so they add residual features on top of the raw (x, y). Fine levels
       can be gated on coarse to fine during step 0. The finest grid spacing bounds the
       resolution the representation can carry."""
    def __init__(self, cfg, levels=None, ch=None):
        super().__init__()
        self.cfg = cfg
        self.levels = tuple(cfg.levels if levels is None else levels)   # base (short-side) sizes
        self.ch = cfg.enc_ch if ch is None else ch                      # NetT passes its own
        # Rectangular grids: each level n -> (n_y=n, n_x=round(aspect*n)), so the long direction of
        # a strip gets proportionally more cells and the physical resolution is about isotropic.
        # aspect=1 (the unit square, and every example that does not set one) gives square grids.
        aspect = float(getattr(cfg, "grid_aspect", 1.0) or 1.0)
        self.grid_hw = [(n, max(3, int(round(aspect * n)))) for n in self.levels]   # (n_y, n_x)
        self.grids = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.ch, ny, nx, dtype=DTYPE)) for (ny, nx) in self.grid_hw])
        self.gates = [1.0] * len(self.levels)     # plain floats, set per iteration in step 0

    @property
    def n_feat(self):
        return self.ch * len(self.levels)

    def set_progress(self, step, it):
        """Coarse-to-fine gates. Only step 0 ramps, and every later (warm) step runs fully on. A
           net whose level count differs from --enc_start (a second, independently sized
           encoding) stays fully on throughout."""
        c = self.cfg
        if step > 0 or len(c.enc_start) != len(self.levels):
            self.gates = [1.0] * len(self.levels)
            return
        self.gates = [min(1.0, max(0.0, (it - s0) / max(1, c.enc_ramp))) if it < s0 + c.enc_ramp
                      else 1.0 for s0 in c.enc_start]

    def _interp_level(self, G, x, y):
        """G: (C, n_y, n_x) control values. x, y: (N,) in [0,1], slight excursions extrapolate.
           Tensor-product quadratic B-spline with per-axis spacing (rectangular grids), evaluated
           as a 3x3 gather. All operations compose with autograd and support double backward,
           which the grad phi energy followed by loss.backward needs. Square grids reduce to the
           isotropic case exactly."""
        ny, nx = G.shape[-2], G.shape[-1]
        sx, sy = float(nx - 2), float(ny - 2)
        ux, uy = x * sx, y * sy
        ix = ux.detach().floor().clamp(0, nx - 3).long()
        iy = uy.detach().floor().clamp(0, ny - 3).long()
        tx = ux - ix.to(DTYPE)
        ty = uy - iy.to(DTYPE)
        wx = _qbspline_w(tx)                       # 3 x (N,)
        wy = _qbspline_w(ty)
        out = 0.0
        for b in range(3):
            for a in range(3):
                out = out + (wy[b] * wx[a]).unsqueeze(0) * G[:, iy + b, ix + a]   # (C, N)
        return out

    def forward(self, x, y):
        """x, y: (N,1) -> features (N, n_feat)."""
        xf, yf = x.reshape(-1), y.reshape(-1)
        feats = []
        for k, G in enumerate(self.grids):
            f = self._interp_level(G, xf, yf)                 # (C, N)
            feats.append(self.gates[k] * f)
        return torch.cat(feats, dim=0).transpose(0, 1)        # (N, n_feat)

    def reg(self):
        return sum((G ** 2).sum() for G in self.grids)

    def fine_max(self):
        return float(self.grids[-1].detach().abs().max())


# --------------------------------------------------------------------------
# network ansatz and autodiff kinematics


class NetMS(nn.Module):
    """One network for the raw outputs [u_hat, v_hat, a_hat] on the encoded input
       (x, y, F(x, y)). The input coordinates are used as given."""
    def __init__(self, cfg):
        super().__init__()
        self.enc = MultiResEncoding(cfg)
        L = [nn.Linear(2 + self.enc.n_feat, cfg.width), nn.GELU()]
        for _ in range(cfg.depth - 1):
            L += [nn.Linear(cfg.width, cfg.width), nn.GELU()]
        last = nn.Linear(cfg.width, 3)
        L += [last]
        self.mlp = nn.Sequential(*L)
        with torch.no_grad():                       # negative phi-channel bias: bulk phi ~ 0
            last.bias[2] = cfg.phi_bias

    def forward(self, x, y):
        feats = self.enc(x, y)
        return self.mlp(torch.cat([x, y, feats], dim=1))

def _fields(net, x, y, cfg, u_delta):
    """The physical fields (u, v, phi, phi0) at the query points, with the displacement boundary
       conditions imposed exactly. The conditions depend on the example, so this dispatches to
       its Problem (see pfte.problems.Problem.fields)."""
    return problem_of(cfg).fields(net, x, y, cfg, u_delta)

def kinematics(net, x, y, cfg, u_delta):
    """Strains, phi and |grad phi|^2 by automatic differentiation, all with respect to the
       physical coordinates. With --geo none the inputs are physical coordinates. With a GeoMap
       the inputs are parametric, the raw gradients are pulled back with J^{-T} (J is detached
       because the geometry does not depend on the network parameters). Returns
       (exx, eyy, exy, phi, |grad phi|^2, detJ, lap_phi): detJ (1 for identity geometry) is
       the quadrature weight, and lap_phi is the Laplacian of phi with --pf_order 4, None
       otherwise."""
    x = x.requires_grad_(True); y = y.requires_grad_(True)
    u, v, phi, phi0 = _fields(net, x, y, cfg, u_delta)
    ou = torch.ones_like(u)
    ux = torch.autograd.grad(u, x, ou, create_graph=True)[0]
    uy = torch.autograd.grad(u, y, ou, create_graph=True)[0]
    vx = torch.autograd.grad(v, x, ou, create_graph=True)[0]
    vy = torch.autograd.grad(v, y, ou, create_graph=True)[0]
    op = torch.ones_like(phi)
    px = torch.autograd.grad(phi, x, op, create_graph=True)[0]
    py = torch.autograd.grad(phi, y, op, create_graph=True)[0]
    gm = getattr(cfg, "geomap", None)
    lap_phi = None
    if getattr(cfg, "pf_order", 2) == 4:
        # Fourth order: the Laplacian phi_xx + phi_yy takes one more autodiff pass (px, py carry
        # create_graph=True). Identity geometry only: with J = I, px and py are already the
        # physical grad phi, so d(px)/dx + d(py)/dy is the physical Laplacian. A curved patch
        # would need the second derivatives of the map in the pullback, which are not
        # implemented. validate() rejects that combination before the run.
        if gm is not None and not gm.is_identity:
            raise ValueError("pf_order 4 requires identity geometry, "
                             "the Laplacian pullback on a curved patch is not implemented")
        pxx = torch.autograd.grad(px, x, torch.ones_like(px), create_graph=True)[0]
        pyy = torch.autograd.grad(py, y, torch.ones_like(py), create_graph=True)[0]
        lap_phi = pxx + pyy
    if gm is None or gm.is_identity:                        # identity: J = I, detJ = 1
        detJ = torch.ones_like(phi)
    else:
        a, b, c, d, detJ = gm.jac(x, y)                     # J = [[a,b],[c,d]], detached
        # Query points outside [0,1]^2 (any point drawn just outside the parametric square) meet
        # an all-zero B-spline basis, so Wf = 0 and the map and its Jacobian are 0/0 = NaN.
        # Those points are masked out downstream (inom = 0, so w = 0), but NaN * 0 = NaN would
        # still spoil the mean. The Jacobian is therefore replaced there by a = b = c = d = 0,
        # detJ = 1, so the pullback gives finite (zero) strains and the mask drops them.
        # In-domain points are not affected.
        bad = ~torch.isfinite(detJ) | (detJ.abs() < 1e-12)
        if bad.any():
            z, o = torch.zeros_like(detJ), torch.ones_like(detJ)
            a, b, c, d = torch.where(bad, z, a), torch.where(bad, z, b), \
                torch.where(bad, z, c), torch.where(bad, z, d)
            detJ = torch.where(bad, o, detJ)
        ux, uy = (d * ux - c * uy) / detJ, (-b * ux + a * uy) / detJ
        vx, vy = (d * vx - c * vy) / detJ, (-b * vx + a * vy) / detJ
        px, py = (d * px - c * py) / detJ, (-b * px + a * py) / detJ
    exx, eyy, exy = ux, vy, 0.5 * (uy + vx)
    gradphi2 = px ** 2 + py ** 2
    return exx, eyy, exy, phi, gradphi2, detJ, lap_phi

def phi_squash(a_hat, cfg):
    """Map the raw phi logit to about [0,1]. The default is a sigmoid. "leaky" is Manav et al.,
       CMAME 2024, eq. (23): a linear core (slope 1/4 for |a| <= 2) with leaky tails (slope
       beta = 1e-3), so the gradient never vanishes and the constant downward drive of AT1
       cannot push the phi channel into sigmoid saturation. Slight excursions outside [0,1] are
       not clamped. The degradation, the crack-surface term and the irreversibility penalty
       push them back. With leaky, set --phi_bias -2 so that the initial phi is 0."""
    if cfg.phi_map == "leaky":
        b = 1e-3
        core = a_hat / 4.0 + 0.5
        lo = b * (a_hat + 2.0)
        hi = b * (a_hat - 2.0) + 1.0
        return torch.where(a_hat < -2.0, lo, torch.where(a_hat > 2.0, hi, core))
    return torch.sigmoid(a_hat)

def phi_forward(net, x, y, cfg):
    """phi at points with gradients enabled (for the orientation field and structure tensor).
       net=None returns phi0."""
    phi0 = phi0_field(x, y, cfg)
    if net is None:
        return phi0
    out = net(x, y)
    return phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)

@torch.no_grad()
def phi_value(net, x, y, cfg):
    """phi at points from a frozen net, value only. net=None returns phi0 (step 0, the notch)."""
    return phi_forward(net, x, y, cfg)

def drive_field(net, x, y, cfg, u_delta, netT=None, t=0.0, chunk=16384):
    """g(phi) * psi_plus of the frozen previous net (detached), the crack driving force. It
       enters the sampler's density indicator so that the process zone ahead of the tip is
       covered by points.

       psi_plus here is always the Amor form, whatever --split the energy uses. This field is a
       proposal density, not physics. It enters only through the sampler's Dgrid, and the
       importance weight w = inom detJ / rho (det J > 0) divides that density back out, so the
       estimator of Pi is unbiased for any strictly positive choice. Only the variance and the point
       placement change. A region with driving but few points rests on the sparse uniform
       stratum and raises the variance, while points where there is no driving only cost a few
       samples. The Amor form is the wider indicator (in pure shear, e1 = -e2, the spectral
       psi_plus is half of it), so it errs on the side of more coverage."""
    outs = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].clone().requires_grad_(True)
        yc = y[i:i + chunk].clone().requires_grad_(True)
        u, v, phi, _ = _fields(net, xc, yc, cfg, u_delta)
        ou = torch.ones_like(u)
        ux = torch.autograd.grad(u, xc, ou, retain_graph=True)[0]
        uy = torch.autograd.grad(u, yc, ou, retain_graph=True)[0]
        vx = torch.autograd.grad(v, xc, ou, retain_graph=True)[0]
        vy = torch.autograd.grad(v, yc, ou)[0]
        gm = getattr(cfg, "geomap", None)
        if gm is not None and not gm.is_identity:
            a, b, c, d, detJ = gm.jac(xc, yc)
            ux, uy = (d * ux - c * uy) / detJ, (-b * ux + a * uy) / detJ
            vx, vy = (d * vx - c * vy) / detJ, (-b * vx + a * vy) / detJ
        exx, eyy, exy = ux, vy, 0.5 * (uy + vx)
        # the indicator includes the thermal strain, so that a purely thermal example (no
        # mechanical load) places its points where the thermal driving is
        a_th = thermal_strain(cfg, netT, x[i:i + chunk], y[i:i + chunk], t)
        exx, eyy, ezz, tr = elastic_parts(exx, eyy, a_th, cfg)
        tr_p = torch.clamp(tr, min=0.0)
        if getattr(cfg, "plane_stress", False):
            dev_xx, dev_yy = exx - tr / 2.0, eyy - tr / 2.0
            dev_con = dev_xx ** 2 + dev_yy ** 2 + 2.0 * exy ** 2
        else:
            dev_xx, dev_yy, dev_zz = exx - tr / 3.0, eyy - tr / 3.0, ezz - tr / 3.0
            dev_con = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2
        psi_p = 0.5 * cfg.K * tr_p ** 2 + cfg.mu * dev_con
        g = (1.0 - phi) ** 2 + cfg.g_floor
        outs.append((g * psi_p).detach())
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------
# energy densities


class DirField:
    """Crack orientation and blend field for the directional split, frozen for one load step and
       computed from phi_prev on a grid in parametric coordinates.

       The structure tensor <grad phi grad phi^T> (Gaussian-blurred outer products) gives the
       orientation, stored as the double angle (cos 2 theta, sin 2 theta) so that the sign of the
       director does not matter. The blend weight is W = coherence * phi presence: W ~ 1 on an
       established band, where the directional split lets the crack both open and slide, and
       W ~ 0 in the bulk, where the spectral split applies. Since everything is frozen from the
       previous step, each step remains the minimization of one energy functional. At step 0
       (prev_model=None) phi0 supplies the orientation."""
    def __init__(self, prev_model, cfg, n=192, sig_cells=2.0, chunk=16384):
        xs = (torch.arange(n, dtype=DTYPE, device=utils.DEVICE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        gx, gy = GX.reshape(-1, 1), GY.reshape(-1, 1)
        pxs, pys, phs = [], [], []
        for i in range(0, gx.shape[0], chunk):
            xc = gx[i:i + chunk].clone().requires_grad_(True)
            yc = gy[i:i + chunk].clone().requires_grad_(True)
            ph = phi_forward(prev_model, xc, yc, cfg)
            px, py = torch.autograd.grad(ph.sum(), [xc, yc])
            pxs.append(px.detach()); pys.append(py.detach()); phs.append(ph.detach())
        px = torch.cat(pxs).reshape(n, n); py = torch.cat(pys).reshape(n, n)
        phig = torch.cat(phs).reshape(n, n)
        J11, J12, J22 = px * px, px * py, py * py
        # A separable Gaussian blur averages over a neighborhood, which makes the director well
        # defined on the band centerline, where grad phi = 0 and the flanks average in.
        k = int(4 * sig_cells) | 1
        t = torch.arange(k, dtype=DTYPE, device=utils.DEVICE) - (k - 1) / 2.0
        ker = torch.exp(-0.5 * (t / sig_cells) ** 2); ker = ker / ker.sum()
        def blur(M):
            M = M.reshape(1, 1, n, n)
            M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, -1, 1), padding=(k // 2, 0))
            M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, 1, -1), padding=(0, k // 2))
            return M.reshape(n, n)
        J11, J12, J22 = blur(J11), blur(J12), blur(J22)
        theta = 0.5 * torch.atan2(2.0 * J12, J11 - J22)
        disc = torch.sqrt(((J11 - J22) * 0.5) ** 2 + J12 ** 2 + 1e-300)
        lam1 = (J11 + J22) * 0.5 + disc
        lam2 = (J11 + J22) * 0.5 - disc
        coh = (lam1 - lam2) / (lam1 + lam2 + 1e-30)
        self.A = torch.cos(2.0 * theta).detach()
        self.B = torch.sin(2.0 * theta).detach()
        self.W = (coh * torch.clamp(phig / 0.5, 0.0, 1.0)).detach()
        self.n = n

    @torch.no_grad()
    def __call__(self, x, y):
        """Bilinear lookup of (cos 2 theta, sin 2 theta, W) at the query points. The fields are
           detached because the energy needs no gradient through the frozen orientation."""
        n = self.n
        u = (x.reshape(-1) * n - 0.5).clamp(0.0, n - 1.0)
        v = (y.reshape(-1) * n - 0.5).clamp(0.0, n - 1.0)
        i0 = u.floor().long().clamp(0, n - 2); j0 = v.floor().long().clamp(0, n - 2)
        tu = (u - i0.to(DTYPE)).unsqueeze(-1); tv = (v - j0.to(DTYPE)).unsqueeze(-1)
        out = []
        for M in (self.A, self.B, self.W):
            m00 = M[j0, i0]; m01 = M[j0, i0 + 1]; m10 = M[j0 + 1, i0]; m11 = M[j0 + 1, i0 + 1]
            m0 = m00.unsqueeze(-1) * (1 - tu) + m01.unsqueeze(-1) * tu
            m1 = m10.unsqueeze(-1) * (1 - tu) + m11.unsqueeze(-1) * tu
            out.append(m0 * (1 - tv) + m1 * tv)
        return out[0], out[1], out[2]

def thermal_strain(cfg, netT, x, y, t):
    """a_th at the query points, or None for an isothermal run. The heat_on test returns before
       the import, so an isothermal run never loads the thermal module. That module imports this
       one, and the lazy import avoids a circular import."""
    if not getattr(cfg, "heat_on", False):
        return None
    from .thermal import thermal_a
    return thermal_a(netT, x, y, cfg, t)

def material_fields(x, y, cfg):
    """(lambda, mu, K, Gc) at the query points: scalars for a homogeneous body, fields for a
       graded or heterogeneous one. See pfte.problems.Problem.material."""
    return problem_of(cfg).material(x, y, cfg)

def elastic_parts(exx, eyy, a_th, cfg):
    """Split the total strain into the elastic part the stress responds to, eps_e = eps - a_th I,
       where a_th = alpha (T - T0) is the detached thermal strain. a_th=None is the isothermal
       case and returns the total strain unchanged.

       The out-of-plane elastic strain ezz_e distinguishes the two 2-D reductions. Under plane
       stress the plate is free to thin, so ezz_e is eliminated and contributes nothing. With
       --alpha_mode ps3d the total eps_zz is held at 0 under plane strain, so cooling puts the
       whole thermal contraction into the elastic channel, ezz_e = -a_th, a tensile driving term.
       The default, --alpha_mode 2d, drops it, as in purely 2-D models."""
    if a_th is None:                                   # isothermal: eps_e == eps
        return exx, eyy, torch.zeros_like(exx), exx + eyy
    exx_e = exx - a_th
    eyy_e = eyy - a_th
    if getattr(cfg, "stress_state", "plane_strain") == "plane_stress":
        ezz_e = torch.zeros_like(a_th)
    else:
        ezz_e = -a_th if getattr(cfg, "alpha_mode", "2d") == "ps3d" else torch.zeros_like(a_th)
    return exx_e, eyy_e, ezz_e, exx_e + eyy_e + ezz_e

def _frac_energy(phi, gradphi2, lap_phi, Gc_loc, cfg):
    """Crack-surface energy density. pf_order 2 is the second-order AT1 or AT2 model. pf_order 4
       is the fourth-order AT2-type model of Borden et al., CMAME 273 (2014), eq. (15),
       (Gc/2) [phi^2/l + (l/2) |grad phi|^2 + (l^3/16) (Lap phi)^2], written in the convention
       used here (phi = 1 - c, and l = 2 l0 with l0 the length scale of Borden et al.)."""
    l = cfg.l
    if getattr(cfg, "pf_order", 2) == 4:
        return 0.5 * Gc_loc * (phi ** 2 / l + 0.5 * l * gradphi2 + (l ** 3 / 16.0) * lap_phi ** 2)
    if cfg.at == 1:
        return (3.0 / 8.0) * Gc_loc * (phi / l + l * gradphi2)          # AT1: Gc/c_w, c_w=8/3
    return 0.5 * Gc_loc * (phi ** 2 / l + l * gradphi2)                 # AT2: Gc/c_w, c_w=2

def energy_terms(exx, eyy, exy, phi, gradphi2, phi_prev, cfg, a_th=None, x=None, y=None,
                 lap_phi=None):
    """The per-point energy densities (elastic, crack-surface, irreversibility).

       Every split is written on the elastic strain eps_e = eps - a_th I and on the local
       material, so the isothermal homogeneous case (a_th=None with scalar moduli) reduces
       exactly to the plain elastic expressions, term for term."""
    lam_l, mu_l, K_l, Gc_loc = material_fields(x, y, cfg)
    if getattr(cfg, "gc_tdep", False) and a_th is not None and cfg.alpha != 0.0:
        # Gc(T) = Gc0 [1 - b1 theta + b2 theta^2], theta = (T - Tref)/Tmax: the toughness falls
        # as the body heats. T is recovered from the thermal strain.
        T_loc = cfg.T0 + a_th / cfg.alpha
        theta = (T_loc - cfg.gc_Tref) / cfg.gc_Tmax
        Gc_loc = Gc_loc * (1.0 - cfg.gc_b1 * theta + cfg.gc_b2 * theta ** 2)
    if getattr(cfg, "gc_grip", 0.0) > 0 and x is not None:
        Gc_loc = Gc_loc * (1.0 + cfg.gc_grip * gc_bump(x, y, cfg))
    if getattr(cfg, "gc_extra_on", False) and x is not None:
        Gc_loc = Gc_loc * problem_of(cfg).gc_extra(x, y, cfg)
    exx_e, eyy_e, ezz_e, tr = elastic_parts(exx, eyy, a_th, cfg)
    tr_p = torch.clamp(tr, min=0.0); tr_m = tr - tr_p
    # In-plane trace, for the in-plane eigenvalues only. tr above includes ezz_e, while e1 and e2
    # are eigenvalues of the 2x2 in-plane block. In the isothermal case (ezz_e = 0) they coincide.
    tr2 = exx_e + eyy_e
    if cfg.split == "hybrid":
        # Hybrid formulation (Ambati et al., Comput. Mech. 2015). Isotropic degradation governs
        # the mechanics, so the crack is soft in all modes, including mode II sliding, while phi
        # is driven by the spectral tension energy only, so compressive and shear states do not
        # drive damage. This is not a single functional. The two couplings are detached so that
        # the gradients equal those of the staggered hybrid scheme, combined in one Adam loop.
        # The value of e_el is g psi_full, the mechanical energy, so the reaction force, the logs
        # and the plateau test remain meaningful. Ambati's crack-interpenetration condition is
        # not applied.
        psi_full = 0.5 * lam_l * tr ** 2 + mu_l * (exx_e ** 2 + eyy_e ** 2 + ezz_e ** 2
                                                   + 2.0 * exy ** 2)
        rad = torch.sqrt(((exx_e - eyy_e) * 0.5) ** 2 + exy ** 2 + 1e-30)
        e1 = tr2 * 0.5 + rad
        e2 = tr2 * 0.5 - rad
        D = (0.5 * lam_l * tr_p ** 2
             + mu_l * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2
                       + torch.clamp(ezz_e, min=0.0) ** 2))
        g_all = (1.0 - phi) ** 2 + cfg.g_floor
        g_det = (1.0 - phi.detach()) ** 2 + cfg.g_floor
        e_el = g_det * psi_full + (g_all - g_all.detach()) * D.detach()
        e_fr = _frac_energy(phi, gradphi2, lap_phi, Gc_loc, cfg)
        e_ir = _irrev_energy(phi, phi_prev, cfg)
        return e_el, e_fr, e_ir
    if cfg.split == "directional":
        # Crack-frame split on the band, where both the opening strain enn+ and the sliding strain
        # ent are degraded, so mode II sliding is free. It is blended by W into the spectral split
        # in the bulk, where only tension drives damage. The orientation and W are frozen for the
        # step (cfg.dirfield).
        A, B, W = cfg.dirfield(x, y)
        th = 0.5 * torch.atan2(B, A)
        nx, ny = torch.cos(th), torch.sin(th)
        tx, ty = -ny, nx
        enn = exx_e * nx * nx + 2.0 * exy * nx * ny + eyy_e * ny * ny
        ett = exx_e * tx * tx + 2.0 * exy * tx * ty + eyy_e * ty * ty
        ent = exx_e * nx * tx + exy * (nx * ty + ny * tx) + eyy_e * ny * ty
        enn_p = torch.clamp(enn, min=0.0); enn_m = enn - enn_p
        psi_p_dir = 0.5 * lam_l * tr_p ** 2 + mu_l * enn_p ** 2 + 2.0 * mu_l * ent ** 2
        psi_m_dir = 0.5 * lam_l * tr_m ** 2 + mu_l * enn_m ** 2 + mu_l * ett ** 2
        rad = torch.sqrt(((exx_e - eyy_e) * 0.5) ** 2 + exy ** 2 + 1e-30)
        e1 = tr2 * 0.5 + rad
        e2 = tr2 * 0.5 - rad
        psi_p_sp = (0.5 * lam_l * tr_p ** 2
                    + mu_l * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2
                              + torch.clamp(ezz_e, min=0.0) ** 2))
        psi_m_sp = (0.5 * lam_l * tr_m ** 2
                    + mu_l * (torch.clamp(e1, max=0.0) ** 2 + torch.clamp(e2, max=0.0) ** 2
                              + torch.clamp(ezz_e, max=0.0) ** 2))
        psi_p = W * psi_p_dir + (1.0 - W) * psi_p_sp
        psi_m = W * psi_m_dir + (1.0 - W) * psi_m_sp
    elif cfg.split == "spectral":
        # Spectral split (Miehe et al., CMAME 2010) on the elastic strain. Only the positive
        # principal strains drive damage. In pure shear (e1 = -e2) psi_p is half of the Amor
        # value, with the same total psi. The out-of-plane elastic strain is the third
        # eigenvalue. It is zero in the isothermal case and under plane stress, but under plane
        # strain with --alpha_mode ps3d a cooling body drives tension through it, so it has its
        # own tension and compression parts.
        rad = torch.sqrt(((exx_e - eyy_e) * 0.5) ** 2 + exy ** 2 + 1e-30)
        e1 = tr2 * 0.5 + rad
        e2 = tr2 * 0.5 - rad
        psi_p = (0.5 * lam_l * tr_p ** 2
                 + mu_l * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2
                           + torch.clamp(ezz_e, min=0.0) ** 2))
        psi_m = (0.5 * lam_l * tr_m ** 2
                 + mu_l * (torch.clamp(e1, max=0.0) ** 2 + torch.clamp(e2, max=0.0) ** 2
                           + torch.clamp(ezz_e, max=0.0) ** 2))
    elif cfg.split == "iso":
        # No split (the isotropic model used by Manav et al., CMAME 2024): the full isotropic
        # strain energy is degraded in every mode, tension, compression and shear alike, so
        # damage is driven isotropically. There is no psi_minus.
        psi_p = 0.5 * lam_l * tr ** 2 + mu_l * (exx_e ** 2 + eyy_e ** 2 + ezz_e ** 2
                                                + 2.0 * exy ** 2)
        psi_m = torch.zeros_like(psi_p)
    else:
        if getattr(cfg, "plane_stress", False):
            # 2-D reduced Amor split: the deviator is taken in-plane (tr/2), consistent with the
            # 2-D bulk modulus K_2D = lam' + mu that cfg.K returns under plane stress.
            dev_xx, dev_yy = exx_e - tr / 2.0, eyy_e - tr / 2.0
            dev_con = dev_xx ** 2 + dev_yy ** 2 + 2.0 * exy ** 2
        else:
            dev_xx, dev_yy, dev_zz = exx_e - tr / 3.0, eyy_e - tr / 3.0, ezz_e - tr / 3.0
            dev_con = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2
        # Amor split (Amor et al., JMPS 2009), extended by gamma_star to the star-convex family
        # (Vicentini et al., Int. J. Fract. 247, 2024): the compressive volumetric energy is
        # subtracted from the driving part and added undegraded to psi_m, which suppresses damage
        # in compression-dominated states. gamma_star = 0 is the Amor split. The sum psi_p + psi_m
        # does not depend on gamma_star. See Vicentini et al. for the admissible range.
        psi_p = (0.5 * K_l * tr_p ** 2 + mu_l * dev_con
                 - cfg.gamma_star * 0.5 * K_l * tr_m ** 2)
        psi_m = (1.0 + cfg.gamma_star) * 0.5 * K_l * tr_m ** 2
    g = (1.0 - phi) ** 2 + cfg.g_floor
    e_el = g * psi_p + psi_m
    e_fr = _frac_energy(phi, gradphi2, lap_phi, Gc_loc, cfg)
    # ir_tol is a dead band of the irreversibility penalty: phi may relax by up to ir_tol below
    # phi_prev, so low-level Monte Carlo noise in phi does not accumulate across load steps.
    # On a crack (phi ~ 1) the energy keeps phi up, so the dead band has no visible effect there.
    e_ir = _irrev_energy(phi, phi_prev, cfg)
    return e_el, e_fr, e_ir

def _irrev_energy(phi, phi_prev, cfg):
    """The irreversibility penalty gamma_ir <phi_prev - ir_tol - phi>_+^2. With --ir_thresh c it
       acts only on the crack set {phi_prev >= c}, as in the crack-set irreversibility of Bourdin
       et al. (discussed by Gerasimov and De Lorenzis, CMAME 354, 2019), except that inside the
       set it penalizes any decrease of phi instead of fixing phi = 1. Below the threshold the
       damage may heal, so end-of-step noise on the band flanks and in the bulk is not frozen in
       and does not accumulate from step to step. c = 0 (the default) applies the penalty
       everywhere. The strip, slab and annulus use c = 0.5."""
    e = cfg.gamma_ir * torch.clamp(phi_prev - cfg.ir_tol - phi, min=0.0) ** 2
    if cfg.ir_thresh > 0 and torch.is_tensor(phi_prev):
        e = e * (phi_prev >= cfg.ir_thresh).to(e.dtype)
    return e


def _blur_grid(M, sig_cells):
    """Separable Gaussian blur of an (n, n) torch grid. It widens the new-band stratum to the
       cross profile of the notch stratum (sigma_band), so the propagated crack gets the same
       point density per unit length as the notch, flanks included."""
    n = M.shape[-1]
    k = int(4 * sig_cells) | 1
    t = torch.arange(k, dtype=DTYPE, device=M.device) - (k - 1) / 2.0
    ker = torch.exp(-0.5 * (t / max(sig_cells, 1e-6)) ** 2); ker = ker / ker.sum()
    M = M.reshape(1, 1, n, n)
    M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, -1, 1), padding=(k // 2, 0))
    M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, 1, -1), padding=(0, k // 2))
    return M.reshape(n, n)


# --------------------------------------------------------------------------
# mesh-free Monte Carlo quadrature


class Sampler:
    """Mixture sampler for the Monte Carlo integration points: a uniform bulk stratum, a crack
       stratum covering the whole current crack, and a damage (process zone) stratum. The density
       rho of the mixture is known exactly (categorical grid densities), so the 1/rho weighting
       is unbiased. Every stratum is weighted by |detJ|, which makes the sampling uniform in
       physical space under a geometry map (with the identity map |detJ| = 1 and parametric and
       physical coordinates coincide). The strata are built from prev_model and stay fixed for
       the whole load step. phi0 is built from physical distances (through to_phys), so the
       strata follow the physical crack on a curved patch."""
    def __init__(self, cfg, prev_model=None, N=None, seed=0):
        self.cfg = cfg
        self.N = cfg.N if N is None else N
        self.sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
        self.rng = np.random.default_rng(seed + 12345)
        wu, wn, wd = cfg.w_unif, cfg.w_notch, cfg.w_damage
        self.gn = cfg.grid_n
        self.cell = 1.0 / self.gn
        gm = getattr(cfg, "geomap", None)
        self.is_curved = gm is not None and not getattr(gm, "is_identity", False)
        self.Jcell = self._jac_grid()          # |detJ| per cell, the physical-density weight
        # area of the body as a fraction of the parametric square. An example whose body does
        # not fill the square reduces it in prepare_sampler, so the bulk density stays normalized.
        self.dom_area = 1.0
        problem_of(cfg).prepare_sampler(self)
        # physically uniform bulk on a curved patch (parametric-uniform Sobol on identity/none)
        self.UnifGrid = (self.Jcell / self.Jcell.sum()) if self.is_curved else None
        self._build_notch_grid(prev_model)     # crack stratum = max(phi0, phi_prev) * |detJ|
        self._build_damage_grid(prev_model)    # Dgrid (process zone) + optional Ngrid_raw (band)
        if self.Dgrid is None:
            wd = 0.0
        s = wu + wn + wd
        self.wu, self.wn, self.wd = wu / s, wn / s, wd / s
        self.nu = int(round(self.wu * self.N))
        self.nd = int(round(self.wd * self.N)) if self.Dgrid is not None else 0
        self.nn_ = self.N - self.nu - self.nd
        # The crack stratum (nn_ points) covers the whole crack, notch and propagated part, at one
        # physical density, so the propagated crack receives the notch density at a fixed budget.
        # The optional band (rho_new > 0) adds n_add points on the propagated crack at an explicit
        # areal density (points per physical area), for extra concentration there.
        self.n_add = 0
        if self.Ngrid_raw is not None and self.cfg.rho_new > 0:
            self.n_add = int(round(self.cfg.rho_new * self.A_new / (self.gn ** 2)))
            self.Ngrid = (self.Ngrid_raw / self.Ngrid_raw.sum()) if self.n_add > 0 else None
        else:
            self.Ngrid = None
        self.Nt = self.N + self.n_add
        self.fu = self.nu / self.Nt
        self.fn = self.nn_ / self.Nt
        self.fd = self.nd / self.Nt
        self.fnew = self.n_add / self.Nt

    def _jac_grid(self):
        """|detJ| at the grid cell centers (n x n). It is 1 for the identity map or no map, where
           parametric and physical coordinates coincide. On a curved patch it is the physical-area
           weight that makes every grid stratum uniform in physical space (|detJ| can vary
           strongly)."""
        n = self.gn
        gm = getattr(self.cfg, "geomap", None)
        if gm is None or getattr(gm, "is_identity", False):
            return np.ones((n, n))
        xs = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        # gm.jac() runs autograd internally on its own leaf graph, so it must not be in no_grad
        _, _, _, _, det = gm.jac(GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE))
        return det.abs().reshape(n, n).cpu().numpy()

    def _build_notch_grid(self, prev_model):
        """Crack stratum: a categorical grid proportional to max(phi0, phi_prev) * |detJ|, which
           gives a physically uniform density over the whole current crack, notch and propagated
           part, so the crack path receives the notch density. phi0 is built from physical
           distances (through to_phys), so the stratum follows the physical crack under any
           geometry map. At step 0 (prev_model=None) it is phi0 alone, the notch."""
        c = self.cfg
        n = self.gn
        xs = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        gx, gy = GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE)
        with torch.no_grad():
            crack = phi0_field(gx, gy, c)
            if prev_model is not None:
                crack = torch.maximum(crack, phi_value(prev_model, gx, gy, c))
            # an example with no pre-existing crack still needs points where one will start
            crack = problem_of(c).crack_stratum(crack, gx, gy, c)
        crack = crack.reshape(n, n).cpu().numpy()
        W = np.clip(crack, 1e-12, None) * self.Jcell   # crack * |detJ|, the physical density
        self.NotchGrid = W / W.sum()                   # point distribution over the whole crack
        self.A_notch = float(((crack > 0.5) * self.Jcell).sum())   # |detJ|-weighted cell count of the band (area x gn^2)

    def _build_damage_grid(self, prev_model):
        c = self.cfg
        self.Dgrid = None; self.Ngrid_raw = None; self.A_new = 0.0
        if prev_model is None or (c.w_damage <= 0 and c.rho_new == 0):
            return
        n = self.gn
        xs = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        gx, gy = GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE)
        phi = phi_value(prev_model, gx, gy, cfg=c)
        phi0 = phi0_field(gx, gy, c)
        if c.w_damage > 0:
            D = phi * (1.0 - phi) + c.eta_damage * phi0
            if c.beta_drive > 0:
                drv = drive_field(prev_model, gx, gy, c, c.ud_prev,
                                  netT=getattr(c, "netT_prev", None),
                                  t=getattr(c, "t_prev", 0.0))
                D = D + c.beta_drive * drv / (drv.max() + 1e-30)
            D = D.reshape(n, n).cpu().numpy()
            D = np.clip(D, 1e-12, None) * self.Jcell            # physical density
            self.Dgrid = D / D.sum()
        if c.rho_new != 0:                                      # optional band (rho_new > 0)
            mask = ((phi0 < 0.4) & (phi > 0.5)).to(DTYPE)
            mask_np = mask.reshape(n, n).cpu().numpy()          # binary new-crack band
            Dn = (phi * (1.0 - phi) + c.eta_new * phi) * mask
            Dn = _blur_grid(Dn.reshape(n, n), c.sigma_band * n).cpu().numpy()
            W = np.clip(Dn, 0.0, None) * self.Jcell
            if W.sum() > 0:
                self.Ngrid_raw = W                              # point distribution (weighted)
                self.A_new = float((mask_np * self.Jcell).sum())  # |detJ|-weighted cell count (area x gn^2)

    def _sample_grid(self, grid, m):
        n = self.gn
        idx = self.rng.choice(n * n, size=m, p=grid.ravel())
        iy, ix = np.divmod(idx, n)
        jx = (ix + self.rng.random(m)) * self.cell
        jy = (iy + self.rng.random(m)) * self.cell
        return torch.tensor(np.stack([jx, jy], 1), dtype=DTYPE, device=utils.DEVICE)

    def sample(self):
        parts = []
        # an example whose body does not fill the parametric square may draw the bulk stratum
        # itself, so that rejected points are replaced rather than lost
        own = problem_of(self.cfg).sample_uniform(self, self.nu)
        if own is not None:
            parts.append(own)
        elif self.is_curved:                        # physically uniform bulk (|detJ|-weighted grid)
            parts.append(self._sample_grid(self.UnifGrid, self.nu))
        else:                                       # identity/none: parametric-uniform Sobol
            parts.append(torch.as_tensor(self.sobol.random(self.nu), dtype=DTYPE).to(utils.DEVICE))
        if getattr(self.cfg, "fixed_notch", False):    # crack stratum = fixed band about the notch
            c = self.cfg
            sx = torch.rand(self.nn_, 1, device=utils.DEVICE) * (c.nx1 - c.nx0) + c.nx0
            sy = torch.full((self.nn_, 1), c.ny, device=utils.DEVICE) + torch.randn(self.nn_, 1, device=utils.DEVICE) * c.sigma_band
            parts.append(torch.cat([sx, sy], dim=1))
        else:                                           # default: adaptive crack stratum (grid)
            parts.append(self._sample_grid(self.NotchGrid, self.nn_))
        if self.nd > 0:
            parts.append(self._sample_grid(self.Dgrid, self.nd))
        if self.n_add > 0:
            parts.append(self._sample_grid(self.Ngrid, self.n_add))
        return torch.cat(parts, dim=0)

    def _grid_density(self, grid, x, y):
        if grid is None:
            return torch.zeros_like(x)
        n = self.gn
        ix = torch.clamp((x / self.cell).long(), 0, n - 1)
        iy = torch.clamp((y / self.cell).long(), 0, n - 1)
        Dg = torch.tensor(grid, dtype=DTYPE, device=utils.DEVICE)
        p = Dg[iy.squeeze(-1), ix.squeeze(-1)].unsqueeze(-1)     # categorical prob of the cell
        return p / (self.cell ** 2)                             # -> density (per area)

    def _notch_density(self, x, y):
        """Density of the fixed notch band used by --fixed_notch: Gaussian in y about ny times
           uniform in x over [nx0, nx1]. It does not depend on phi and is symmetric about y = ny,
           unlike the adaptive crack stratum, whose density follows the current phi."""
        c = self.cfg
        sig, L = c.sigma_band, c.nx1 - c.nx0
        s2pi = float(np.sqrt(2.0 * np.pi)); s2 = float(np.sqrt(2.0))
        gy = torch.exp(-((y - c.ny) ** 2) / (2 * sig ** 2)) / (sig * s2pi)
        a_lo = (c.nx0 - x) / (sig * s2); a_hi = (c.nx1 - x) / (sig * s2)
        return (1.0 / L) * gy * 0.5 * (torch.erf(a_hi) - torch.erf(a_lo))

    def rho_and_mask(self, pts):
        c = self.cfg
        x, y = pts[:, 0:1], pts[:, 1:2]
        inom = problem_of(c).in_domain(x, y, c).to(DTYPE)
        unif = self._grid_density(self.UnifGrid, x, y) if self.is_curved else inom / self.dom_area
        notch = (self._notch_density(x, y) if getattr(c, "fixed_notch", False)
                 else self._grid_density(self.NotchGrid, x, y))
        rho = (self.fu * unif + self.fn * notch
               + self.fd * self._grid_density(self.Dgrid, x, y)
               + self.fnew * self._grid_density(self.Ngrid, x, y))
        return rho, inom

def estimate_Pi(net, pts, sampler, cfg, phi_prev_vals, u_delta, netT=None, t_now=0.0):
    x, y = pts[:, 0:1].clone(), pts[:, 1:2].clone()
    exx, eyy, exy, phi, gradphi2, detJ, lap_phi = kinematics(net, x, y, cfg, u_delta)
    a_th = thermal_strain(cfg, netT, pts[:, 0:1], pts[:, 1:2], t_now)
    e_el, e_fr, e_ir = energy_terms(exx, eyy, exy, phi, gradphi2, phi_prev_vals, cfg, a_th=a_th,
                                    x=pts[:, 0:1], y=pts[:, 1:2], lap_phi=lap_phi)
    # Pi = int_param f |detJ| dxi ~ E_rho[f |detJ| / rho]  (detJ = 1 with --geo none)
    rho, inom = sampler.rho_and_mask(pts)
    w = (inom * detJ / rho.clamp_min(1e-300)).detach()
    Pi_el = (e_el * w).mean()
    Pi_fr = (e_fr * w).mean()
    Pi_ir = (e_ir * w).mean()
    return Pi_el, Pi_fr, Pi_ir, phi.detach()


# --------------------------------------------------------------------------
# reaction force


def reaction_force(net, cfg, phi_prev_model, u_delta_nd, seed=999, netT=None, t_now=0.0):
    """F = dPi_el/du_delta,nd / U_ref, the energy-consistent reaction in N per unit thickness
       (the energy is per unit thickness of the 2-D body). The fracture and irreversibility
       terms do not depend on u_delta, so the gradient of the total energy equals that of the
       elastic energy.

       An example whose load is not carried by the displacement ansatz computes the reaction
       itself (see Problem.reaction_force)."""
    own = problem_of(cfg).reaction_force(net, cfg, phi_prev_model, u_delta_nd, seed)
    if own is not None:
        return own
    smp = Sampler(cfg, prev_model=phi_prev_model, N=cfg.N_meter, seed=seed)
    pts = smp.sample()
    phi_prev = phi_value(phi_prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
    ud = torch.tensor(float(u_delta_nd), device=utils.DEVICE, requires_grad=True)
    Pi_el, _, _, _ = estimate_Pi(net, pts, smp, cfg, phi_prev, ud, netT=netT, t_now=t_now)
    # allow_unused: a body carrying no displacement load at all (a quenched slab, loaded only by
    # its own thermal contraction) has no path from the energy to delta, and no reaction to report
    dPi = torch.autograd.grad(Pi_el, ud, allow_unused=True)[0]
    return 0.0 if dPi is None else float(dPi) / cfg.U_ref


# --------------------------------------------------------------------------
# load stepping


def mech_adam(net, cfg):
    """The mechanics optimizer: MLP weights and encoding grids with their own learning rates."""
    groups = [{"params": net.mlp.parameters(), "lr": cfg.lr},
              {"params": net.enc.parameters(), "lr": cfg.lr_enc}]
    return torch.optim.Adam(groups, eps=cfg.adam_eps)


def solve_step(net, prev_model, cfg, ud, save_dir, logger, step, n_iter, netT=None, t_now=0.0):
    """Minimize the total energy Pi over the MLP weights and the encoding grids at a fixed load,
       warm-started from the previous step, with Adam and a plateau stop. The encoding has its
       own learning rate and a small L2 penalty that keeps rarely sampled cells from drifting.
       netT and t_now supply the temperature at which the mechanics is solved, frozen for the
       step. With --accept 1 the end state is checked against the start state
       (pfte/stepcheck.py), and a rejected step is continued in chunks of --accept_chunk
       iterations, up to --accept_extra more. This is off by default, and a step that passes
       the check is not changed."""
    if cfg.adam_persist and getattr(net, "_adam_persist", None) is not None:
        # --adam_persist: the moments carry over from the previous step, so the second moment is
        # already adapted to the sampling noise when the step starts. The rates are reset to the
        # configured values in case a warm-up or polish phase left them scaled, and so is eps,
        # because load_state_dict on --resume restores the eps stored in the checkpoint.
        opt = net._adam_persist
        for g, b0 in zip(opt.param_groups, (cfg.lr, cfg.lr_enc)):
            g["lr"] = b0
            g["eps"] = cfg.adam_eps
    else:
        opt = mech_adam(net, cfg)
        if cfg.adam_persist:
            net._adam_persist = opt
    lr_base = [g["lr"] for g in opt.param_groups]          # for --lr_warmup
    sampler = Sampler(cfg, prev_model=prev_model, seed=cfg.seed + 1000 * step)
    if sampler.n_add > 0:
        logger.info(f"  sampler: +{sampler.n_add} adaptive pts on the new band (total {sampler.Nt})")
    pts = sampler.sample()
    phi_prev = phi_value(prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
    hist = []
    t0 = time.time()
    terms = None

    def iterate(it, last, plateau=True):
        """One Adam update. Returns True when the plateau test stops the step."""
        nonlocal pts, phi_prev, terms
        net.enc.set_progress(step, it)
        if it > 0 and it % cfg.resample_every == 0:
            # Redraw the points only (the Sobol and rng streams advance). The density grids stay
            # the ones built from prev_model at the start of the step, which also avoids
            # rebuilding the drive grid at every iteration.
            pts = sampler.sample()
            phi_prev = phi_value(prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
        opt.zero_grad()
        Pi_el, Pi_fr, Pi_ir, _ = estimate_Pi(net, pts, sampler, cfg, phi_prev,
                                             cfg.u_delta_nd_step, netT=netT, t_now=t_now)
        Pi = Pi_el + Pi_fr + Pi_ir
        loss = Pi
        # boundary conditions that cannot be imposed exactly on a curved patch enter here
        Pi_bc = cfg.problem.extra_loss(net, cfg, cfg.u_delta_nd_step)
        if Pi_bc is not None:
            loss = loss + Pi_bc
        if cfg.lambda_reg > 0:
            loss = loss + cfg.lambda_reg * sum((p ** 2).sum() for p in net.mlp.parameters())
        if cfg.enc_reg > 0:
            loss = loss + cfg.enc_reg * net.enc.reg()
        if cfg.lr_final > 0 and n_iter > 1:
            # --lr_final: exponential decay of both rates across the step to lr_final / lr of
            # their start values, times the warm-up ramp if any. The stationary spread of Adam
            # scales with lr x gradient noise, so a decaying step ends with less noise. The
            # continuation iterations of --accept (it >= n_iter) extend the same exponential
            # decay below lr_final.
            f = min(1.0, (it + 1) / cfg.lr_warmup) if cfg.lr_warmup > 0 else 1.0
            dec = (cfg.lr_final / cfg.lr) ** (it / (n_iter - 1))
            for g, b0 in zip(opt.param_groups, lr_base):
                g["lr"] = b0 * f * dec
        elif cfg.lr_warmup > 0 and it < cfg.lr_warmup:
            # linear ramp from 0. A fresh Adam takes steps of about lr on parameters dominated by
            # noise, which disturbs the field at the start of the step.
            f = (it + 1) / cfg.lr_warmup
            for g, b0 in zip(opt.param_groups, lr_base):
                g["lr"] = b0 * f
        elif cfg.lr_warmup > 0 and it == cfg.lr_warmup:
            for g, b0 in zip(opt.param_groups, lr_base):
                g["lr"] = b0
        loss.backward(); opt.step()
        hist.append(float(Pi.detach()))
        # the energy split of the current state, for energies.txt. Detached references only, so
        # there is no host sync inside the loop and no graph is kept alive.
        terms = (Pi_el.detach(), Pi_fr.detach(), Pi_ir.detach(),
                 None if Pi_bc is None else Pi_bc.detach())
        if it % cfg.log_every == 0 or last:
            bc = "" if Pi_bc is None else f" bc {float(Pi_bc.detach()):.3e}"
            logger.info(f"  step {step:3d} delta {cfg.U_ref*ud:.3e}  it {it:5d}  "
                        f"Pi {hist[-1]:.5e}  (el {float(Pi_el.detach()):.4e} "
                        f"fr {float(Pi_fr.detach()):.4e} ir {float(Pi_ir.detach()):.3e}{bc})  "
                        f"gfine {net.enc.fine_max():.2e}  gate {net.enc.gates[-1]:.2f}  "
                        f"{time.time()-t0:5.0f}s")
        # plateau early-stop (only once all levels are gated on)
        if plateau and len(hist) > cfg.plateau_win and net.enc.gates[-1] >= 1.0:
            recent = hist[-cfg.plateau_win:]
            if (max(recent) - min(recent)) / (abs(np.mean(recent)) + 1e-30) < cfg.plateau_tol:
                logger.info(f"  step {step} plateau @ it {it}")
                return True
        return False

    for it in range(n_iter):
        if iterate(it, it == n_iter - 1):
            break
    if cfg.polish_iters > 0:
        # Annealed end of step. The points are still redrawn at every iteration, polish_N x N of
        # them, while the learning rates decay to zero on a cosine. This stochastic-approximation
        # end averages the sampling noise out instead of ending on one draw from the noise floor
        # of the optimizer. A fixed point set is not used, because the encoding has enough
        # parameters to overfit it. Off by default.
        psm = Sampler(cfg, prev_model=prev_model, N=int(round(cfg.polish_N * cfg.N)),
                      seed=cfg.seed + 1000 * step + 777)
        base = [g["lr"] for g in opt.param_groups]
        net.enc.set_progress(step + 1, 0)
        P = cfg.polish_iters
        for j in range(P):
            ppts = psm.sample()
            pprev = phi_value(prev_model, ppts[:, 0:1], ppts[:, 1:2], cfg)
            fac = 0.5 * (1.0 + math.cos(math.pi * j / P))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * fac
            opt.zero_grad()
            Pi_el, Pi_fr, Pi_ir, _ = estimate_Pi(net, ppts, psm, cfg, pprev,
                                                 cfg.u_delta_nd_step, netT=netT, t_now=t_now)
            Pi = Pi_el + Pi_fr + Pi_ir
            loss = Pi
            Pi_bc = cfg.problem.extra_loss(net, cfg, cfg.u_delta_nd_step)
            if Pi_bc is not None:
                loss = loss + Pi_bc
            if cfg.lambda_reg > 0:
                loss = loss + cfg.lambda_reg * sum((p ** 2).sum() for p in net.mlp.parameters())
            if cfg.enc_reg > 0:
                loss = loss + cfg.enc_reg * net.enc.reg()
            loss.backward(); opt.step()
            hist.append(float(Pi.detach()))
            terms = (Pi_el.detach(), Pi_fr.detach(), Pi_ir.detach(),
                     None if Pi_bc is None else Pi_bc.detach())
            if j == 0 or j == P - 1 or (j + 1) % cfg.log_every == 0:
                logger.info(f"  step {step:3d} polish {j:5d}/{P}  Pi {hist[-1]:.5e}  (el "
                            f"{float(Pi_el.detach()):.4e} fr {float(Pi_fr.detach()):.4e} ir "
                            f"{float(Pi_ir.detach()):.3e})  lr x{fac:.3f}  pts {ppts.shape[0]}  "
                            f"{time.time()-t0:5.0f}s")
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0
    if cfg.accept and prev_model is not None:
        # step-end acceptance: a rejected end state is continued
        from .stepcheck import check_step_end
        extra = 0
        while True:
            net.enc.set_progress(step + 1, 0)
            ok, line = check_step_end(net, prev_model, cfg, ud, netT, t_now)
            logger.info(f"  step {step} end check after {len(hist)} its: {line}")
            if ok:
                break
            if extra >= cfg.accept_extra:
                logger.info(f"  step {step} end state kept although not accepted: the budget of "
                            f"{cfg.accept_extra} its is spent")
                break
            it0 = len(hist)
            for it in range(it0, it0 + cfg.accept_chunk):
                iterate(it, it == it0 + cfg.accept_chunk - 1, plateau=False)
            extra += cfg.accept_chunk
    net.enc.set_progress(step + 1, 0)              # gates fully on for output and the next step
    return hist, tuple(0.0 if t is None else float(t) for t in terms)


def run_cli(problem_name, argv=None, description=None, defaults=None):
    """Parse a command line for one example and run it. This is the whole body of an entry
       script.

       `defaults` presets the flags for one case of that example, so the script reproduces the
       published run with no arguments while every flag can still be set on the command line."""
    problem = utils.get_problem(problem_name)
    ap = build_parser(problem, description=description)
    if defaults:
        ap.set_defaults(**defaults)
    return run(ap.parse_args(argv), problem)


def validate(cfg):
    """Reject option combinations the solver does not implement, before the run starts."""
    assert len(cfg.enc_start) == len(cfg.levels), "--enc_start must list one start per level"
    if cfg.pf_order == 4:
        assert cfg.at == 2, "--pf_order 4 is the AT2-type fourth-order model (Borden), use --at 2"
        assert cfg.geo in ("identity", "none"), \
            "--pf_order 4 needs identity or none geometry, the Laplacian pullback on a curved " \
            "patch is not implemented"
    if cfg.geo != "none":
        assert cfg.split != "directional", "the directional split requires --geo none"
        assert cfg.grip_release <= 0, "--grip_release requires --geo none"
    if getattr(cfg, "fixed_notch", False):
        # the fixed band is drawn in the parametric square directly from nx0/nx1/ny, which for
        # other examples are physical notch coordinates, not parametric ones
        assert cfg.problem_name == "sen", \
            "--fixed_notch is the notched-square band and does not apply to " \
            f"'{cfg.problem_name}' (its notch line is not in the parametric square)"


def run(a, problem):
    """Run one example end to end. `a` is the parsed command line, `problem` its Problem."""
    cfg = Cfg(a, problem=problem)
    validate(cfg)
    os.makedirs(a.out, exist_ok=True)
    logger = get_logger(a.out)
    logger.info(f"device={utils.DEVICE}  dtype={utils.DTYPE}  problem={cfg.problem_name}  "
                f"cfg={json.dumps(cfg.to_dict())}")
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    cfg.geomap = problem.build_geomap(cfg, logger)
    problem.validate(cfg, logger)                           # fail fast, before any training
    plot_geometry(cfg, a.out)                               # patch figure (once, fixed geometry)
    plot_geometry_bc(cfg, a.out)                            # physical space + BC / control net
    plot_parametric(cfg, a.out)                             # parametric [0,1]^2 reference domain
    problem.plot_domain(cfg, a.out)                         # example-specific geometry figure
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    net = NetMS(cfg).to(utils.DEVICE)
    n_grid = sum(G.numel() for G in net.enc.grids)
    logger.info(f"encoding: levels={list(cfg.levels)} ch={cfg.enc_ch} "
                f"grid params={n_grid} (finest h={1.0/(cfg.levels[-1]-2):.4f} = "
                f"{1.0/(cfg.levels[-1]-2)/cfg.l:.2f}l)")
    if problem.setup(net, cfg, a.out, logger):              # an example with no load stepping
        logger.info(f"done -> {a.out}")
        return cfg, net, None
    schedule, times = problem.schedules(cfg)
    FD = []
    EN = []                    # the energy history, read back from its file on --resume (below)
    prev_model = None
    start_step = 0
    netT = prevT_model = prevT_state = None
    if cfg.heat_on:
        from .thermal import NetT, audit_T, fit_T_step0, solve_T_step, solve_T_steady
        if cfg.T_analytic:
            # the temperature is prescribed in closed form, with no network and no solves. The
            # mechanics reads the analytic lift through thermal_a(netT=None).
            logger.info(f"T analytic (prescribed lift, no T-net)  T0={cfg.T0} "
                        f"dT_final={cfg.dT_final}  t: {times[0]:.3e} -> {times[-1]:.3e} s")
        else:
            netT = NetT(cfg).to(utils.DEVICE)
            nT = sum(G.numel() for G in netT.enc.grids)
            logger.info(f"T-net: levels={list(cfg.levels_T)} ch={cfg.enc_ch_T} grid params={nT}  "
                        f"T0={cfg.T0} dT_final={cfg.dT_final} kdeg={cfg.kdeg}  "
                        f"t: {times[0]:.3e} -> {times[-1]:.3e} s (power {cfg.time_power})")
    ckpt = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt):
        ck = load_ckpt(ckpt, net, netT)
        start_step, FD = ck["step"] + 1, ck["FD"]
        if ck["prev"] is not None:
            prev_model = NetMS(cfg).to(utils.DEVICE); prev_model.load_state_dict(ck["prev"])
            for p in prev_model.parameters(): p.requires_grad_(False)
            prev_model.eval()
        if cfg.heat_on and ck.get("prevT") is not None:
            prevT_state = ck["prevT"]
            prevT_model = NetT(cfg).to(utils.DEVICE); prevT_model.load_state_dict(prevT_state)
            for p in prevT_model.parameters(): p.requires_grad_(False)
            prevT_model.eval()
        net.enc.set_progress(1, 0)
        if cfg.adam_persist and ck.get("adam") is not None:
            # --adam_persist: continue with the stored moments rather than a fresh Adam
            net._adam_persist = mech_adam(net, cfg)
            net._adam_persist.load_state_dict(ck["adam"])
            logger.info("  resumed the persistent Adam state from the checkpoint")
        cfg.grip_released = tuple(bool(b) for b in
                                  ck["cfg"].get("grip_released", [False, False, False, False]))
        # the energy history is kept in its own file rather than in the checkpoint. Only the
        # steps already done are kept, so a resume that redoes a step overwrites it instead of
        # duplicating it.
        en_path = os.path.join(a.out, "energies.txt")
        if os.path.exists(en_path):
            prev_en = np.atleast_2d(np.loadtxt(en_path))
            if prev_en.size:
                EN = [row.tolist() for row in prev_en if row[0] < start_step]
        logger.info(f"resumed at step {start_step}  grip_released={cfg.grip_released}")

    for step in range(start_step, len(schedule)):
        ud = schedule[step]
        t_now = times[step]
        t_prev = times[step - 1] if step > 0 else times[0]
        cfg.u_delta_nd_step = ud                          # load parameter of the current step
        cfg.ud_prev = schedule[step - 1] if step > 0 else 0.0   # previous load, for drive_field
        cfg.t_prev, cfg.netT_prev = t_prev, prevT_model   # what the sampler's drive field reads
        cfg.t_now, cfg.netT_now = t_now, netT             # what an example's plot hook reads
        update_grip_release(cfg, prev_model, logger)      # opt-in corner-gate release
        if cfg.split == "directional":
            cfg.dirfield = DirField(prev_model, cfg)      # frozen orientation field for this step
        # Step 0 gets --iters0 and step 1 --iters1 (default --iters). In the heated notched
        # plate and the slab and annulus schedules step 0 is a zero-load state at t = 0 and step 1
        # the first temperature change. In the strip step 0 already carries the developed front
        # and is a full solve.
        n_iter = cfg.iters0 if step == 0 else (cfg.iters1 if step == 1 else cfg.iters)
        logger.info(f"=== load step {step}/{len(schedule)-1}  delta={cfg.U_ref*ud:.4e}  "
                    + (f"t={t_now:.4e}s  " if cfg.heat_on else "") + f"iters={n_iter} ===")

        # The thermal block runs first, so the mechanics is solved at a temperature already
        # converged for this step. The heat block is the minimization of one functional (with
        # the hybrid split, the mechanics follows the gradients of the staggered hybrid scheme).
        if cfg.heat_on and cfg.T_analytic:
            # prescribed field: nothing to solve, only report the state the mechanics will see
            tmin, tmax, tmean = audit_T(None, cfg, t_now)
            logger.info(f"    T(analytic) in [{tmin:.2f}, {tmax:.2f}] mean {tmean:.2f}")
        elif cfg.heat_on and cfg.T_steady:
            # steady and linear in its boundary data: solved once at full load and scaled per
            # step (thermal.T_effective). Nothing to do after the pre-solve.
            if step == start_step and prevT_model is None:
                smpT = Sampler(cfg, prev_model=prev_model, seed=cfg.seed + 3)
                solve_T_steady(netT, cfg, smpT, logger, cfg.iters_T0)
                prevT_state = copy.deepcopy(netT.state_dict())
                prevT_model = NetT(cfg).to(utils.DEVICE); prevT_model.load_state_dict(prevT_state)
                for p in prevT_model.parameters(): p.requires_grad_(False)
                prevT_model.eval()
            tmin, tmax, tmean = audit_T(netT, cfg, t_now)
            logger.info(f"    T(frac {t_now:.3f}) in [{tmin:.2f}, {tmax:.2f}] "
                        f"mean {tmean:.2f} K")
        elif cfg.heat_on:
            smpT = Sampler(cfg, prev_model=prev_model, seed=cfg.seed + 1000 * step + 3)
            if step == start_step and prevT_model is None:
                if t_now == 0.0 and cfg.T_zero_init:
                    # at t = 0 the head weights start at zero, so with --T_head_bias 0 (the
                    # default) T equals the lift T_D, the example's initial state, and no fit is
                    # needed
                    logger.info("  [T0] start state = lift T_D at t = 0 (zero-initialized head), "
                                "no fit needed")
                else:
                    # cold start: fit the field to the reference state of the example at the
                    # first time, so step 1 does not start from an arbitrary temperature
                    fit_T_step0(netT, cfg, smpT, t_now, logger, cfg.iters_T0)
            else:
                solve_T_step(netT, prevT_model, prev_model, cfg, smpT, t_now, t_prev,
                             logger, step, cfg.iters_T1 if step == 1 else cfg.iters_T)
            tmin, tmax, tmean = audit_T(netT, cfg, t_now)
            logger.info(f"    T in [{tmin:.2f}, {tmax:.2f}] mean {tmean:.2f} K")
            prevT_state = copy.deepcopy(netT.state_dict())
            prevT_model = NetT(cfg).to(utils.DEVICE); prevT_model.load_state_dict(prevT_state)
            for p in prevT_model.parameters(): p.requires_grad_(False)
            prevT_model.eval()

        _, terms = solve_step(net, prev_model, cfg, ud, a.out, logger, step, n_iter,
                              netT=netT, t_now=t_now)

        F = reaction_force(net, cfg, prev_model, ud, netT=netT, t_now=t_now)
        FD.append([cfg.U_ref * ud, F])
        EN.append([step, cfg.U_ref * ud, t_now, *terms])
        logger.info(f"    -> F = {F:.4f} N per unit thickness")

        # freeze the current net (MLP and encoding) as phi_prev and sampler source for the next
        # step
        prev_state = copy.deepcopy(net.state_dict())
        prev_model = NetMS(cfg).to(utils.DEVICE); prev_model.load_state_dict(prev_state)
        for p in prev_model.parameters(): p.requires_grad_(False)
        prev_model.eval()
        prev_model.enc.set_progress(1, 0)

        ck_extra = dict(netT=netT, prevT_state=(prevT_state if cfg.heat_on else None),
                        t_now=t_now,
                        opt=(getattr(net, "_adam_persist", None) if cfg.adam_persist else None))
        save_ckpt(ckpt, net, prev_state, step, ud, FD, cfg, **ck_extra)   # rolling (--resume)
        last = (step == len(schedule) - 1)
        step_dir = os.path.join(a.out, f"step_{step:03d}")
        if cfg.ckpt_every > 0 and (step % cfg.ckpt_every == 0 or last):
            os.makedirs(step_dir, exist_ok=True)              # per-step snapshot folder
            save_ckpt(os.path.join(step_dir, "ckpt.pt"), net, prev_state, step, ud, FD, cfg,
                      **ck_extra)
        if cfg.plot_every > 0:
            plot_now = (step % cfg.plot_every == 0) or last
        else:
            plot_now = (step % max(1, len(schedule) // 12) == 0) or last
        if plot_now:
            os.makedirs(step_dir, exist_ok=True)
            smp = Sampler(cfg, prev_model=prev_model, seed=cfg.seed + 7)
            plot_phi(net, cfg, step_dir, f"s{step:03d}", ud, sample_pts=smp.sample().cpu().numpy())
            plot_linecut(net, cfg, step_dir, f"s{step:03d}", ud)
            plot_fields(net, cfg, step_dir, f"s{step:03d}", ud)
            plot_FD_solo(FD, cfg, step_dir)                   # curve alone -> step folder
            if cfg.heat_on:
                plot_T(netT, cfg, step_dir, f"s{step:03d}", t_now)
            problem.plot(net, cfg, step_dir, f"s{step:03d}", ud)
        plot_FD_compare(FD, cfg, a.out, a.fem)                # compare curve -> run root only
        np.savetxt(os.path.join(a.out, "FD.txt"), np.array(FD), header="delta(mm)  F(N)")
        # The energy history at full precision. The examples without a mechanical load report
        # F = 0 by construction, so for them the energies, not FD.txt, record the response.
        np.savetxt(os.path.join(a.out, "energies.txt"), np.array(EN),
                   header="step  delta(mm)  t(s)  Pi_el  Pi_fr  Pi_ir  Pi_bc")

    logger.info("=" * 70)
    FD = np.array(FD)
    if len(FD):
        ipk = int(np.argmax(FD[:, 1]))
        logger.info(f"peak F {FD[ipk,1]:.3f} N per unit thickness @ delta {FD[ipk,0]:.3e}")
    plot_FD_compare(FD.tolist(), cfg, a.out, a.fem)
    plot_phi(net, cfg, a.out, "final", schedule[-1])
    plot_linecut(net, cfg, a.out, "final", schedule[-1])
    problem.finalize(net, cfg, a.out, logger)
    logger.info(f"done -> {a.out}")
    return cfg, net, FD
