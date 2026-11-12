"""
Step-end acceptance of the mechanical solve.

A load step ends on the last iterate of its budget, or of its plateau, and the irreversibility of
the next step then retains that state. An iterate that happens to sit on a transient of the
optimizer is therefore carried into every later step. The check here compares the start state of
the step (phi_prev, the accepted state of the previous step) with its end state under the
conditions of the step, on one regular grid and with nothing taken from the minimization itself:

  - the functional of the step, el + fr + ir, at both states,
  - the phase-field Euler-Lagrange residual
        r = Gc (phi/l - l Lap phi) - 2 (1 - phi) D - 2 gamma_ir (phi_prev - ir_tol - phi)_+
    (D the spectral tension energy that drives phi in the hybrid split), as its rms in units of
    Gc/l over one region for both states, the union of their bands phi > 0.05,
  - the largest violation of irreversibility, max (phi_prev - ir_tol - phi)_+.

An end state is rejected when the functional rises above (1 + accept_pi) times that of the start
state and the band residual above accept_r times that of the start state, or when the violation
exceeds its tolerance. The
residual alone also rises on a regular step whose damage grows strongly, so it is not used by
itself, and the functional alone is not a convergence measure of the hybrid update. A rejected
step is continued rather than replaced by the iterate of lowest loss.

The check is written for the second-order AT2 model with the hybrid split only. It applies the
irreversibility penalty everywhere, as with --ir_thresh 0, so with --ir_thresh > 0 its functional
and residual differ from those of the step, and healing below the threshold, which the step
allows, counts as a violation. It also assumes a thermal run with ezz_e = 0 (plane stress or
--alpha_mode 2d) and a toughness without the temperature or boundary factors of energy_terms
(--gc_tdep, --gc_grip and the example's own Gc factor).

The grid derivatives assume an affine map from the parametric square to a W x H rectangle (the
quenched slab and the glass strip). Curved patches are refused.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import numpy as np
import torch

from . import utils
from .utils import DTYPE
from .solver import _fields, phi_value, thermal_strain, material_fields, elastic_parts

DEV = utils.DEVICE


def spectral_D(exx, eyy, exy, a_th, cfg, lam_l, mu_l):
    """The tension part of the spectral split, the energy that drives phi in the hybrid split."""
    exx_e, eyy_e, ezz_e, tr = elastic_parts(exx, eyy, a_th, cfg)
    tr_p = torch.clamp(tr, min=0.0)
    tr2 = exx_e + eyy_e
    rad = torch.sqrt(((exx_e - eyy_e) * 0.5) ** 2 + exy ** 2 + 1e-30)
    e1 = tr2 * 0.5 + rad
    e2 = tr2 * 0.5 - rad
    return (0.5 * lam_l * tr_p ** 2
            + mu_l * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2
                      + torch.clamp(ezz_e, min=0.0) ** 2))


def grid_evaluation(cfg, net, prev, netT, t, ud, ngx, ngy, chunk=16384):
    """Energies on a regular parametric grid (uniform quadrature, the integral = mean of
       f detJ over the unit square) and the phase-field Euler-Lagrange residual field.
       Returns (el, fr, ir), the residual field and the phase field, both (ngy, ngx)."""
    W, H = cfg.strip_w, cfg.strip_h
    xs = (np.arange(ngx) + 0.5) / ngx
    ys = (np.arange(ngy) + 0.5) / ngy
    GX, GY = np.meshgrid(xs, ys)
    X = torch.tensor(GX.reshape(-1, 1), dtype=DTYPE, device=DEV)
    Y = torch.tensor(GY.reshape(-1, 1), dtype=DTYPE, device=DEV)
    n = X.shape[0]
    acc = np.zeros(3)
    R = np.zeros(n); PHI = np.zeros(n)
    l = cfg.l
    for i0 in range(0, n, chunk):
        x = X[i0:i0 + chunk].clone().requires_grad_(True)
        y = Y[i0:i0 + chunk].clone().requires_grad_(True)
        u, v, phi, _ = _fields(net, x, y, cfg, ud)
        ou = torch.ones_like(u)
        ux = torch.autograd.grad(u, x, ou, create_graph=True)[0]
        uy = torch.autograd.grad(u, y, ou, create_graph=True)[0]
        vx = torch.autograd.grad(v, x, ou, create_graph=True)[0]
        vy = torch.autograd.grad(v, y, ou, create_graph=True)[0]
        px = torch.autograd.grad(phi, x, ou, create_graph=True)[0]
        py = torch.autograd.grad(phi, y, ou, create_graph=True)[0]
        pxx = torch.autograd.grad(px, x, ou, create_graph=True)[0]
        pyy = torch.autograd.grad(py, y, ou, create_graph=True)[0]
        # the strip map is affine: physical derivatives are the parametric ones over W and H
        exx, eyy, exy = ux / W, vy / H, 0.5 * (uy / H + vx / W)
        gradphi2 = (px / W) ** 2 + (py / H) ** 2
        lap = pxx / W ** 2 + pyy / H ** 2
        a_th = thermal_strain(cfg, netT, x, y, t)
        lam_l, mu_l, _, Gc = material_fields(x, y, cfg)
        with torch.no_grad():
            phi_prev = phi_value(prev, x, y, cfg)
            D = spectral_D(exx, eyy, exy, a_th, cfg, lam_l, mu_l)
            psi_full = (0.5 * lam_l * (exx - a_th + eyy - a_th) ** 2
                        + mu_l * ((exx - a_th) ** 2 + (eyy - a_th) ** 2 + 2.0 * exy ** 2))
            g = (1.0 - phi) ** 2 + cfg.g_floor
            e_el = g * psi_full
            e_fr = 0.5 * Gc * (phi ** 2 / l + l * gradphi2)
            pen = torch.clamp(phi_prev - cfg.ir_tol - phi, min=0.0)
            e_ir = cfg.gamma_ir * pen ** 2
            r = Gc * (phi / l - l * lap) - 2.0 * (1.0 - phi) * D - 2.0 * cfg.gamma_ir * pen
            acc += np.array([float(e_el.sum()), float(e_fr.sum()), float(e_ir.sum())])
            R[i0:i0 + chunk] = r.squeeze(-1).cpu().numpy()
            PHI[i0:i0 + chunk] = phi.squeeze(-1).cpu().numpy()
    Pi = acc / n * (W * H)
    return Pi, R.reshape(ngy, ngx), PHI.reshape(ngy, ngx)


def residual_metrics(R, mask, Gc_l):
    """rms of the Euler-Lagrange residual in units of Gc/l, over the domain and over `mask`,
       and its 99th percentile over the domain."""
    rn = np.abs(R) / Gc_l
    return (float(np.sqrt((rn ** 2).mean())),
            float(np.sqrt((rn[mask] ** 2).mean())) if mask.any() else 0.0,
            float(np.percentile(rn, 99)))


def accept_grid(cfg):
    """Grid of the check: spacing accept_h (mm) on the W x H rectangle."""
    return (max(8, int(round(cfg.strip_w / cfg.accept_h))),
            max(8, int(round(cfg.strip_h / cfg.accept_h))))


def compare_states(cfg, net, prev, netT, t, ud, ngx, ngy):
    """Start state (prev) against end state (net) under one step's conditions on one grid."""
    Ps, Rs, PHs = grid_evaluation(cfg, prev, prev, netT, t, ud, ngx, ngy)
    Pe, Re, PHe = grid_evaluation(cfg, net, prev, netT, t, ud, ngx, ngy)
    mask = (PHs > 0.05) | (PHe > 0.05)
    Gc_l = cfg.Gc / cfg.l
    ms, me = residual_metrics(Rs, mask, Gc_l), residual_metrics(Re, mask, Gc_l)
    max_d = float(np.clip(PHs - cfg.ir_tol - PHe, 0.0, None).max())
    return dict(Pi_s=float(Ps.sum()), Pi_e=float(Pe.sum()), Ps=Ps, Pe=Pe,
                rdom_s=ms[0], rdom_e=me[0], rband_s=ms[1], rband_e=me[1],
                r99_s=ms[2], r99_e=me[2], band_mm2=float(mask.mean()) * cfg.strip_w * cfg.strip_h,
                phi_max_e=float(PHe.max()), max_d=max_d)


def check_step_end(net, prev, cfg, ud, netT, t_now):
    """(accepted, one log line). The network's parameters are frozen for the evaluation and
       restored afterwards. Nothing random is drawn, so a step that passes continues exactly as
       it would without the check."""
    assert hasattr(cfg, "strip_w") and str(cfg.geo).startswith("strip"), \
        "the step-end check needs an affine strip map (the slab or the strip)"
    flags = [p.requires_grad for p in net.parameters()]
    was_training = net.training
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    try:
        ngx, ngy = accept_grid(cfg)
        m = compare_states(cfg, net, prev, netT, t_now, ud, ngx, ngy)
    finally:
        for p, f in zip(net.parameters(), flags):
            p.requires_grad_(f)
        net.train(was_training)
    rise_pi = m["Pi_e"] > (1.0 + cfg.accept_pi) * m["Pi_s"]
    rise_r = m["rband_e"] > cfg.accept_r * m["rband_s"]
    viol = m["max_d"] > cfg.accept_dmax
    ok = not ((rise_pi and rise_r) or viol)
    line = ("Pi %.5e -> %.5e (%+.1f %%)  r_band %.4f -> %.4f (x%.2f)  max d %.2e  -> %s"
            % (m["Pi_s"], m["Pi_e"], 100.0 * (m["Pi_e"] / m["Pi_s"] - 1.0) if m["Pi_s"] else float("nan"),
               m["rband_s"], m["rband_e"],
               m["rband_e"] / m["rband_s"] if m["rband_s"] > 0 else float("inf"), m["max_d"],
               "accepted" if ok else "rejected"))
    return ok, line
