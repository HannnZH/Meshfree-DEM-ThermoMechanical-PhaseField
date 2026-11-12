"""Staggered AT2 phase-field finite element reference (scikit-fem) for the directionally
quenched glass strip of Yuse and Sano (1993), the `yusesano` problem of the mesh-free solver,
built on the kernels of pf_fem.py in this directory.

The model is the mesh-free solver's strip, term by term (pfte/problems_thermal.py,
YuseSanoProblem):
  - the strip [0, W] x [0, H] (30 x 16 mm), plane stress, AT2 with the hybrid split, degraded
    elasticity and degraded thermal force, g = (1 - phi)^2 + g_floor;
  - the imposed moving front T(x, t) = Tc + dT - dT exp(-max(x - V t, 0) / (D / V)), stress free
    at Tc + dT, t_k = (travel / V) k / nsteps, k = 0 .. nsteps (state 0 is the developed front at
    the leading edge, as in the mesh-free solver);
  - u = 0 on the trailing edge x = W and v = 0 at its midpoint, everything else traction free;
  - the initial crack from (0, y_c) to (L_n, y_c), y_c = H/2 + offset (L_n = --crack_len,
    offset = --crack_off), seeded with the AT2 profile
    exp(-d / l) and held as a lower bound of phi at every state.
Irreversibility: the bound phi >= phi of the previous state (with the current tensile energy as
the driving force, by an active set, the default), or the mesh-free solver's one-sided penalty
(--irr penalty). Each state is solved by alternating u and phi until phi stops changing
(--stag_tol, cap --stag_max).

Outputs (in --out): ENERGY.txt per state (Pi_el, Pi_fr, staggered iterations, tip x), the phase
field on the nodal grid every --save_every states and at the end (phi_step_NNN.npy, indexed
[iy, ix], with grid_x.npy, grid_y.npy), the crack path y(x) of the final state (path_final.csv),
phi_final.png, config.txt.

Usage, with the settings of the reference in the paper (0.1 mm of front travel per state):

    python fem/pf_fem_strip.py --dT 165 --h 0.025 --nsteps 240

--nsteps 120 and 480 give 0.2 and 0.05 mm per state. With the default length scale l = 0.1 mm,
h = 0.025 mm is l/4.

Options not used by the reference of the paper (--model kflp, --at 1, --irr penalty, --noise,
--phinoise, --jitter) are kept for further study.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_fem import FastKdeg, FastPhi, FastThermalRHS, psi_plus, LINSOLVE  # noqa: E402
from skfem import MeshQuad, Basis, ElementVector, ElementQuad1, solve, condense  # noqa: E402


class Cfg:
    pass


def strip_mesh(W, H, h):
    nx = max(4, int(math.ceil(W / h)))
    ny = max(4, int(math.ceil(H / h)))
    x = np.linspace(0.0, W, nx + 1)
    y = np.linspace(0.0, H, ny + 1)
    m0 = MeshQuad.init_tensor(x, y)
    return MeshQuad(np.array([m0.p[0], m0.p[1]]), m0.t), x, y


def dist_to_segment(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    s = np.clip(((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(px - (x0 + s * dx), py - (y0 + s * dy))


def run(a):
    os.makedirs(a.out, exist_ok=True)
    cfg = Cfg()
    cfg.g_floor = a.g_floor
    cfg.alpha_mode = "2d"
    cfg.split = "hybrid"
    cfg.l = a.l
    l, gf = a.l, a.g_floor
    m, cx, cy = strip_mesh(a.W, a.H, a.h)
    # node -> nominal grid index, taken before any jitter so the outputs stay on the regular grid
    ix = np.clip(np.rint(m.p[0] / (cx[1] - cx[0])).astype(int), 0, len(cx) - 1)
    iy = np.clip(np.rint(m.p[1] / (cy[1] - cy[0])).astype(int), 0, len(cy) - 1)
    if a.jitter > 0.0:
        # --jitter: every node moved by a uniform random fraction of the element size, boundary
        # nodes only along their boundary, so the structured mesh loses its preferred directions
        # (0, 45, 90 degrees) while keeping its size and topology
        rng = np.random.default_rng(a.jitter_seed)
        pj = m.p.copy()
        onx = (np.abs(pj[0]) < 1e-9) | (np.abs(pj[0] - a.W) < 1e-9)
        ony = (np.abs(pj[1]) < 1e-9) | (np.abs(pj[1] - a.H) < 1e-9)
        pj[0] += np.where(onx, 0.0, rng.uniform(-a.jitter, a.jitter, pj.shape[1]) * (cx[1] - cx[0]))
        pj[1] += np.where(ony, 0.0, rng.uniform(-a.jitter, a.jitter, pj.shape[1]) * (cy[1] - cy[0]))
        m = MeshQuad(pj, m.t)
    nx, ny = len(cx) - 1, len(cy) - 1
    iu = Basis(m, ElementVector(ElementQuad1()), intorder=2)
    ip = Basis(m, ElementQuad1(), intorder=2)
    xq = ip.global_coordinates().value[0]
    lam = a.E * a.nu / (1.0 - a.nu ** 2)                   # plane stress
    mu = a.E / (2.0 * (1.0 + a.nu))
    fk = FastKdeg(iu, cfg, lam, mu)
    # AT2: FastPhi as it is. AT1 (--at 1): w(phi) = phi, c_w = 8/3, the equation
    # (2 Gc l / c_w)(-lap phi) + 2 H phi = 2 H - Gc / (c_w l), phi >= 0. FastPhi built with
    # Gc' = 2 Gc / c_w carries the Laplacian and a (Gc'/l) mass term. Shifting H removes that
    # mass term from A and puts the AT1 source into b:
    # A = A'(H - Gc/(c_w l)), b = b'(H - Gc/(2 c_w l)).
    cw = 8.0 / 3.0
    if a.at == 1:
        fp1 = FastPhi(ip, cfg, 2.0 * a.Gc / cw)
        sA, sb = a.Gc / (cw * l), a.Gc / (2.0 * cw * l)
        # where H = 0 (AT1's elastic phase) A is a pure Neumann Laplacian, singular: a mass term
        # of 1e-6 of the source scale makes it regular. The solution there is then large and
        # negative, and the lower bound of the active set returns phi = phi_prev there
        sA = sA * (1.0 - 1e-6)
        class _AT1:
            def A(self, H):
                return fp1.A(H - sA)
            def b(self, H):
                return fp1.b(H - sb)
        fp = _AT1()
    else:
        fp = FastPhi(ip, cfg, a.Gc)
    kf = a.model == "kflp"
    if kf:
        # The complete model of Kumar et al. (2020) with a Drucker-Prager strength surface, as
        # Zeng and Dolbow (2026) (arXiv 2602.09031, eqs 16-29) use it for this strip. In d = 1 - v:
        #   (3 delta Gc l / 4)(-lap d) + 2 H d = 2 H - c_hat - 3 delta Gc / (8 l),
        #   H = psi (full, no split) where I1 >= 0 and 0 where I1 < 0 (their v (1 - sgn I1) psi
        #   term cancels the elastic driving force in compression), c_hat = beta2 sqrt(J2) +
        #   beta1 I1 of the degraded stress, 0 <= d <= 1, and d >= d_prev only once
        #   d_prev >= 1 - v_th (their v_th = 0.05).
        # FastPhi with Gc' = 3 delta Gc / 4 carries the Laplacian. A = A'(H - 3 delta Gc/(8 l))
        # removes its mass term, and b = b'(H - c_hat/2 - 3 delta Gc/(16 l)) is the source.
        assert a.irr == "bound", "kflp: bound only"
        sts = a.sig_ts
        scs = a.cs_ratio * sts
        # Drucker-Prager: scs = 3 shs sts / (3 shs - 2 sts)
        shs = 2.0 * scs * sts / (3.0 * (scs - sts))
        Kb = a.E / (3.0 * (1.0 - 2.0 * a.nu))
        psi_ts, psi_hs = sts ** 2 / (2.0 * a.E), shs ** 2 / (2.0 * Kb)
        r8 = 1.0 + 3.0 * a.h / (8.0 * l)                      # eq 29, quadrilateral mesh correction
        kd = r8 ** -2 * (0.2201 * shs + 0.1295 * sts) / shs * 3.0 * a.Gc / (16.0 * psi_ts * l) + 0.6297 / r8
        dG8l = kd * a.Gc / (8.0 * l)
        kb1 = -dG8l / shs + 2.0 * psi_hs / (3.0 * shs)
        kb2 = (-math.sqrt(3.0) * (3.0 * shs - sts) / (shs * sts) * dG8l - 2.0 * psi_hs / (math.sqrt(3.0) * shs)
               + 2.0 * math.sqrt(3.0) * psi_ts / sts)
        fpk = FastPhi(ip, cfg, 0.75 * kd * a.Gc)
        sAk = 3.0 * dG8l * (1.0 - 1e-6)                       # 3 delta Gc/(8 l), as AT1's

        def kf_drive(u, phi_q, c_qp):
            du = iu.interpolate(u).grad
            exx, eyy = du[0, 0] - c_qp, du[1, 1] - c_qp
            exy = 0.5 * (du[0, 1] + du[1, 0])
            tr = exx + eyy
            psi = 0.5 * lam * tr ** 2 + mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
            g = (1.0 - phi_q) ** 2 + gf
            sxx, syy, sxy = g * (lam * tr + 2.0 * mu * exx), g * (lam * tr + 2.0 * mu * eyy), g * 2.0 * mu * exy
            I1 = sxx + syy
            J2 = (sxx ** 2 + syy ** 2 - sxx * syy) / 3.0 + sxy ** 2
            H = np.where(I1 >= 0.0, psi, 0.0)
            chat = kb2 * np.sqrt(np.maximum(J2, 0.0)) + kb1 * I1
            return H, H - 0.5 * chat - 1.5 * dG8l
    fth = FastThermalRHS(iu, cfg, lam, mu)
    dx = ip.dx

    # supports: u = 0 on x = W, v = 0 at (W, H/2)
    right = iu.get_dofs(lambda x: x[0] > a.W - 1e-9 * a.W)
    pin = iu.get_dofs(lambda x: (x[0] > a.W - 1e-9 * a.W) & (np.abs(x[1] - 0.5 * a.H) < 0.5 * a.h))
    D = np.unique(np.concatenate([right.all("u^1"), pin.all("u^2")[:1]]))

    # the initial crack, AT2 profile, a lower bound at every state
    yc = 0.5 * a.H + a.crack_off
    dn = dist_to_segment(m.p[0], m.p[1], 0.0, yc, a.crack_len, yc)
    phi0 = np.exp(-dn / l) if (a.at == 2 and not kf) else np.clip(1.0 - dn / (2.0 * l), 0.0, None) ** 2
    phi0[phi0 < 1e-12] = 0.0


    def to_grid(v):
        g = np.zeros((ny + 1, nx + 1))
        g[iy, ix] = v
        return g

    T0 = a.Tc + a.dT
    t_final = a.travel / a.Vpull
    times = [t_final * k / a.nsteps for k in range(a.nsteps + 1)]

    def c_of(t):
        s = np.maximum(xq - a.Vpull * t, 0.0)
        T = (a.Tc + a.dT) - a.dT * np.exp(-(a.Vpull / a.Dtherm) * s)
        return a.alpha * (T - T0)

    tol_s = a.stag_tol
    cap = a.stag_max
    b = 0.5 * a.H
    Kc = math.sqrt(a.E * a.Gc)
    R = a.alpha * a.E * a.dT * math.sqrt(b) / Kc
    print(f"[strip_fem] mesh {nx} x {ny} quads (h={a.h:g} mm = {a.h/l:.2f} l) u-dofs={iu.N} "
          f"phi-dofs={ip.N} solver={getattr(LINSOLVE, 'name', '?')} irr={a.irr}"
          + (f" gamma_ir={a.gamma_ir:g}" if a.irr == "penalty" else ""), flush=True)
    print(f"[strip_fem] dT={a.dT:g} R={R:.3f} P={b*a.Vpull/a.Dtherm:.2f} l={l:g} "
          + (f"sigma_c(AT2)={math.sqrt(27*a.E*a.Gc/(256*l)):.1f} MPa" if a.at == 2 else
             f"sigma_c(AT1)={math.sqrt(3*a.E*a.Gc/(8*l)):.1f} MPa") + f"  {a.nsteps} states, "
          f"{a.travel/a.nsteps:.4g} mm/step, stag tol {tol_s:g} cap {cap}", flush=True)
    if kf:
        print(f"[strip_fem] model kflp (Zeng and Dolbow eqs 16-29): sigma_ts {sts:g} sigma_cs {scs:g} sigma_hs {shs:.2f} MPa, "
              f"delta {kd:.4f} (eq 29, h/l {a.h/l:.2f}), beta1 {kb1:+.4e} beta2 {kb2:+.4e}, v_th {a.v_th:g}, "
              f"l_max 3GcE/(8 sts^2) = {3*a.Gc*a.E/(8*sts**2):.4f} mm", flush=True)

    rng_noise = np.random.default_rng(a.noise_seed)
    rng_pn = np.random.default_rng(a.phinoise_seed)
    phi = phi0.copy()
    if a.init_phi:
        # restart: the saved state K as phi, continued from state k0 = K + 1. Exact for both
        # irreversibility treatments up to the float32 of the saved grid: a state needs only phi
        # of the previous one (u is re-solved from phi at every staggered iteration). The noise
        # streams (--noise, --phinoise) restart from their seeds, and the saved field precedes
        # the --phinoise perturbation, so a run with either is not continued bitwise.
        phi = np.maximum(np.load(a.init_phi).astype(float)[iy, ix], phi0)
        print(f"[strip_fem] restart from {a.init_phi} (phi_max {phi.max():.3f}) at state {a.k0}", flush=True)
    u = iu.zeros()
    EN = []
    t0 = time.time()
    for k in range(a.k0, a.nsteps + 1):
        tk = time.time()
        c_qp = c_of(times[k])
        phi_prev = np.maximum(phi, phi0)
        if kf:   # irreversible only once broken (Zeng and Dolbow: v <= v_th), the notch always
            phi_prev = np.maximum(np.where(phi >= 1.0 - a.v_th, phi, 0.0), phi0)
        act_pen = None
        for it in range(cap):
            phi_q = ip.interpolate(phi).value
            g_qp = (1.0 - phi_q) ** 2 + gf
            u = solve(*condense(fk(phi_q), fth(c_qp, g_qp), x=iu.zeros(), D=D), solver=LINSOLVE)
            if kf:
                psi_q, src_q = kf_drive(u, phi_q, c_qp)
            else:
                psi_q = psi_plus(iu, u, cfg, lam, mu, c_qp)
            if a.noise > 0.0:
                # a fresh random factor on the driving force at every staggered iteration, the
                # counterpart of the mesh-free solver's per-iteration resampling of the
                # integration points
                psi_q = psi_q * (1.0 + a.noise * rng_noise.uniform(-1.0, 1.0, psi_q.shape))
            if a.irr == "penalty":
                pprev_q = ip.interpolate(phi_prev).value
                act = np.zeros(pprev_q.shape, dtype=bool) if act_pen is None else act_pen
                for _ in range(30):
                    g_act = a.gamma_ir * act
                    phi_new = solve(fp.A(psi_q + g_act), fp.b(psi_q) + fp.b(g_act * pprev_q),
                                    solver=LINSOLVE)
                    act_new = ip.interpolate(phi_new).value < pprev_q
                    if np.array_equal(act_new, act):
                        break
                    act = act_new
                act_pen = act
                phi_new = np.maximum(phi_new, phi0)       # the initial crack stays
            else:
                if kf:
                    Ap, bp = fpk.A(psi_q - sAk), fpk.b(src_q)
                else:
                    Ap, bp = fp.A(psi_q), fp.b(psi_q)
                phi_new = solve(Ap, bp, solver=LINSOLVE)
                active = np.zeros(ip.N, dtype=bool)
                upper = np.zeros(ip.N, dtype=bool)
                for _ in range(30):
                    viol = (phi_new < phi_prev - 1e-9) & ~active
                    # AT1 needs the box phi <= 1 as well (its linear source drives phi past 1)
                    over = (phi_new > 1.0 + 1e-9) & ~upper if (a.at == 1 or kf) else np.zeros(ip.N, dtype=bool)
                    if not viol.any() and not over.any():
                        break
                    active |= viol
                    upper |= over & ~active
                    xfix = phi_prev.copy()
                    xfix[upper] = 1.0
                    if (active | upper).all():
                        # every node on a bound (AT1 in its elastic phase): nothing left to solve,
                        # and an empty system is what Pardiso rejects
                        phi_new = xfix
                        break
                    phi_new = solve(*condense(Ap, bp, x=xfix, D=np.where(active | upper)[0]),
                                    solver=LINSOLVE)
            dchg = np.linalg.norm(phi_new - phi) / (np.linalg.norm(phi_new) + 1e-5)
            phi = phi_new
            if dchg < tol_s:
                break
        fi = ip.interpolate(phi)
        pq, gq = fi.value, fi.grad
        if kf:   # regularized surface energy with its delta (the work of c_hat is not included)
            e_fr = 0.375 * kd * (pq / l + l * (gq[0] ** 2 + gq[1] ** 2))
        elif a.at == 2:
            e_fr = (pq ** 2 + l * l * (gq[0] ** 2 + gq[1] ** 2)) / (2.0 * l)
        else:
            e_fr = (pq / l + l * (gq[0] ** 2 + gq[1] ** 2)) / cw
        Pi_fr = float((dx * a.Gc * e_fr).sum())
        du = iu.interpolate(u).grad
        exx, eyy = du[0, 0] - c_qp, du[1, 1] - c_qp
        exy = 0.5 * (du[0, 1] + du[1, 0])
        psi = 0.5 * lam * (exx + eyy) ** 2 + mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
        Pi_el = float((dx * ((1.0 - pq) ** 2 + gf) * psi).sum())
        pg = to_grid(phi)
        cols = np.where(pg.max(axis=0) > 0.9)[0]
        tip = float(cx[cols.max()]) if cols.size else 0.0
        EN.append([k, times[k], a.Vpull * times[k], Pi_el, Pi_fr, it + 1, tip])
        if k % a.save_every == 0 or k == a.nsteps:
            np.save(os.path.join(a.out, f"phi_step_{k:03d}.npy"), pg.astype(np.float32))
        print(f"  state {k:3d}/{a.nsteps} t={times[k]:7.3f}s front={a.Vpull*times[k]:6.3f} "
              f"tip={tip:6.3f} stag={it+1:3d} Pi_el={Pi_el:.5e} Pi_fr/Gc={Pi_fr/a.Gc:8.3f} mm "
              f"phi_max={phi.max():.3f} {time.time()-tk:6.1f}s", flush=True)
        if a.phinoise > 0.0:
            # the accepted state is the converged one plus a smooth random field (amplitude
            # phinoise, Gaussian correlation length phinoise_len) weighted by 4 phi (1 - phi)
            G = gaussian_filter(rng_pn.standard_normal((ny + 1, nx + 1)), a.phinoise_len / a.h)
            G /= G.std() + 1e-30
            phi = np.clip(phi + a.phinoise * 4.0 * phi * (1.0 - phi) * G[iy, ix], 0.0, 1.0)
            phi = np.maximum(phi, phi0)

    np.savetxt(os.path.join(a.out, "ENERGY.txt"), np.array(EN),
               header="state t(s) front_x(mm) Pi_el(mJ) Pi_fr(mJ) stag_it tip_x(mm)")
    np.save(os.path.join(a.out, "grid_x.npy"), cx)
    np.save(os.path.join(a.out, "grid_y.npy"), cy)
    pg = to_grid(phi)
    # crack path of the final state: the phi-weighted center of the ridge in each column
    path = []
    for j in range(nx + 1):
        col = pg[:, j]
        if col.max() > 0.9:
            w = np.where(col > 0.5, col, 0.0)
            path.append((cx[j], float((w * cy).sum() / w.sum()), float(col.max())))
    np.savetxt(os.path.join(a.out, "path_final.csv"), np.array(path), delimiter=",",
               header="x,y,phi_max", comments="")
    with open(os.path.join(a.out, "config.txt"), "w") as f:
        for kk, vv in sorted(vars(a).items()):
            f.write(f"{kk} = {vv}\n")
        f.write(f"mesh = {nx} x {ny}\nR = {R!r}\n")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(pg, origin="lower", extent=(0, a.W, 0, a.H), cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.savefig(os.path.join(a.out, "phi_final.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[strip_fem] done -> {a.out}  ({(time.time()-t0)/3600:.2f} h)  final tip {EN[-1][6]:.3f} mm",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--W", type=float, default=30.0)
    ap.add_argument("--H", type=float, default=16.0)
    ap.add_argument("--h", type=float, default=0.05, help="element size (mm)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="random node shift as a fraction of h (0 = the structured mesh)")
    ap.add_argument("--jitter_seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="random factor 1 +- noise on the driving force at every staggered "
                         "iteration (0 = off)")
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--model", default="at", choices=["at", "kflp"],
                    help="at = the variational AT model of --at (default). kflp = the complete "
                         "model of Zeng and Dolbow (2026) with an independent Drucker-Prager "
                         "strength")
    ap.add_argument("--sig_ts", type=float, default=80.0,
                    help="kflp: uniaxial tensile strength (MPa)")
    ap.add_argument("--cs_ratio", type=float, default=20.0,
                    help="kflp: compressive / tensile strength")
    ap.add_argument("--v_th", type=float, default=0.05,
                    help="kflp: irreversible once v = 1 - phi <= v_th")
    ap.add_argument("--phinoise", type=float, default=0.0,
                    help="smooth random perturbation of phi added at the end of every state "
                         "(0 = off)")
    ap.add_argument("--phinoise_len", type=float, default=0.1,
                    help="its Gaussian correlation length (mm)")
    ap.add_argument("--phinoise_seed", type=int, default=777)
    ap.add_argument("--l", type=float, default=0.1)
    ap.add_argument("--at", type=int, default=2, choices=[1, 2],
                    help="2 = AT2, the mesh-free solver's model (default). 1 = AT1")
    ap.add_argument("--E", type=float, default=70e3)
    ap.add_argument("--nu", type=float, default=0.22)
    ap.add_argument("--Gc", type=float, default=8.25e-3)
    ap.add_argument("--alpha", type=float, default=7.7e-6)
    ap.add_argument("--g_floor", type=float, default=1e-6)
    ap.add_argument("--dT", type=float, default=165.0,
                    help="temperature jump across the front in K (default 165)")
    ap.add_argument("--Tc", type=float, default=23.0,
                    help="bath temperature (only differences enter, default 23)")
    ap.add_argument("--Vpull", type=float, default=0.4, help="front speed in mm/s (default 0.4)")
    ap.add_argument("--Dtherm", type=float, default=0.47,
                    help="thermal diffusivity in mm^2/s (default 0.47)")
    ap.add_argument("--travel", type=float, default=24.0,
                    help="front travel over the run in mm (default 24)")
    ap.add_argument("--nsteps", type=int, default=480)
    ap.add_argument("--crack_len", type=float, default=5.0)
    ap.add_argument("--crack_off", type=float, default=0.25)
    ap.add_argument("--irr", default="bound", choices=["bound", "penalty"])
    ap.add_argument("--gamma_ir", type=float, default=20.625,
                    help="irreversibility penalty weight for --irr penalty (default 250 Gc/l at "
                         "l = 0.1, as in the mesh-free solver)")
    ap.add_argument("--stag_tol", type=float, default=1e-3)
    ap.add_argument("--stag_max", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--init_phi", default=None,
                    help="restart: a saved phi_step_K.npy (grid) to continue from")
    ap.add_argument("--k0", type=int, default=0, help="restart: the first state to solve (K + 1)")
    ap.add_argument("--out", default="runs_fem/strip")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
