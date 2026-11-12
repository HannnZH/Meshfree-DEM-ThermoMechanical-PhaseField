r"""Staggered AT2 phase-field finite element reference (scikit-fem) for the quench problems: the
quenched ceramic slab of Tang et al. (2016) with --geom slab, and the quenched ceramic annulus
with --geom ring (the default). It reuses the thermo-mechanical kernels of pf_fem.py in this
directory: degraded elasticity, the AT2 phase field, transient backward Euler heat conduction
and the thermal eigenstrain force, all reassembled from precomputed kernels.

Common to both geometries:
  - a structured quadrilateral mesh of element size h, in (x, y) for the slab and in
    (r, theta) mapped to the circle for the annulus;
  - cooling of the quenched surfaces either by convection (--bc robin, the default, with a
    facet mass matrix assembled once) or by the bath temperature imposed there (--bc dirichlet);
  - symmetry planes on the cuts and no mechanical load;
  - plane stress with the in-plane thermal eigenstrain (--stress plane_strain changes the Lame
    constant only, the thermal strain stays in-plane);
  - the Weibull meso-structure of Tang et al. (2016), drawn exactly as the mesh-free solver
    draws it (pfte.problems_thermal._weibull_fields: numpy's Weibull stream, E first, then the
    strength), looked up by nearest cell at each quadrature point, converted to a toughness
    through the AT2 peak stress and rescaled to the mean toughness --gc_mean
    (pfte.problems_thermal._calibrate_gc). The startup line prints gc_scale, which equals the
    mesh-free solver's for the same material and Weibull options, so both methods see the same
    realization;
  - the cubic time stepping t_k = t_start + (t_final - t_start) ((k - 1) / (nsteps - 1))^power
    with one backward Euler step per state (--clock uniform gives equal steps).

Annulus: Ri = 2.5 mm, Ro = 10 mm, both rims cooled, v = 0 on theta = 0 and u = 0 on
theta = pi/2. Slab: the quarter [0, W] x [0, H] of the 50 x 10 mm plate, cooled on x = 0 (end
face) and y = 0 (long face), with u = 0 on x = W and v = 0 on y = H as in the mesh-free solver.

Each state is solved by alternating u and phi until phi stops changing (--stag_tol, cap
--stag_max). Irreversibility is the history field of Miehe et al. (2010) (--irr hist, the
default), the bound phi >= phi of the previous state with the current tensile energy as the
driving force (--irr bound), or the one-sided penalty of the mesh-free solver (--irr penalty).

Outputs (in --out): energies per state (ENERGY.txt), the irreversibility violation per state
(VIOLATION.txt), crack counts per state at threshold 0.7 (COUNTS.txt) and over a threshold sweep
at the end (COUNTS_SWEEP.txt: annulus bore/outer at r = Ri + 2l and Ro - 2l, slab long face/end
face at depth 2l), the count against depth (slab) or radius (annulus) (NR.txt), the fields on
the structured grid (phi_final.npy, T_final.npy, grid_*.npy), phi every tenth state
(phi_step_NNN.npy), config.txt and a rendering (phi_final.png).

Usage, with the settings of the references in the paper. The slab at a temperature drop of 580 K:

    python fem/pf_fem_ring.py --geom slab --h 0.025 --nsteps 100 --bc dirichlet \
        --irr bound --T0 873 --het_seed 1

For 280, 380 and 480 K use --T0 573, 673 and 773 with --het_seed 2, 3 and 2. The annulus at
580 K:

    python fem/pf_fem_ring.py --h 0.025 --nsteps 100 --bc dirichlet --irr bound \
        --stag_max 1000 --T0 873

For 280, 380 and 480 K use --T0 573, 673 and 773, with Weibull field 1 (the default --het_seed)
throughout. The bath temperature is --T_env (default 293 K), and h = 0.025 mm is half the
default length scale l = 0.05 mm. Give each variant its own --out.

Options not used by the references of the paper (--model kflp, --at 1, --scheme tangella,
--hist dg0, --noise, --closure, --init_phi, --eval_phi) are kept for further study. The
comparison on the crack set at 380 K adds --ir_thresh 0.5 to the slab command (--T0 673
--het_seed 3).

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_fem import (FastKdeg, FastPhi, FastHeat, FastThermalRHS, psi_plus, LINSOLVE)  # noqa: E402
from skfem import (MeshQuad, Basis, FacetBasis, ElementVector, ElementQuad1,        # noqa: E402
                   BilinearForm, LinearForm, asm, solve, condense)


# ------------------------------------------------------------------ Weibull field (Tang et al.)
def weibull_fields(E_mean, S_mean, m, seed, ny, nx):
    """The mesh-free solver's draw (pfte.problems_thermal._weibull_fields): numpy's rng.weibull
       stream, E first, then the strength."""
    rng = np.random.default_rng(seed)
    E = E_mean * rng.weibull(m, (ny, nx))
    S = S_mean * rng.weibull(m, (ny, nx))
    return E, S


def calibrate_gc(l, gc_mean, Efield, Sfield, at=2):
    """Strength to toughness through the homogeneous peak stress, then one constant rescale
       of the mean to gc_mean (pfte.problems_thermal._calibrate_gc). AT2 uses
       sigma_c^2 = 27 Gc E / (256 l), AT1 uses sigma_c^2 = 3 Gc E / (8 l)."""
    G = ((256.0 / 27.0) if at == 2 else (8.0 / 3.0)) * l * Sfield ** 2 / Efield
    scale = gc_mean / float(G.mean()) if gc_mean > 0 else 1.0
    return G * scale, scale, float(Sfield.mean() * math.sqrt(scale))


def lookup(field, s, t, nx, ny):
    """Nearest cell by position: s, t are the fractions of the box in x and y."""
    ix = np.clip((s * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((t * ny).astype(np.int64), 0, ny - 1)
    return field[iy, ix]


# ------------------------------------------------------------------ meshes
def quarter_annulus_mesh(ri, ro, h):
    """Tensor grid in (r, theta) mapped to the first quadrant. nth is set by the outer arc."""
    nr = max(4, int(math.ceil((ro - ri) / h)))
    nth = max(4, int(math.ceil(0.5 * math.pi * ro / h)))
    r = np.linspace(ri, ro, nr + 1)
    th = np.linspace(0.0, 0.5 * math.pi, nth + 1)
    m0 = MeshQuad.init_tensor(r, th)
    rr, tt = m0.p[0], m0.p[1]
    p = np.array([rr * np.cos(tt), rr * np.sin(tt)])
    return MeshQuad(p, m0.t), r, th


def slab_mesh(W, H, h):
    nx = max(4, int(math.ceil(W / h)))
    ny = max(4, int(math.ceil(H / h)))
    x = np.linspace(0.0, W, nx + 1)
    y = np.linspace(0.0, H, ny + 1)
    m0 = MeshQuad.init_tensor(x, y)
    return MeshQuad(np.array([m0.p[0], m0.p[1]]), m0.t), x, y


# ------------------------------------------------------------------ clock
def clock(nsteps, t_start, t_final, power):
    return [0.0] + [t_start + (t_final - t_start) * ((k - 1) / (nsteps - 1)) ** power
                    for k in range(1, nsteps + 1)]


# ------------------------------------------------------------------ crack counting
def line_crossings(grid, c0, c1, at0, thr, merge_arc, scale):
    """Distinct phi ridges cut by the line c0 = at0, sampled along c1 in arc length (arc =
       c1 * scale), with peaks above thr merged within merge_arc mm. grid is indexed [i0, i1]
       on the coordinates c0, c1."""
    i = int(np.clip(np.searchsorted(c0, at0) - 1, 0, len(c0) - 2))
    fr = (at0 - c0[i]) / (c0[i + 1] - c0[i])
    v = (1.0 - fr) * grid[i] + fr * grid[i + 1]
    n = len(v)
    peaks = [k for k in range(1, n - 1) if v[k] >= thr and v[k] >= v[k - 1] and v[k] > v[k + 1]]
    keep = []
    for k in sorted(peaks, key=lambda j: -v[j]):
        if all(abs(c1[k] - c1[j]) * scale > merge_arc for j in keep):
            keep.append(k)
    return len(keep)


# ------------------------------------------------------------------ driver
class Cfg:
    pass


def evaluate_fields(a, k, tk, c_qp, ip, iu, m, fk, fth, D, cfg, lam, mu, gf, l, Gc_qp, dx, c0, c1):
    """--eval_phi / --eval_state: the reduced energy E(phi) = min_u Pi_el(u, phi) + Pi_fr(phi) of
       given phase fields at state k, on this mesh, material field and temperature. Each file is
       either a .npz on a Cartesian grid (arrays GX, GY, phi in physical mm) or a grid .npy written
       by this script (phi_step_NNN.npy on the grid c0 x c1, slab only). It compares states of one problem,
       for example from two solvers, by the same functional, not the paths that led to them."""
    from scipy.interpolate import RegularGridInterpolator
    rows = []
    for path in a.eval_phi.split(","):
        if path.endswith(".npz"):
            z = np.load(path)
            GX, GY, P = z["GX"], z["GY"], z["phi"]
            gx = GX[0, :] if GX.ndim == 2 else GX
            gy = GY[:, 0] if GY.ndim == 2 else GY
            f = RegularGridInterpolator((gy, gx), P, bounds_error=False, fill_value=0.0)
            phi = f(np.stack([m.p[1], m.p[0]], 1))
        else:
            P = np.load(path)
            f = RegularGridInterpolator((c0, c1), P, bounds_error=False, fill_value=0.0)
            phi = f(np.stack([m.p[0], m.p[1]], 1))
        phi = np.clip(phi, 0.0, 1.0)
        fi = ip.interpolate(phi)
        pq, gq = fi.value, fi.grad
        g_qp = (1.0 - pq) ** 2 + gf
        u = solve(*condense(fk(pq), fth(c_qp, g_qp), x=iu.zeros(), D=D), solver=LINSOLVE)
        du = iu.interpolate(u).grad
        exx, eyy = du[0, 0] - c_qp, du[1, 1] - c_qp
        exy = 0.5 * (du[0, 1] + du[1, 0])
        psi = 0.5 * lam * (exx + eyy) ** 2 + mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
        Pi_el = float((dx * g_qp * psi).sum())
        e_fr = (pq ** 2 + l * l * (gq[0] ** 2 + gq[1] ** 2)) / (2.0 * l)
        Pi_fr = float((dx * Gc_qp * e_fr).sum())
        rows.append((path, Pi_el, Pi_fr, Pi_el + Pi_fr, float(phi.max())))
        print(f"[eval] state {k} t={tk:.4e}s  {os.path.basename(path)}: Pi_el={Pi_el:.6e} "
              f"Pi_fr={Pi_fr:.6e} E={Pi_el + Pi_fr:.6e} mJ  (Pi_fr/Gc {Pi_fr / a.gc_mean:.3f} mm, "
              f"phi_max {phi.max():.3f})", flush=True)
    with open(os.path.join(a.out, "EVAL.txt"), "a") as fo:
        for r in rows:
            fo.write(f"{k} {tk:.6e} {r[1]:.9e} {r[2]:.9e} {r[3]:.9e} {r[4]:.4f} {r[0]}\n")


def run(a):
    os.makedirs(a.out, exist_ok=True)
    cfg = Cfg()
    cfg.g_floor = 1e-6
    cfg.kdeg = False
    cfg.heat_lump = True
    cfg.alpha_mode = "2d"
    cfg.split = "hybrid"
    cfg.l = a.l
    nu = a.nu
    l = a.l
    slab = a.geom == "slab"

    if slab:
        W, Hs = a.W, a.H                                   # Hs: the height (H is the history)
        m, c0, c1 = slab_mesh(W, Hs, a.h)                 # c0 = x nodes, c1 = y nodes
        n0, n1 = len(c0) - 1, len(c1) - 1
    else:
        m, c0, c1 = quarter_annulus_mesh(a.ri, a.ro, a.h)  # c0 = r nodes, c1 = theta nodes
        n0, n1 = len(c0) - 1, len(c1) - 1
    iu = Basis(m, ElementVector(ElementQuad1()), intorder=2)
    ip = Basis(m, ElementQuad1(), intorder=2)
    xy = ip.global_coordinates().value                     # (2, NE, NQ)
    xq, yq = xy[0], xy[1]

    # ---- material: the Weibull draw of the solver, looked up at the quadrature points
    if slab:
        ny = max(1, int(round(Hs / a.het_cell)))
        nx = max(1, int(round(W / a.het_cell)))
        Ef, Sf = weibull_fields(a.E, a.sigma_t, a.het_m, a.het_seed, ny, nx)
        Gf, gc_scale, s_impl = calibrate_gc(a.l, a.gc_mean, Ef, Sf, a.at)
        E_qp = lookup(Ef, xq / W, yq / Hs, nx, ny)
        Gc_qp = lookup(Gf, xq / W, yq / Hs, nx, ny)
        S_qp = lookup(Sf, xq / W, yq / Hs, nx, ny)
    else:
        ny = max(1, int(round(2.0 * a.ro / a.het_cell)))
        nx = max(1, int(round(a.ro / a.het_cell)))
        Ef, Sf = weibull_fields(a.E, a.sigma_t, a.het_m, a.het_seed, ny, nx)
        Gf, gc_scale, s_impl = calibrate_gc(a.l, a.gc_mean, Ef, Sf, a.at)
        E_qp = lookup(Ef, xq / a.ro, (yq + a.ro) / (2.0 * a.ro), nx, ny)
        Gc_qp = lookup(Gf, xq / a.ro, (yq + a.ro) / (2.0 * a.ro), nx, ny)
        S_qp = lookup(Sf, xq / a.ro, (yq + a.ro) / (2.0 * a.ro), nx, ny)
    kf = a.model == "kflp"
    if kf:
        # kflp takes the toughness as an input of its own, uniform at the mean. The Weibull
        # strength field S_qp is its independent strength, with no AT2 strength -> toughness map
        Gc_qp = np.full_like(E_qp, a.gc_mean)
    if a.stress == "plane_strain":
        lam = E_qp * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # plane strain (Tangella et al.)
    else:
        lam = E_qp * nu / (1.0 - nu ** 2)                 # plane stress (mesh-free solver)
    mu = E_qp / (2.0 * (1.0 + nu))
    rho_cp = a.rho * a.cp * 1e-6                          # mJ/(mm^3 K)
    kcond = a.kcond                                       # mJ/(s mm K)
    alpha = a.alpha

    fk = FastKdeg(iu, cfg, lam, mu)
    # AT2: FastPhi as it is. AT1 (--at 1, the same construction as in pf_fem_strip.py):
    # w(phi) = phi, c_w = 8/3, the equation
    # (2 Gc l / c_w)(-lap phi) + 2 H phi = 2 H - Gc / (c_w l), 0 <= phi <= 1.
    # FastPhi built with Gc' = 2 Gc / c_w carries the Laplacian and a (Gc'/l) mass term. Shifting
    # H removes that mass term from A and puts the AT1 source into b. Gc may be a field.
    cw = 8.0 / 3.0
    if a.at == 1:
        if a.irr != "bound":
            raise SystemExit("--at 1 is implemented with --irr bound only")
        fp1 = FastPhi(ip, cfg, 2.0 * Gc_qp / cw)
        # where H = 0 (the elastic phase) A is a pure Neumann Laplacian, singular: a mass term of
        # 1e-6 of the source scale makes it regular, and the bound returns phi = phi_prev there
        sA, sb = Gc_qp / (cw * l) * (1.0 - 1e-6), Gc_qp / (2.0 * cw * l)

        class _AT1:
            def A(self, H):
                return fp1.A(H - sA)

            def b(self, H):
                return fp1.b(H - sb)
        fp = _AT1()
    else:
        fp = FastPhi(ip, cfg, Gc_qp)
    if kf:
        # The complete model of Zeng and Dolbow (2026) (KFLP, Drucker-Prager strength by external
        # microforces, their eqs 16-29), as in pf_fem_strip.py --model kflp, here with the strength
        # and modulus fields. In phi = 1 - v:
        #   (3 delta Gc l / 4)(-lap phi) + 2 H phi = 2 H - c_hat - 3 delta Gc / (8 l),
        # H = psi (full, no split) where I1 >= 0 and 0 where I1 < 0, c_hat = beta2 sqrt(J2) +
        # beta1 I1 of the degraded stress (plane stress invariants), 0 <= phi <= 1, and
        # phi >= phi_prev only once phi_prev >= 1 - v_th.
        if a.irr != "bound" or a.stress != "plane_stress":
            raise SystemExit("--model kflp: --irr bound and plane stress only")
        sts = S_qp                                            # uniaxial tensile strength (MPa)
        scs = a.cs_ratio * sts
        # Drucker-Prager: scs = 3 shs sts / (3 shs - 2 sts)
        shs = 2.0 * scs * sts / (3.0 * (scs - sts))
        Kb = E_qp / (3.0 * (1.0 - 2.0 * nu))
        psi_ts, psi_hs = sts ** 2 / (2.0 * E_qp), shs ** 2 / (2.0 * Kb)
        r8 = 1.0 + 3.0 * a.h / (8.0 * l)                      # eq 29, quadrilateral mesh correction
        kd = r8 ** -2 * (0.2201 * shs + 0.1295 * sts) / shs * 3.0 * Gc_qp / (16.0 * psi_ts * l) + 0.6297 / r8
        dG8l = kd * Gc_qp / (8.0 * l)
        kb1 = -dG8l / shs + 2.0 * psi_hs / (3.0 * shs)
        kb2 = (-math.sqrt(3.0) * (3.0 * shs - sts) / (shs * sts) * dG8l - 2.0 * psi_hs / (math.sqrt(3.0) * shs)
               + 2.0 * math.sqrt(3.0) * psi_ts / sts)
        fpk = FastPhi(ip, cfg, 0.75 * kd * Gc_qp)
        sAk = 3.0 * dG8l * (1.0 - 1e-6)                       # 3 delta Gc/(8 l), as AT1's
        l_max = float((3.0 * Gc_qp * E_qp / (8.0 * sts ** 2)).min())

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
    heat = FastHeat(ip, cfg, kcond, rho_cp)
    fth = FastThermalRHS(iu, cfg, lam, mu)

    # ---- convection on the cooled surfaces: facet mass matrix and load, assembled once.
    # Facets are tested at their midpoints; the tolerance stays below half an element so the
    # first facet of each cut does not join.
    tol = 0.1 * a.h
    if slab:
        cooled = m.facets_satisfying(lambda x: (x[0] < tol) | (x[1] < tol), boundaries_only=True)
    else:
        cooled = m.facets_satisfying(lambda x: (np.abs(np.hypot(x[0], x[1]) - a.ro) < tol)
                                     | (np.abs(np.hypot(x[0], x[1]) - a.ri) < tol),
                                     boundaries_only=True)
    fb = FacetBasis(m, ElementQuad1(), facets=cooled, intorder=2)

    @BilinearForm
    def mass_b(u, v, w):
        return u * v

    @LinearForm
    def one_b(v, w):
        return v

    Mb = asm(mass_b, fb).tocsr()
    bb = asm(one_b, fb)
    arc = float(bb.sum())

    # ---- symmetry cuts
    if slab:
        cutx = iu.get_dofs(lambda x: x[0] > W - 1e-9 * W)      # x = W: u = 0
        cuty = iu.get_dofs(lambda x: x[1] > Hs - 1e-9 * Hs)    # y = H: v = 0
        D = np.unique(np.concatenate([cutx.all("u^1"), cuty.all("u^2")]))
    else:
        cut0 = iu.get_dofs(lambda x: x[1] < 1e-9 * a.ro)       # theta = 0: v = 0
        cut1 = iu.get_dofs(lambda x: x[0] < 1e-9 * a.ro)       # theta = pi/2: u = 0
        D = np.unique(np.concatenate([cut0.all("u^2"), cut1.all("u^1")]))

    if a.clock == "uniform":
        times = list(np.linspace(0.0, a.t_final, a.nsteps + 1))
    else:
        times = clock(a.nsteps, a.t_start, a.t_final, a.time_power)
    # Dirichlet cooling (Tangella et al.): the bath temperature imposed on the cooled surfaces
    if a.bc == "dirichlet":
        if slab:
            Tdofs = ip.get_dofs(lambda x: (x[0] < 1e-9 * W) | (x[1] < 1e-9 * W)).all()
        else:
            Tdofs = ip.get_dofs(lambda x: (np.abs(np.hypot(x[0], x[1]) - a.ro) < 1e-6 * a.ro)
                                | (np.abs(np.hypot(x[0], x[1]) - a.ri) < 1e-6 * a.ro)).all()
        Tdofs = np.unique(Tdofs)
    hcell = a.h if slab else 0.5 * math.pi * a.ro / n1
    print(f"[ring_fem] {a.geom}: mesh {n0} x {n1} quads (h={hcell:.4f} mm = {hcell/a.l:.2f} l)  "
          f"u-dofs={iu.N} phi-dofs={ip.N}  solver={getattr(LINSOLVE, 'name', '?')}  "
          f"scheme={a.scheme} hist={a.hist} irr={a.irr}"
          + (f" gamma_ir={a.gamma_ir:g} ir_tol={a.ir_tol:g}" if a.irr == "penalty" else ""))
    print(f"[ring_fem] Weibull box ({ny},{nx}) cell={a.het_cell:g} m={a.het_m:g} seed={a.het_seed}"
          f"  E mean/CV={Ef.mean()/1e3:.2f}GPa/{Ef.std()/Ef.mean():.3f}  gc_scale={gc_scale:.10f}"
          f"  implied strength={s_impl:.2f} MPa (AT{a.at})  Gc mean(qp)={Gc_qp.mean():.5f}")
    if kf:
        print(f"[ring_fem] model kflp (Zeng and Dolbow eqs 16-29): sigma_ts field mean {sts.mean():.1f} "
              f"[{sts.min():.1f}, {sts.max():.1f}] MPa, cs_ratio {a.cs_ratio:g}, sigma_hs mean {shs.mean():.1f} MPa, "
              f"delta mean {kd.mean():.4f} (eq 29, h/l {a.h/l:.2f}), v_th {a.v_th:g}, Gc {a.gc_mean:g} uniform; "
              f"l {l:g} vs l_max = 3 Gc E/(8 sigma_ts^2) = {l_max:.3f} mm (min over the field)")
    print(f"[ring_fem] quench T0={a.T0:g} K into {a.T_env:g} K, h={a.hconv:g} mJ/(s mm2 K), "
          f"cooled length {arc:.3f} mm, kappa={kcond/rho_cp:.3f} mm2/s, {a.nsteps} states "
          f"{a.t_start:g}..{a.t_final:g} s, l={a.l:g}")
    print(f"[ring_fem] first state: dt={times[1]:.2e} s, boundary layer sqrt(kappa t)="
          f"{math.sqrt(kcond/rho_cp*times[1]):.4f} mm = {math.sqrt(kcond/rho_cp*times[1])/a.l:.2f} l")

    T = ip.zeros() + a.T0
    phi = ip.zeros()
    H = np.zeros(ip.dx.shape)
    u = iu.zeros()
    if a.init_phi:
        # start from another solver's phase field (a .npz with arrays GX, GY, phi on a Cartesian
        # grid): interpolated to the nodes, and the history field is set so that this field is
        # what the phi equation returns at the start, H = Gc phi / (2 l (1 - phi))
        from scipy.interpolate import RegularGridInterpolator
        z = np.load(a.init_phi)
        GX, GY, P = z["GX"], z["GY"], z["phi"]
        gx = GX[0, :] if GX.ndim == 2 else GX
        gy = GY[:, 0] if GY.ndim == 2 else GY
        f = RegularGridInterpolator((gy, gx), P, bounds_error=False, fill_value=0.0)
        phi = np.clip(f(np.stack([m.p[1], m.p[0]], 1)), 0.0, 0.999)
        pq0 = ip.interpolate(phi).value
        H = Gc_qp * pq0 / (2.0 * l * np.maximum(1.0 - pq0, 1e-3))
        print(f"[ring_fem] initial phase field from {a.init_phi}: nodal max {phi.max():.3f}, "
              f"mean {phi.mean():.4f}")
    gf = cfg.g_floor
    dx = ip.dx
    EN, CN, VI = [], [], []
    # nodal values onto the structured grid by the node coordinates, whatever init_tensor's order
    if slab:
        i0 = np.clip(np.rint(m.p[0] / (c0[1] - c0[0])).astype(int), 0, n0)
        i1 = np.clip(np.rint(m.p[1] / (c1[1] - c1[0])).astype(int), 0, n1)
    else:
        rn = np.hypot(m.p[0], m.p[1])
        tn = np.arctan2(m.p[1], m.p[0])
        i0 = np.clip(np.rint((rn - a.ri) / (c0[1] - c0[0])).astype(int), 0, n0)
        i1 = np.clip(np.rint(tn / (c1[1] - c1[0])).astype(int), 0, n1)
    assert len(np.unique(i0 * (n1 + 1) + i1)) == m.p.shape[1], "grid index collision"

    def to_grid(v):
        g = np.zeros((n0 + 1, n1 + 1))
        g[i0, i1] = v[:m.p.shape[1]]
        return g

    def counts(pg, thr):
        """(count on the first cooled surface, count on the second) at depth 2l."""
        if slab:
            n_long = line_crossings(pg.T, c1, c0, 2.0 * l, thr, a.merge, 1.0)   # y = 2l along x
            n_end = line_crossings(pg, c0, c1, 2.0 * l, thr, a.merge, 1.0)      # x = 2l along y
            return n_long, n_end
        rb, ro_ = a.ri + 2.0 * l, a.ro - 2.0 * l
        return (line_crossings(pg, c0, c1, rb, thr, a.merge, rb),
                line_crossings(pg, c0, c1, ro_, thr, a.merge, ro_))

    t0 = time.time()
    rng_noise = np.random.default_rng(12345 + a.het_seed)
    tol_s = a.stag_tol if a.stag_tol > 0 else (1e-2 if a.scheme == "tangella" else 1e-3)
    cap = a.stag_max if a.stag_max > 0 else (1000 if a.scheme == "tangella" else 50)
    for k in range(1, len(times)):
        tk = time.time()
        dt = times[k] - times[k - 1]
        phi_qp = ip.interpolate(phi).value
        if a.bc == "dirichlet":
            xT = ip.zeros() + a.T0
            xT[Tdofs] = a.T_env
            T = solve(*condense(heat.A(phi_qp, dt), heat.b(T, dt), x=xT, D=Tdofs), solver=LINSOLVE)
        else:
            A = heat.A(phi_qp, dt) + a.hconv * Mb
            b = heat.b(T, dt) + a.hconv * a.T_env * bb
            T = solve(A, b, solver=LINSOLVE)
        c_qp = alpha * (ip.interpolate(T).value - a.T0)
        if a.eval_state and k == a.eval_state:
            assert a.at == 2, "--eval_phi evaluates the AT2 functional only"
            evaluate_fields(a, k, times[k], c_qp, ip, iu, m, fk, fth, D, cfg, lam, mu, gf, l, Gc_qp, dx, c0, c1)
            return
        if k < a.k0:                                      # thermal history only, up to the start
            continue
        phi_prev = phi.copy()                             # the converged field of the last state
        if kf:   # irreversible only once broken (Zeng and Dolbow: v <= v_th)
            phi_prev = np.where(phi >= 1.0 - a.v_th, phi, 0.0)
        elif a.ir_thresh > 0:
            # crack-set irreversibility, the mesh-free solver's --ir_thresh: only damage at or above
            # the threshold is held, below it the field may heal
            phi_prev = np.where(phi >= a.ir_thresh, phi, 0.0)
        closed = None
        act_pen = None                                    # the penalty's active set (--irr penalty)
        for it in range(cap):
            phi_q = ip.interpolate(phi).value
            if a.closure and closed is not None:
                # crack closure of Ambati et al. (2015): where the elastic strain is compression
                # dominated the damaged material transmits the full stiffness, so closed faces do
                # not interpenetrate and a crack in the compressive core is not driven by it
                phi_q = np.where(closed, 0.0, phi_q)
            g_qp = (1.0 - phi_q) ** 2 + gf
            Km = fk(phi_q)
            f_ext = fth(c_qp, g_qp)
            u_new = solve(*condense(Km, f_ext, x=iu.zeros(), D=D), solver=LINSOLVE)
            du_rel = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-5)
            u = u_new
            if kf:
                psi_q, src_q = kf_drive(u, phi_q, c_qp)
            else:
                psi_q = psi_plus(iu, u, cfg, lam, mu, c_qp)
            if a.noise > 0:
                # a fresh random factor on the driving force at every staggered iteration, the
                # counterpart of the mesh-free solver's per-iteration resampling: it perturbs
                # the unstable uniform band so that the iteration can leave it
                psi_q = psi_q * (1.0 + a.noise * rng_noise.uniform(-1.0, 1.0, psi_q.shape))
            if a.closure:
                dug = iu.interpolate(u).grad
                e11, e22 = dug[0, 0] - c_qp, dug[1, 1] - c_qp
                e12 = 0.5 * (dug[0, 1] + dug[1, 0])
                psi_full = 0.5 * lam * (e11 + e22) ** 2 + mu * (e11 ** 2 + e22 ** 2 + 2.0 * e12 ** 2)
                closed = (psi_full - psi_q) > psi_q          # psi_minus > psi_plus
            if a.hist == "dg0":
                # Tangella et al. project the history field onto one value per element
                psi_q = np.repeat(psi_q.mean(axis=1, keepdims=True), psi_q.shape[1], axis=1)
            if a.irr == "hist":
                # history field of Miehe et al. (2010): the driving force is the largest tensile
                # energy seen
                H = np.maximum(H, psi_q)
                phi_new = solve(fp.A(H), fp.b(H), solver=LINSOLVE)
            elif a.irr == "penalty":
                # the irreversibility of the mesh-free solver: the one-sided quadratic penalty
                # gamma_ir (phi_prev - tau - phi)_+^2 added to the phase-field energy, with the
                # current tensile energy as the driving force. Its variation adds 2 gamma_ir to the
                # mass coefficient and 2 gamma_ir (phi_prev - tau) to the load wherever the penalty
                # is active, which FastPhi's 2H kernels carry directly; the active set of quadrature
                # points is iterated until it no longer changes.
                pprev_q = ip.interpolate(phi_prev).value - a.ir_tol
                # start from the active set of the previous staggered iteration of this state
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
            else:
                # bound constraint phi >= phi of the previous state, the current tensile energy
                # as the driving force (as Bourdin et al. (2000) enforce it), by an active set:
                # nodes that fall below the bound are fixed at it and the solve repeated until no
                # further node violates
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
            if a.scheme == "tangella":
                # the loop of Tangella et al. (2022): both the displacement and the phase
                # field must have stopped changing, in the relative L2 norm, to 1e-2
                if max(dchg, du_rel) < tol_s:
                    break
            elif dchg < tol_s:
                break
        # energies of the state
        fi = ip.interpolate(phi)
        pq, gq = fi.value, fi.grad
        # the irreversibility violation of the state against the last one, as the mesh-free
        # solver's check measures it: the largest decrease and its area integral
        d_q = np.clip(ip.interpolate(phi_prev).value - pq, 0.0, None)
        VI.append([k, times[k], float(np.clip(phi_prev - phi, 0.0, None).max()),
                   float(d_q.max()), float((dx * d_q).sum())])
        if kf:   # regularized surface energy with its delta (the work of c_hat is not included)
            e_fr = 0.375 * kd * (pq / l + l * (gq[0] ** 2 + gq[1] ** 2))
        elif a.at == 2:
            e_fr = (pq ** 2 + l * l * (gq[0] ** 2 + gq[1] ** 2)) / (2.0 * l)
        else:
            e_fr = (pq / l + l * (gq[0] ** 2 + gq[1] ** 2)) / cw
        Pi_fr = float((dx * Gc_qp * e_fr).sum())
        Pi_fr_mean = float((dx * a.gc_mean * e_fr).sum())
        du = iu.interpolate(u).grad
        exx, eyy = du[0, 0] - c_qp, du[1, 1] - c_qp
        exy = 0.5 * (du[0, 1] + du[1, 0])
        psi = 0.5 * lam * (exx + eyy) ** 2 + mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
        Pi_el = float((dx * ((1.0 - pq) ** 2 + gf) * psi).sum())
        EN.append([k, times[k], Pi_el, Pi_fr, Pi_fr_mean, float(T.min()), float(T.max()), it + 1])
        pg = to_grid(phi)
        n1_, n2_ = counts(pg, 0.7)
        CN.append([k, times[k], n1_, n2_])
        if k % 10 == 0 or k == len(times) - 1:
            np.save(os.path.join(a.out, f"phi_step_{k:03d}.npy"), pg)
        lab = ("long/end" if slab else "bore/outer")
        print(f"  state {k:3d}/{a.nsteps} t={times[k]:.3e}s dt={dt:.2e} stag={it+1:2d} "
              f"phi_max={phi.max():.3f} viol n/qp={VI[-1][2]:.1e}/{VI[-1][3]:.1e} Pi_fr/Gc={Pi_fr/a.gc_mean:7.3f} mm  "
              f"{lab}={n1_}/{n2_} "
              f"T=[{T.min():.1f},{T.max():.1f}] {time.time()-tk:6.1f}s", flush=True)

    # ---- outputs
    np.savetxt(os.path.join(a.out, "ENERGY.txt"), np.array(EN),
               header="step t(s) Pi_el(mJ) Pi_fr_localGc(mJ) Pi_fr_meanGc(mJ) T_min T_max stag_it")
    np.savetxt(os.path.join(a.out, "VIOLATION.txt"), np.array(VI),
               header="step t(s) max_decrease_nodal max_decrease_qp int_decrease_dA(mm2)")
    np.savetxt(os.path.join(a.out, "COUNTS.txt"), np.array(CN),
               header="step t(s) " + ("long_face end_face" if slab else "bore outer") + " (thr 0.7, depth 2l)")
    pg = to_grid(phi)
    sweep = [[thr] + list(counts(pg, thr)) for thr in (0.6, 0.7, 0.8, 0.9)]
    np.savetxt(os.path.join(a.out, "COUNTS_SWEEP.txt"), np.array(sweep),
               header=("thr long_face end_face" if slab else "thr bore outer") + "   (merge %.2f mm)" % a.merge)
    if slab:
        depths = np.linspace(2.0 * l, Hs - 2.0 * l, 12)
        NR = [[d] + [line_crossings(pg.T, c1, c0, d, thr, a.merge, 1.0) for thr in (0.6, 0.7, 0.8, 0.9)]
              for d in depths]
        hdr = "depth_from_long_face  N(thr 0.6 0.7 0.8 0.9)"
    else:
        radii = np.linspace(a.ri + 2.0 * l, a.ro - 2.0 * l, 12)
        NR = [[rr] + [line_crossings(pg, c0, c1, rr, thr, a.merge, rr) for thr in (0.6, 0.7, 0.8, 0.9)]
              for rr in radii]
        hdr = "r  N(thr 0.6 0.7 0.8 0.9)"
    np.savetxt(os.path.join(a.out, "NR.txt"), np.array(NR), header=hdr)
    np.save(os.path.join(a.out, "phi_final.npy"), pg)
    np.save(os.path.join(a.out, "T_final.npy"), to_grid(T))
    np.save(os.path.join(a.out, "grid_r.npy" if not slab else "grid_x.npy"), c0)
    np.save(os.path.join(a.out, "grid_th.npy" if not slab else "grid_y.npy"), c1)
    with open(os.path.join(a.out, "config.txt"), "w") as f:
        for kk, vv in sorted(vars(a).items()):
            f.write(f"{kk} = {vv}\n")
        f.write(f"gc_scale = {gc_scale!r}\nmesh = {n0} x {n1}\nu_dofs = {iu.N}\n")
    # rendering in physical coordinates
    if slab:
        X, Y = np.meshgrid(c0, c1, indexing="ij")
        fig, ax = plt.subplots(figsize=(10, 2.4))
    else:
        R, TH = np.meshgrid(c0, c1, indexing="ij")
        X, Y = R * np.cos(TH), R * np.sin(TH)
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.pcolormesh(X, Y, pg, cmap="magma", vmin=0, vmax=1, shading="gouraud")
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(a.out, "phi_final.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[ring_fem] done -> {a.out}  ({(time.time()-t0)/3600:.2f} h)  "
          f"Pi_fr/Gc = {EN[-1][3]/a.gc_mean:.3f} mm (local Gc), {EN[-1][4]/a.gc_mean:.3f} mm (mean Gc)"
          f"  {lab} at 0.7: {CN[-1][2]}/{CN[-1][3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geom", default="ring", choices=["ring", "slab"])
    ap.add_argument("--ri", type=float, default=2.5)
    ap.add_argument("--ro", type=float, default=10.0)
    ap.add_argument("--W", type=float, default=25.0, help="slab quarter length (mm)")
    ap.add_argument("--H", type=float, default=5.0, help="slab quarter height (mm)")
    ap.add_argument("--h", type=float, default=0.02,
                    help="element size (mm), along the outer arc for the annulus")
    ap.add_argument("--l", type=float, default=0.05)
    ap.add_argument("--E", type=float, default=370e3)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=7.5e-6)
    ap.add_argument("--kcond", type=float, default=31.0)
    ap.add_argument("--rho", type=float, default=3980.0)
    ap.add_argument("--cp", type=float, default=880.0)
    ap.add_argument("--hconv", type=float, default=50.0,
                    help="heat transfer coefficient for --bc robin in mJ/(s mm^2 K), equal to "
                         "kW/(m^2 K) (default 50)")
    ap.add_argument("--T0", type=float, default=873.0,
                    help="initial temperature in K (default 873)")
    ap.add_argument("--T_env", type=float, default=293.0)
    ap.add_argument("--sigma_t", type=float, default=180.0)
    ap.add_argument("--het_m", type=float, default=15.0)
    ap.add_argument("--het_cell", type=float, default=0.025)
    ap.add_argument("--het_seed", type=int, default=1,
                    help="realization of the Weibull field (default 1)")
    ap.add_argument("--gc_mean", type=float, default=0.04247)
    ap.add_argument("--nsteps", type=int, default=60, help="number of time states (default 60)")
    ap.add_argument("--t_start", type=float, default=1e-4)
    ap.add_argument("--t_final", type=float, default=0.2)
    ap.add_argument("--time_power", type=float, default=3.0)
    ap.add_argument("--stag_tol", type=float, default=0.0,
                    help="relative change of phi (and of u for tangella) to stop the staggered "
                         "loop; 0 = 1e-3 (alt) or 1e-2 (tangella)")
    ap.add_argument("--stag_max", type=int, default=0, help="cap on the staggered iterations; "
                    "0 = 50 (alt) or 1000 (tangella)")
    ap.add_argument("--scheme", default="alt", choices=["alt", "tangella"],
                    help="alt (default): u then phi alternated until phi stops changing. tangella: "
                         "the loop of Tangella et al. (2022), u and phi both converged to 1e-2")
    ap.add_argument("--hist", default="qp", choices=["qp", "dg0"],
                    help="history field per quadrature point, or one value per element (DG0)")
    ap.add_argument("--gamma_ir", type=float, default=36000.0,
                    help="weight of the irreversibility penalty (--irr penalty), in the units of "
                         "the energy density (default 36000, as in the mesh-free solver's quench "
                         "examples)")
    ap.add_argument("--ir_tol", type=float, default=0.0,
                    help="dead band of the penalty (--irr penalty)")
    ap.add_argument("--ir_thresh", type=float, default=0.0,
                    help="> 0: the lower bound phi_prev applies only where phi_prev >= ir_thresh "
                         "(the mesh-free solver's crack-set irreversibility); 0 = everywhere "
                         "(default)")
    ap.add_argument("--irr", default="hist", choices=["hist", "bound", "penalty"],
                    help="irreversibility: the history field of Miehe et al. (2010) (default, as "
                         "in pf_fem.py), the bound phi >= phi_prev with the current driving force "
                         "(as Bourdin et al. (2000) enforce it), or the one-sided quadratic "
                         "penalty of the mesh-free solver with the current driving force "
                         "(--gamma_ir, --ir_tol)")
    ap.add_argument("--merge", type=float, default=0.4,
                    help="merge length for the crossing count (mm)")
    ap.add_argument("--init_phi", default="",
                    help="phase field to start from: a .npz with arrays GX, GY, phi on a "
                         "Cartesian grid")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="relative amplitude of a fresh uniform random factor on the driving force "
                         "at every staggered iteration (0 = off)")
    ap.add_argument("--k0", type=int, default=1,
                    help="first state to solve; earlier states advance the temperature only "
                         "(default 1)")
    ap.add_argument("--closure", action="store_true",
                    help="crack closure condition of Ambati et al. (2015) in the displacement "
                         "solve")
    ap.add_argument("--bc", default="robin", choices=["robin", "dirichlet"],
                    help="robin (default): convection with --hconv. dirichlet: the bath "
                         "temperature --T_env imposed on the cooled surfaces (Bourdin et al. "
                         "2014)")
    ap.add_argument("--stress", default="plane_stress", choices=["plane_stress", "plane_strain"])
    ap.add_argument("--clock", default="cubic", choices=["cubic", "uniform"],
                    help="cubic (default): the time stepping of the quench examples, with "
                         "--time_power. uniform: nsteps equal steps to t_final")
    ap.add_argument("--eval_phi", default="",
                    help="comma-separated phase fields (.npz with arrays GX, GY, phi, or a grid "
                         ".npy of this script, slab only) whose reduced energy is evaluated at "
                         "--eval_state, then exit")
    ap.add_argument("--eval_state", type=int, default=0, help="state of --eval_phi (0 = off)")
    ap.add_argument("--model", default="at", choices=["at", "kflp"],
                    help="at = the variational AT model of --at (default). kflp = the complete "
                         "model of Zeng and Dolbow (2026), with the Weibull strength field as "
                         "strength and --gc_mean as toughness")
    ap.add_argument("--cs_ratio", type=float, default=10.0,
                    help="kflp: compressive / tensile strength")
    ap.add_argument("--v_th", type=float, default=0.05,
                    help="kflp: irreversible once v = 1 - phi <= v_th")
    ap.add_argument("--at", type=int, default=2, choices=[1, 2],
                    help="2 = AT2, the mesh-free solver's model (default). 1 = AT1 (--irr bound "
                         "only)")
    ap.add_argument("--out", default="runs_fem/ring")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
