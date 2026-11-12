"""
Run-time figures: phi maps, displacement and strain fields, F-delta curves, patch views.

The figures are written during the run so its progress can be inspected. The network evaluators
(kinematics, _fields) are imported inside the functions that need them: solver.py imports this
module, so importing them at module level would create a circular import.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import utils
from .geometry import phi0_field, to_phys
from .utils import DTYPE, problem_of


def _render_grid(cfg, ng):
    """Parametric meshgrid whose point count is scaled by the domain aspect (nx ~ aspect*ny) so
       the physical pixel is square. Otherwise an elongated body renders its short-axis
       boundary layers at a resolution coarser by the aspect ratio, and they alias into
       spurious lines. Returns GX, GY of shape (ny, nx), plus nx, ny. aspect=1 gives (ng, ng),
       the isotropic grid."""
    asp = problem_of(cfg).domain_aspect(cfg)
    if asp >= 1.0:
        nx, ny = min(int(round(ng * asp)), 2400), ng
    else:
        nx, ny = ng, min(int(round(ng / asp)), 2400)
    xs = torch.linspace(0, 1, nx, device=utils.DEVICE)
    ys = torch.linspace(0, 1, ny, device=utils.DEVICE)
    GX, GY = torch.meshgrid(xs, ys, indexing="xy")   # (ny, nx)
    return GX, GY, nx, ny

def _domain_figsize(cfg, ax_in=5.4, cbar_in=1.5, cap=22.0):
    """Figure size matched to the physical domain aspect, so an elongated body fills the frame
       instead of appearing as a thin strip in a nearly square figure. ax_in is the short
       physical side in inches, and the long side scales with the aspect, capped at `cap`.
       cbar_in reserves room for the colorbar. aspect=1 gives (6.9, 5.4)."""
    asp = problem_of(cfg).domain_aspect(cfg)
    if asp >= 1.0:                          # wide body
        return (min(ax_in * asp, cap) + cbar_in, ax_in)
    return (ax_in + cbar_in, min(ax_in / asp, cap))    # tall body

def _add_cbar(ax, m, label):
    """Colorbar whose height matches the data axes (make_axes_locatable). With aspect='equal'
       data axes, a default full-height colorbar adds large top and bottom margins under
       bbox_inches='tight', which makes the field appear small. The bar has a fixed width of
       0.16 in, so its thickness is the same at every aspect."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed
    cax = make_axes_locatable(ax).append_axes("right", size=Fixed(0.16), pad=Fixed(0.12))
    ax.figure.colorbar(m, cax=cax, label=label)

def _grid_eval(net, cfg, ng=500, chunk=45000):
    """Shear strain and phi on the render grid, in physical coordinates. ng=500 gives a pixel
       of 0.2 l at l=0.01, fine enough to render the crack band without aliasing. The autodiff
       evaluation is chunked so that the ng^2 points fit on the GPU for any ng. The grid is
       aspect-scaled, so short-axis features of an elongated body resolve like long-axis ones."""
    from .solver import kinematics
    GX, GY, nx, ny = _render_grid(cfg, ng)
    xf, yf = GX.reshape(-1, 1), GY.reshape(-1, 1)
    exys, phis = [], []
    for i in range(0, xf.shape[0], chunk):
        exx, eyy, exy, phi, gp2, _, _ = kinematics(net, xf[i:i + chunk].clone(),
                                                   yf[i:i + chunk].clone(), cfg, cfg.u_delta_nd)
        exys.append(exy.detach()); phis.append(phi.detach())
    to = lambda ts: torch.cat(ts).reshape(ny, nx).cpu().numpy()
    with torch.no_grad():
        XP, YP = to_phys(xf, yf, cfg)                # physical axes (curvilinear under --geo)
    XP = XP.reshape(ny, nx).cpu().numpy(); YP = YP.reshape(ny, nx).cpu().numpy()
    out = dict(exy=to(exys), phi=to(phis))
    blank = problem_of(cfg).render_mask(cfg, XP, YP)     # pixels outside the body (Problem.render_mask) are left blank
    if blank is not None:
        out = {k: np.where(blank, np.nan, v) for k, v in out.items()}
    return XP, YP, out

def _crack_phys(cfg):
    """The pre-existing cracks as physical segments for the white guide lines on phi plots:
       every segment of cfg.cracks, at its true inclination. cfg.cracks are in physical
       coordinates, the same ones dist_to_notch measures against, so they are drawn directly
       and not mapped through to_phys (which expects parametric coordinates in [0,1])."""
    return [((ax, bx), (ay, by)) for (ax, ay, bx, by) in (getattr(cfg, "cracks", None) or [])]

def plot_phi(net, cfg, save_dir, tag, ud, sample_pts=None):
    """Writes a plain phi map (phi_{tag}.png). If sample_pts is given, also writes
       phi_{tag}_pts.png with the integration points overlaid. ng=500 with shading='gouraud'
       gives a smooth render. Flat or nearest shading shows jagged crack edges at any
       resolution."""
    GX, GY, F = _grid_eval(net, cfg)
    seeds = _crack_phys(cfg)
    pts_phys = None
    if sample_pts is not None:
        if getattr(cfg, "geomap", None) is not None:
            with torch.no_grad():
                sx, sy = cfg.geomap(torch.as_tensor(sample_pts[:, 0:1], dtype=DTYPE, device=utils.DEVICE),
                                    torch.as_tensor(sample_pts[:, 1:2], dtype=DTYPE, device=utils.DEVICE))
            pts_phys = np.concatenate([sx.cpu().numpy(), sy.cpu().numpy()], axis=1)
        else:
            pts_phys = sample_pts
    for with_pts in ([False, True] if pts_phys is not None else [False]):
        fig, ax = plt.subplots(figsize=_domain_figsize(cfg))
        m = ax.pcolormesh(GX, GY, F["phi"], cmap="magma", shading="gouraud", vmin=0, vmax=1)
        _add_cbar(ax, m, "phi")
        if with_pts:
            ax.scatter(pts_phys[:, 0], pts_phys[:, 1], s=1.0, c="c", alpha=0.25, linewidths=0)
        for sx, sy in seeds:                            # no guide line without a pre-existing crack
            ax.plot(sx, sy, "w--", lw=0.8, alpha=0.6)
        problem_of(cfg).overlay(cfg, ax)
        extra = "  + integration points" if with_pts else ""
        ax.set_title(f"phi  (delta={cfg.U_ref*ud:.3e})  encoding levels={list(cfg.levels)}{extra}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        fn = f"phi_{tag}_pts.png" if with_pts else f"phi_{tag}.png"
        fig.savefig(os.path.join(save_dir, fn), dpi=140, bbox_inches="tight"); plt.close(fig)

def plot_T(netT, cfg, save_dir, tag, t):
    """The temperature field, on a diverging map centered on T0. Warmer and cooler regions
       read as the two directions away from the stress-free reference temperature, so the
       color shows the sign of the thermal strain."""
    from .thermal import T_effective
    GX, GY, nx, ny = _render_grid(cfg, 220)
    with torch.no_grad():
        T = T_effective(netT, GX.reshape(-1, 1), GY.reshape(-1, 1), cfg, t)
        XP, YP = to_phys(GX.reshape(-1, 1), GY.reshape(-1, 1), cfg)
    T = T.reshape(ny, nx).cpu().numpy()
    X = XP.reshape(ny, nx).cpu().numpy(); Y = YP.reshape(ny, nx).cpu().numpy()
    blank = problem_of(cfg).render_mask(cfg, X, Y)
    if blank is not None:
        T = np.where(blank, np.nan, T)
    half = max(abs(np.nanmin(T) - cfg.T0), abs(np.nanmax(T) - cfg.T0), 1e-12)
    fig, ax = plt.subplots(figsize=_domain_figsize(cfg))
    m = ax.pcolormesh(X, Y, T, cmap="coolwarm", shading="gouraud",
                      vmin=cfg.T0 - half, vmax=cfg.T0 + half)
    _add_cbar(ax, m, "T (K)")
    problem_of(cfg).overlay(cfg, ax)
    ax.set_title(f"T  (t={t:.4e} s)   range [{np.nanmin(T):.1f}, {np.nanmax(T):.1f}] K")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    fig.savefig(os.path.join(save_dir, f"T_{tag}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_FD_compare(FD, cfg, save_dir, fem_path):
    """DEM force-displacement curve, with the FEM curve of fem_path overlaid when that file
       exists, written to the run directory every step. The examples without a mechanical
       load record F = 0 at delta = 0, and their response is in energies.txt instead."""
    FD = np.array(FD)
    plt.figure(figsize=(6.5, 5))
    if os.path.exists(fem_path):
        fem = np.loadtxt(fem_path)
        plt.plot(fem[:, 0], fem[:, 1], "-", lw=1.3, alpha=0.55,
                 label=f"{os.path.basename(fem_path)} (FEM)")
    if len(FD):
        plt.plot(FD[:, 0], FD[:, 1], "o-", ms=4, lw=1.6, label="DEM")
    plt.xlabel("Displacement delta (mm)"); plt.ylabel("Force (N)")
    plt.title(f"DEM vs FEM (l={cfg.l})")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "FD_vs_FEM.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_FD_solo(FD, cfg, save_dir):
    """DEM force-displacement curve alone, written into the per-step folder on plot steps."""
    FD = np.array(FD)
    plt.figure(figsize=(6.5, 5))
    if len(FD):
        plt.plot(FD[:, 0], FD[:, 1], "o-", ms=4, lw=1.6, color="tab:green")
    plt.xlabel("Displacement delta (mm)"); plt.ylabel("Force (N)")
    plt.title(f"DEM F-delta (l={cfg.l})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "FD_solo.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_fields(net, cfg, save_dir, tag, ud, ng=500, chunk=45000):
    """Maps of the displacements u, v and the shear strain exy, with the phi=0.5 crack contour
       drawn on exy. ng=500 with chunked evaluation gives a smooth render."""
    from .solver import _fields, kinematics
    GX, GY, nx, ny = _render_grid(cfg, ng)
    xf, yf = GX.reshape(-1, 1), GY.reshape(-1, 1)
    us, vs, exys, phis = [], [], [], []
    for i in range(0, xf.shape[0], chunk):
        xc, yc = xf[i:i + chunk].clone(), yf[i:i + chunk].clone()
        exx, eyy, exy, phi, gp2, _, _ = kinematics(net, xc, yc, cfg, ud)
        with torch.no_grad():
            u, v, _, _ = _fields(net, xf[i:i + chunk].clone(), yf[i:i + chunk].clone(),
                                 cfg, ud)
        us.append(u.detach()); vs.append(v.detach())
        exys.append(exy.detach()); phis.append(phi.detach())
    to = lambda ts: torch.cat(ts).reshape(ny, nx).cpu().numpy()
    U, V, EXY, PHI = to(us), to(vs), to(exys), to(phis)
    with torch.no_grad():
        XP, YP = to_phys(xf, yf, cfg)
    X = XP.reshape(ny, nx).cpu().numpy(); Y = YP.reshape(ny, nx).cpu().numpy()
    blank = problem_of(cfg).render_mask(cfg, X, Y)      # pixels outside the body (Problem.render_mask) are left blank
    if blank is not None:
        U, V, EXY, PHI = (np.where(blank, np.nan, A) for A in (U, V, EXY, PHI))
    # Palette: phi=magma, u/v=viridis, T=coolwarm, exy=viridis with symmetric +- limits
    # (zero shear is mid-green, the sign is the dark or bright side). The phi=0.5 contour
    # is drawn on exy only, so the displacement maps stay clean.
    for F, lab, fn, cmap, sym, ctr in [(U, "u (mm)", f"field_u_{tag}.png", "viridis", False, False),
                                       (V, "v (mm)", f"field_v_{tag}.png", "viridis", False, False),
                                       (EXY, "exy", f"field_exy_{tag}.png", "viridis", True, True)]:
        fig, ax = plt.subplots(figsize=_domain_figsize(cfg))
        if sym:
            vv = np.nanpercentile(np.abs(F), 99.5); kw = dict(vmin=-vv, vmax=vv)
        else:
            kw = dict(vmin=np.nanpercentile(F, 0.5), vmax=np.nanpercentile(F, 99.5))
        m = ax.pcolormesh(X, Y, F, cmap=cmap, shading="gouraud", **kw)
        _add_cbar(ax, m, lab)
        if ctr:
            # phi=0.5 crack contour, on exy only
            ax.contour(X, Y, PHI, levels=[0.5], colors="w", linewidths=0.7)
            ax.set_title(f"{lab}  (delta={cfg.U_ref*ud:.3e})  white = phi=0.5")
        else:
            ax.set_title(f"{lab}  (delta={cfg.U_ref*ud:.3e})")
        problem_of(cfg).overlay(cfg, ax)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        fig.savefig(os.path.join(save_dir, fn), dpi=140, bbox_inches="tight"); plt.close(fig)

def plot_linecut(net, cfg, save_dir, tag, ud):
    """u along three vertical parametric lines (xi = 0.25, 0.55, 0.85) across eta = 0.5, to
       show whether the displacement jump is present and how sharp it is. It is meaningful for
       the notched plate, where eta = 0.5 is the notch line."""
    from .solver import _fields
    yy = torch.linspace(0.30, 0.70, 500, device=utils.DEVICE).reshape(-1, 1)
    plt.figure(figsize=(6.5, 5))
    for xv, cc in [(0.25, "tab:red"), (0.55, "tab:orange"), (0.85, "tab:blue")]:
        xx = torch.full_like(yy, xv)
        with torch.no_grad():
            uu = _fields(net, xx, yy, cfg, ud)[0].cpu().numpy()
        plt.plot(yy.cpu().numpy(), uu, color=cc, label=f"x={xv}")
    plt.axvline(0.5, ls=":", c="k", alpha=0.5, label="notch line y=0.5")
    plt.xlabel("y"); plt.ylabel("u(x,y)")
    plt.title(f"u linecut across the crack ({tag})")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"linecut_{tag}.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_geometry(cfg, save_dir, ng=400):
    """One figure of the patch, written once per run since the geometry is fixed: the
       physical images of the knot lines (the element boundaries of the patch) and the
       control net (points and polygon). When |det J| varies (distorted or curved patches) it
       is drawn as a smooth field behind the lines. When |det J| is essentially constant (the
       identity square, det J = 1) the fill is skipped and the lines are drawn on a white
       background. Does nothing under --geo none."""
    gm = getattr(cfg, "geomap", None)
    if gm is None:
        return
    xs = torch.linspace(0, 1, ng, device=utils.DEVICE)
    GXi, GEta = torch.meshgrid(xs, xs, indexing="xy")
    xi, eta = GXi.reshape(-1, 1), GEta.reshape(-1, 1)
    with torch.no_grad():
        xp, yp = gm(xi, eta)
    _, _, _, _, det = gm.jac(xi, eta)
    X = xp.reshape(ng, ng).cpu().numpy(); Y = yp.reshape(ng, ng).cpu().numpy()
    D = det.reshape(ng, ng).cpu().numpy()
    ku = np.unique(gm.ku.cpu().numpy()); kv = np.unique(gm.kv.cpu().numpy())
    cx = gm.cx.cpu().numpy(); cy = gm.cy.cpu().numpy()
    t = torch.linspace(0, 1, 200, device=utils.DEVICE).reshape(-1, 1)
    varies = (D.max() - D.min()) / (abs(D.mean()) + 1e-30) > 1e-3

    fig, ax = plt.subplots(figsize=(7.4, 6))
    if varies:
        m = ax.pcolormesh(X, Y, D, cmap="twilight_shifted", shading="gouraud")   # continuous
        plt.colorbar(m, ax=ax, label="|det J|  (param->physical area factor)")
        detstr = f"detJ in [{D.min():.3f}, {D.max():.3f}]"
    else:
        detstr = f"detJ = {D.mean():.3f} (uniform)"
    if not gm.is_identity:                                   # knot lines unless the map is identity
        with torch.no_grad():
            for kval in ku:
                px, py = gm(torch.full_like(t, float(kval)), t)
                ax.plot(px.cpu().numpy(), py.cpu().numpy(), "-", c="0.15", lw=0.9)
            for kval in kv:
                px, py = gm(t, torch.full_like(t, float(kval)))
                ax.plot(px.cpu().numpy(), py.cpu().numpy(), "-", c="0.15", lw=0.9)
    for i in range(cx.shape[0]):                             # control net
        ax.plot(cx[i, :], cy[i, :], "-", c="tab:red", lw=0.6, alpha=0.6)
    for j in range(cx.shape[1]):
        ax.plot(cx[:, j], cy[:, j], "-", c="tab:red", lw=0.6, alpha=0.6)
    ax.plot(cx.ravel(), cy.ravel(), "o", c="tab:red", ms=3.5, label="control points")
    ax.set_title(f"geometry patch: knot lines + control net  (geo={cfg.geo}, deg {gm.pu}x{gm.pv}, {detstr})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", "box"); ax.legend(fontsize=8)
    fig.savefig(os.path.join(save_dir, "geometry_patch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_geometry_bc(cfg, save_dir):
    """Physical geometry with the iso-parametric lines, the boundary of the patch and its
       NURBS control net. Does nothing under --geo none."""
    gm = getattr(cfg, "geomap", None)
    if gm is None:
        return

    def P(xa, ea):
        with torch.no_grad():
            X, Y = gm(torch.as_tensor(xa, dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1),
                      torch.as_tensor(ea, dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1))
        return X.cpu().numpy().ravel(), Y.cpu().numpy().ravel()

    t = np.linspace(0, 1, 240)
    fig, ax = plt.subplots(figsize=(6.4, 7))
    if not gm.is_identity:                                   # iso-parametric lines unless the
        for e in np.linspace(0, 1, 19):                      # map is the identity
            x, y = P(t, np.full_like(t, e)); ax.plot(x, y, "0.8", lw=0.5)
        for xi in np.linspace(0, 1, 10):
            x, y = P(np.full_like(t, xi), t); ax.plot(x, y, "0.8", lw=0.5)
    for xi in (0.0, 1.0):
        x, y = P(np.full_like(t, xi), t); ax.plot(x, y, "0.15", lw=1.6)
    for e in (0.0, 1.0):
        x, y = P(t, np.full_like(t, e)); ax.plot(x, y, "0.15", lw=1.6)
    cx = gm.cx.cpu().numpy(); cy = gm.cy.cpu().numpy()     # NURBS control net
    for i in range(cx.shape[0]):
        ax.plot(cx[i, :], cy[i, :], "-", c="0.45", lw=0.7, alpha=0.7, zorder=1)
    for j in range(cx.shape[1]):
        ax.plot(cx[:, j], cy[:, j], "-", c="0.45", lw=0.7, alpha=0.7, zorder=1)
    ax.plot(cx.ravel(), cy.ravel(), "s", c="darkorange", ms=4, label="control points", zorder=5)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", "box")
    ax.set_title(f"geometry: boundary and control net  (geo={cfg.geo})")
    ax.set_xlabel("x (physical)"); ax.set_ylabel("y (physical)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "geometry_bc.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_parametric(cfg, save_dir, ng=300):
    """The parametric domain [0,1]^2, the square on which the network, encoding, sampler and
       quadrature live (the physical shape comes from the geometry map). Draws a uniform
       iso-parametric grid and the crack seed (phi0 map)."""
    xs = torch.linspace(0, 1, ng)
    GX, GY = torch.meshgrid(xs, xs, indexing="xy")
    with torch.no_grad():
        p0 = phi0_field(GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE), cfg)
    p0 = p0.reshape(ng, ng).cpu().numpy()
    from matplotlib.colors import LinearSegmentedColormap
    wcmap = LinearSegmentedColormap.from_list("phi0_white", ["white", "#8000a0", "#1a0030"])
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.set_facecolor("white")
    ax.pcolormesh(GX.numpy(), GY.numpy(), p0, shading="gouraud", cmap=wcmap, vmin=0, vmax=1)
    for g in np.linspace(0, 1, 13):                          # uniform iso-parametric grid
        ax.axvline(g, c="0.75", lw=0.5); ax.axhline(g, c="0.75", lw=0.5)
    ax.set_xlabel("xi"); ax.set_ylabel("eta")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal", "box")
    ax.set_title(f"parametric domain [0,1]$^2$  (geo={cfg.geo})", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "geometry_parametric.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
