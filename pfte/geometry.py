"""
Geometry: the isogeometric patch, the crack seeds and the treatment of the fixed corners.

The physical domain is one B-spline or NURBS patch, so curved bodies are represented exactly
and the Jacobian comes from automatic differentiation rather than from a mesh. Fields, encoding
and quadrature all live on the parametric square. Everything that needs the physical position
of a point (the crack seed, the corner discs, the plot axes) goes through to_phys.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from . import utils
from .utils import DTYPE, problem_of


# --------------------------------------------------------------------------
# B-spline / NURBS patch


def _bspline_basis(t, knots, p):
    """Cox-de Boor recursion, vectorized and composable with autodiff (safe for a double
       backward pass). t (N,), open knot vector (nk,), degree p -> (N, m) basis values,
       m = nk-p-1. The degree-0 boxcar is piecewise constant, so its gradient is zero almost
       everywhere, as for the detached floor index in the encoding's _interp_level. All
       dependence on t that carries derivatives is in the polynomial factors of the
       recursion. The right endpoint belongs to the last span, where the derivative is
       one-sided."""
    nk = knots.numel()
    m = nk - p - 1
    t = t.reshape(-1, 1)
    # Span membership is half-open, t in [k_i, k_{i+1}), except for the last nonempty span,
    # which is closed, [k_{m-1}, k_m]. Queries at or above the domain end are moved just
    # inside the last span (detached, used for index selection only).
    hi = knots[m]
    tq = torch.where(t >= hi, hi - 1e-12 * (1.0 + hi.abs()), t)
    B = ((tq >= knots[:-1]) & (tq < knots[1:])).to(t.dtype)          # (N, nk-1)
    for k in range(1, p + 1):
        d1 = knots[k:nk - 1] - knots[:nk - k - 1]                    # left span lengths
        d2 = knots[k + 1:nk] - knots[1:nk - k]                       # right span lengths
        w1 = torch.where(d1 > 0, (t - knots[:nk - k - 1]) / torch.where(d1 > 0, d1, torch.ones_like(d1)),
                         torch.zeros_like(t * d1))
        w2 = torch.where(d2 > 0, (knots[k + 1:nk] - t) / torch.where(d2 > 0, d2, torch.ones_like(d2)),
                         torch.zeros_like(t * d2))
        B = w1 * B[:, :nk - k - 1] + w2 * B[:, 1:nk - k]
    return B                                                          # (N, m)

class GeoMap(nn.Module):
    """Single-patch B-spline or NURBS geometry map (xi,eta) in [0,1]^2 -> physical (x,y).
       The map is fixed (buffers, not parameters): the geometry is data, not a trainable
       field. Fields, encoding and sampler live on the parametric domain. The energy
       quadrature carries |det J| and physical gradients use J^{-T}. jac() builds J on a
       separate detached graph, because J does not depend on the network parameters and
       enters the energy as a constant. The in-graph forward() is used where the map itself
       must be differentiated, for example the crack-seed distance through to_phys."""
    def __init__(self, pu, pv, knots_u, knots_v, ctrl, weights=None):
        super().__init__()
        self.is_identity = False                  # set by build_geomap, skips jac in kinematics
        self.pu, self.pv = int(pu), int(pv)
        ku = torch.as_tensor(knots_u, dtype=DTYPE)
        kv = torch.as_tensor(knots_v, dtype=DTYPE)
        C = torch.as_tensor(ctrl, dtype=DTYPE)                        # (mu, mv, 2)
        W = torch.ones(C.shape[0], C.shape[1], dtype=DTYPE) if weights is None \
            else torch.as_tensor(weights, dtype=DTYPE)
        assert ku.numel() - self.pu - 1 == C.shape[0], "knots_u/ctrl size mismatch"
        assert kv.numel() - self.pv - 1 == C.shape[1], "knots_v/ctrl size mismatch"
        self.register_buffer("ku", ku)
        self.register_buffer("kv", kv)
        self.register_buffer("cx", C[..., 0])
        self.register_buffer("cy", C[..., 1])
        self.register_buffer("w", W)

    def forward(self, xi, eta):
        """(N,1),(N,1) -> physical (N,1),(N,1), rational (NURBS) if weights != 1."""
        Bu = _bspline_basis(xi, self.ku, self.pu)                     # (N, mu)
        Bv = _bspline_basis(eta, self.kv, self.pv)                    # (N, mv)
        Wf = torch.einsum("ni,ij,nj->n", Bu, self.w, Bv)
        X = torch.einsum("ni,ij,nj->n", Bu, self.w * self.cx, Bv) / Wf
        Y = torch.einsum("ni,ij,nj->n", Bu, self.w * self.cy, Bv) / Wf
        return X.reshape(-1, 1), Y.reshape(-1, 1)

    def jac(self, xi, eta):
        """Entries of J and det J at the query points, detached (constants in the energy
           graph). They are built on a separate graph, kept apart from the field autodiff."""
        xg = xi.detach().reshape(-1, 1).clone().requires_grad_(True)
        yg = eta.detach().reshape(-1, 1).clone().requires_grad_(True)
        xp, yp = self.forward(xg, yg)
        o = torch.ones_like(xp)
        a, b = torch.autograd.grad(xp, [xg, yg], o, retain_graph=True)   # dx/dxi, dx/deta
        c, d = torch.autograd.grad(yp, [xg, yg], o)                      # dy/dxi, dy/deta
        det = a * d - b * c
        return a.detach(), b.detach(), c.detach(), d.detach(), det.detach()

    @property
    def corners_phys(self):
        with torch.no_grad():
            z = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=DTYPE, device=self.ku.device)
            e = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=DTYPE, device=self.ku.device)
            xp, yp = self.forward(z, e)
        return tuple((float(xp[i]), float(yp[i])) for i in range(4))

def _open_uniform_knots(m, p):
    """Open uniform knot vector for m control points of degree p."""
    ni = m - p                                                        # interior spans
    core = np.linspace(0.0, 1.0, ni + 1)
    return np.concatenate([np.zeros(p), core, np.ones(p)])

def _greville(knots, p):
    """Greville abscissae. Control values placed there reproduce linear functions exactly,
       which gives an exact identity patch."""
    k = np.asarray(knots)
    m = len(k) - p - 1
    return np.array([k[i + 1:i + p + 1].mean() for i in range(m)])

def build_ring_patch(Ri, Ro, quarter=False):
    """Half-annulus (right half, x >= 0) of a thick-walled ring as a single NURBS patch.
       xi (u direction) is radial (degree 1, Ri -> Ro). eta (v direction) is angular
       (rational quadratic, two 90 deg arcs sweeping -90 deg -> 0 deg -> +90 deg, the right
       semicircle from (0,-r) through (r,0) to (0,+r)). Hence eta = 0.5 is the horizontal
       axis (angle 0), eta > 0.5 maps to y > 0 and eta < 0.5 to y < 0, and eta = 0 and
       eta = 1 both map onto the symmetry line x = 0. Returns (pu, pv, knots_u, knots_v,
       ctrl[mu][mv][2], weights[mu][mv]).

       With quarter=True the patch is the upper-right quadrant instead: one 90 deg rational
       quadratic arc from (r,0) to (0,r), so eta = 0 is the +x axis and eta = 1 the +y axis,
       both symmetry lines. Halving the swept arc halves the number of cells needed in the
       angular direction at a given h/l. The radial direction spans the same wall thickness
       either way, so the saving over the half-annulus is a factor 2. On this patch
       eta = 0.5 is the 45 deg line, not the horizontal axis."""
    s = 2.0 ** -0.5
    if quarter:
        unit = np.array([[1., 0.], [1., 1.], [0., 1.]])                    # one 90-deg arc
        wv = np.array([1.0, s, 1.0])
        knots_v = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    else:
        unit = np.array([[0., -1.], [1., -1.], [1., 0.], [1., 1.], [0., 1.]])  # semicircle
        wv = np.array([1.0, s, 1.0, s, 1.0])                         # rational-quadratic circle
        knots_v = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1], dtype=float)  # two 90-deg Bezier arcs
    pv, pu = 2, 1
    knots_u = np.array([0, 0, 1, 1], dtype=float)                     # linear radial, 2 layers
    radii = np.array([float(Ri), float(Ro)])
    mv = unit.shape[0]
    ctrl = np.empty((2, mv, 2)); W = np.empty((2, mv))
    for i in range(2):
        ctrl[i] = radii[i] * unit                     # exact annulus: r_i * unit circle
        W[i] = wv
    return pu, pv, knots_u, knots_v, ctrl, W


def build_geomap(spec, logger=None, ring=(5.0, 20.0)):
    """Builds the geometry map named by --geo. 'none' -> None. 'identity' -> Greville identity
       patch (the full spline machinery with the map equal to the identity up to round-off, a
       regression check). 'distort' -> the same unit square with the boundary and the y=0.5
       notch midline held pointwise invariant and the interior distorted, so J != I with
       identical physics (a null test of the pullback). The end rows and columns of control
       points have zero offset, so the edges stay exact, and the offsets are antisymmetric
       about the middle control row, so the map is odd and the midline stays exact.
       'ring' -> half-annulus with radii ring=(Ri,Ro). 'ring_quarter' -> the upper-right
       quadrant of the same annulus (one 90 deg arc). 'strip:W,H' -> the rectangle
       [0,W]x[0,H]. 'square:a' -> the square [-a,a]^2. Otherwise a JSON file
       {pu, pv, knots_u, knots_v, ctrl[mu][mv][2], weights?}."""
    if spec is None or spec == "none":
        return None
    if spec in ("ring", "ring_quarter"):
        pu, pv, ku, kv, ctrl, W = build_ring_patch(*ring, quarter=(spec == "ring_quarter"))
        gm = GeoMap(pu, pv, ku, kv, ctrl, W)
    elif spec.startswith("strip:"):
        # A rectangle [0,W]x[0,H]: the same degree-2 Greville patch as 'identity' with its
        # control points scaled, so the map is affine, J = diag(W,H) is constant and |det J| = W*H.
        # One patch at any aspect ratio. It is not flagged as identity, so the pullback is
        # evaluated.
        w, h = (float(v) for v in spec.split(":", 1)[1].split(","))
        p, m = 2, 7
        knots = _open_uniform_knots(m, p)
        g = _greville(knots, p)
        GX0, GY0 = np.meshgrid(g, g, indexing="ij")
        gm = GeoMap(p, p, knots, knots, np.stack([GX0 * w, GY0 * h], axis=-1))
    elif spec.startswith("square:"):
        # A square [-a,a]^2 as a degree-1 2x2 patch. Bilinear on a rectangle is affine, so
        # J = diag(2a,2a) exactly.
        a_ = float(spec.split(":", 1)[1])
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        ctrl = np.stack([np.array([[-a_, -a_], [a_, a_]]),      # x depends on xi only
                         np.array([[-a_, a_], [-a_, a_]])], axis=-1)   # y on eta only
        gm = GeoMap(1, 1, knots, knots, ctrl)
    elif spec in ("identity", "distort"):
        p, m = 2, 7
        knots = _open_uniform_knots(m, p)
        g = _greville(knots, p)
        GX0, GY0 = np.meshgrid(g, g, indexing="ij")                   # ctrl[i,j] at (g_i, g_j)
        CX, CY = GX0.copy(), GY0.copy()
        if spec == "distort":
            # The offsets are s(g_i)*q(g_j). s vanishes at the ends, so the fixed end rows
            # and columns keep the boundary exact. q = sin(2*pi*t) is odd under g_j -> 1-g_j
            # (the Greville set is symmetric), so the control offsets are antisymmetric about
            # the middle row, the map is odd about eta=0.5 and the y=0.5 notch midline is
            # pointwise invariant (keeping the notch stratum and phi0 aligned). Elsewhere
            # J != I, which is the null test of the pullback.
            q = np.sin(2.0 * np.pi * GY0)
            CX = GX0 + 0.07 * np.sin(np.pi * GX0) * q
            CY = GY0 + 0.05 * np.sin(2.0 * np.pi * GX0) * q
        ctrl = np.stack([CX, CY], axis=-1)
        gm = GeoMap(p, p, knots, knots, ctrl)
    else:
        with open(spec) as f:
            d = json.load(f)
        gm = GeoMap(d["pu"], d["pv"], d["knots_u"], d["knots_v"], d["ctrl"],
                    d.get("weights"))
    gm = gm.to(utils.DEVICE)
    # The identity map reproduces the physical coordinates to machine precision, so it is
    # flagged and kinematics skips the Jacobian autodiff pass, which would change nothing.
    gm.is_identity = (spec == "identity")
    # validity and distortion report: det J on a Sobol cloud (area = integral of det J)
    smp = torch.as_tensor(qmc.Sobol(d=2, scramble=True, seed=4).random(1 << 14),
                          dtype=DTYPE, device=utils.DEVICE)
    a, b, c, d_, det = gm.jac(smp[:, 0:1], smp[:, 1:2])
    stretch = torch.sqrt(torch.stack([a, b, c, d_]) ** 2).max()
    msg = (f"geomap '{spec}': det J in [{float(det.min()):.4f}, {float(det.max()):.4f}], "
           f"area~{float(det.mean()):.6f}, max |J| entry {float(stretch):.3f}, "
           f"corners {gm.corners_phys}")
    assert float(det.min()) > 0.0, f"geometry map not invertible: {msg}"
    if logger:
        logger.info(msg)
    return gm


# --------------------------------------------------------------------------
# crack seeds and corner treatment


def to_phys(x, y, cfg):
    """Parametric -> physical coordinates. Every such conversion goes through this function.
       With --geo none the parametric coordinates are the physical ones. With a GeoMap the
       notch distance, the corner discs and the plot axes are all mapped here."""
    gm = getattr(cfg, "geomap", None)
    if gm is None or gm.is_identity:
        return x, y
    return gm(x, y)

SEN_NOTCH = "0,0.5,0.5,0.5"

DEFAULT_CRACKS = "0.25,0.35,0.35,0.45;0.45,0.45,0.55,0.55;0.65,0.55,0.75,0.65"

def parse_cracks(s):
    """"ax,ay,bx,by;..." -> [(ax,ay,bx,by), ...] physical crack segments."""
    out = []
    for seg in str(s).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        v = [float(t) for t in seg.split(",")]
        assert len(v) == 4, f"each crack needs 4 numbers ax,ay,bx,by; got {seg!r}"
        out.append(tuple(v))
    assert out, "no cracks parsed"
    return out

def dist_to_notch(x, y, cfg):
    """Minimum point-to-segment distance over all cfg.cracks, in physical space. The projection
       parameter t is restricted to [0,1], so segments of any orientation are handled."""
    xp, yp = to_phys(x, y, cfg)
    d2 = None
    for (ax, ay, bx, by) in cfg.cracks:
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey + 1e-30
        t = torch.clamp(((xp - ax) * ex + (yp - ay) * ey) / L2, 0.0, 1.0)
        cx, cy = ax + t * ex, ay + t * ey
        dj = (xp - cx) ** 2 + (yp - cy) ** 2
        d2 = dj if d2 is None else torch.minimum(d2, dj)
    return torch.sqrt(d2 + 1e-30)

def crack_profile(d, cfg):
    """The optimal phi profile of the model at distance d from a crack, the shape a fully
       developed crack takes. It seeds pre-existing cracks so that no load step has to grow
       them."""
    if getattr(cfg, "pf_order", 2) == 4:
        # Fourth-order optimal profile of Borden et al. (2014), Eq. 13 with phi=1-c, l0=l/2:
        # exp(-2d/l)(1+2d/l). It is C1 at the crack (zero slope at d=0), unlike the kink of
        # the second-order profile exp(-d/l).
        r = d / cfg.l
        return torch.exp(-2.0 * r) * (1.0 + 2.0 * r)
    if getattr(cfg, "at", 2) == 1:
        # AT1 optimal profile with compact support 2l (the AT2 exponential is not stationary
        # for AT1)
        return torch.clamp(1.0 - d / (2.0 * cfg.l), min=0.0) ** 2
    return torch.exp(-d / cfg.l)


def phi0_field(x, y, cfg):
    """The crack seed of the running example (see pfte.problems.Problem.phi0)."""
    return problem_of(cfg).phi0(x, y, cfg)

GRIP_CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

def grip_corners(cfg):
    """Physical points protected by the corner toughening. By default these are the corners
       of the patch. An example whose loaded edges meet its free flanks elsewhere returns its
       own points from pfte.problems.Problem.grip_points."""
    own = problem_of(cfg).grip_points(cfg)
    if own is not None:
        return own
    gm = getattr(cfg, "geomap", None)
    if gm is None or gm.is_identity:
        return GRIP_CORNERS
    return gm.corners_phys

def phi_gate(x, y, cfg):
    """Optional corner gate: multiplies the learnable phi by a factor that vanishes at the
       points of grip_corners (by default the four corners of the patch) and recovers within
       about grip_l*l. These corners are
       stress-singular points where damage nucleates spuriously. The edges themselves stay
       damageable, so a crack can reach them. The gate is smooth (C-infinity), so
       |grad phi|^2 stays well defined. grip_l <= 0 disables it. Corners whose grip_released
       flag is set (see --grip_release) are skipped. When --gc_grip > 0 the variational
       corner toughening is used instead and the gate is off."""
    if cfg.grip_l <= 0 or getattr(cfg, "gc_grip", 0.0) > 0:
        return 1.0
    w = cfg.grip_l * cfg.l
    rel = getattr(cfg, "grip_released", (False, False, False, False))
    xp, yp = to_phys(x, y, cfg)
    g = 1.0
    for k, (cx, cy) in enumerate(grip_corners(cfg)):
        if k < len(rel) and rel[k]:         # an example may declare more grip points than four
            continue
        r = torch.sqrt((xp - cx) ** 2 + (yp - cy) ** 2 + 1e-30)
        g = g * torch.tanh(r / w)
    return g

@torch.no_grad()
def update_grip_release(cfg, prev_model, logger):
    """Optional (--grip_release > 0) release of the corner gates when a crack arrives. The
       gate of a corner stays on until phi_prev exceeds the threshold somewhere in the ring
       [1.3, 2.5]*grip_l*l just outside its disc. The gate of that corner is then switched off
       for the rest of the run, so the final separation at the corner is physical. The check
       runs once per load step on the frozen state of the previous step, so each step remains
       one variational minimization. The released flags are saved with the checkpoint."""
    if cfg.grip_l <= 0 or cfg.grip_release <= 0 or cfg.gc_grip > 0 or prev_model is None:
        return
    from .solver import phi_value          # imported here because solver imports this module
    pts = grip_corners(cfg)                 # the same points phi_gate gates, not a fixed four
    rel = list(cfg.grip_released) + [False] * max(0, len(pts) - len(cfg.grip_released))
    w = cfg.grip_l * cfg.l
    for k, (cx, cy) in enumerate(pts):
        if rel[k]:
            continue
        n = 4000
        r = (1.3 + 1.2 * torch.rand(n, 1, device=utils.DEVICE)) * w
        th = 2.0 * math.pi * torch.rand(n, 1, device=utils.DEVICE)
        xs = cx + r * torch.cos(th)
        ys = cy + r * torch.sin(th)
        m = ((xs >= 0.0) & (xs <= 1.0) & (ys >= 0.0) & (ys <= 1.0)).reshape(-1)
        if int(m.sum()) == 0:
            continue
        ph = phi_value(prev_model, xs[m].reshape(-1, 1), ys[m].reshape(-1, 1), cfg)
        if float(ph.max()) > cfg.grip_release:
            rel[k] = True
            logger.info(f"  grip gate released at corner ({cx:.0f},{cy:.0f}): "
                        f"phi_max {float(ph.max()):.2f} > {cfg.grip_release} "
                        f"in the ring around it")
    cfg.grip_released = tuple(rel)

def gc_bump(x, y, cfg):
    """Toughening bump for --gc_grip: 1 at the points of grip_corners, decaying to 0 outside
       discs of radius about grip_l*l, plus any edge band the example declares through
       Problem.grip_edges. It is the complement of the corner gate profile, and grip_l sets
       the toughening radius. The bump is smooth, so e_fr stays C-infinity."""
    w = max(cfg.grip_l, 1e-6) * cfg.l
    xp, yp = to_phys(x, y, cfg)
    g = 1.0
    for cx, cy in grip_corners(cfg):
        r = torch.sqrt((xp - cx) ** 2 + (yp - cy) ** 2 + 1e-30)
        g = g * torch.tanh(r / w)
    g = g * problem_of(cfg).grip_edges(xp, yp, cfg)     # 1.0 unless the example toughens strips
    return 1.0 - g
