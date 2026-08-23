"""
E-Init (ellipsoid initialization) for point-cloud registration, extended to
similarity transforms.

Based on: A. Kolpakov and M. Werman, "An approach to robust ICP initialization"
(arXiv:2212.05332). The original method initializes ICP for two point clouds
related by a *rigid* motion  Q = O.P.S  (rotation O, unknown correspondence /
permutation S) by matching the covariance ellipsoids of the two clouds and
testing the finite sign-flip ambiguity of the principal axes.

Part B extension (this file): our two clouds come from independent monocular SfM
reconstructions, so each carries an ARBITRARY relative scale -- they are related
by a *similarity*  Q = s.O.P.S, not a rigid motion. The extension is small and
principled:

    if   E_Q = s^2 . O . E_P . O^T          (covariance transforms with scale^2)
    then the scale falls entirely into the EIGENVALUES (lambda_Q = s^2 lambda_P)
    and leaves the EIGENVECTORS (principal-axis directions) untouched.

So the rotation is recovered exactly by the original method (scale-invariant),
and the scale is recovered for free from the eigenvalue ratio
    s = sqrt( geomean_i( lambda_target,i / lambda_source,i ) ).

We return the similarity (s, R, t) that maps the SOURCE cloud onto the TARGET
cloud:  p_target ~= s * R @ p_source + t.
"""

from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Ellipsoid primitives
# ---------------------------------------------------------------------------
def covariance_ellipsoid(X: np.ndarray):
    """Barycenter, centered points and covariance (inertia) matrix of a cloud.

    X : (N, 3) points. Returns (b, Xc, E) with E = Xc^T Xc / N  (3x3, SPD).
    """
    X = np.asarray(X, dtype=np.float64)
    b = X.mean(axis=0)
    Xc = X - b
    E = (Xc.T @ Xc) / len(Xc)
    return b, Xc, E


def eig_frame(E: np.ndarray):
    """Eigen-decomposition of an SPD matrix, eigenvalues DESCENDING.

    Returns (w, V) with w[0] >= w[1] >= w[2] and V's columns the matching
    orthonormal eigenvectors, forced to a right-handed frame (det(V) = +1) so
    that the remaining ambiguity is exactly the sign-flip group we test below.
    """
    w, V = np.linalg.eigh(E)              # ascending, V orthonormal
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if np.linalg.det(V) < 0:              # enforce right-handed
        V[:, 2] *= -1.0
    return w, V


# ---------------------------------------------------------------------------
# Nearest-neighbour scoring (used to pick among the finite candidate set)
# ---------------------------------------------------------------------------
def _nn_score(src_xyz: np.ndarray, tgt_tree: cKDTree, trim: float = 0.8) -> float:
    """Mean trimmed nearest-neighbour distance from src points into a target tree.

    Trimming (keep the closest `trim` fraction) makes the score robust to the
    non-overlapping tail when the two clouds only partially overlap.
    """
    d, _ = tgt_tree.query(src_xyz, k=1, workers=-1)
    d.sort()
    k = max(1, int(trim * len(d)))
    return float(d[:k].mean())


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def e_init_similarity(
    source: np.ndarray,
    target: np.ndarray,
    allow_reflection: bool = False,
    score_samples: int = 20000,
    trim: float = 0.8,
    seed: int = 0,
    up_source: np.ndarray | None = None,
    up_target: np.ndarray | None = None,
):
    """Initialize a similarity transform aligning `source` onto `target`.

    Parameters
    ----------
    source, target : (N,3), (M,3) point clouds (may differ in cardinality/scale).
    allow_reflection : if False (default) only proper rotations (det=+1) are
        tested -- correct for merging real physical scenes. If True, all 2^3
        sign flips (the reflection group Ref(3)) are tested, as in the paper.
    score_samples : #points subsampled from each cloud when scoring candidates.
    trim : fraction of closest NN distances kept when scoring (partial-overlap
        robustness).

    Returns
    -------
    dict with keys:
        s, R, t        : the similarity  p_tgt ~= s * R @ p_src + t
        T              : 4x4 matrix embedding [[s*R, t],[0,1]] (open3d-ready)
        scale_per_axis : sqrt(lambda_tgt/lambda_src) for each axis (their spread
                         is a diagnostic of how similarity-like the relation is)
        eig_source, eig_target : the eigenvalue triples
        score          : NN score of the chosen candidate
        n_candidates   : how many candidates were tested
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)

    b_s, _, E_s = covariance_ellipsoid(src)
    b_t, _, E_t = covariance_ellipsoid(tgt)
    w_s, V_s = eig_frame(E_s)
    w_t, V_t = eig_frame(E_t)

    # Scale from eigenvalue ratio: for a true similarity all three per-axis
    # sqrt-ratios are equal. Under PARTIAL overlap one axis inflates (the extent
    # present in one cloud but not the other), so we take the MEDIAN per-axis
    # ratio -- robust to a single inflated axis -- as the scale estimate. Their
    # spread is itself a diagnostic of how similarity-like / how well-overlapped
    # the two clouds are.
    scale_per_axis = np.sqrt(w_t / w_s)
    s = float(np.median(scale_per_axis))

    # Candidate rotations R = V_t @ D @ V_s^T for sign matrices D. With both
    # frames right-handed, det(R) = det(D); keep det(D)=+1 for proper rotations.
    signs = list(itertools.product((1.0, -1.0), repeat=3))
    candidates = []
    for sx, sy, sz in signs:
        D = np.diag([sx, sy, sz])
        if not allow_reflection and np.linalg.det(D) < 0:
            continue
        R = V_t @ D @ V_s.T
        # Gravity gate: a correct alignment maps source-up to target-up (same
        # hemisphere). Reject candidates that flip up->down (the upside-down
        # family) -- they are geometrically near-symmetric to the truth so NN /
        # ICP overlap cannot tell them apart, but gravity can.
        if up_source is not None and up_target is not None:
            if float((R @ up_source) @ up_target) <= 0.0:
                continue
        candidates.append(R)
    if not candidates:                       # gate removed everything -> ignore it
        candidates = [V_t @ np.diag(d) @ V_s.T for d in signs
                      if allow_reflection or np.linalg.det(np.diag(d)) > 0]

    # Score each candidate by trimmed NN distance on subsampled clouds.
    src_s = _subsample(src, score_samples, rng)
    tgt_s = _subsample(tgt, score_samples, rng)
    tgt_tree = cKDTree(tgt_s)

    scored = []
    for R in candidates:
        t = b_t - s * R @ b_s
        moved = (s * R @ src_s.T).T + t
        score = _nn_score(moved, tgt_tree, trim=trim)
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t
        scored.append(dict(s=s, R=R, t=t, T=T, score=score))

    scored.sort(key=lambda c: c["score"])   # ascending NN distance
    best = dict(scored[0])
    best.update(
        scale_per_axis=scale_per_axis,
        eig_source=w_s,
        eig_target=w_t,
        n_candidates=len(candidates),
        candidates=scored,       # all candidates, NN-sorted, for ICP re-scoring
    )
    return best


def _align_vectors(u_from: np.ndarray, u_to: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector u_from onto unit u_to."""
    u_from = u_from / (np.linalg.norm(u_from) + 1e-12)
    u_to = u_to / (np.linalg.norm(u_to) + 1e-12)
    v = np.cross(u_from, u_to)
    c = float(np.dot(u_from, u_to))
    s = np.linalg.norm(v)
    if s < 1e-8:                                  # parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        # 180 deg about any axis perpendicular to u_from
        perp = np.array([1.0, 0, 0])
        if abs(u_from[0]) > 0.9:
            perp = np.array([0, 1.0, 0])
        axis = np.cross(u_from, perp); axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def gravity_yaw_candidates(source, target, up_source, up_target, s, n_yaw=24):
    """Candidate similarities that fix scale (s) + up-alignment (gravity) and
    sweep the one remaining DOF: yaw about the target up-axis.

    Use when the ellipsoid's horizontal axes are degenerate (near-symmetric
    footprint), so E-Init's yaw is unreliable -- we brute-force it instead.
    Returns a list of dicts(R, t, T, yaw_deg), to be ICP-scored by the caller.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    b_s, b_t = src.mean(0), tgt.mean(0)
    R0 = _align_vectors(up_source, up_target)     # maps source-up onto target-up
    u = up_target / (np.linalg.norm(up_target) + 1e-12)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    out = []
    for theta in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False):
        Ryaw = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        R = Ryaw @ R0
        t = b_t - s * R @ b_s
        T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
        out.append(dict(R=R, t=t, T=T, yaw_deg=float(np.degrees(theta))))
    return out


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotation matrices."""
    Rrel = R_a.T @ R_b
    c = (np.trace(Rrel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
