"""
Piecewise-similarity refinement with smooth blending (Part B, optional stage).

After the global E-Init + ICP similarity alignment, two independent monocular
reconstructions can still differ by RESIDUAL SCALE DRIFT: the correct local scale
varies smoothly along the capture path (measured on columns: ~0.87 at one end ->
~1.00 at the other). A single global similarity cannot absorb a gradient. This
stage corrects it the way the eye wants to: split the (already globally-aligned)
source along its principal axis, fit a LOCAL similarity per segment, and blend the
per-segment transforms into a SMOOTH per-point deformation (no seams).

It is deliberately low-DOF and heavily smoothed (a handful of segments, linearly
blended) -- it corrects a slow physical drift, it does NOT free-form warp. It only
fixes the MISALIGNMENT component; genuinely missing coverage stays missing.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def _decompose(T):
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    R = T[:3, :3] / s
    # re-orthonormalize (numerical safety) before quaternion
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    return s, R, T[:3, 3].copy()


def piecewise_refine(src_xyz, tgt_pcd, axis, n_seg=6, overlap=0.5,
                     max_corr_dist=1.0, min_pts=2000):
    """Return smoothly-deformed source points aligning them to tgt per-region.

    src_xyz : (N,3) source points ALREADY in the target frame (post global ICP).
    axis    : 0/1/2, the principal (path) axis to segment along.
    n_seg   : number of segments; overlap : window padding as a fraction of a
              segment width (for stable local fits); min_pts : skip tiny segments.
    """
    S = np.asarray(src_xyz, dtype=np.float64)
    tgt = o3d.geometry.PointCloud(tgt_pcd)
    x = S[:, axis]
    lo, hi = np.percentile(x, [1, 99])
    edges = np.linspace(lo, hi, n_seg + 1)
    w = (edges[1] - edges[0])
    centers, transforms = [], []

    for i in range(n_seg):
        c = 0.5 * (edges[i] + edges[i + 1])
        m = (x >= edges[i] - overlap * w) & (x < edges[i + 1] + overlap * w)
        centers.append(c)
        if m.sum() < min_pts:
            transforms.append(np.eye(4)); continue
        seg = o3d.geometry.PointCloud()
        seg.points = o3d.utility.Vector3dVector(S[m])
        reg = o3d.pipelines.registration.registration_icp(
            seg, tgt, max_corr_dist, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = np.asarray(reg.transformation)
        s = float(np.cbrt(np.linalg.det(T[:3, :3])))
        # guard each local fit: reject an implausible correction (keep identity)
        transforms.append(T if (0.7 < s < 1.4 and reg.fitness > 0.3) else np.eye(4))

    centers = np.array(centers)
    scales = np.array([_decompose(T)[0] for T in transforms])
    quats = Rotation.from_matrix([_decompose(T)[1] for T in transforms])
    trans = np.array([_decompose(T)[2] for T in transforms])
    slerp = Slerp(centers, quats)

    # per-point: interpolate (scale, rotation, translation) from bracketing centers
    xc = np.clip(x, centers[0], centers[-1])
    idx = np.clip(np.searchsorted(centers, xc) - 1, 0, len(centers) - 2)
    c0, c1 = centers[idx], centers[idx + 1]
    a = (xc - c0) / (c1 - c0 + 1e-12)                      # blend weight in [0,1]
    s_i = np.exp((1 - a) * np.log(scales[idx]) + a * np.log(scales[idx + 1]))
    R_i = slerp(xc).as_matrix()
    t_i = (1 - a)[:, None] * trans[idx] + a[:, None] * trans[idx + 1]
    out = (s_i[:, None] * np.einsum('nij,nj->ni', R_i, S)) + t_i
    return out, dict(centers=centers.tolist(), scales=scales.tolist())
