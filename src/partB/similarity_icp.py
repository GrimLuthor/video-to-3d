"""
Refinement + baseline for two-cloud registration (Part B).

- refine_similarity(): scaled ICP (Umeyama each iteration) that refines the
  E-Init similarity. Uses open3d's point-to-point estimation with_scaling=True,
  so it re-estimates (scale, rotation, translation) from actual correspondences
  -- this in particular corrects the scale bias E-Init inherits from noise
  inflating the covariance eigenvalues.

- fpfh_ransac_init(): the standard global-registration baseline (FPFH features +
  RANSAC). It is RIGID (no scale), so we feed it clouds already scale-normalized
  by the eigenvalue-ratio scale, giving a fair rotation/translation comparison
  against E-Init at a fraction of E-Init's simplicity/cost.
"""

from __future__ import annotations

import time
import numpy as np
import open3d as o3d


def _prep(pcd, voxel):
    down = pcd.voxel_down_sample(voxel) if voxel and voxel > 0 else pcd
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return down


def refine_similarity(source, target, T_init, voxel, max_corr_dist=None,
                      max_iter=100, refine_scale=True):
    """Two-stage ICP refinement of an initial similarity T_init (source->target).

    Stage 1 -- RIGID point-to-plane ICP at the fixed E-Init scale. We do NOT let
    scale float here: free-scale (Umeyama) ICP from a rough/low-overlap start
    collapses (shrinks the source into a blob, fitness -> 1 but geometry gone).
    Stage 1 gets rotation + translation robustly.

    Stage 2 (refine_scale, only when stage 1 is already well-aligned) -- a
    with_scaling pass from the stage-1 pose with a TIGHT correspondence distance,
    to correct the residual scale (the eigenvalue-ratio scale is a moment estimate;
    this finds the correspondence-optimal uniform scale, removing the "matches in
    the center, diverges at the edges" error). Guarded: accepted only if it stays
    well-aligned and the correction is modest -- so a low-overlap case (meonot)
    that never reaches stage 2, or a collapse, is rejected and stage 1 is kept.

    Returns dict(T, fitness, inlier_rmse, scale, seconds).
    """
    src0 = o3d.geometry.PointCloud(source).transform(T_init)  # into target frame
    src = _prep(src0, voxel)
    tgt = _prep(target, voxel)
    if max_corr_dist is None:
        max_corr_dist = voxel * 3.0
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)

    t0 = time.time()
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), crit)
    T_stage = np.asarray(reg.transformation)
    fitness, rmse = float(reg.fitness), float(reg.inlier_rmse)

    if refine_scale and fitness > 0.3:
        # Use a GENEROUS correspondence distance here: a residual scale error
        # pushes the edge points apart, and they must stay corresponded for
        # Umeyama to see (and correct) the scale. A tight distance would exclude
        # exactly those edges and find no correction. High overlap keeps it safe.
        crit2 = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        reg2 = o3d.pipelines.registration.registration_icp(
            src, tgt, max_corr_dist * 2.0, reg.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=True), crit2)
        s_corr = float(np.cbrt(np.linalg.det(np.asarray(reg2.transformation)[:3, :3])))
        # accept only a modest correction that doesn't lose alignment (else collapse)
        if 0.7 < s_corr < 1.4 and reg2.fitness >= 0.9 * fitness:
            T_stage = np.asarray(reg2.transformation)
            fitness, rmse = float(reg2.fitness), float(reg2.inlier_rmse)

    dt = time.time() - t0
    T = T_stage @ np.asarray(T_init)
    scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
    return dict(T=T, fitness=fitness, inlier_rmse=rmse, scale=scale, seconds=dt)


def fpfh_ransac_init(source, target, voxel):
    """Baseline global registration (FPFH + RANSAC), RIGID.

    Assumes the two clouds are already at the same scale. Returns
    dict(T, fitness, inlier_rmse, seconds).
    """
    def feats(pcd):
        down = _prep(pcd, voxel)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 5.0, max_nn=100))
        return down, fpfh

    src, src_f = feats(source)
    tgt, tgt_f = feats(target)
    dist = voxel * 1.5

    t0 = time.time()
    reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))
    dt = time.time() - t0
    return dict(T=np.asarray(reg.transformation), fitness=float(reg.fitness),
                inlier_rmse=float(reg.inlier_rmse), seconds=dt)
