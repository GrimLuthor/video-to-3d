"""Stage 1 step 1.4: chain per-pair relative poses into a single global
trajectory, fixing the scale-drift problem described in plan.md.

`cv2.recoverPose` only ever returns a unit-norm translation per pair, so
naively composing R,t across pairs produces a trajectory where every step
has an arbitrary, mutually-inconsistent "baseline" of 1. We recover a
consistent scale per pair using the classic 3-view trick: points already
triangulated (in world coordinates) from pair (i-1, i) that are also seen in
pair (i, i+1) tell us what frame i+1's baseline scale must be, via a
least-squares fit of the local (unit-baseline) triangulation onto the
already-known world points.

That same shared-point match doubles as multi-frame track-building: when a
point seen in pair (i, i+1) is recognized as the same physical point already
tracked from pair (i-1, i), we extend its existing track (add one more
observation) instead of minting a brand-new 3D point. Points a SIFT feature
survives across several consecutive pairs end up with 3+ observations
spanning real time gaps, not just 2 -- which is what gives Stage 2 bundle
adjustment (bundle_adjustment.py) actual leverage to pull separate
time-segments into a globally consistent reconstruction. Without this, every
point only ever has exactly 2 observations (its one triangulating pair), and
BA has nothing to reconcile distant frames with (see bundle_adjustment.py's
docstring and the pre/post-BA comparison in run_full_clip_chain.py).
"""

import cv2
import numpy as np

from pairwise_pose import estimate_pairwise_pose


def triangulate_local(K, R_rel, t_rel, pts_a, pts_b):
    """Triangulate matches into camera-A's own coordinate frame (P1 = K[I|0])."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R_rel, t_rel])
    pts4d = cv2.triangulatePoints(P1, P2, pts_a.T, pts_b.T)
    return (pts4d[:3] / pts4d[3]).T


def match_common_points(pts_prev_b, pts_curr_a, tol=0.5):
    """Match inlier points shared between two pairwise results at frame i:
    pts_prev_b are frame-i points from pair (i-1, i); pts_curr_a are frame-i
    points from pair (i, i+1). Both come from SIFT run on the identical
    frame-i image data, so shared keypoints land on (near-)identical pixel
    coordinates -- matched here by nearest-neighbor within `tol` px.

    Returns (idx_prev, idx_curr): parallel index arrays into the two inputs.
    """
    if len(pts_prev_b) == 0 or len(pts_curr_a) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx_prev, idx_curr = [], []
    used_prev = set()
    for i, p in enumerate(pts_curr_a):
        dists = np.linalg.norm(pts_prev_b - p, axis=1)
        j = int(np.argmin(dists))
        if dists[j] < tol and j not in used_prev:
            idx_prev.append(j)
            idx_curr.append(i)
            used_prev.add(j)
    return np.array(idx_prev, dtype=int), np.array(idx_curr, dtype=int)


def fit_scale(local_pts_unit, target_world_pts):
    """Least-squares scalar s minimizing ||s * local_pts_unit - target_world_pts||^2."""
    num = np.sum(local_pts_unit * target_world_pts)
    den = np.sum(local_pts_unit * local_pts_unit)
    if den < 1e-12:
        return None
    return float(num / den)


def chain_poses(frames, K, min_common_points=6):
    """Sequentially process consecutive frame pairs, returning:
      - cameras: list of dicts {R, t} mapping world -> camera-i (t already
        scaled consistently across the whole chain, up to one global,
        unrecoverable overall scale factor)
      - points_world: (M, 3) array, accumulated 3D points in world coords
      - diagnostics: list of per-pair dicts (inlier ratio, rotation_deg,
        n_matches, scale, n_common_points_used, n_new_points,
        n_continued_points) for the Stage 1 checkpoints
      - observations: list of (point_idx, cam_idx, x, y) -- which camera saw
        which point at which pixel, for Stage 2 bundle adjustment. Points
        whose track survived multiple pairs have 3+ observations here.
    """
    n = len(frames)
    if n < 3:
        raise ValueError("chain_poses needs at least 3 frames to fit scale; "
                          "use estimate_pairwise_pose directly for 2 frames")

    cameras = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    points_world = []
    diagnostics = []
    observations = []  # (point_idx, cam_idx, x, y), for bundle adjustment (Stage 2)

    prev_result = None
    prev_scale = 1.0  # first pair defines the arbitrary global scale unit
    # point_idx_for_b[k] = points_world index for the track that produced the
    # k-th inlier of the *previous* pair's pts_b_inliers (frame i, as "b" of
    # pair (i-1, i)) -- lets the next pair recognize the same physical point
    # as a track continuation rather than a fresh point.
    prev_point_idx_for_b = None

    for i in range(n - 1):
        result = estimate_pairwise_pose(frames[i], frames[i + 1], K)
        if result is None:
            diagnostics.append({"pair": (i, i + 1), "status": "failed_pose_estimation"})
            # Can't proceed past a broken link without re-anchoring; stop here.
            break

        R_i = cameras[i]["R"]
        t_i = cameras[i]["t"]

        n_inliers = len(result.pts_a_inliers)
        is_continued = np.zeros(n_inliers, dtype=bool)
        continued_point_idx = np.full(n_inliers, -1, dtype=int)

        if prev_result is None:
            # First pair: no scale reference yet, use the arbitrary unit baseline.
            scale = 1.0
            n_common = 0
        else:
            idx_prev, idx_curr = match_common_points(
                prev_result.pts_b_inliers, result.pts_a_inliers
            )
            n_common = len(idx_prev)

            is_continued[idx_curr] = True
            continued_point_idx[idx_curr] = prev_point_idx_for_b[idx_prev]

            if n_common >= min_common_points:
                local_unit_pts = triangulate_local(K, result.R, result.t,
                                                    result.pts_a_inliers,
                                                    result.pts_b_inliers)
                # target: known world points (the continued tracks' existing
                # positions), expressed in frame i's local camera frame
                # (X_cam_i = R_i @ X_world + t_i)
                target_local = (R_i @ np.array(points_world)[continued_point_idx[idx_curr]].T).T + t_i.ravel()
                fitted = fit_scale(local_unit_pts[idx_curr], target_local)
                scale = fitted if fitted is not None else prev_scale
            else:
                scale = prev_scale  # not enough overlap; carry forward last known scale

        t_rel_scaled = result.t * scale
        R_new = result.R @ R_i
        t_new = result.R @ t_i + t_rel_scaled
        cameras.append({"R": R_new, "t": t_new})

        # Only triangulate + add points for matches that AREN'T continuing an
        # existing track -- continued ones keep their original (first-pair)
        # triangulated position as the initial guess; bundle adjustment is
        # what reconciles it with the new observation, not a re-triangulation
        # here.
        new_mask = ~is_continued
        n_new = int(new_mask.sum())
        start_idx = len(points_world)
        if n_new > 0:
            # cv2.triangulatePoints errors on empty input, so this only runs
            # when there's at least one genuinely new point -- an all-
            # continuation pair (every inlier already tracked) is common
            # once tracks get long, especially with small inter-frame motion.
            local_pts_new = triangulate_local(K, result.R, result.t,
                                               result.pts_a_inliers[new_mask],
                                               result.pts_b_inliers[new_mask]) * scale
            world_pts_new = (R_i.T @ (local_pts_new - t_i.ravel()).T).T
            points_world.extend(world_pts_new.tolist())
        new_point_idx = np.arange(start_idx, len(points_world))

        # point_idx aligned to this pair's pts_b_inliers (frame i+1), for the
        # *next* iteration's continuation matching.
        point_idx_for_b = np.full(n_inliers, -1, dtype=int)
        point_idx_for_b[is_continued] = continued_point_idx[is_continued]
        point_idx_for_b[new_mask] = new_point_idx

        # Observations: every inlier gets a frame-(i+1) observation. Only
        # NEW points also get a frame-i observation -- continued points'
        # frame-i observation was already recorded as a frame-i "b"
        # observation in the previous iteration (same physical camera index).
        for k in range(n_inliers):
            xb, yb = result.pts_b_inliers[k]
            observations.append((int(point_idx_for_b[k]), i + 1, float(xb), float(yb)))
        for k in np.where(new_mask)[0]:
            xa, ya = result.pts_a_inliers[k]
            observations.append((int(point_idx_for_b[k]), i, float(xa), float(ya)))

        diagnostics.append({
            "pair": (i, i + 1),
            "status": "ok",
            "n_matches": result.n_matches,
            "match_inlier_ratio": result.match_inlier_ratio,
            "n_pose_inliers": result.n_pose_inliers,
            "rotation_deg": result.rotation_deg,
            "scale": scale,
            "n_common_points_used": n_common,
            "n_new_points": int(new_mask.sum()),
            "n_continued_points": int(is_continued.sum()),
        })

        prev_result = result
        prev_scale = scale
        prev_point_idx_for_b = point_idx_for_b

    return cameras, np.array(points_world), diagnostics, observations
