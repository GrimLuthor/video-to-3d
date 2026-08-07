"""Stage 2: global bundle adjustment over the Stage 1 chained trajectory.

chain_poses() (Stage 1, step 1.4) only ever reconciles each new pair's pose
against its immediate predecessor via a local scale fit -- small errors
accumulate along the chain, visible as the by-time clustering/banding in
the point-cloud plots (each frame pair's points sit in their own little
island instead of blending continuously with their neighbors).

Bundle adjustment fixes this the classic SfM way: refine every camera pose
and every 3D point *jointly*, minimizing total reprojection error summed
over every (point, camera) observation at once, instead of one link at a
time. This is a nonlinear least-squares problem (Levenberg-Marquardt /
trust-region), solved here with scipy.optimize.least_squares using a
sparse Jacobian -- with tens of thousands of points, a dense Jacobian
would be far too large (rows = 2 * n_observations, cols = 6 * n_cameras +
3 * n_points), so we tell scipy exactly which entries can possibly be
nonzero (each observation only depends on its own camera's 6 params and
its own point's 3 params) and it exploits that for both the finite-
difference Jacobian estimate and the linear solve.

Camera 0 is held fixed (world origin/identity) to anchor the
reconstruction -- everything else (camera 1..N poses, all point positions)
is free to move. This still leaves the reconstruction's overall *scale* as
an unconstrained direction (monocular SfM's classic gauge freedom: scaling
every camera translation and every point position by the same factor
doesn't change any projected pixel), but since the initial guess (from
chain_poses' local scale-propagation) is already close to a globally
consistent scale, the optimizer has no gradient pushing it to drift along
that direction in practice.

Each point here only has 2 observations (it was triangulated from exactly
one consecutive frame pair, see chain_poses.py) rather than a proper
multi-frame track. That's a real limitation -- more observations per point
would constrain the optimization better -- but it's still a genuine global
joint refinement instead of the current link-by-link chaining, and is the
natural place to extend from later if needed.

Reprojection error alone cannot see monocular scale drift: a slowly,
smoothly growing scale along a chain-like camera sequence (each camera only
linked to nearby-in-time neighbors, never back to distant ones) barely
moves any pixel's reprojection, so it's an almost-flat direction of this
cost function -- this is a textbook limitation of monocular SfM/VO, not a
bug (see run_full_clip_chain.py's before/after plots on longer_walk.mp4,
where BA left a ~12x scale runaway completely untouched). Since this
footage is a one-way walk with no loop closure available (no revisited
viewpoint to re-anchor scale against), the practical mitigation here is a
soft "roughly constant walking pace" prior: an extra residual per
consecutive triplet of cameras penalizing the relative change in step
length between camera_i -> camera_i+1 and camera_i+1 -> camera_i+2. This
doesn't recover true metric scale (nothing here can, without an external
reference or a real loop closure) but directly opposes the compounding
multiplicative drift.
"""

import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

N_CAM_PARAMS = 6  # 3 rotation (Rodrigues) + 3 translation


def _poses_to_vector(cameras):
    params = np.zeros((len(cameras), N_CAM_PARAMS))
    for i, cam in enumerate(cameras):
        rvec, _ = cv2.Rodrigues(cam["R"])
        params[i, :3] = rvec.ravel()
        params[i, 3:] = cam["t"].ravel()
    return params.ravel()


def _vector_to_poses(vec, n_cameras):
    params = vec.reshape(n_cameras, N_CAM_PARAMS)
    cameras = []
    for i in range(n_cameras):
        R, _ = cv2.Rodrigues(params[i, :3])
        cameras.append({"R": R, "t": params[i, 3:].reshape(3, 1)})
    return cameras


def _project(K, R, t, points):
    """Project Nx3 world points into a camera with pose R,t (world->cam)."""
    cam_pts = (R @ points.T).T + t.ravel()
    proj = (K @ cam_pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def _unpack(free_params, n_free_cameras, n_points, fixed_cam0):
    cam_vec = np.concatenate([_poses_to_vector([fixed_cam0]), free_params[:n_free_cameras * N_CAM_PARAMS]])
    cam_params = cam_vec.reshape(n_free_cameras + 1, N_CAM_PARAMS)
    points = free_params[n_free_cameras * N_CAM_PARAMS:].reshape(n_points, 3)
    return cam_params, points


def _reprojection_residuals(cam_params, points, cam_indices, point_indices, obs_2d, K):
    res = np.zeros((len(obs_2d), 2))
    for cam_idx in np.unique(cam_indices):
        mask = cam_indices == cam_idx
        rvec = cam_params[cam_idx, :3]
        tvec = cam_params[cam_idx, 3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        proj = _project(K, R, tvec, points[point_indices[mask]])
        res[mask] = proj - obs_2d[mask]
    return res.ravel()


def _camera_centers(cam_params):
    """cam_params: (n_cameras, 6) -> (n_cameras, 3) world positions
    (C = -R^T t) for every camera, in trajectory order."""
    n = len(cam_params)
    centers = np.zeros((n, 3))
    for i in range(n):
        R, _ = cv2.Rodrigues(cam_params[i, :3])
        centers[i] = -R.T @ cam_params[i, 3:]
    return centers


def _smoothness_residuals(cam_params, weight):
    """One residual per consecutive camera triplet, penalizing the relative
    change in step length assuming roughly constant walking pace (see module
    docstring). Dimensionless ratio scaled by `weight` (pixel-equivalent
    units) so it can be balanced against reprojection residuals.
    """
    if weight == 0 or len(cam_params) < 3:
        return np.zeros(0)
    centers = _camera_centers(cam_params)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    eps = 1e-9
    ratios = steps[1:] / np.maximum(steps[:-1], eps) - 1.0
    return weight * ratios


def _residuals(free_params, n_free_cameras, n_points, cam_indices, point_indices,
                obs_2d, K, fixed_cam0, smoothness_weight=0.0):
    cam_params, points = _unpack(free_params, n_free_cameras, n_points, fixed_cam0)
    reproj = _reprojection_residuals(cam_params, points, cam_indices, point_indices, obs_2d, K)
    smooth = _smoothness_residuals(cam_params, smoothness_weight)
    return np.concatenate([reproj, smooth])


def _build_sparsity(n_free_cameras, n_points, cam_indices, point_indices, n_smoothness=0):
    """Vectorized (row, col) pair construction -- with tens of thousands of
    observations, building this via per-element sparse-matrix assignment
    (e.g. lil_matrix[i, j] = 1 in a loop) is far too slow; coo_matrix from
    flat row/col arrays is effectively instant.
    """
    n_obs = len(cam_indices)
    m = n_obs * 2 + n_smoothness
    n = n_free_cameras * N_CAM_PARAMS + n_points * 3

    rows = [np.empty(0, dtype=int)]
    cols = [np.empty(0, dtype=int)]
    row_pair = np.arange(n_obs)  # residual row pair index -> rows 2*i, 2*i+1

    # camera 0 is fixed (excluded from free_params), so its observations
    # contribute rows with no camera-column entries -- only point columns.
    free_mask = cam_indices > 0
    free_cam_col0 = (cam_indices[free_mask] - 1) * N_CAM_PARAMS
    rows_free = row_pair[free_mask]
    for s in range(N_CAM_PARAMS):
        rows.append(2 * rows_free)
        cols.append(free_cam_col0 + s)
        rows.append(2 * rows_free + 1)
        cols.append(free_cam_col0 + s)

    point_col0 = n_free_cameras * N_CAM_PARAMS + point_indices * 3
    for s in range(3):
        rows.append(2 * row_pair)
        cols.append(point_col0 + s)
        rows.append(2 * row_pair + 1)
        cols.append(point_col0 + s)

    # Smoothness residual k (0-indexed) depends on cameras k, k+1, k+2 (all
    # cameras, 0-indexed including the fixed camera0). Mark all 6 params of
    # whichever of those are free -- over-marking a few entries that turn
    # out not to matter is harmless for jac_sparsity, just slightly less
    # efficient, and this term is tiny (~n_cameras rows) next to the
    # reprojection term anyway.
    if n_smoothness > 0:
        smooth_row0 = n_obs * 2
        for k in range(n_smoothness):
            for cam_idx in (k, k + 1, k + 2):
                if cam_idx == 0:
                    continue  # fixed camera0, no free columns
                col0 = (cam_idx - 1) * N_CAM_PARAMS
                rows.append(np.full(N_CAM_PARAMS, smooth_row0 + k))
                cols.append(col0 + np.arange(N_CAM_PARAMS))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=int)
    return coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()


def bundle_adjust(cameras, points_world, observations, K, smoothness_weight=2.0,
                  max_nfev=300, loss="linear", f_scale=3.0, verbose=True):
    """Refine cameras (list of {R,t}) and points_world (N,3) jointly by
    minimizing total reprojection error over `observations`
    (point_idx, cam_idx, x, y), plus a soft constant-walking-pace prior
    (see module docstring) controlled by `smoothness_weight` (0 disables it,
    recovering plain bundle adjustment). Camera 0 stays fixed as the world
    anchor.

    `loss`/`f_scale`: passed to least_squares. Default "linear" (plain L2) is
    right for the old chain path, whose observations are already tightly
    RANSAC/cheirality-filtered. The incremental-SfM path (incremental_sfm.py)
    instead builds many wide-baseline multi-view tracks that still carry a fat
    tail of mismatches, so it calls this with loss="huber" -- there the earlier
    "huber converged worse" finding does NOT apply (that was on synthetic
    noise-free data where every residual is an inlier and huber only starved
    gradient; with real outliers huber is the correct choice). `f_scale` is the
    soft inlier/outlier threshold in pixels.

    Returns (cameras_refined, points_world_refined, info) where info has
    the pre/post reprojection RMSE and step-length relative-change stats.
    """
    n_cameras = len(cameras)
    n_points = len(points_world)
    n_free_cameras = n_cameras - 1
    n_smoothness = max(0, n_cameras - 2) if smoothness_weight else 0

    obs = np.asarray(observations, dtype=float)
    point_indices = obs[:, 0].astype(int)
    cam_indices = obs[:, 1].astype(int)
    obs_2d = obs[:, 2:4]
    n_reproj_res = len(obs) * 2

    free_cam_vec = _poses_to_vector(cameras[1:])
    x0 = np.concatenate([free_cam_vec, points_world.ravel()])

    def fun(params):
        return _residuals(params, n_free_cameras, n_points, cam_indices,
                           point_indices, obs_2d, K, cameras[0], smoothness_weight)

    def reproj_rmse(residual_vec):
        return np.sqrt(np.mean(residual_vec[:n_reproj_res] ** 2))

    def step_ratio_stats(residual_vec):
        if n_smoothness == 0:
            return None
        return np.abs(residual_vec[n_reproj_res:] / smoothness_weight).mean()

    res0 = fun(x0)
    rmse0 = reproj_rmse(res0)
    step_dev0 = step_ratio_stats(res0)
    if verbose:
        msg = (f"  BA initial reprojection RMSE: {rmse0:.4f} px "
               f"({len(observations)} observations, {n_points} points, "
               f"{n_free_cameras} free cameras)")
        if step_dev0 is not None:
            msg += f", mean |step ratio - 1| = {step_dev0:.3f}"
        print(msg)

    sparsity = _build_sparsity(n_free_cameras, n_points, cam_indices, point_indices, n_smoothness)

    # Plain L2 (loss="linear"), not a robust loss: observations here are
    # already RANSAC- and cheirality-filtered inliers from chain_poses (see
    # pairwise_pose.py), so gross outliers are largely pre-removed, and a
    # robust loss (tried huber) actually converged *worse* on a synthetic
    # test -- with typical initial reprojection errors well above a
    # plausible outlier threshold, huber down-weights the vast majority of
    # residuals from the start, starving the optimizer of gradient and
    # causing premature xtol/ftol termination far from the true optimum.
    # Verified on synthetic noise-free data that linear loss converges to
    # ~0 RMSE in under 10 iterations.
    #
    # x_scale: "jac" (auto column-norm scaling) is what makes the pure
    # reprojection case converge fast -- but with the smoothness prior
    # added, it's actively harmful: verified on a synthetic drifting-camera
    # case that "jac" scaling makes the smoothness residual's contribution
    # to the rescaled trust-region problem negligible regardless of its
    # weight (its gradient is tiny compared to reprojection's in pixel
    # units), so the solver reports zero possible improvement and quits via
    # xtol on iteration 1 without moving at all. Plain x_scale=1.0 fixes
    # this (confirmed the same synthetic case then reduces step-ratio
    # deviation by >95% for a negligible reprojection cost).
    t0 = time.time()
    result = least_squares(
        fun, x0, jac_sparsity=sparsity, method="trf",
        x_scale=1.0 if smoothness_weight else "jac", tr_solver="lsmr",
        loss=loss, f_scale=f_scale,
        max_nfev=max_nfev, verbose=2 if verbose else 0,
    )
    if verbose:
        print(f"  BA solve took {time.time() - t0:.1f}s")

    rmse1 = reproj_rmse(result.fun)
    step_dev1 = step_ratio_stats(result.fun)
    # per-observation pixel error (magnitude of each 2-vector residual); its
    # median is a robust quality measure that ignores the outlier tail the raw
    # RMSE is dominated by -- the meaningful number under a huber loss.
    px_err = np.linalg.norm(result.fun[:n_reproj_res].reshape(-1, 2), axis=1)
    median_px = float(np.median(px_err))
    if verbose:
        msg = (f"  BA final reprojection RMSE: {rmse1:.4f} px "
               f"(median {median_px:.3f} px)")
        if step_dev1 is not None:
            msg += f", mean |step ratio - 1| = {step_dev1:.3f}"
        print(msg)

    free_cams = _vector_to_poses(result.x[:n_free_cameras * N_CAM_PARAMS], n_free_cameras)
    cameras_refined = [cameras[0]] + free_cams
    points_refined = result.x[n_free_cameras * N_CAM_PARAMS:].reshape(n_points, 3)

    return cameras_refined, points_refined, {
        "rmse_before": rmse0, "rmse_after": rmse1, "median_px_after": median_px,
        "step_ratio_dev_before": step_dev0, "step_ratio_dev_after": step_dev1,
    }
