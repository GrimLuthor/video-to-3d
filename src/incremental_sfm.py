"""Incremental structure-from-motion over multi-frame tracks.

This replaces chain_poses.py's sequential two-view chaining (which is really
visual odometry, and drifts in scale on a long one-way walk) with the way SfM
is actually done classically (Bundler/VisualSFM/COLMAP, no CNNs):

  1. Seed from a single well-conditioned pair (wide enough baseline for stable
     triangulation), setting the one unrecoverable global scale.
  2. Repeatedly register the next best un-registered frame by PnP against the
     already-triangulated 3D points it observes -- pose is solved *against the
     existing fixed-scale map*, so there's no per-pair unit-baseline scale to
     propagate and nothing for scale to drift along.
  3. Triangulate each track from *all* the registered views that see it (a
     multi-view DLT over a wide baseline), not from a single adjacent pair.
  4. Bundle-adjust periodically during the build and once globally at the end.

Because tracks tie temporally-distant frames together with wide baselines, the
bundle-adjustment graph is richly connected instead of a chain, so relative
scale between distant cameras is now an observable, well-constrained direction
of the reprojection cost -- which is what removes the scale drift without any
loop closure or constant-pace prior.

Reuses pairwise_pose (seed pose), feature_tracks (matching/tracks) and
bundle_adjustment.bundle_adjust (the solver, with smoothness_weight=0).
"""

import time

import cv2
import numpy as np

from bundle_adjustment import bundle_adjust


def _P(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def triangulate_multiview(proj_mats, pts2d):
    """Linear (DLT) triangulation of one 3D point from N>=2 views.
    proj_mats: list of 3x4 P; pts2d: list of (x,y). Returns (3,) world point."""
    A = []
    for P, (x, y) in zip(proj_mats, pts2d):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _center(R, t):
    return (-R.T @ t.reshape(3)).ravel()


def _max_tri_angle(centers, X):
    """Max pairwise angle (deg) between rays from the given camera centers to
    point X. Small angle => poorly conditioned depth (nearly parallel rays)."""
    rays = X[None, :] - np.asarray(centers)
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms < 1e-9):
        return 0.0
    rays = rays / norms[:, None]
    best = 0.0
    for a in range(len(rays)):
        for b in range(a + 1, len(rays)):
            c = np.clip(rays[a] @ rays[b], -1, 1)
            best = max(best, np.degrees(np.arccos(c)))
    return best


def _reproj_ok(poses, pts2d, K, X, max_err):
    for (R, t), (x, y) in zip(poses, pts2d):
        cam = R @ X + t.reshape(3)
        if cam[2] <= 0:  # cheirality
            return False
        proj = K @ cam
        proj = proj[:2] / proj[2]
        if np.hypot(proj[0] - x, proj[1] - y) > max_err:
            return False
    return True


def _try_triangulate(track, poses_by_frame, K, min_angle, max_err):
    """Triangulate one track from all its currently-registered observations.
    Returns the 3D point or None if too few views / ill-conditioned / bad
    reprojection."""
    frames = [f for f in track if f in poses_by_frame]
    if len(frames) < 2:
        return None
    poses = [poses_by_frame[f] for f in frames]
    pts2d = [track[f] for f in frames]
    proj = [_P(K, R, t) for (R, t) in poses]
    X = triangulate_multiview(proj, pts2d)
    centers = [_center(R, t) for (R, t) in poses]
    if _max_tri_angle(centers, X) < min_angle:
        return None
    if not _reproj_ok(poses, pts2d, K, X, max_err):
        return None
    return X


def _choose_seed(tracks, features, K, candidate_pairs, min_angle=2.0, verbose=True):
    """Pick the seed pair maximizing well-triangulated correspondences.

    For each candidate (i,j), recover its two-view pose (unit baseline) and
    triangulate the tracks seen in both; score = number of points passing the
    triangulation-angle + cheirality test. A wide-baseline, high-parallax pair
    wins, which is what makes the initial map (and thus the global scale) stable.
    """
    frames_of_track = [set(t) for t in tracks]
    best = None
    for (i, j) in candidate_pairs:
        common = [ti for ti, fs in enumerate(frames_of_track) if i in fs and j in fs]
        if len(common) < 15:
            continue
        pts_i = np.array([tracks[ti][i] for ti in common], np.float64)
        pts_j = np.array([tracks[ti][j] for ti in common], np.float64)
        E, mask = cv2.findEssentialMat(pts_i, pts_j, K,
                                       method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
                                       prob=0.999, threshold=1.0)
        if E is None:
            continue
        m = mask.ravel().astype(bool)
        if m.sum() < 15:
            continue
        _, R, t, _ = cv2.recoverPose(E, pts_i[m], pts_j[m], K)
        Pi = _P(K, np.eye(3), np.zeros(3))
        Pj = _P(K, R, t)
        Ci, Cj = np.zeros(3), _center(R, t)
        score = 0
        for xi, xj in zip(pts_i[m], pts_j[m]):
            X = triangulate_multiview([Pi, Pj], [xi, xj])
            if X[2] <= 0:
                continue
            camj = R @ X + t.ravel()
            if camj[2] <= 0:
                continue
            if _max_tri_angle([Ci, Cj], X) >= min_angle:
                score += 1
        if best is None or score > best[0]:
            best = (score, i, j, R, t)
    if best is None:
        raise RuntimeError("no usable seed pair found")
    if verbose:
        print(f"  seed pair ({best[1]},{best[2]}): {best[0]} well-conditioned points")
    return best[1], best[2], best[3], best[4]


def _assemble_for_ba(poses_by_frame, track_point, tracks):
    """Pack current state into bundle_adjust's (cameras, points, observations)
    with contiguous, temporally-ordered camera indices (index 0 = earliest
    registered frame, held fixed as the gauge anchor)."""
    frames_sorted = sorted(poses_by_frame)
    frame_to_cam = {f: c for c, f in enumerate(frames_sorted)}
    cameras = [{"R": poses_by_frame[f][0], "t": poses_by_frame[f][1].reshape(3, 1)}
               for f in frames_sorted]

    tids = sorted(track_point)
    track_to_pt = {tid: p for p, tid in enumerate(tids)}
    points = np.array([track_point[tid] for tid in tids], float)

    observations = []
    for tid in tids:
        p = track_to_pt[tid]
        for f, xy in tracks[tid].items():
            if f in frame_to_cam:
                observations.append((p, frame_to_cam[f], float(xy[0]), float(xy[1])))
    return cameras, points, observations, frames_sorted, tids


def _run_ba(poses_by_frame, track_point, tracks, K, max_nfev, loss="linear",
            f_scale=3.0, verbose=False):
    cams, points, obs, frames_sorted, tids = _assemble_for_ba(poses_by_frame, track_point, tracks)
    if len(obs) < 10 or len(cams) < 2:
        return
    cams_ba, points_ba, info = bundle_adjust(
        cams, points, obs, K, smoothness_weight=0.0, max_nfev=max_nfev,
        loss=loss, f_scale=f_scale, verbose=verbose)
    for f, cam in zip(frames_sorted, cams_ba):
        poses_by_frame[f] = (cam["R"], cam["t"].reshape(3, 1))
    for tid, p in zip(tids, points_ba):
        track_point[tid] = p
    return info


def _median_per_frame_step(poses_by_frame):
    """Median camera-center displacement *per frame* between temporally
    adjacent registered cameras -- a robust estimate of this reconstruction's
    own motion scale, used to bound where the next camera may plausibly land."""
    frames = sorted(poses_by_frame)
    if len(frames) < 5:
        return None
    steps = []
    for a, b in zip(frames[:-1], frames[1:]):
        gap = b - a
        if gap > 0:
            d = np.linalg.norm(_center(*poses_by_frame[b]) - _center(*poses_by_frame[a]))
            steps.append(d / gap)
    if len(steps) < 4:
        return None
    return float(np.median(steps))


def _pose_jump_ok(frame, R, tvec, poses_by_frame, guard_factor):
    """Reject a PnP pose whose camera center is implausibly far from the
    nearest already-registered frame, given the reconstruction's own median
    per-frame motion. This is the guard against a glass/reflection mismatch
    placing a camera (and everything registered after it against the corrupted
    structure) wildly wrong -- the discrete blow-up that stretches the cloud.
    Self-scaling (relative to this video's median step), so it is inert on a
    clean walk where no camera comes near the budget. Returns (ok, info)."""
    med = _median_per_frame_step(poses_by_frame)
    if med is None or med <= 0:
        return True, None  # too early / degenerate: no reliable scale yet
    C_new = _center(R, tvec)
    fn = min(poses_by_frame, key=lambda f: abs(f - frame))  # nearest in time
    gap = max(1, abs(frame - fn))
    dist = float(np.linalg.norm(C_new - _center(*poses_by_frame[fn])))
    budget = guard_factor * gap * med
    return dist <= budget, (dist, budget, med, gap, fn)


def _try_register(frame, tracks_of_frame, track_point, tracks, poses_by_frame, K,
                  max_reproj, min_corr, guard_factor=12.0, verbose=False):
    """PnP-register one frame against the current 3D map. Returns True on
    success (pose written into poses_by_frame). Solving pose against the
    existing fixed-scale points is what keeps scale consistent -- there is no
    per-pair baseline to drift.

    A pose that PnP accepts but that lands the camera implausibly far from its
    temporal neighbour (relative to the median per-frame motion) is rejected by
    the jump guard rather than allowed to corrupt every camera solved after it."""
    corr = [tid for tid in tracks_of_frame[frame] if tid in track_point]
    if len(corr) < min_corr:
        return False
    obj = np.array([track_point[tid] for tid in corr], np.float64)
    img = np.array([tracks[tid][frame] for tid in corr], np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, None, reprojectionError=max_reproj,
        iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < min_corr:
        return False
    R, _ = cv2.Rodrigues(rvec)
    if guard_factor > 0:
        ok_jump, jinfo = _pose_jump_ok(frame, R, tvec, poses_by_frame, guard_factor)
        if not ok_jump:
            if verbose and jinfo:
                d, b, m, g, fn = jinfo
                print(f"    [guard] reject frame {frame}: center jump {d:.2f} > "
                      f"budget {b:.2f} ({guard_factor:g}x * gap {g} * med step "
                      f"{m:.3f}); nearest registered frame {fn}")
            return False
    poses_by_frame[frame] = (R, tvec.reshape(3, 1))
    return True


def _triangulate_frame_tracks(frame, tracks_of_frame, track_point, tracks,
                              poses_by_frame, K, min_tri_angle, max_reproj):
    """Triangulate every not-yet-placed track this newly-registered frame
    observes, from all its registered views (wide-baseline multi-view DLT)."""
    for tid in tracks_of_frame[frame]:
        if tid in track_point:
            continue
        X = _try_triangulate(tracks[tid], poses_by_frame, K, min_tri_angle, max_reproj)
        if X is not None:
            track_point[tid] = X


def _filter_outlier_points(poses_by_frame, track_point, tracks, K, max_median_err):
    """Drop points whose median reprojection error over their registered views
    exceeds `max_median_err` -- the surviving-mismatch tail that inflates RMSE
    and litters the cloud with stray 3D points. Returns the number removed."""
    bad = []
    for tid, X in track_point.items():
        errs = []
        for f, (x, y) in ((f, tracks[tid][f]) for f in tracks[tid] if f in poses_by_frame):
            R, t = poses_by_frame[f]
            cam = R @ X + t.reshape(3)
            if cam[2] <= 0:
                errs.append(1e9)
                continue
            proj = K @ cam
            proj = proj[:2] / proj[2]
            errs.append(np.hypot(proj[0] - x, proj[1] - y))
        if len(errs) < 2 or np.median(errs) > max_median_err:
            bad.append(tid)
    for tid in bad:
        del track_point[tid]
    return len(bad)


def reconstruct(features, tracks, K, window=8, min_tri_angle=2.0, max_reproj=4.0,
                min_pnp_corr=12, ba_every=15, pose_jump_guard=12.0, verbose=True):
    """Incremental SfM. Returns (cameras, points_world, observations,
    point_frame) where cameras is a list of {R,t} in temporal (frame) order,
    observations is (point_idx, cam_idx, x, y) into that camera ordering, and
    point_frame[k] is the median frame index of point k's track (a "time along
    the clip" label for plotting / depth-bucket drift diagnostics)."""
    n_frames = max(max(t) for t in tracks) + 1

    # candidate seed pairs: track co-visibility with a real baseline (gap>=2),
    # capped in gap by the matching window (beyond it, few shared tracks).
    covis = {}
    for t in tracks:
        fs = sorted(t)
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                if 2 <= fs[b] - fs[a] <= window:
                    covis[(fs[a], fs[b])] = covis.get((fs[a], fs[b]), 0) + 1
    candidate_pairs = [p for p, c in sorted(covis.items(), key=lambda kv: -kv[1])[:40]]

    i0, j0, R0, t0 = _choose_seed(tracks, features, K, candidate_pairs,
                                  min_angle=min_tri_angle, verbose=verbose)
    poses_by_frame = {i0: (np.eye(3), np.zeros((3, 1))), j0: (R0, t0.reshape(3, 1))}
    track_point = {}

    # seed points
    for tid, t in enumerate(tracks):
        X = _try_triangulate(t, poses_by_frame, K, min_tri_angle, max_reproj)
        if X is not None:
            track_point[tid] = X
    if verbose:
        print(f"  seed: {len(poses_by_frame)} cams, {len(track_point)} points")

    tracks_of_frame = [[] for _ in range(n_frames)]
    for tid, t in enumerate(tracks):
        for f in t:
            tracks_of_frame[f].append(tid)

    failed = set()  # frames PnP couldn't place yet (retried later, relaxed)
    since_ba = 0
    while True:
        # next-best view: unregistered, not-yet-failed frame with the most
        # 2D-3D correspondences to the current map.
        best_f, best_n = None, min_pnp_corr - 1
        for f in range(n_frames):
            if f in poses_by_frame or f in failed:
                continue
            n_corr = sum(1 for tid in tracks_of_frame[f] if tid in track_point)
            if n_corr > best_n:
                best_f, best_n = f, n_corr
        if best_f is None:
            break

        if not _try_register(best_f, tracks_of_frame, track_point, tracks,
                             poses_by_frame, K, max_reproj, min_pnp_corr,
                             guard_factor=pose_jump_guard, verbose=verbose):
            failed.add(best_f)  # revisit in the relaxed recovery pass below
            continue
        _triangulate_frame_tracks(best_f, tracks_of_frame, track_point, tracks,
                                  poses_by_frame, K, min_tri_angle, max_reproj)

        since_ba += 1
        if since_ba >= ba_every:
            info = _run_ba(poses_by_frame, track_point, tracks, K, max_nfev=60, verbose=False)
            if verbose and info:
                print(f"  [{len(poses_by_frame)}/{n_frames} cams] intermediate BA "
                      f"RMSE {info['rmse_before']:.2f}->{info['rmse_after']:.2f}px, "
                      f"{len(track_point)} points")
            since_ba = 0

    # Recovery pass: retry every still-unregistered frame with relaxed PnP now
    # that the map is much larger (a frame that lacked correspondences early on
    # often has plenty once its neighbours are in). Repeat until it stops
    # helping -- each recovered frame can enable the next.
    while True:
        recovered = 0
        for f in range(n_frames):
            if f in poses_by_frame:
                continue
            if _try_register(f, tracks_of_frame, track_point, tracks, poses_by_frame,
                             K, max_reproj * 2, max(6, min_pnp_corr // 2),
                             guard_factor=pose_jump_guard, verbose=verbose):
                _triangulate_frame_tracks(f, tracks_of_frame, track_point, tracks,
                                          poses_by_frame, K, min_tri_angle, max_reproj)
                recovered += 1
        if verbose and recovered:
            print(f"  recovery pass: +{recovered} frames "
                  f"({len(poses_by_frame)}/{n_frames} registered)")
        if recovered == 0:
            break
        _run_ba(poses_by_frame, track_point, tracks, K, max_nfev=60, verbose=False)

    if verbose:
        print(f"  registered {len(poses_by_frame)}/{n_frames} frames, "
              f"{len(track_point)} points; final robust BA + outlier filtering...")
    t_ba = time.time()
    # Robust (huber) final BA down-weights the mismatch tail, then drop points
    # whose median reprojection error is still high, then one more clean BA.
    _run_ba(poses_by_frame, track_point, tracks, K, max_nfev=200,
            loss="huber", f_scale=2.0, verbose=verbose)
    n_removed = _filter_outlier_points(poses_by_frame, track_point, tracks, K,
                                       max_median_err=2.5)
    if verbose:
        print(f"  filtered {n_removed} outlier points; final clean BA...")
    _run_ba(poses_by_frame, track_point, tracks, K, max_nfev=200,
            loss="huber", f_scale=2.0, verbose=verbose)
    if verbose:
        print(f"  final BA + filtering done in {time.time()-t_ba:.1f}s")

    # pack outputs in temporal camera order
    frames_sorted = sorted(poses_by_frame)
    frame_to_cam = {f: c for c, f in enumerate(frames_sorted)}
    cameras = [{"R": poses_by_frame[f][0], "t": poses_by_frame[f][1].reshape(3, 1)}
               for f in frames_sorted]

    tids = sorted(track_point)
    points_world = np.array([track_point[tid] for tid in tids], float)
    point_frame = np.array([int(np.median(list(tracks[tid].keys()))) for tid in tids])
    observations = []
    for p, tid in enumerate(tids):
        for f, xy in tracks[tid].items():
            if f in frame_to_cam:
                observations.append((p, frame_to_cam[f], float(xy[0]), float(xy[1])))

    # frames_sorted[c] is the video-frame index (into the extracted-frame list)
    # of camera c -- needed by Stage 3 (plane-sweep MVS) to re-fetch the actual
    # image behind each pose.
    return cameras, points_world, observations, point_frame, frames_sorted
