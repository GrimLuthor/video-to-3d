"""Stage 3: plane-sweep multi-view stereo (dense per-pixel depth).

Given calibrated, scale-consistent poses from Stage 1/2 (incremental_sfm.py),
recover a dense depth map for a reference view by the classic plane-sweep:
hypothesize a family of fronto-parallel depth planes in the reference camera,
warp each neighbouring (source) view onto the reference image through the
homography that plane induces, score photo-consistency (ZNCC) per pixel, and
keep the depth whose warp best matches. No CNNs -- this is the multi-view,
calibrated-geometry version of the homography warping done in ex4.py (there the
homography came from optical flow for a 2D mosaic; here it comes from real 3D
geometry + a depth hypothesis).

Plane-induced homography (ref pixel -> src pixel), fronto-parallel plane at
depth d in the reference camera (normal n = [0,0,1]):

    H(d) = K R_rel K^-1  +  (K t_rel) [0,0,1] / d

where (R_rel, t_rel) is the reference->source relative pose. Warping the source
with WARP_INVERSE_MAP then samples, for every reference pixel, the source pixel
its ray would hit if the scene were at depth d.
"""

import cv2
import numpy as np

_EZ = np.array([[0.0, 0.0, 1.0]])  # picks the homogeneous coord of a ref pixel


def relative_pose(cam_ref, cam_src):
    """Both cameras are world->cam {R,t}. Returns (R_rel, t_rel) mapping
    reference-camera coords to source-camera coords: X_src = R_rel X_ref + t_rel."""
    R_r, t_r = cam_ref["R"], cam_ref["t"].reshape(3)
    R_s, t_s = cam_src["R"], cam_src["t"].reshape(3)
    R_rel = R_s @ R_r.T
    t_rel = t_s - R_rel @ t_r
    return R_rel, t_rel


def plane_homography(K, R_rel, t_rel, depth):
    """3x3 homography mapping a reference pixel to the corresponding source
    pixel for a fronto-parallel plane at `depth` in the reference camera."""
    return K @ R_rel @ np.linalg.inv(K) + (K @ t_rel).reshape(3, 1) @ _EZ / depth


def depth_hypotheses(d_min, d_max, n_planes):
    """Depth samples uniform in inverse depth (disparity-like) -- puts more
    planes near the camera where depth resolution matters most."""
    inv = np.linspace(1.0 / d_max, 1.0 / d_min, n_planes)
    return (1.0 / inv)[::-1]  # ascending depth


def depth_range_from_points(points_world, cam_ref, K, img_shape, lo=2.0, hi=98.0,
                            near_frac=0.15):
    """Depth range of the sparse points that fall in front of, and project
    inside, the reference view -- bounds the sweep to where the scene is.

    The near bound is deliberately NOT just the `lo` percentile: near-foreground
    objects (a bin, a parked car you walk past) carry few SIFT points compared
    to a big textured far wall, so they sink below that percentile and would be
    excluded from the sweep entirely (then get clamped to the far range and
    misplaced). We extend the near bound down to `near_frac` * median depth so
    the sweep always covers near space -- cheap because inverse-depth plane
    sampling concentrates planes near the camera anyway.
    """
    R, t = cam_ref["R"], cam_ref["t"].reshape(3)
    Xc = (R @ points_world.T).T + t          # points in ref-camera coords
    z = Xc[:, 2]
    front = z > 1e-6
    proj = (K @ Xc[front].T).T
    uv = proj[:, :2] / proj[:, 2:3]
    h, w = img_shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    zc = z[front][inside]
    if len(zc) < 10:
        zc = z[front]
    d_lo = min(np.percentile(zc, lo), near_frac * np.median(zc))
    d_lo = max(0.3, d_lo)  # absolute near clip
    return float(d_lo), float(np.percentile(zc, hi))


def _warp_source(src_gray, H, size):
    """Warp a source view into the reference grid via H (ref->src) with inverse
    mapping. Returns (warped float32, valid mask bool)."""
    w, h = size
    warped = cv2.warpPerspective(src_gray, H, (w, h),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones_like(src_gray, dtype=np.float32)
    vmask = cv2.warpPerspective(ones, H, (w, h),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(np.float32), vmask > 0.999


def _zncc_cost(ref, warped, valid, window):
    """Per-pixel ZNCC cost (1 - zncc, in [0,2]) over a `window`x`window` box.
    Invalid pixels get cost 2.0 (worst)."""
    k = (window, window)
    mu_a = cv2.boxFilter(ref, -1, k)
    mu_b = cv2.boxFilter(warped, -1, k)
    a2 = cv2.boxFilter(ref * ref, -1, k)
    b2 = cv2.boxFilter(warped * warped, -1, k)
    ab = cv2.boxFilter(ref * warped, -1, k)
    var_a = np.maximum(a2 - mu_a * mu_a, 0)
    var_b = np.maximum(b2 - mu_b * mu_b, 0)
    cov = ab - mu_a * mu_b
    zncc = cov / (np.sqrt(var_a * var_b) + 1e-6)
    cost = 1.0 - zncc
    cost[~valid] = 2.0
    return cost.astype(np.float32)


def _guided_filter(guide, src, radius, eps):
    """He et al. edge-aware guided filter (all box filters). `guide` in [0,1].
    Used for cost-volume aggregation: smooths cost within regions of uniform
    reference appearance while preserving discontinuities at real image edges."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    mean_Ip = cv2.boxFilter(guide * src, -1, k)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(guide * guide, -1, k) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def _aggregate_sources(costs, best_frac):
    """Robust across-view aggregation of per-source cost maps (list of (H,W),
    invalid entries = +inf). Per pixel, average only the best (lowest-cost)
    fraction of the valid sources -- a source where this pixel is occluded
    scores badly and is dropped, instead of poisoning the mean."""
    stack = np.stack(costs, axis=0)                 # (S,H,W), inf where invalid
    stack.sort(axis=0)                              # ascending; inf sinks to top
    S = stack.shape[0]
    k = max(1, int(np.ceil(S * best_frac)))
    top = stack[:k]                                 # (k,H,W) best-k per pixel
    finite = np.isfinite(top)
    cnt = finite.sum(axis=0)
    summ = np.where(finite, top, 0.0).sum(axis=0)
    out = np.full(cnt.shape, np.nan, np.float32)
    ok = cnt > 0
    out[ok] = summ[ok] / cnt[ok]
    return out


def plane_sweep(ref_gray, source_grays, K, cam_ref, cam_srcs, depths,
                window=7, best_frac=0.6, cost_agg_radius=6, cost_agg_eps=1e-4,
                speckle_win=5):
    """Compute a dense depth map for the reference view.

    ref_gray / source_grays: float32 grayscale images (same size).
    depths: 1D array of depth hypotheses (ascending).
    best_frac: fraction of sources averaged per pixel (occlusion robustness).
    cost_agg_radius: edge-aware guided-filter radius for cost aggregation
        (0 disables -- reverts to raw winner-take-all).
    speckle_win: final median-filter window for speckle cleanup (0 disables).

    Returns (depth_map, confidence, best_cost) each (H,W) float32; depth_map is
    NaN where no depth had enough valid source support.
    """
    h, w = ref_gray.shape
    D = len(depths)
    cost_vol = np.full((D, h, w), np.nan, np.float32)
    guide = (ref_gray / 255.0).astype(np.float32)

    rel = [relative_pose(cam_ref, cs) for cs in cam_srcs]
    for di, d in enumerate(depths):
        per_source = []
        for (R_rel, t_rel), src in zip(rel, source_grays):
            H = plane_homography(K, R_rel, t_rel, d)
            warped, valid = _warp_source(src, H, (w, h))
            cost = _zncc_cost(ref_gray, warped, valid, window)
            cost[~valid] = np.inf
            per_source.append(cost)
        agg = _aggregate_sources(per_source, best_frac)   # (H,W), NaN where none
        if cost_agg_radius > 0:
            # edge-aware spatial aggregation; fill NaN with the worst cost first
            slice_ = np.where(np.isfinite(agg), agg, 2.0).astype(np.float32)
            agg = _guided_filter(guide, slice_, cost_agg_radius, cost_agg_eps)
        cost_vol[di] = agg

    # winner-take-all over depth
    valid_any = np.isfinite(cost_vol).any(axis=0)
    filled = np.where(np.isfinite(cost_vol), cost_vol, np.inf)
    best_idx = np.argmin(filled, axis=0)
    best_cost = np.take_along_axis(filled, best_idx[None], axis=0)[0]

    depth_map = _subpixel_depth(filled, best_idx, depths)
    depth_map[~valid_any] = np.nan

    conf = _ratio_confidence(filled, best_idx)
    conf[~valid_any] = 0.0

    if speckle_win and speckle_win >= 3:
        # median filter smooths salt-and-pepper depth without needing contrib
        # modules; only applied where depth is valid.
        dm = np.where(np.isfinite(depth_map), depth_map, 0).astype(np.float32)
        dm = cv2.medianBlur(dm, speckle_win)
        depth_map = np.where(np.isfinite(depth_map), dm, np.nan).astype(np.float32)

    return depth_map.astype(np.float32), conf.astype(np.float32), best_cost.astype(np.float32)


def patchmatch_depth(ref_gray, source_grays, K, cam_ref, cam_srcs, d_min, d_max,
                     iterations=3, window=7, best_frac=0.6, n_refine=3,
                     init_planes=32, seed=0, verbose=False):
    """Depth by (fronto-parallel) PatchMatch stereo instead of a brute-force
    plane sweep. Each pixel holds a *continuous* depth hypothesis; we start from
    random depths and repeatedly (a) PROPAGATE each pixel's depth to its
    neighbours (good depths spread across the image -- this is what fills
    textureless regions and gives spatial coherence the per-pixel winner-take-all
    plane sweep lacks) and (b) randomly REFINE with a shrinking radius. Photo-
    consistency is the same multi-view ZNCC + robust best-K aggregation as the
    plane sweep. Continuous depth + propagation => thinner surfaces and less
    speckle than discrete planes.

    Cost is comparable-or-cheaper than the plane sweep: ~iterations*(4 prop +
    n_refine) cost evaluations vs. one per depth plane (~128).

    Returns (depth NaN where unsupported, confidence in [0,1]).
    """
    h, w = ref_gray.shape
    rng = np.random.default_rng(seed)
    Kinv = np.linalg.inv(K)
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1).astype(np.float64)
    dirs = pix @ Kinv.T                      # (h,w,3) viewing rays, z-comp = 1
    rel = [relative_pose(cam_ref, cs) for cs in cam_srcs]
    inv_min, inv_max = 1.0 / d_max, 1.0 / d_min

    def cost_of(Dc):
        """Aggregated multi-view ZNCC cost of each pixel under per-pixel depth Dc."""
        Xc = dirs * Dc[..., None]
        per = []
        for (R_rel, t_rel), src in zip(rel, source_grays):
            Xs = Xc @ R_rel.T + t_rel
            z = Xs[..., 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                mapx = (Xs[..., 0] * K[0, 0] / z + K[0, 2]).astype(np.float32)
                mapy = (Xs[..., 1] * K[1, 1] / z + K[1, 2]).astype(np.float32)
            warped = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            valid = (z > 1e-6) & (mapx >= 0) & (mapx < w) & (mapy >= 0) & (mapy < h)
            c = _zncc_cost(ref_gray, warped, valid, window)
            c[~valid] = np.inf
            per.append(c)
        return _aggregate_sources(per, best_frac)

    # Coherent initialisation from a coarse plane sweep (random init + parallel
    # propagation floods far too slowly to converge -> salt-and-pepper). PatchMatch
    # then refines this to continuous depth.
    D0, _, _ = plane_sweep(ref_gray, source_grays, K, cam_ref, cam_srcs,
                           depth_hypotheses(d_min, d_max, init_planes),
                           window=window, best_frac=best_frac,
                           cost_agg_radius=0, speckle_win=0)
    D = D0.copy()
    bad = ~np.isfinite(D)
    D[bad] = 1.0 / rng.uniform(inv_min, inv_max, size=int(bad.sum()))
    curr = cost_of(D)

    def update(Dc, strict=True):
        nonlocal D, curr
        c = cost_of(Dc)
        cmp = (c < curr) if strict else (c <= curr)
        better = np.isfinite(c) & (~np.isfinite(curr) | cmp)
        D = np.where(better, Dc, D)
        curr = np.where(better, c, curr)

    for it in range(iterations):
        # Jump-flood propagation, STRICT (only spread a depth if it *strictly*
        # lowers cost). Accepting ties over-propagated the dominant background
        # plane into thin foreground objects and across depth edges -> car/fence
        # flattened onto the far plane, corners merged. Strict + smaller steps
        # keeps foreground: a near object's own depth fits it better, so the
        # wall's depth can't invade it. Textureless regions just keep their
        # coarse-plane-sweep init depth (no worse than the plane sweep).
        for step in (4, 2, 1):
            for ax, sh in [(1, step), (1, -step), (0, step), (0, -step)]:
                update(np.roll(D, sh, axis=ax), strict=True)
        r = (inv_max - inv_min) / 4.0                        # random refine, shrinking
        for _ in range(n_refine):
            invd = np.clip(1.0 / D + rng.uniform(-r, r, size=(h, w)), inv_min, inv_max)
            update(1.0 / invd, strict=True)
            r *= 0.5
        if verbose:
            m = np.isfinite(curr)
            print(f"    PM iter {it+1}/{iterations}: mean cost "
                  f"{np.nanmean(np.where(m, curr, np.nan)):.3f}, valid {m.mean():.1%}",
                  flush=True)

    depth = np.where(np.isfinite(curr), D, np.nan).astype(np.float32)
    conf = np.where(np.isfinite(curr), np.clip(1.0 - curr, 0.0, 1.0), 0.0).astype(np.float32)
    return depth, conf


def backproject_to_world(depth, cam, K):
    """Back-project every pixel's depth into world coordinates. Returns
    (HW,3) world points and (HW,) the per-pixel depth used."""
    h, w = depth.shape
    Kinv = np.linalg.inv(K)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([u.ravel(), v.ravel(), np.ones(h * w)], axis=0).astype(np.float64)
    dirs = (Kinv @ pix).T                       # (HW,3), z-component = 1
    z = depth.reshape(-1).astype(np.float64)
    pts_cam = dirs * z[:, None]                 # camera-frame points
    R, t = cam["R"], cam["t"].reshape(3)
    pts_world = (R.T @ (pts_cam - t).T).T
    return pts_world, z


def project_world(points, cam, K):
    """Project world points into a camera. Returns (N,2) pixels and (N,) depth."""
    R, t = cam["R"], cam["t"].reshape(3)
    pc = (R @ points.T).T + t
    z = pc[:, 2]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def geometric_consistency(ref_depth, ref_cam, neighbor_depths, neighbor_cams, K,
                          rel_thresh=0.02, min_support=2):
    """Keep a reference-view depth only where >= min_support neighbour views
    independently agree with it. For each reference pixel: back-project to a 3D
    point, project it into each neighbour, and compare the neighbour's OWN
    estimated depth there against the projected depth; agreement within
    `rel_thresh` (relative) counts as support.

    This is the denoiser for large low-confidence areas: a per-view artifact
    (e.g. the horizontal aperture strip) has no cross-view agreement and is
    dropped, while genuine geometry that several views recovered survives even
    if any single view's confidence was low.

    Returns (filtered_depth NaN where unsupported, support_count int map).
    """
    h, w = ref_depth.shape
    valid = np.isfinite(ref_depth)
    pts_world, _ = backproject_to_world(np.where(valid, ref_depth, 1.0), ref_cam, K)
    support = np.zeros(h * w, dtype=int)
    for nd, ncam in zip(neighbor_depths, neighbor_cams):
        uv, z_proj = project_world(pts_world, ncam, K)
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z_proj > 0)
        z_nbr = np.full(h * w, np.nan)
        idx = np.where(inb)[0]
        z_nbr[idx] = nd[v[idx], u[idx]]
        agree = inb & np.isfinite(z_nbr) & \
            (np.abs(z_nbr - z_proj) / np.maximum(z_proj, 1e-6) < rel_thresh)
        support += agree.astype(int)
    support = support.reshape(h, w)
    filtered = np.where(valid & (support >= min_support), ref_depth, np.nan)
    return filtered.astype(np.float32), support


def free_space_cull(xyz, cameras, depth_maps, K, tol=0.06, min_violations=3):
    """Remove points that violate free space (ghosts / depth-smear floaters).

    For each point, over all cameras, count:
      - VIOLATIONS: the point projects *in front of* that camera's measured
        surface by more than `tol` (relative) -> it would occlude something the
        camera clearly saw, so it's spurious there.
      - SUPPORT: the point matches that camera's measured surface within `tol`
        -> that camera actually sees this surface.
    A genuine surface is measured consistently (high support) while a ghost is
    mostly an in-front violation with little support. So we cull only when
    violations >= min_violations AND violations exceed support -- i.e. more
    cameras see the point as a floater than as a real surface. This protects
    real geometry from the occasional false violation caused by depth-map noise
    (which was over-culling ~half the cloud when we thresholded on violations
    alone). Points merely *behind* what a camera sees are just occluded (neither
    support nor violation) and are kept.

    xyz: (N,3) world points. cameras: list of {R,t} by camera index.
    depth_maps: dict {camera_index -> (H,W) depth map at K's resolution}.
    Returns (keep_mask, violation_counts, support_counts).
    """
    n = len(xyz)
    violations = np.zeros(n, dtype=int)
    support = np.zeros(n, dtype=int)
    for cam_idx, dm in depth_maps.items():
        R, t = cameras[cam_idx]["R"], cameras[cam_idx]["t"].reshape(3)
        Pc = (R @ xyz.T).T + t
        z = Pc[:, 2]
        proj = (K @ Pc.T).T
        with np.errstate(invalid="ignore", divide="ignore"):
            u = proj[:, 0] / proj[:, 2]
            v = proj[:, 1] / proj[:, 2]
        h, w = dm.shape
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        inb = (z > 1e-6) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        idx = np.where(inb)[0]
        d_meas = np.full(n, np.nan)
        d_meas[idx] = dm[vi[idx], ui[idx]]
        with np.errstate(invalid="ignore"):
            rel = (z - d_meas) / d_meas          # < 0 => in front of surface
        valid = np.isfinite(d_meas)
        violations += (valid & (rel < -tol)).astype(int)
        support += (valid & (np.abs(rel) <= tol)).astype(int)
    keep = (violations < min_violations) | (support >= violations)
    return keep, violations, support


def _subpixel_depth(cost_vol, best_idx, depths):
    """Parabolic interpolation of the cost minimum across the two neighbouring
    depth planes -> sub-plane depth, removing the depth-discretisation staircase."""
    D, h, w = cost_vol.shape
    depth_map = depths[best_idx].astype(np.float64)
    interior = (best_idx > 0) & (best_idx < D - 1)
    ii, jj = np.where(interior)
    bi = best_idx[ii, jj]
    c0 = cost_vol[bi - 1, ii, jj]
    c1 = cost_vol[bi, ii, jj]
    c2 = cost_vol[bi + 1, ii, jj]
    denom = (c0 - 2 * c1 + c2)
    good = np.isfinite(denom) & (np.abs(denom) > 1e-6)
    delta = np.zeros_like(denom)
    delta[good] = 0.5 * (c0[good] - c2[good]) / denom[good]
    delta = np.clip(delta, -1, 1)
    # interpolate in depth space between neighbouring planes
    d_lo = depths[np.maximum(bi - 1, 0)]
    d_hi = depths[np.minimum(bi + 1, D - 1)]
    d_ctr = depths[bi]
    step = np.where(delta >= 0, d_hi - d_ctr, d_ctr - d_lo)
    depth_map[ii, jj] = d_ctr + delta * step
    return depth_map


def _ratio_confidence(cost_vol, best_idx):
    """1 - best/second-best cost, masking the winning plane's neighbourhood so
    the 'second best' is a genuinely different depth."""
    D, h, w = cost_vol.shape
    tmp = cost_vol.copy()
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for off in (-1, 0, 1):
        idx = np.clip(best_idx + off, 0, D - 1)
        tmp[idx, ii, jj] = np.inf
    second = np.min(tmp, axis=0)
    best = np.take_along_axis(cost_vol, best_idx[None], axis=0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = 1.0 - best / second
    conf[~np.isfinite(conf)] = 0.0
    return np.clip(conf, 0, 1)
