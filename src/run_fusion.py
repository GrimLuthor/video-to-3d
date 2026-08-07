"""Stage 3->4: multi-view depth fusion into a dense coloured point cloud.

For each reference frame in the selected range: compute a permissive plane-sweep
depth map, keep only depths >= min_support neighbour views agree with
(cross-view consistency), back-project the survivors to world coordinates with
colour from the reference image, and accumulate. Depth maps are cached per
camera so each is computed once even though it serves as both a reference and a
neighbour. The merged cloud is voxel-downsampled (averaging duplicates) and
written as a binary PLY plus a preview PNG.

Every reference view keeps only its verified fraction, but the fractions from
all frames overlap and accumulate, so the fused cloud is dense even though any
single depth map is sparse after filtering.

Usage:
    python run_fusion.py [tag] [--cam-start A] [--cam-end B] [--every N]
                         [--scale S] [--planes D] [--min-support M]
                         [--rel-thresh R] [--voxel V]
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_frames import extract_frames
import calibration
import plane_sweep as ps
from run_plane_sweep_poc import scale_K, pick_neighbors, pick_neighbors_multibaseline

OUT_DIR = Path(__file__).parent.parent / "output"


def write_ply(path, xyz, rgb):
    n = len(xyz)
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        arr.tofile(f)


def statistical_clean(xyz, rgb, nb_neighbors=20, std_ratio=2.0):
    """Remove floater outliers: drop points whose mean distance to their
    nb_neighbors nearest neighbours is > std_ratio std-devs above the global
    mean (Open3D statistical outlier removal). Catches the isolated 'spray' of
    spurious depths that pass cross-view consistency but sit in empty space."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).astype(np.uint8)


def voxel_downsample(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = inv.max() + 1
    cnt = np.zeros(n)
    np.add.at(cnt, inv, 1)
    sxyz = np.zeros((n, 3)); np.add.at(sxyz, inv, xyz)
    srgb = np.zeros((n, 3)); np.add.at(srgb, inv, rgb.astype(np.float64))
    return sxyz / cnt[:, None], (srgb / cnt[:, None]).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag", nargs="?", default="longer_walk_incr")
    p.add_argument("--cam-start", type=int, default=0)
    p.add_argument("--cam-end", type=int, default=None, help="exclusive; default = all")
    p.add_argument("--every", type=int, default=1, help="use every Nth camera as a reference")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--planes", type=int, default=96)
    p.add_argument("--patchmatch", action="store_true",
                   help="use PatchMatch depth (continuous + propagation, cleaner/thinner) "
                        "instead of the brute-force plane sweep")
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--neighbors", type=int, default=6)
    p.add_argument("--single-baseline", action="store_true",
                   help="use only the closest N source views (old behaviour); "
                        "default is multi-baseline: tight views for near objects "
                        "+ wide views for far surfaces")
    # Defaults = the "B" setting chosen on the stride-15 data (2026-08-05):
    # min_support 3 kills floaters, rel_thresh 1.5% keeps foliage. With a smaller
    # stride (more per-surface support) a looser rel_thresh (~2%, "mid") gives
    # richer foliage without the floaters -- revisit these when stride changes.
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--rel-thresh", type=float, default=0.015)
    p.add_argument("--voxel", type=float, default=0.04)
    p.add_argument("--no-visibility-cull", action="store_true",
                   help="skip free-space/visibility cull (ghost/depth-smear removal)")
    p.add_argument("--vis-tol", type=float, default=0.06,
                   help="relative depth margin to count as 'in front' (lower = more aggressive)")
    p.add_argument("--vis-min-viol", type=int, default=3,
                   help="min in-front violations before a point can be culled")
    p.add_argument("--save-depths", action="store_true",
                   help="pickle the per-camera depth maps (<tag>_depths<suffix>.pkl) "
                        "so the cull can be re-tuned offline via tune_cull.py")
    p.add_argument("--depths-from", default=None,
                   help="reuse precomputed depth maps <tag>_depths<SUFFIX>.pkl instead of "
                        "recomputing them -- lets fusion knobs (consistency, cull) be "
                        "iterated offline in minutes instead of a full ~40min re-fuse")
    p.add_argument("--no-clean", action="store_true", help="skip statistical outlier removal")
    p.add_argument("--clean-std", type=float, default=2.0,
                   help="lower = more aggressive floater removal")
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    with open(OUT_DIR / f"{args.tag}_reconstruction.pkl", "rb") as f:
        b = pickle.load(f)
    cameras, frame_indices, K_full = b["cameras"], b["frame_indices"], b["K"]
    points_world = b["points_world"]
    n_cams = len(cameras)
    cam_end = args.cam_end if args.cam_end is not None else n_cams

    frames = extract_frames(Path(b["video_path"]), stride=b["stride"])
    frames = calibration.undistort_frames(frames, K_full)
    s = args.scale
    K = scale_K(K_full, s)

    def color_bgr(cam_order):
        img = frames[frame_indices[cam_order]]
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def gray(cam_order):
        return cv2.cvtColor(color_bgr(cam_order), cv2.COLOR_BGR2GRAY).astype(np.float32)

    def neighbors_of(cam_order):
        if args.single_baseline:
            return pick_neighbors(frame_indices, cam_order, args.neighbors)
        return pick_neighbors_multibaseline(frame_indices, cam_order)

    depth_cache = {}
    if args.depths_from is not None:
        with open(OUT_DIR / f"{args.tag}_depths{args.depths_from}.pkl", "rb") as fh:
            depth_cache = pickle.load(fh)
        print(f"loaded {len(depth_cache)} precomputed depth maps from "
              f"{args.tag}_depths{args.depths_from}.pkl (offline re-fuse; no depth recompute)")

    def get_depth(cam_order):
        if cam_order in depth_cache:
            return depth_cache[cam_order]
        if args.depths_from is not None:
            return None  # only the saved maps are available in offline mode
        ref_gray = gray(cam_order)
        nbrs = neighbors_of(cam_order)
        src = [gray(c) for c in nbrs]
        cs = [cameras[c] for c in nbrs]
        d_min, d_max = ps.depth_range_from_points(points_world, cameras[cam_order], K, ref_gray.shape)
        if args.patchmatch:
            depth, _ = ps.patchmatch_depth(ref_gray, src, K, cameras[cam_order], cs,
                                           d_min, d_max, window=args.window)
        else:
            depth, _, _ = ps.plane_sweep(
                ref_gray, src, K, cameras[cam_order], cs,
                ps.depth_hypotheses(d_min, d_max, args.planes),
                window=args.window, cost_agg_radius=0, speckle_win=0)
        depth_cache[cam_order] = depth
        return depth

    refs = list(range(args.cam_start, cam_end, args.every))

    # Precompute all needed depth maps up front with a progress bar / ETA (this is
    # the long stage). Skipped when reusing saved maps (--depths-from).
    if args.depths_from is None:
        needed = sorted(set(refs) | {c for r in refs for c in neighbors_of(r)})
        method = "PatchMatch" if args.patchmatch else f"plane-sweep ({args.planes} planes)"
        print(f"\n=== Stage 3: computing {len(needed)} depth maps ({method}, scale {s}) ===",
              flush=True)
        td = time.time()
        for k, cam in enumerate(needed, 1):
            get_depth(cam)
            el = time.time() - td
            eta = el / k * (len(needed) - k)
            print(f"  depth {k}/{len(needed)} (cam {cam}) | {el:.0f}s elapsed, ~{eta:.0f}s left",
                  flush=True)
        print(f"=== depth stage done in {time.time()-td:.0f}s ===", flush=True)

    print(f"\n=== Stage 4: fusing {len(refs)} reference views "
          f"(cams {args.cam_start}..{cam_end-1} step {args.every}) ===", flush=True)
    all_xyz, all_rgb = [], []
    t0 = time.time()
    for i, ref in enumerate(refs):
        d_ref = get_depth(ref)
        if d_ref is None:
            continue
        nbrs = neighbors_of(ref)
        nbr_pairs = [(c, get_depth(c)) for c in nbrs]
        nbr_pairs = [(c, d) for c, d in nbr_pairs if d is not None]
        if not nbr_pairs:
            continue
        filtered, _ = ps.geometric_consistency(
            d_ref, cameras[ref], [d for _, d in nbr_pairs], [cameras[c] for c, _ in nbr_pairs],
            K, rel_thresh=args.rel_thresh, min_support=args.min_support)
        mask = np.isfinite(filtered).reshape(-1)
        if not mask.any():
            continue
        pts, _ = ps.backproject_to_world(np.nan_to_num(filtered, nan=1.0), cameras[ref], K)
        rgb = cv2.cvtColor(color_bgr(ref), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        all_xyz.append(pts[mask])
        all_rgb.append(rgb[mask])
        if (i + 1) % 5 == 0 or i == len(refs) - 1:
            print(f"  [{i+1}/{len(refs)}] ref cam {ref}: +{mask.sum()} pts "
                  f"({sum(len(x) for x in all_xyz)} total, {time.time()-t0:.0f}s)")

    if args.save_depths:
        dpath = OUT_DIR / f"{args.tag}_depths{args.out_suffix}.pkl"
        with open(dpath, "wb") as fh:
            pickle.dump(depth_cache, fh)
        print(f"saved {len(depth_cache)} depth maps to {dpath.name} (for tune_cull.py)")

    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    stats = {"reference_views": len(refs), "cameras_used": len(depth_cache),
             "raw_fused": int(len(xyz))}
    print(f"raw fused: {len(xyz)} points; voxel-downsampling at {args.voxel}...")
    xyz, rgb = voxel_downsample(xyz, rgb, args.voxel)
    stats["after_downsample"] = int(len(xyz))
    print(f"after downsample: {len(xyz)} points")

    if not args.no_visibility_cull:
        n0 = len(xyz)
        # save the pre-cull cloud so the cull can be A/B'd in Open3D
        write_ply(OUT_DIR / f"{args.tag}_dense{args.out_suffix}_nocull.ply", xyz, rgb)
        keep, _, _ = ps.free_space_cull(xyz, cameras, depth_cache, K,
                                        tol=args.vis_tol, min_violations=args.vis_min_viol)
        xyz, rgb = xyz[keep], rgb[keep]
        stats["visibility_cull_removed"] = int(n0 - len(xyz))
        stats["visibility_cull_tol"] = args.vis_tol
        print(f"after free-space cull (tol {args.vis_tol}, >={args.vis_min_viol} viol): "
              f"{len(xyz)} points ({n0 - len(xyz)} ghost/smear points removed, "
              f"using {len(depth_cache)} depth maps; pre-cull saved as _nocull)")

    if not args.no_clean:
        n0 = len(xyz)
        xyz, rgb = statistical_clean(xyz, rgb, std_ratio=args.clean_std)
        stats["outlier_removed"] = int(n0 - len(xyz))
        print(f"after outlier removal (std {args.clean_std}): {len(xyz)} points "
              f"({n0 - len(xyz)} floaters removed)")

    stats["final_points"] = int(len(xyz))
    ply_path = OUT_DIR / f"{args.tag}_dense{args.out_suffix}.ply"
    write_ply(ply_path, xyz, rgb)
    with open(OUT_DIR / f"{args.tag}_dense{args.out_suffix}_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved {ply_path}")
    _preview(xyz, rgb, cameras, args.tag, args.out_suffix)


def _preview(xyz, rgb, cameras, tag, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.array([(-c["R"].T @ c["t"]).ravel() for c in cameras])
    # trim depth outliers along each axis for a sane view box
    keep = np.ones(len(xyz), bool)
    for a in range(3):
        lo, hi = np.percentile(xyz[:, a], [1, 99])
        keep &= (xyz[:, a] >= lo) & (xyz[:, a] <= hi)
    xyz, rgb = xyz[keep], rgb[keep]
    idx = np.random.default_rng(0).choice(len(xyz), min(120000, len(xyz)), replace=False)

    fig = plt.figure(figsize=(16, 10))
    for k, (el, az) in enumerate([(-70, -90), (-60, -60)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], c=rgb[idx] / 255.0, s=1)
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "r-", lw=1, label="trajectory")
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
    out = OUT_DIR / f"{tag}_dense{suffix}_preview.png"
    plt.tight_layout(); plt.savefig(out, dpi=130); print(f"Saved {out}")


if __name__ == "__main__":
    main()
