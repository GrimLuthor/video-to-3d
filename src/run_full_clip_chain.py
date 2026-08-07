"""Run the real Stage 1 chain (chain_poses, real epipolar geometry +
triangulation -- not ex4's 2D mosaic trick) across the *whole* pilot clip at
full 1920x1080 resolution, then run Stage 2 bundle adjustment on top and
compare before/after. Colors the recovered point cloud by which pair
contributed each point -- this directly shows the drift bundle adjustment
is meant to fix: before BA, points from different times form disjoint
islands instead of blending continuously along the camera's path.

Uses the real video-mode calibration from calibration.py (checkerboard
displayed on a screen, recorded at the same 1920x1080 settings as this
clip -- see calibrate_from_video.py), not the earlier still-photo-derived
approximation.

Usage:
    python run_full_clip_chain.py <video_path> <stride> [output_tag]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_frames import extract_frames
from chain_poses import chain_poses
from bundle_adjustment import bundle_adjust
import calibration

DEFAULT_VIDEO_PATH = Path(__file__).parent.parent / "capture" / "20260705_141208.mp4"
OUT_DIR = Path(__file__).parent.parent / "output"

DEFAULT_STRIDE = 40  # covers the whole ~36s pilot clip in ~27 sampled frames


def pair_idx_per_point_from_diagnostics(diagnostics, n_points):
    """Labels each point by the pair that *first created* it (n_new_points),
    not every pair that observed it -- continued tracks don't add new
    points_world entries, see chain_poses.py.
    """
    pair_idx_per_point = []
    for d in diagnostics:
        if d.get("status") != "ok":
            continue
        i0, i1 = d["pair"]
        n_pts = d["n_new_points"]
        pair_idx_per_point.extend([i0] * n_pts)
    pair_idx_per_point = np.array(pair_idx_per_point)
    assert len(pair_idx_per_point) == n_points, (len(pair_idx_per_point), n_points)
    return pair_idx_per_point


def print_track_length_histogram(observations, n_points):
    counts = np.zeros(n_points, dtype=int)
    for point_idx, _cam_idx, _x, _y in observations:
        counts[point_idx] += 1
    print(f"Track length (observations per point): mean={counts.mean():.2f}, "
          f"max={counts.max()}, points with 3+ observations: "
          f"{(counts >= 3).sum()} / {n_points} ({(counts >= 3).mean():.1%})")


def plot_reconstruction(points_world, pair_idx_per_point, cameras, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.array([(-cam["R"].T @ cam["t"]).ravel() for cam in cameras])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    depths = points_world[:, 2]
    thresh = np.percentile(np.abs(depths), 95)
    mask = np.abs(depths) < thresh
    sc = ax.scatter(points_world[mask, 0], points_world[mask, 1], points_world[mask, 2],
                     c=pair_idx_per_point[mask], cmap="viridis", s=2, alpha=0.6)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "r-o", markersize=3, label="camera trajectory")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("originating frame pair index (time along clip)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (~depth)")
    ax.legend()
    ax.set_title(title)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def print_depth_buckets(depths, pair_idx_per_point, n_buckets=6):
    print("Median |Z| (depth proxy) by pair-index bucket:")
    bucket_edges = np.linspace(pair_idx_per_point.min(), pair_idx_per_point.max() + 1, n_buckets + 1)
    for i in range(n_buckets):
        b_mask = (pair_idx_per_point >= bucket_edges[i]) & (pair_idx_per_point < bucket_edges[i + 1])
        if b_mask.sum() > 0:
            print(f"  pairs [{bucket_edges[i]:.0f},{bucket_edges[i+1]:.0f}): "
                  f"median|Z|={np.median(np.abs(depths[b_mask])):.2f}  n={b_mask.sum()}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video_path", nargs="?", type=Path, default=DEFAULT_VIDEO_PATH)
parser.add_argument("stride", nargs="?", type=int, default=DEFAULT_STRIDE)
parser.add_argument("output_tag", nargs="?", default=None,
                     help="Prefix for output files, defaults to the video's stem.")
parser.add_argument("--smoothness-weight", type=float, default=20.0,
                     help="Constant-walking-pace BA prior weight (0 disables it). "
                          "See bundle_adjustment.py; validated on synthetic data "
                          "at this scale to cut scale-drift step-ratio deviation "
                          "by >90%% for a ~0.05px reprojection-error cost.")
parser.add_argument("--force-rechain", action="store_true",
                     help="Ignore any cached chain_poses() result and recompute it "
                          "(chaining is the expensive part -- by default we cache it "
                          "so bundle-adjustment settings like --smoothness-weight "
                          "can be retuned without repaying that cost).")
args = parser.parse_args()
tag = args.output_tag or args.video_path.stem
cache_path = OUT_DIR / f"{tag}_chain_cache.pkl"

if cache_path.exists() and not args.force_rechain:
    print(f"Loading cached chain_poses() result from {cache_path} "
          f"(pass --force-rechain to recompute)...")
    with open(cache_path, "rb") as f:
        cameras, points_world, diagnostics, observations, K = pickle.load(f)
    print(f"Loaded {len(cameras)} cameras, {len(points_world)} points, "
          f"{len(observations)} observations")
else:
    t0 = time.time()
    frames = extract_frames(args.video_path, stride=args.stride)
    h, w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames at {w}x{h}, stride={args.stride} "
          f"(covers ~{len(frames)*args.stride/29.85:.1f}s of the clip) in {time.time()-t0:.1f}s")

    K = calibration.scale_K(w, h)
    frames = calibration.undistort_frames(frames, K)

    t0 = time.time()
    cameras, points_world, diagnostics, observations = chain_poses(frames, K)
    print(f"Chained {len(cameras)} cameras, {len(points_world)} points, "
          f"{len(observations)} observations in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((cameras, points_world, diagnostics, observations, K), f)
    print(f"Cached chain result to {cache_path}")

pair_idx_per_point = pair_idx_per_point_from_diagnostics(diagnostics, len(points_world))
print_track_length_histogram(observations, len(points_world))

np.save(OUT_DIR / f"{tag}_points.npy", points_world)
np.save(OUT_DIR / f"{tag}_pair_idx.npy", pair_idx_per_point)
np.save(OUT_DIR / f"{tag}_camera_centers.npy",
        np.array([(-cam["R"].T @ cam["t"]).ravel() for cam in cameras]))

plot_reconstruction(points_world, pair_idx_per_point, cameras,
                     f"{tag}: Stage 1 (chained, pre-BA), colored by time",
                     OUT_DIR / f"{tag}_colored_by_time.png")
print()
print_depth_buckets(points_world[:, 2], pair_idx_per_point)

print("\n" + "=" * 60)
print("STAGE 2: BUNDLE ADJUSTMENT")
print("=" * 60)
t0 = time.time()
cameras_ba, points_world_ba, ba_info = bundle_adjust(
    cameras, points_world, observations, K, smoothness_weight=args.smoothness_weight)
print(f"Bundle adjustment done in {time.time()-t0:.1f}s "
      f"(RMSE {ba_info['rmse_before']:.3f}px -> {ba_info['rmse_after']:.3f}px, "
      f"step-ratio dev {ba_info['step_ratio_dev_before']} -> {ba_info['step_ratio_dev_after']})")

np.save(OUT_DIR / f"{tag}_points_ba.npy", points_world_ba)
np.save(OUT_DIR / f"{tag}_camera_centers_ba.npy",
        np.array([(-cam["R"].T @ cam["t"]).ravel() for cam in cameras_ba]))

plot_reconstruction(points_world_ba, pair_idx_per_point, cameras_ba,
                     f"{tag}: Stage 2 (post bundle adjustment), colored by time",
                     OUT_DIR / f"{tag}_colored_by_time_ba.png")
print()
print_depth_buckets(points_world_ba[:, 2], pair_idx_per_point)
