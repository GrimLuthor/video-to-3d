"""End-to-end incremental-SfM runner (the scale-drift fix for longer_walk.mp4).

Pipeline: extract frames -> undistort -> SIFT features -> windowed multi-view
matching + track building (feature_tracks.py) -> incremental reconstruction +
bundle adjustment (incremental_sfm.py). Prints the same depth-bucket drift
diagnostic and colored-by-time plot as run_full_clip_chain.py so results are
directly comparable to the old chained pipeline.

The windowed matching (the expensive, recomputable part) is cached to
<tag>_tracks_cache.pkl; pass --force-rematch to recompute.

Usage:
    python run_incremental_sfm.py <video_path> <stride> [output_tag] [--window W]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_frames import extract_frames
from feature_tracks import extract_features, match_windowed, build_tracks
from incremental_sfm import reconstruct
import calibration

OUT_DIR = Path(__file__).parent.parent / "output"


def plot_reconstruction(points_world, point_frame, cameras, title, out_path):
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
                    c=point_frame[mask], cmap="viridis", s=2, alpha=0.6)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "r-o", markersize=3,
            label="camera trajectory")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("median frame index of track (time along clip)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (~depth)")
    ax.legend(); ax.set_title(title)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def print_scale_drift_diagnostic(cameras, points_world, point_frame):
    """The honest scale-drift test, independent of world orientation: (a) camera
    step length over time -- a roughly constant walking pace should give roughly
    constant steps; monocular scale drift makes them grow monotonically; and
    (b) true point->nearest-camera viewing distance bucketed by time. The old
    chained pipeline drifted ~12x (monotonic); a fixed reconstruction shows flat
    steps and non-monotonic viewing distance (that's real scene depth, not drift)."""
    centers = np.array([(-cam["R"].T @ cam["t"]).ravel() for cam in cameras])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    k = max(1, len(steps) // 3)
    print(f"Camera step length (flat => no scale drift): first third median="
          f"{np.median(steps[:k]):.3f}, middle={np.median(steps[k:2*k]):.3f}, "
          f"last third={np.median(steps[2*k:]):.3f}")
    d = np.linalg.norm(points_world[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    edges = np.linspace(point_frame.min(), point_frame.max() + 1, 7)
    meds = []
    print("True viewing distance (point -> nearest camera) by time bucket:")
    for i in range(6):
        m = (point_frame >= edges[i]) & (point_frame < edges[i + 1])
        if m.sum():
            md = np.median(d[m]); meds.append(md)
            print(f"  frames [{edges[i]:.0f},{edges[i+1]:.0f}): median dist={md:.2f}  n={m.sum()}")
    if len(meds) >= 2:
        print(f"  --> viewing-distance ratio max/min = {max(meds)/min(meds):.2f}x "
              f"(non-monotonic = real depth; old chained pipeline was a monotonic ~12x)")


def print_depth_buckets(depths, point_frame, n_buckets=6):
    print("Median |Z| (depth proxy) by time bucket (drift shows as growth across buckets):")
    edges = np.linspace(point_frame.min(), point_frame.max() + 1, n_buckets + 1)
    meds = []
    for i in range(n_buckets):
        m = (point_frame >= edges[i]) & (point_frame < edges[i + 1])
        if m.sum() > 0:
            med = np.median(np.abs(depths[m]))
            meds.append(med)
            print(f"  frames [{edges[i]:.0f},{edges[i+1]:.0f}): median|Z|={med:.2f}  n={m.sum()}")
    if len(meds) >= 2:
        print(f"  --> drift ratio (max/min bucket) = {max(meds)/min(meds):.2f}x "
              f"(old chained pipeline was ~12x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("stride", nargs="?", type=int, default=15)
    parser.add_argument("output_tag", nargs="?", default=None)
    parser.add_argument("--window", type=int, default=8,
                        help="Match each frame to the next W frames (W=1 is the old "
                             "chain). Larger W = longer tracks / wider baselines.")
    parser.add_argument("--min-tri-angle", type=float, default=2.0)
    parser.add_argument("--pose-jump-guard", type=float, default=12.0,
                        help="Reject a PnP pose whose camera center jumps more than "
                             "this many times the median per-frame motion from its "
                             "nearest registered neighbour (guards against glass/"
                             "reflection mismatches corrupting the whole back half). "
                             "Self-scaling; 12x clears legit indoor motion (library "
                             "peaks ~7.6x) while catching catastrophes (prep_academy "
                             "was ~76x); 0 disables.")
    parser.add_argument("--force-rematch", action="store_true")
    args = parser.parse_args()

    tag = args.output_tag or (args.video_path.stem + "_incr")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / f"{tag}_tracks_cache.pkl"

    if cache_path.exists() and not args.force_rematch:
        print(f"Loading cached features/tracks from {cache_path} (--force-rematch to redo)...")
        with open(cache_path, "rb") as f:
            features, tracks, K = pickle.load(f)
        print(f"  {len(features)} frames, {len(tracks)} tracks")
    else:
        t0 = time.time()
        frames = extract_frames(args.video_path, stride=args.stride)
        h, w = frames[0].shape[:2]
        print(f"Extracted {len(frames)} frames at {w}x{h}, stride={args.stride} in {time.time()-t0:.1f}s")
        K = calibration.scale_K(w, h)
        frames = calibration.undistort_frames(frames, K)

        t0 = time.time()
        features = extract_features(frames)
        print(f"SIFT on {len(features)} frames in {time.time()-t0:.1f}s "
              f"(mean {np.mean([len(f['xy']) for f in features]):.0f} kp/frame)")
        t0 = time.time()
        verified = match_windowed(features, K, window=args.window)
        tracks = build_tracks(features, verified)
        print(f"Matching+tracks in {time.time()-t0:.1f}s")
        # drop raw cv2 keypoints before pickling (not picklable / not needed downstream)
        features_slim = [{"xy": f["xy"]} for f in features]
        with open(cache_path, "wb") as f:
            pickle.dump((features_slim, tracks, K), f)
        features = features_slim
        print(f"Cached to {cache_path}")

    print("\n" + "=" * 60 + "\nINCREMENTAL RECONSTRUCTION\n" + "=" * 60)
    t0 = time.time()
    cameras, points_world, observations, point_frame, frame_indices = reconstruct(
        features, tracks, K, window=args.window, min_tri_angle=args.min_tri_angle,
        pose_jump_guard=args.pose_jump_guard)
    print(f"Reconstruction (incl. BA) in {time.time()-t0:.1f}s: "
          f"{len(cameras)} cameras, {len(points_world)} points, {len(observations)} observations")

    np.save(OUT_DIR / f"{tag}_points.npy", points_world)
    np.save(OUT_DIR / f"{tag}_point_frame.npy", point_frame)
    np.save(OUT_DIR / f"{tag}_camera_centers.npy",
            np.array([(-cam["R"].T @ cam["t"]).ravel() for cam in cameras]))

    # Full reconstruction bundle for downstream stages (Stage 3 plane-sweep MVS
    # needs the actual poses R,t and which video frame each camera is).
    bundle = {
        "frame_indices": list(frame_indices),  # camera c -> extracted-frame index
        "cameras": [{"R": c["R"], "t": c["t"]} for c in cameras],
        "K": K,
        "points_world": points_world,
        "point_frame": point_frame,
        "observations": observations,
        "stride": args.stride,
        "video_path": str(args.video_path),
    }
    with open(OUT_DIR / f"{tag}_reconstruction.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved reconstruction bundle to {tag}_reconstruction.pkl")

    print()
    print_scale_drift_diagnostic(cameras, points_world, point_frame)
    print()
    print_depth_buckets(points_world[:, 2], point_frame)
    plot_reconstruction(points_world, point_frame, cameras,
                        f"{tag}: incremental SfM (windowed tracks + BA)",
                        OUT_DIR / f"{tag}_colored_by_time.png")


if __name__ == "__main__":
    main()
