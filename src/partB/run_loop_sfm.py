"""Loop-closure-enabled incremental SfM runner (Part B; reuses Part A read-only).

Same pipeline as Part A's run_incremental_sfm.py -- extract -> undistort -> SIFT
-> windowed tracks -> incremental SfM + BA -- with ONE addition: after the
windowed matching it also runs `loop_closure.match_loops` (bag-of-words place
recognition of temporally-distant revisits + essential-matrix verification) and
FEEDS THE LOOP MATCHES INTO THE SAME build_tracks. A physical point seen at the
start and again at the end of a loop becomes one wide-baseline track, so the
global bundle adjustment pins the loop closed and the reconstruction is globally
scale-consistent -- the thing plain windowed matching cannot give a trajectory
that returns to where it started (an object orbit, a building loop).

Nothing in Part A is modified; everything new lives in Final/src/partB/.

    python Final/src/partB/run_loop_sfm.py capture/bin_1.mp4 5 bin_1_loop --window 10
    python Final/src/partB/run_loop_sfm.py capture/bin_1.mp4 5 bin_1_noloop --window 10 --no-loop   # ablation

The expensive matching (features + windowed + loop) is cached to
<tag>_loopsfm_cache.pkl; pass --force-rematch to recompute.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]  # Final/src
sys.path.insert(0, str(SRC))
from extract_frames import extract_frames                     # noqa: E402
from feature_tracks import extract_features, match_windowed, build_tracks  # noqa: E402
from incremental_sfm import reconstruct                       # noqa: E402
from run_incremental_sfm import (plot_reconstruction, print_scale_drift_diagnostic,  # noqa: E402
                                 print_depth_buckets)
import calibration                                            # noqa: E402
from loop_closure import match_loops                          # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "output"
CAP_DIR = Path(__file__).resolve().parents[2] / "capture"


def print_loop_diagnostic(cameras, tracks, window, n_loop_tracks_hint):
    """Report whether the trajectory actually forms (and closes) a loop, and how
    many wide-baseline loop tracks span past the matching window."""
    centers = np.array([(-c["R"].T @ c["t"]).ravel() for c in cameras])
    if len(centers) < 3:
        return
    centroid = centers.mean(axis=0)
    radius = np.median(np.linalg.norm(centers - centroid, axis=1))
    first_last = float(np.linalg.norm(centers[0] - centers[-1]))
    # how far the path wraps around the centroid (sum of turn angles ~ 360 deg for
    # a full orbit); cheap proxy = total angular sweep of the camera bearing.
    bearings = np.arctan2(centers[:, 2] - centroid[2], centers[:, 0] - centroid[0])
    sweep = np.abs(np.diff(np.unwrap(bearings))).sum()
    spans = np.array([max(t) - min(t) for t in tracks]) if tracks else np.zeros(0)
    n_wide = int((spans > window).sum())
    print("Loop diagnostic:")
    print(f"  orbit radius (median cam->centroid) = {radius:.3f}; first<->last camera "
          f"gap = {first_last:.3f} ({first_last/max(radius,1e-9):.2f} x radius)")
    print(f"  angular sweep around centroid ~ {np.degrees(sweep):.0f} deg "
          f"(~360 => a full orbit/loop)")
    print(f"  tracks spanning > window ({window}) frames = {n_wide} "
          f"(these are the loop/wide-baseline constraints; {n_loop_tracks_hint} "
          f"loop-closure pairs were added)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video name (in capture/, with or without .mp4) or a path")
    ap.add_argument("stride", nargs="?", type=int, default=8)
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min-tri-angle", type=float, default=2.0)
    ap.add_argument("--pose-jump-guard", type=float, default=12.0)
    ap.add_argument("--no-loop", action="store_true",
                    help="disable loop-closure matching (ablation = plain Part A pipeline)")
    ap.add_argument("--loop-top-k", type=int, default=6,
                    help="candidate distant revisits retrieved per frame")
    ap.add_argument("--loop-min-inliers", type=int, default=30,
                    help="essential-matrix inlier floor to accept a loop closure (strict)")
    ap.add_argument("--min-gap", type=int, default=None,
                    help="min frame gap for a loop candidate (default window+1)")
    ap.add_argument("--force-rematch", action="store_true")
    args = ap.parse_args()

    vp = Path(args.video)
    if not vp.exists():
        vp = CAP_DIR / (args.video if args.video.endswith(".mp4") else args.video + ".mp4")
    tag = args.tag or (vp.stem + "_loop")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / f"{tag}_loopsfm_cache.pkl"

    if cache_path.exists() and not args.force_rematch:
        print(f"Loading cached features/matches from {cache_path} (--force-rematch to redo)...")
        with open(cache_path, "rb") as f:
            features, windowed, loop_verified, K = pickle.load(f)
        print(f"  {len(features)} frames, {len(windowed)} windowed + "
              f"{len(loop_verified)} loop-closure pairs")
    else:
        t0 = time.time()
        frames = extract_frames(vp, stride=args.stride)
        h, w = frames[0].shape[:2]
        print(f"Extracted {len(frames)} frames {w}x{h}, stride={args.stride} in {time.time()-t0:.1f}s")
        K = calibration.scale_K(w, h)
        frames = calibration.undistort_frames(frames, K)

        t0 = time.time()
        features = extract_features(frames)
        print(f"SIFT on {len(features)} frames in {time.time()-t0:.1f}s "
              f"(mean {np.mean([len(f['xy']) for f in features]):.0f} kp/frame)")

        t0 = time.time()
        windowed = match_windowed(features, K, window=args.window)
        print(f"Windowed matching in {time.time()-t0:.1f}s")

        print("Loop-closure matching (Part B addition)...")
        loop_verified = match_loops(features, K, window=args.window, min_gap=args.min_gap,
                                    top_k=args.loop_top_k, min_inliers=args.loop_min_inliers)

        features = [{"xy": f["xy"]} for f in features]  # drop desc before pickling
        with open(cache_path, "wb") as f:
            pickle.dump((features, windowed, loop_verified, K), f)
        print(f"Cached to {cache_path}")

    use_loop = not args.no_loop
    verified = windowed + (loop_verified if use_loop else [])
    print(f"\nBuilding tracks from {len(windowed)} windowed"
          + (f" + {len(loop_verified)} loop-closure" if use_loop else " (loop closure DISABLED)")
          + " pairs...")
    tracks = build_tracks(features, verified)

    print("\n" + "=" * 60 + "\nINCREMENTAL RECONSTRUCTION"
          + ("  (loop closure ON)" if use_loop else "  (loop closure OFF)")
          + "\n" + "=" * 60)
    t0 = time.time()
    cameras, points_world, observations, point_frame, frame_indices = reconstruct(
        features, tracks, K, window=args.window, min_tri_angle=args.min_tri_angle,
        pose_jump_guard=args.pose_jump_guard)
    print(f"Reconstruction (incl. BA) in {time.time()-t0:.1f}s: {len(cameras)} cameras, "
          f"{len(points_world)} points, {len(observations)} observations")

    np.save(OUT_DIR / f"{tag}_points.npy", points_world)
    np.save(OUT_DIR / f"{tag}_point_frame.npy", point_frame)
    np.save(OUT_DIR / f"{tag}_camera_centers.npy",
            np.array([(-c["R"].T @ c["t"]).ravel() for c in cameras]))
    bundle = {
        "frame_indices": list(frame_indices),
        "cameras": [{"R": c["R"], "t": c["t"]} for c in cameras],
        "K": K,
        "points_world": points_world,
        "point_frame": point_frame,
        "observations": observations,
        "stride": args.stride,
        "video_path": str(vp),
        "loop_closure": use_loop,
        "n_loop_pairs": len(loop_verified) if use_loop else 0,
    }
    with open(OUT_DIR / f"{tag}_reconstruction.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved reconstruction bundle to {tag}_reconstruction.pkl")

    print()
    print_loop_diagnostic(cameras, tracks, args.window,
                          len(loop_verified) if use_loop else 0)
    print()
    print_scale_drift_diagnostic(cameras, points_world, point_frame)
    print()
    print_depth_buckets(points_world[:, 2], point_frame)
    plot_reconstruction(points_world, point_frame, cameras,
                        f"{tag}: incremental SfM ({'loop closure' if use_loop else 'no loop'})",
                        OUT_DIR / f"{tag}_colored_by_time.png")


if __name__ == "__main__":
    main()
