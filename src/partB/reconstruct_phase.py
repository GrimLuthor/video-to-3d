"""
Reconstruct a PHASE-OFFSET frame subset of one video, to get two independent
same-scene point clouds for the refinement experiment (Part B).

Phase-0 uses frames {0, stride, 2*stride, ...}; phase-k uses {k, k+stride, ...}.
The two subsets cover the same scene but each SfM seeds from a different frame,
so each lands in its own arbitrary coordinate frame + scale -- exactly the
refinement-registration setting, with guaranteed high overlap and near ground
truth (the two must coincide when correctly aligned).

Reuses the Part A modules read-only (extract nothing from run_incremental_sfm;
it hard-codes phase 0). No Part A files are modified.

    python Final/src/partB/reconstruct_phase.py columns 8 0 --window 10
    python Final/src/partB/reconstruct_phase.py columns 8 4 --window 10
    # -> output/columns_p0_reconstruction.pkl , output/columns_p4_reconstruction.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1]        # Final/src
sys.path.insert(0, str(SRC))
from feature_tracks import extract_features, match_windowed, build_tracks
from incremental_sfm import reconstruct
import calibration

OUT_DIR = Path(__file__).resolve().parents[2] / "output"
CAP_DIR = Path(__file__).resolve().parents[2] / "capture"


def extract_phase(video_path, stride, phase):
    """Frames where idx % stride == phase (a stride-strided subset offset by phase)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == phase:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames from {video_path} (stride {stride}, phase {phase})")
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video name (in capture/, with or without .mp4) or a path")
    ap.add_argument("stride", type=int)
    ap.add_argument("phase", type=int)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min-tri-angle", type=float, default=2.0)
    ap.add_argument("--pose-jump-guard", type=float, default=12.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    vp = Path(args.video)
    if not vp.exists():
        vp = CAP_DIR / (args.video if args.video.endswith(".mp4") else args.video + ".mp4")
    tag = args.tag or f"{vp.stem}_p{args.phase}"
    OUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    frames = extract_phase(vp, args.stride, args.phase)
    h, w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames {w}x{h} (stride {args.stride}, phase "
          f"{args.phase}) in {time.time()-t0:.1f}s")
    K = calibration.scale_K(w, h)
    frames = calibration.undistort_frames(frames, K)

    t0 = time.time()
    features = extract_features(frames)
    print(f"SIFT on {len(features)} frames in {time.time()-t0:.1f}s")
    t0 = time.time()
    verified = match_windowed(features, K, window=args.window)
    tracks = build_tracks(features, verified)
    print(f"Matching+tracks in {time.time()-t0:.1f}s ({len(tracks)} tracks)")

    print("\n" + "=" * 60 + "\nINCREMENTAL RECONSTRUCTION\n" + "=" * 60)
    t0 = time.time()
    cameras, points_world, observations, point_frame, frame_indices = reconstruct(
        features, tracks, K, window=args.window, min_tri_angle=args.min_tri_angle,
        pose_jump_guard=args.pose_jump_guard)
    print(f"Reconstruction in {time.time()-t0:.1f}s: {len(cameras)} cameras, "
          f"{len(points_world)} points")

    bundle = {
        "frame_indices": list(frame_indices),
        "cameras": [{"R": c["R"], "t": c["t"]} for c in cameras],
        "K": K,
        "points_world": points_world,
        "point_frame": point_frame,
        "observations": observations,
        "stride": args.stride,
        "phase": args.phase,
        "video_path": str(vp),
    }
    with open(OUT_DIR / f"{tag}_reconstruction.pkl", "wb") as f:
        pickle.dump(bundle, f)
    np.save(OUT_DIR / f"{tag}_camera_centers.npy",
            np.array([(-c["R"].T @ c["t"]).ravel() for c in cameras]))
    print(f"Saved {tag}_reconstruction.pkl  ("
          f"export a viewable cloud with:  python Final/src/partB/sparse_ply.py {tag})")


if __name__ == "__main__":
    main()
