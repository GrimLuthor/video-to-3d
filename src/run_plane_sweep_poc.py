"""Stage 3 proof-of-concept: dense depth for ONE reference view via plane-sweep
MVS, on top of the Stage 1/2 reconstruction bundle.

Loads <tag>_reconstruction.pkl (poses + frame indices + K + sparse points),
re-extracts and undistorts the frames, picks a reference camera + a few
temporal-neighbour source cameras, bounds the depth sweep with the sparse
cloud, runs plane_sweep.plane_sweep, and saves a reference / depth / confidence
figure for a sanity check before scaling to all views + fusion (Stage 4).

Usage:
    python run_plane_sweep_poc.py [tag] [--ref-cam N] [--scale S] [--planes D]
                                  [--window W] [--neighbors K]
"""

import argparse
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

OUT_DIR = Path(__file__).parent.parent / "output"


def scale_K(K, s):
    Ks = K.copy()
    Ks[0, 0] *= s; Ks[1, 1] *= s; Ks[0, 2] *= s; Ks[1, 2] *= s
    return Ks


def pick_neighbors(frame_indices, ref_cam, n_neighbors, min_gap=2, max_gap=7):
    """Cameras whose frame index is a moderate baseline away from the reference
    (skip ~adjacent frames: too little parallax; skip far frames: occlusion /
    appearance change). Returns camera-order indices, closest gaps first."""
    f_ref = frame_indices[ref_cam]
    cands = [(abs(frame_indices[c] - f_ref), c) for c in range(len(frame_indices))
             if c != ref_cam and min_gap <= abs(frame_indices[c] - f_ref) <= max_gap]
    cands.sort()
    return [c for _, c in cands[:n_neighbors]]


def pick_neighbors_multibaseline(frame_indices, ref_cam, target_gaps=(1, 2, 4, 7)):
    """Source views spanning a range of baselines on BOTH sides of the reference.
    A single baseline can't serve the whole frame: near objects (a bin you walk
    past) need TIGHT baselines (small appearance change) while the far wall needs
    WIDE baselines (depth precision). Supplying both lets the robust best-K cost
    aggregation pick, per pixel, whichever baseline actually matches -- tight
    wins on near content, wide wins on far. Returns camera-order indices."""
    f_ref = frame_indices[ref_cam]
    chosen, result = set(), []
    for g in target_gaps:
        for sign in (+1, -1):
            target = f_ref + sign * g
            best, best_d = None, 1e9
            for c in range(len(frame_indices)):
                if c == ref_cam or c in chosen:
                    continue
                d = abs(frame_indices[c] - target)
                if d < best_d and abs(frame_indices[c] - f_ref) >= 1:
                    best_d, best = d, c
            if best is not None and best_d <= 1:  # a camera actually near this gap
                chosen.add(best); result.append(best)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag", nargs="?", default="longer_walk_incr")
    p.add_argument("--ref-cam", type=int, default=None, help="camera-order index; default = middle")
    p.add_argument("--scale", type=float, default=0.5, help="working-resolution downscale")
    p.add_argument("--planes", type=int, default=96)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--neighbors", type=int, default=6)
    p.add_argument("--min-gap", type=int, default=2, help="min source-view frame gap (baseline)")
    p.add_argument("--max-gap", type=int, default=7, help="max source-view frame gap (baseline)")
    p.add_argument("--multibaseline", action="store_true",
                   help="use both tight and wide source baselines (near+far coverage)")
    p.add_argument("--cost-agg-radius", type=int, default=6,
                   help="edge-aware cost aggregation radius (0 = raw winner-take-all)")
    p.add_argument("--patchmatch", action="store_true",
                   help="use PatchMatch depth (continuous + propagation) instead of plane sweep")
    p.add_argument("--out-suffix", default="", help="appended to the output filename")
    args = p.parse_args()

    with open(OUT_DIR / f"{args.tag}_reconstruction.pkl", "rb") as f:
        b = pickle.load(f)
    cameras, frame_indices, K_full = b["cameras"], b["frame_indices"], b["K"]
    points_world = b["points_world"]
    ref_cam = args.ref_cam if args.ref_cam is not None else len(cameras) // 2
    print(f"{len(cameras)} cameras; reference = camera {ref_cam} (video frame {frame_indices[ref_cam]})")

    # frames must be undistorted with the SAME K the poses were computed on
    frames = extract_frames(Path(b["video_path"]), stride=b["stride"])
    frames = calibration.undistort_frames(frames, K_full)

    s = args.scale
    K = scale_K(K_full, s)

    def gray(cam_order):
        img = frames[frame_indices[cam_order]]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ref_gray = gray(ref_cam)
    if args.multibaseline:
        nbrs = pick_neighbors_multibaseline(frame_indices, ref_cam)
    else:
        nbrs = pick_neighbors(frame_indices, ref_cam, args.neighbors,
                              min_gap=args.min_gap, max_gap=args.max_gap)
    if not nbrs:
        raise SystemExit("no suitable neighbour cameras found")
    print(f"source cameras: {nbrs} (video frames {[frame_indices[c] for c in nbrs]})")
    source_grays = [gray(c) for c in nbrs]
    cam_srcs = [cameras[c] for c in nbrs]

    d_min, d_max = ps.depth_range_from_points(points_world, cameras[ref_cam], K, ref_gray.shape)
    print(f"depth range from sparse cloud: [{d_min:.2f}, {d_max:.2f}]")

    t0 = time.time()
    if args.patchmatch:
        depth_map, conf = ps.patchmatch_depth(
            ref_gray, source_grays, K, cameras[ref_cam], cam_srcs, d_min, d_max,
            window=args.window, verbose=True)
        method = "PatchMatch"
    else:
        depths = ps.depth_hypotheses(d_min, d_max, args.planes)
        depth_map, conf, _ = ps.plane_sweep(
            ref_gray, source_grays, K, cameras[ref_cam], cam_srcs, depths,
            window=args.window, cost_agg_radius=args.cost_agg_radius)
        method = f"plane-sweep ({args.planes} planes)"
    valid = np.isfinite(depth_map)
    print(f"{method} in {time.time()-t0:.1f}s; valid depth at {valid.mean():.1%} of pixels")

    _visualize(frames[frame_indices[ref_cam]], depth_map, conf, args.tag, ref_cam,
               args.out_suffix)


def _visualize(ref_bgr, depth_map, conf, tag, ref_cam, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    ax[0].imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)); ax[0].set_title("reference frame")
    im1 = ax[1].imshow(depth_map, cmap="turbo"); ax[1].set_title("depth (all valid pixels)")
    plt.colorbar(im1, ax=ax[1], shrink=0.7)

    # confidence shown as OPACITY (not a hard mask): low-confidence pixels fade
    # but stay visible over the grayscale frame -- so we can see the raw signal
    # is present, rather than discarding it.
    # background gray must match the (downscaled) depth-map resolution, else the
    # overlay is stretched and misaligned.
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_AREA)
    ax[2].imshow(gray, cmap="gray")
    denom = np.nanpercentile(conf, 95) or 1.0
    alpha = np.clip(conf / denom, 0, 1)
    alpha[~np.isfinite(depth_map)] = 0
    im2 = ax[2].imshow(depth_map, cmap="turbo", alpha=alpha)
    ax[2].set_title("depth, opacity = confidence")
    plt.colorbar(im2, ax=ax[2], shrink=0.7)
    for a in ax:
        a.axis("off")
    out = OUT_DIR / f"{tag}_planesweep_cam{ref_cam}{suffix}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120); print(f"Saved {out}")


if __name__ == "__main__":
    main()
