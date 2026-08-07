"""Stage 3 step: cross-view geometric-consistency validation.

Computes a permissive (aggregation-off) depth map for a reference view AND for
several neighbour views, then keeps only the reference depths that >= min_support
neighbours independently agree with (plane_sweep.geometric_consistency). This is
the denoiser for large low-confidence regions: the horizontal aperture strip
(a per-view artifact) has no cross-view agreement and is dropped, while genuine
geometry recovered by multiple views survives even where single-view confidence
was low.

Saves a reference / raw-depth / consistency-filtered figure for comparison.

Usage:
    python run_consistency_poc.py [tag] [--ref-cam N] [--scale S] [--planes D]
                                  [--views V] [--min-support M] [--rel-thresh R]
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
from run_plane_sweep_poc import scale_K, pick_neighbors

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag", nargs="?", default="longer_walk_incr")
    p.add_argument("--ref-cam", type=int, default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--planes", type=int, default=96)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--views", type=int, default=4, help="# neighbour views to cross-check against")
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--rel-thresh", type=float, default=0.02)
    args = p.parse_args()

    with open(OUT_DIR / f"{args.tag}_reconstruction.pkl", "rb") as f:
        b = pickle.load(f)
    cameras, frame_indices, K_full = b["cameras"], b["frame_indices"], b["K"]
    points_world = b["points_world"]
    ref_cam = args.ref_cam if args.ref_cam is not None else len(cameras) // 2

    frames = extract_frames(Path(b["video_path"]), stride=b["stride"])
    frames = calibration.undistort_frames(frames, K_full)
    s = args.scale
    K = scale_K(K_full, s)

    def gray(cam_order):
        img = frames[frame_indices[cam_order]]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def compute_depth(cam_order):
        """Permissive (aggregation-off) plane-sweep depth for one view, using
        its own temporal-neighbour source views."""
        ref_gray = gray(cam_order)
        nbrs = pick_neighbors(frame_indices, cam_order, 6)
        d_min, d_max = ps.depth_range_from_points(points_world, cameras[cam_order], K, ref_gray.shape)
        depths = ps.depth_hypotheses(d_min, d_max, args.planes)
        depth, _, _ = ps.plane_sweep(
            ref_gray, [gray(c) for c in nbrs], K, cameras[cam_order],
            [cameras[c] for c in nbrs], depths, window=args.window,
            cost_agg_radius=0, speckle_win=0)  # keep it truthful; no smoothing
        return depth

    check_cams = pick_neighbors(frame_indices, ref_cam, args.views)
    print(f"reference cam {ref_cam} (frame {frame_indices[ref_cam]}); "
          f"cross-checking against cams {check_cams} (frames {[frame_indices[c] for c in check_cams]})")

    t0 = time.time()
    ref_depth = compute_depth(ref_cam)
    neighbor_depths = [compute_depth(c) for c in check_cams]
    print(f"computed {1+len(check_cams)} depth maps in {time.time()-t0:.1f}s")

    filtered, support = ps.geometric_consistency(
        ref_depth, cameras[ref_cam], neighbor_depths, [cameras[c] for c in check_cams],
        K, rel_thresh=args.rel_thresh, min_support=args.min_support)

    raw_valid = np.isfinite(ref_depth).mean()
    kept = np.isfinite(filtered).mean()
    print(f"raw valid {raw_valid:.1%} -> consistency-filtered {kept:.1%} "
          f"(>= {args.min_support} of {len(check_cams)} views agree, rel_thresh {args.rel_thresh})")

    _visualize(frames[frame_indices[ref_cam]], ref_depth, filtered, support,
               len(check_cams), args.min_support, args.tag, ref_cam)


def _visualize(ref_bgr, ref_depth, filtered, support, n_views, min_support, tag, ref_cam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = np.nanpercentile(ref_depth, 2)
    vmax = np.nanpercentile(ref_depth, 98)
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    ax[0].imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)); ax[0].set_title("reference frame")
    ax[1].imshow(ref_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax[1].set_title("raw single-view depth (permissive)")
    im2 = ax[2].imshow(filtered, cmap="turbo", vmin=vmin, vmax=vmax)
    ax[2].set_title(f"cross-view consistency-filtered (>= {min_support}/{n_views} agree)")
    plt.colorbar(im2, ax=ax[2], shrink=0.7)
    for a in ax:
        a.axis("off")
    out = OUT_DIR / f"{tag}_consistency_cam{ref_cam}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120); print(f"Saved {out}")


if __name__ == "__main__":
    main()
