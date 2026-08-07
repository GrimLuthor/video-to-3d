"""Stage 1 end-to-end runner: video -> frames -> chained poses -> checkpoints.

Usage:
    python run_stage1_pilot.py <pilot_video_path> [stride]

Prints the Stage 1 step 1.3/1.5 go/no-go checkpoints from plan.md and saves
a trajectory + point cloud plot to Final/output/.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibration import K_REFERENCE, scale_K, scale_K_rotated, undistort_frames
from extract_frames import extract_frames
from chain_poses import chain_poses

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def resolve_K(width, height):
    """Try landscape scaling first, then a 90-degree rotated fit, else raise."""
    try:
        return scale_K(width, height)
    except ValueError:
        return scale_K_rotated(width, height)


def main(video_path, stride=5):
    frames = extract_frames(video_path, stride=stride)
    h, w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames at {w}x{h} (w x h), stride={stride}")

    K = resolve_K(w, h)
    print(f"Using K:\n{K}")
    frames = undistort_frames(frames, K)

    cameras, points_world, diagnostics, observations = chain_poses(frames, K)

    print("\n--- STEP 1.3 / 1.5 CHECKPOINTS ---")
    ok_pairs = [d for d in diagnostics if d.get("status") == "ok"]
    failed_pairs = [d for d in diagnostics if d.get("status") != "ok"]
    if failed_pairs:
        print(f"WARNING: {len(failed_pairs)} pair(s) failed pose estimation "
              f"(chain stopped early): {failed_pairs}")

    inlier_ratios = [d["match_inlier_ratio"] for d in ok_pairs]
    rotations = [d["rotation_deg"] for d in ok_pairs]
    scales = [d["scale"] for d in ok_pairs]
    n_common = [d["n_common_points_used"] for d in ok_pairs]

    print(f"Pairs processed: {len(ok_pairs)} / {len(frames) - 1}")
    print(f"Median inlier ratio: {np.median(inlier_ratios):.2%}  (want > 30%)")
    print(f"Median rotation/pair: {np.median(rotations):.2f} deg  "
          f"(want small & consistent for consecutive video frames)")
    print(f"Median common points used for scale fit: {np.median(n_common):.0f} "
          f"(want >= 6; below that, scale was carried forward, not re-fit)")
    print(f"Scale factors per pair (should be roughly stable, not wildly "
          f"jumping): {[f'{s:.3f}' for s in scales]}")
    print(f"Recovered point cloud size: {len(points_world)} points")

    if np.median(inlier_ratios) < 0.30:
        print("FAIL: inlier ratio below 30% -- check Step 1.0 (K/FOV match) "
              "or recapture more smoothly.")
    if np.median(rotations) > 15:
        print("WARNING: rotation per pair looks large for consecutive video "
              "frames -- check camera was translating, not rotating.")

    # Plot trajectory + point cloud for visual go/no-go (step 1.5).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    centers = np.array([(-cam["R"].T @ cam["t"]).ravel() for cam in cameras])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points_world) > 0:
        depths = points_world[:, 2]
        thresh = np.percentile(np.abs(depths), 95)
        mask = np.abs(depths) < thresh
        ax.scatter(points_world[mask, 0], points_world[mask, 1],
                   points_world[mask, 2], s=1, alpha=0.5, label="points")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "r-o",
            markersize=3, label="camera trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Stage 1 pilot: chained trajectory + sparse point cloud")
    out_path = OUTPUT_DIR / "stage1_pilot_reconstruction.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved trajectory/point-cloud plot to {out_path}")
    print("Visually check: does the trajectory roughly match the path you "
          "walked/drove? Does the point cloud show recognizable structure?")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <pilot_video_path> [stride]")
        sys.exit(1)
    video_path = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(video_path, stride)
