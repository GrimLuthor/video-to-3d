"""Extends run_ex4_check.py: (1) print the *signed* per-pair rotation to
check whether the sloped panorama edges are a consistent drift (camera
roll/wobble) or scene-geometry confound, and (2) render a full 24-viewpoint
wiggle video for a visual parallax check.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import ex4

VIDEO_PATH = Path(__file__).parent.parent / "capture" / "20260705_141208.mp4"
OUT_DIR = Path(__file__).parent.parent / "output" / "ex4_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_FRAME = 0
END_FRAME = 300
NUM_VIEWPOINTS = 24
FPS = 10

t0 = time.time()
frames = ex4.load_video_frames(str(VIDEO_PATH), start_frame=START_FRAME, end_frame=END_FRAME)
print(f"Loaded {len(frames)} frames in {time.time()-t0:.1f}s")

frames = ex4.detect_and_fix_direction(frames)

t0 = time.time()
H_successive = []
for i in range(len(frames) - 1):
    H = ex4.estimate_horizon_homography(frames[i], frames[i + 1])
    H_successive.append(H)
print(f"Computed {len(H_successive)} pairwise homographies in {time.time()-t0:.1f}s")

# --- Signed rotation check (the actual question) ---
signed_rot_deg = [np.degrees(np.arctan2(H[1, 0], H[0, 0])) for H in H_successive]
cum_rot = np.cumsum(signed_rot_deg)
print("\n--- ROTATION / WOBBLE CHECK ---")
print(f"Per-pair rotation (deg): mean={np.mean(signed_rot_deg):+.4f} "
      f"std={np.std(signed_rot_deg):.4f} min={min(signed_rot_deg):+.3f} max={max(signed_rot_deg):.3f}")
print(f"Fraction of pairs with same sign as the mean: "
      f"{np.mean(np.sign(signed_rot_deg) == np.sign(np.mean(signed_rot_deg))):.0%}")
print(f"Cumulative rotation across window: {cum_rot[0]:+.3f} deg -> {cum_rot[-1]:+.3f} deg "
      f"(net drift: {cum_rot[-1]-cum_rot[0]:+.3f} deg over {len(H_successive)} pairs)")

ref_idx = len(frames) // 2
homographies = ex4.accumulate_homographies(H_successive, ref_idx)

t0 = time.time()
panoramas = ex4.generate_panoramas(frames, homographies, num_viewpoints=NUM_VIEWPOINTS,
                                    centered=True, use_pyramid_blending=True)
print(f"\nGenerated {len(panoramas)} panoramas in {time.time()-t0:.1f}s, size={panoramas[0].shape}")

padded = [ex4.pad_to_macro_block(p) for p in panoramas]
# ping-pong: forward then backward (excluding the two endpoints on the way back)
# for a smooth continuous wiggle instead of a hard jump-cut loop.
sequence = padded + padded[-2:0:-1]

h, w = padded[0].shape[:2]
out_path = OUT_DIR / "panorama_wiggle_24views.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
for frame in sequence:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()
print(f"Saved {len(sequence)}-frame wiggle video ({w}x{h} @ {FPS}fps) to {out_path}")
