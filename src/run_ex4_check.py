"""One-off sanity check: run the ex4 stereo-mosaicing pipeline (from last
semester, in Final/ex4.py) on the newly captured video, just to eyeball
whether the motion/parallax is good before investing in calibration + the
Stage 1 SfM pipeline. Not part of the Stage 1/2/3 pipeline itself.
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import ex4

VIDEO_PATH = Path(__file__).parent.parent / "capture" / "20260705_141208.mp4"
OUT_DIR = Path(__file__).parent.parent / "output" / "ex4_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use a limited window of the video to keep runtime/memory reasonable for a
# quick check -- full-resolution 1920x1080, first ~10s.
START_FRAME = 0
END_FRAME = 300
NUM_VIEWPOINTS = 8

t0 = time.time()
frames = ex4.load_video_frames(str(VIDEO_PATH), start_frame=START_FRAME, end_frame=END_FRAME)
print(f"Loaded {len(frames)} frames in {time.time()-t0:.1f}s, shape={frames[0].shape}")

frames = ex4.detect_and_fix_direction(frames)

t0 = time.time()
H_successive = []
for i in range(len(frames) - 1):
    H = ex4.estimate_horizon_homography(frames[i], frames[i + 1])
    H_successive.append(H)
print(f"Computed {len(H_successive)} pairwise homographies in {time.time()-t0:.1f}s")

ref_idx = len(frames) // 2
homographies = ex4.accumulate_homographies(H_successive, ref_idx)

# Report net translation across the whole window -- the actual "how much
# parallax do we have" number.
net_dx = homographies[-1][0, 2] - homographies[0][0, 2]
print(f"Net horizontal shift across window (reference-frame coords): {net_dx:.1f} px")
per_pair_dx = [H[0, 2] for H in H_successive]
print(f"Per-pair dx: min={min(per_pair_dx):.2f} max={max(per_pair_dx):.2f} "
      f"mean={np.mean(per_pair_dx):.2f} std={np.std(per_pair_dx):.2f}")

t0 = time.time()
panoramas = ex4.generate_panoramas(frames, homographies, num_viewpoints=NUM_VIEWPOINTS,
                                    centered=True, use_pyramid_blending=True)
print(f"Generated {len(panoramas)} panoramas in {time.time()-t0:.1f}s, "
      f"size={panoramas[0].shape}")

for i, pano in enumerate(panoramas):
    Image.fromarray(pano).save(OUT_DIR / f"viewpoint_{i:02d}.png")
print(f"Saved {len(panoramas)} viewpoint panoramas to {OUT_DIR}")
