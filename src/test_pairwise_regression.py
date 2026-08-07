"""Regression check: the refactored estimate_pairwise_pose() should reproduce
the published ex2/report.tex numbers on ex2's own stereo pair. This is the
Stage 1 "verifiable result" we can check today, before any new footage
exists: keypoints ~13934/11488, matches ~1714, RANSAC inliers ~1144 (66.74%).
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from calibration import K_REFERENCE
from pairwise_pose import detect_and_match, estimate_pairwise_pose

EX2_DIR = Path(__file__).parent.parent.parent / "ex2"

img1 = cv2.imread(str(EX2_DIR / "stereo" / "1.jpeg"))
img2 = cv2.imread(str(EX2_DIR / "stereo" / "2.jpeg"))
assert img1 is not None and img2 is not None, "could not load ex2 stereo pair"
print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")

pts_a, pts_b, n_kp_a, n_kp_b = detect_and_match(img1, img2)
print(f"Keypoints: {n_kp_a} / {n_kp_b}  (report: 13934 / 11488)")
print(f"Good matches: {len(pts_a)}  (report: 1714)")

result = estimate_pairwise_pose(img1, img2, K_REFERENCE)
print(f"RANSAC inliers: {result.n_ransac_inliers} "
      f"({result.match_inlier_ratio:.2%})  (report: 1144, 66.74%)")
print(f"Rotation: {result.rotation_deg:.2f} deg")
print(f"Pose-consistent inliers: {result.n_pose_inliers}")

assert n_kp_a == 13934 and n_kp_b == 11488, "keypoint counts diverged from report"
assert len(pts_a) == 1714, "match count diverged from report"
assert result.n_ransac_inliers == 1144, "RANSAC inlier count diverged from report"
print("\nPASS: refactored pipeline reproduces ex2/report.tex exactly.")
