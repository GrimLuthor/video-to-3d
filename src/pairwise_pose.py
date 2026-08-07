"""Pairwise two-view pose estimation, refactored from ex2 steps 2-5 into a
reusable function so it can be applied to every consecutive frame pair in a
video sequence instead of a single stereo pair.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PairwisePoseResult:
    R: np.ndarray                # 3x3 relative rotation, cam_a -> cam_b
    t: np.ndarray                 # 3x1 relative translation, unit norm (scale unknown)
    pts_a_inliers: np.ndarray     # Nx2, pose-consistent inlier points in image A
    pts_b_inliers: np.ndarray     # Nx2, pose-consistent inlier points in image B
    n_matches: int                # good matches after ratio test
    n_ransac_inliers: int         # inliers from findEssentialMat RANSAC
    n_pose_inliers: int           # inliers cheirality-checked by recoverPose
    rotation_deg: float

    @property
    def match_inlier_ratio(self):
        return self.n_ransac_inliers / self.n_matches if self.n_matches else 0.0


def detect_and_match(img_a, img_b, ratio_thresh=0.75):
    """SIFT detection + Lowe ratio-test matching. Same as ex2 step2."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des_a, des_b, k=2)

    good = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance]

    pts_a = np.float64([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float64([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b, len(kp_a), len(kp_b)


def hartley_normalize(pts):
    """Same as ex2 step3."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    normalized = (T @ pts_h.T).T
    return normalized[:, :2], T


def estimate_pairwise_pose(img_a, img_b, K, ratio_thresh=0.75):
    """Full ex2 steps 2-5 pipeline (matching -> normalize -> RANSAC+E -> pose)
    applied to one image pair. Normalization (step 3) is computed for
    numerical-conditioning parity with ex2, even though cv2.findEssentialMat
    operates on the raw pixel points internally.

    Returns a PairwisePoseResult, or None if there aren't enough matches to
    attempt pose estimation.
    """
    pts_a, pts_b, n_kp_a, n_kp_b = detect_and_match(img_a, img_b, ratio_thresh)
    n_matches = len(pts_a)
    if n_matches < 8:
        return None

    _pts_a_norm, _T_a = hartley_normalize(pts_a)
    _pts_b_norm, _T_b = hartley_normalize(pts_b)

    E, mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    pts_a_in = pts_a[inlier_mask]
    pts_b_in = pts_b[inlier_mask]
    n_ransac_inliers = int(inlier_mask.sum())
    if n_ransac_inliers < 5:
        return None

    n_pose_inliers, R, t, pose_mask = cv2.recoverPose(E, pts_a_in, pts_b_in, K)
    pose_mask = pose_mask.ravel().astype(bool)

    rotation_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    return PairwisePoseResult(
        R=R,
        t=t,
        pts_a_inliers=pts_a_in[pose_mask],
        pts_b_inliers=pts_b_in[pose_mask],
        n_matches=n_matches,
        n_ransac_inliers=n_ransac_inliers,
        n_pose_inliers=int(pose_mask.sum()),
        rotation_deg=float(rotation_deg),
    )
