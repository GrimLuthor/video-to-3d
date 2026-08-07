"""Shared camera calibration for the S22 Ultra.

Calibrated directly in VIDEO mode (1920x1080, 16:9) via
calibrate_from_video.py, using a checkerboard displayed on a screen
(recorded with the same resolution/aspect/EIS-off settings as the real
street footage). Superseded the original ex2 still-photo calibration
(2000x1500, 4:3, Zhang's method, see ex2/report.tex) because phone video
mode uses a different sensor crop than stills -- fx/fy here (~1665) is
meaningfully higher than the still-photo K scaled to this resolution would
give (~1280), confirming the crop really does differ. Unlike the still
calibration (which found ~zero distortion, likely ISP-corrected for JPEG),
video mode shows small but real distortion. k3 is fixed to 0: an initial
free fit put it at an implausible ~-7 because the calibration board never
reached the frame's corners/edges, leaving k3 unconstrained; see
calibrate_from_video.py's automatic k3-plausibility check.
Reprojection error: 0.23px over 40 views (after blur/outlier filtering).
"""

import cv2
import numpy as np

CALIB_WIDTH = 1920
CALIB_HEIGHT = 1080

K_REFERENCE = np.array([
    [1663.1433,    0.0,  990.1991],
    [   0.0,    1669.1356, 559.3829],
    [   0.0,       0.0,      1.0],
])

DIST_COEFFS = np.array([0.02644756, -0.14776052, 0.01094967, 0.00394756, 0.0])


def scale_K(width, height, K=K_REFERENCE, calib_width=CALIB_WIDTH, calib_height=CALIB_HEIGHT):
    """Scale the reference K to a different resolution of the same aspect ratio.

    Raises if the aspect ratio doesn't match (in either orientation), since
    that means a different sensor crop/orientation was used and the
    reference K does not transfer without a fresh calibration or an
    explicit 90-degree correction (see `scale_K_rotated`).

    IMPORTANT: `width, height` must be the actual numpy array dimensions
    (frame.shape[1], frame.shape[0]) of the images you'll process, not an
    assumed label. ex2's own stereo pair is a cautionary example: the report
    describes them as "2000x1500" (landscape) but cv2.imread loads them as
    2000 tall x 1500 wide (portrait) — likely EXIF-rotation vs calibration
    orientation mismatch. It didn't break ex2's checkpoints (RANSAC is
    somewhat tolerant of a slightly-off principal point), but a wrong
    cx/cy is still a real bias on the recovered geometry. Always check
    frame.shape before trusting any K.
    """
    ref_aspect = calib_width / calib_height
    new_aspect = width / height
    if abs(ref_aspect - new_aspect) > 1e-2:
        raise ValueError(
            f"Aspect ratio mismatch: calibrated at {calib_width}x{calib_height} "
            f"(aspect {ref_aspect:.4f}) but got {width}x{height} (aspect "
            f"{new_aspect:.4f}). This likely means a different camera "
            f"mode/crop/orientation was used (e.g. 16:9 video vs 4:3 stills, "
            f"or portrait vs landscape) — the reference K does not transfer "
            f"as-is. If this is just a 90-degree rotation, use "
            f"scale_K_rotated(). Otherwise recalibrate for this mode."
        )
    scale = width / calib_width
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy
    return K_scaled


def scale_K_rotated(width, height, K=K_REFERENCE, calib_width=CALIB_WIDTH, calib_height=CALIB_HEIGHT):
    """Like scale_K, but for frames rotated 90 degrees relative to the
    calibration orientation (e.g. calibrated landscape, capturing portrait).
    Swaps fx/fy and cx/cy accordingly before scaling.
    """
    K_swapped = np.array([
        [K[1, 1], 0.0, K[1, 2]],
        [0.0, K[0, 0], K[0, 2]],
        [0.0, 0.0, 1.0],
    ])
    return scale_K(width, height, K=K_swapped, calib_width=calib_height, calib_height=calib_width)


def undistort_frames(frames, K, dist_coeffs=DIST_COEFFS):
    """Undistort a list of frames with the given (already-resolution-scaled)
    K. Passes K as its own new-camera-matrix, so K stays valid for the
    undistorted output (no crop/recentering) -- same approach ex2_plan.md
    describes ("undistort before anything else").

    No-ops (returns frames as-is) when dist_coeffs is all zero, since the
    ex2 still-photo calibration found none and callers may still be using
    that path.
    """
    if not np.any(dist_coeffs):
        return frames
    return [cv2.undistort(f, K, dist_coeffs) for f in frames]
