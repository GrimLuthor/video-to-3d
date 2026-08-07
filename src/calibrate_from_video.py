"""Zhang's-method camera calibration from a video of a screen-displayed
checkerboard (see generate_checkerboard.py) instead of a printed one.

Does this specifically in VIDEO mode -- extracting frames from a video shot
with the same resolution/aspect ratio/stabilization settings as the real
street footage -- since ex2's calibration.py K was derived from still
photos, and phone video mode can use a different sensor crop (see the
caveat in plan.md and the "K is still ... provisional" note in
run_full_clip_chain.py).

Two-pass fit: a first calibrateCamera() pass is only used to score each
view's reprojection error; views that are motion-blurred or otherwise bad
get dropped, and the final K/dist_coeffs come from refitting on the
survivors. This matters because a handheld phone video will always include
some blurry frames mixed in with the good ones, and just a handful of those
is enough to drag distortion coefficients (especially k3) to implausible
values while barely denting the overall reprojection error.

Usage:
    python calibrate_from_video.py <calibration_video_path> --square-size-mm <mm> [--cols 9] [--rows 6]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_frames import extract_frames

OUT_DIR = Path(__file__).parent.parent / "output"
DEBUG_DIR = OUT_DIR / "calibration_debug"

MAX_DETECTIONS = 40  # cap for calibrateCamera speed; evenly subsampled if exceeded
BLUR_DROP_FRACTION = 0.3  # drop the blurriest 30% of candidates before subsampling
OUTLIER_ERROR_MULTIPLIER = 2.0  # drop views with per-view error > this * median
MIN_VIEWS = 12
MAX_PLAUSIBLE_K3 = 1.0  # |k3| beyond this triggers a refit with k3 fixed to 0


def sharpness_score(gray, corners):
    """Variance of the Laplacian over the checkerboard's bounding box (with
    a small margin) -- a standard blur proxy. Higher = sharper.
    """
    x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
    pad = int(0.15 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return cv2.Laplacian(crop, cv2.CV_64F).var()


def find_corners(frames, cols, rows):
    """Runs findChessboardCorners + cornerSubPix on every frame. Returns
    the successfully-detected corner arrays, their frame indices, and a
    blur-proxy sharpness score per detection.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    img_points = []
    used_indices = []
    sharpness = []

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)
        used_indices.append(i)
        sharpness.append(sharpness_score(gray, corners))
        print(f"  frame {i}: corners found ({len(img_points)} total)")

    return img_points, used_indices, sharpness


def drop_blurriest(used_indices, img_points, sharpness, fraction):
    n_keep = max(MIN_VIEWS, int(round(len(sharpness) * (1 - fraction))))
    if n_keep >= len(sharpness):
        return used_indices, img_points
    order = np.argsort(sharpness)[::-1][:n_keep]  # sharpest first
    order = sorted(order)  # restore time order for even subsampling later
    return [used_indices[i] for i in order], [img_points[i] for i in order]


def subsample_evenly(items, max_n):
    if len(items) <= max_n:
        return items
    idx = np.linspace(0, len(items) - 1, max_n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def per_view_errors(obj_points, img_points, rvecs, tvecs, K, dist_coeffs):
    errors = []
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_coeffs)
        errors.append(float(np.linalg.norm(imgp - projected, axis=2).mean()))
    return np.array(errors)


def save_debug_images(frames, used_indices, img_points, cols, rows, n=6):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    step = max(1, len(used_indices) // n)
    for k in range(0, len(used_indices), step):
        frame_idx = used_indices[k]
        vis = frames[frame_idx].copy()
        cv2.drawChessboardCorners(vis, (cols, rows), img_points[k], True)
        cv2.imwrite(str(DEBUG_DIR / f"detected_{frame_idx:05d}.png"), vis)
    print(f"Saved detection debug images to {DEBUG_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--square-size-mm", type=float, required=True,
                         help="Physical size of one checkerboard square, "
                              "measured with a ruler on the screen.")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners, horizontal")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners, vertical")
    parser.add_argument("--stride", type=int, default=3,
                         help="Frame stride for extraction (calibration clips "
                              "are short, so a small stride is fine).")
    args = parser.parse_args()

    print(f"Extracting frames from {args.video_path} (stride={args.stride})...")
    frames = extract_frames(args.video_path, stride=args.stride)
    h, w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames, each {w}x{h} (w x h)")
    print("Confirm this matches the resolution/orientation of your real "
          "street footage before trusting the result.\n")

    print("Detecting checkerboard corners...")
    img_points, used_indices, sharpness = find_corners(frames, args.cols, args.rows)

    if len(img_points) < 10:
        print(f"\nOnly {len(img_points)} successful detections -- too few for "
              f"a reliable calibration (want 15+). Retake the calibration "
              f"video: fill more of the frame with the board, reduce motion "
              f"blur, and cover more angles/positions.")
        sys.exit(1)

    n_before_blur_filter = len(img_points)
    used_indices, img_points = drop_blurriest(used_indices, img_points, sharpness, BLUR_DROP_FRACTION)
    print(f"\nDropped blurriest candidates: {n_before_blur_filter} -> {len(img_points)} "
          f"(keeping the sharpest ~{int((1 - BLUR_DROP_FRACTION) * 100)}%)")

    if len(img_points) > MAX_DETECTIONS:
        paired = list(zip(used_indices, img_points))
        paired = subsample_evenly(paired, MAX_DETECTIONS)
        used_indices, img_points = zip(*paired)
        used_indices, img_points = list(used_indices), list(img_points)
        print(f"Subsampled to {len(img_points)} evenly-spaced (in time) detections "
              f"for calibrateCamera.")

    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_size_mm
    obj_points = [objp] * len(img_points)

    print(f"\nPass 1: cv2.calibrateCamera on {len(img_points)} detections...")
    ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    print(f"  Pass 1 reprojection error: {ret1:.4f} px")

    errors = per_view_errors(obj_points, img_points, rvecs1, tvecs1, K1, dist1)
    median_err = np.median(errors)
    keep_mask = errors <= median_err * OUTLIER_ERROR_MULTIPLIER
    n_keep = max(MIN_VIEWS, keep_mask.sum())
    if keep_mask.sum() < MIN_VIEWS:
        # keep the best MIN_VIEWS views instead of an empty/too-small set
        keep_idx = np.argsort(errors)[:MIN_VIEWS]
        keep_mask = np.zeros(len(errors), dtype=bool)
        keep_mask[keep_idx] = True

    n_dropped = (~keep_mask).sum()
    print(f"  Per-view error: median={median_err:.3f}px, "
          f"dropping {n_dropped} outlier view(s) with error > "
          f"{OUTLIER_ERROR_MULTIPLIER}x median")

    final_obj_points = [o for o, k in zip(obj_points, keep_mask) if k]
    final_img_points = [p for p, k in zip(img_points, keep_mask) if k]
    final_used_indices = [idx for idx, k in zip(used_indices, keep_mask) if k]

    print(f"\nPass 2 (refit on survivors): cv2.calibrateCamera on "
          f"{len(final_img_points)} detections...")
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        final_obj_points, final_img_points, (w, h), None, None
    )

    # A free-fit k3 easily blows up to an implausible value when the board
    # never covers the far corners/edges of the frame (where distortion is
    # actually measurable) -- the mean reprojection error barely notices
    # since it's dominated by the well-covered central region. This phone's
    # still-photo calibration (ex2/report.tex) found ~zero distortion, so a
    # |k3| this large is a fit artifact, not real optics. Refit with k3
    # pinned to 0 rather than trusting it.
    k3 = dist_coeffs.ravel()[4]
    if abs(k3) > MAX_PLAUSIBLE_K3:
        print(f"\nk3={k3:.3f} is implausible for this phone (still-photo calib "
              f"found ~zero distortion) -- likely unconstrained due to the "
              f"board never reaching the frame's corners/edges. Refitting "
              f"with k3 fixed to 0...")
        ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            final_obj_points, final_img_points, (w, h), None, None,
            flags=cv2.CALIB_FIX_K3
        )
        print(f"  Refit reprojection error: {ret:.4f} px")

    print("\n" + "=" * 60)
    print("CALIBRATION RESULT (after outlier removal)")
    print("=" * 60)
    print(f"Reprojection error: {ret:.4f} px "
          f"({'OK, < 0.5px' if ret < 0.5 else 'HIGH -- retake calibration video' if ret > 1.0 else 'borderline, consider retaking'})")
    print(f"Resolution: {w}x{h}")
    print("K =")
    print(K)
    print(f"dist_coeffs = {dist_coeffs.ravel()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / "video_calibration_result.npz"
    np.savez(npz_path, K=K, dist_coeffs=dist_coeffs, width=w, height=h,
              reprojection_error=ret)
    print(f"\nSaved raw result to {npz_path}")

    save_debug_images(frames, final_used_indices, final_img_points, args.cols, args.rows)

    print("\nTo use this in calibration.py, replace K_REFERENCE / "
          "CALIB_WIDTH / CALIB_HEIGHT / DIST_COEFFS with:")
    print(f"""
CALIB_WIDTH = {w}
CALIB_HEIGHT = {h}

K_REFERENCE = np.array([
    [{K[0,0]:.4f}, {K[0,1]:.4f}, {K[0,2]:.4f}],
    [{K[1,0]:.4f}, {K[1,1]:.4f}, {K[1,2]:.4f}],
    [{K[2,0]:.4f}, {K[2,1]:.4f}, {K[2,2]:.4f}],
])

DIST_COEFFS = np.array({list(dist_coeffs.ravel())})
""")


if __name__ == "__main__":
    main()
