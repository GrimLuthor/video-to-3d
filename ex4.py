"""
Exercise 4: Stereo Mosaicing

Implements the stereo mosaicing algorithm with:
- Robust homography estimation (horizon pairs method)
- Barcode blending (Peleg et al.)
- Laplacian pyramid blending for seamless transitions
"""

import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from PIL import Image

###############################################################
###############################################################
###############################################################
DIRECTION_FRAME_GAP = 8


def estimate_horizon_homography(img1, img2):
    """
    Estimates rigid homography between two frames using Lucas-Kanade & Shi-Tomasi.
    
    Implements robust "Horizon Slope" logic:
    1. Detect many features (Shi-Tomasi)
    2. Track with Lucas-Kanade optical flow
    3. Filter out vertical motion (handles waterfalls/dynamic scenes)
    4. Estimate rotation from horizontally-distant point pairs
    5. Use median statistics for robustness (implicit RANSAC)
    
    Returns:
        3x3 homography matrix (rigid: rotation + translation only)
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # 1. Shi-Tomasi Corner Detection (many points, will filter heavily)
    feature_params = dict(
        maxCorners=2000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )
    
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    if p0 is None:
        return np.eye(3)

    # 2. Lucas-Kanade Optical Flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    
    # Select good points (status == 1 means LK found the point)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) < 10:
        return np.eye(3)

    h, w = img1.shape[:2]
    
    # 3. Vertical Filter (handles dynamic scenes like waterfalls)
    # Reject tracks with significant vertical motion
    deltas = good_new - good_old
    max_vertical_drift = h * 0.02 
    
    stable_mask = np.abs(deltas[:, 1]) < max_vertical_drift
    
    if np.sum(stable_mask) > 10:
        src_pts = good_old[stable_mask]
        dst_pts = good_new[stable_mask]
    else:
        # Fallback if too few points remain
        src_pts = good_old
        dst_pts = good_new

    # 4. Horizon Pairs Logic - estimate rotation from distant point pairs
    n_samples = min(len(src_pts), 300)
    indices = np.random.choice(len(src_pts), n_samples, replace=False)
    
    rotations = []
    
    min_dist_x = w * 0.25  # Points must be horizontally distant
    max_dist_y = h * 0.10  # But vertically close (horizon-like)
    
    for i in range(len(indices) - 1):
        idx_a = indices[i]
        for j in range(i + 1, min(i + 5, len(indices))): 
            idx_b = indices[j]
            
            p1_a, p1_b = src_pts[idx_a], src_pts[idx_b]
            p2_a, p2_b = dst_pts[idx_a], dst_pts[idx_b]
            
            dx = abs(p1_a[0] - p1_b[0])
            dy = abs(p1_a[1] - p1_b[1])
            
            if dx > min_dist_x and dy < max_dist_y:
                # Calculate angle change between point pairs
                angle1 = np.arctan2(p1_b[1] - p1_a[1], p1_b[0] - p1_a[0])
                angle2 = np.arctan2(p2_b[1] - p2_a[1], p2_b[0] - p2_a[0])
                rotations.append(angle2 - angle1)

    # 5. Robust Median Rotation
    if len(rotations) > 0:
        best_rotation = np.median(rotations)
        # Safety clamp to prevent flipping (max 3 degrees)
        if abs(best_rotation) > np.radians(3.0): 
            best_rotation = 0.0
    else:
        best_rotation = 0.0

    # 6. Construct Final Homography Matrix
    cos_t = np.cos(best_rotation)
    sin_t = np.sin(best_rotation)
    
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])
    
    # Apply rotation to source points, then find median translation
    src_hom = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    src_rotated = (R @ src_hom.T).T[:, :2]
    
    translations = dst_pts - src_rotated
    tx = np.median(translations[:, 0])
    ty = np.median(translations[:, 1])
    
    H_final = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0,      0,     1]
    ])
    
    return H_final


def detect_and_fix_direction(frames, num_samples=5):
    """
    Detects if video scans right-to-left and reverses frames if needed.
    
    Uses frame gap to ensure sufficient motion for reliable detection.
    Stereo mosaicing assumes left-to-right camera motion.
    
    Args:
        frames: List of video frames
        num_samples: Number of frame pairs to sample for direction estimation
        
    Returns:
        frames (possibly reversed) in left-to-right order
    """
    frame_gap = DIRECTION_FRAME_GAP
    step = max(1, (len(frames) - frame_gap) // (num_samples + 1))
    dx_values = []
    
    print(f"\n[Direction Detection] gap={frame_gap}, samples={num_samples}")
    
    for i in range(0, min(num_samples * step, len(frames) - frame_gap), step):
        H = estimate_horizon_homography(frames[i], frames[i + frame_gap])
        dx = H[0, 2]
        dx_values.append(dx)
    
    dx_avg = sum(dx_values) / len(dx_values)
    
    # Positive dx = scene moves right = camera moves right-to-left
    if dx_avg > 0:
        print(f"  -> R2L detected (avg dx={dx_avg:+.1f}), reversing frames")
        return frames[::-1]
    
    print(f"  -> L2R detected (avg dx={dx_avg:+.1f}), keeping order")
    return frames
###############################################################
###############################################################
###############################################################

class PyramidBlender:
    """
    Implements Laplacian Pyramid Blending.
    Used for Barcode Blending - blends odd and even mosaics seamlessly.
    
    Reference: Burt & Adelson, "A Multiresolution Spline", 1983
    """
    def __init__(self, levels=4):
        self.levels = levels

    def _get_gaussian_pyramid(self, img, levels):
        """Build Gaussian pyramid by repeated downsampling."""
        pyr = [img.astype(np.float32)]
        for _ in range(levels):
            img = cv2.pyrDown(img)
            pyr.append(img)
        return pyr

    def _get_laplacian_pyramid(self, gaussian_pyr):
        """Build Laplacian pyramid: L[i] = G[i] - expand(G[i+1])"""
        laplacian_pyr = []
        for i in range(len(gaussian_pyr) - 1):
            h, w = gaussian_pyr[i].shape[:2]
            upsampled = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=(w, h))
            laplacian = gaussian_pyr[i] - upsampled
            laplacian_pyr.append(laplacian)
        laplacian_pyr.append(gaussian_pyr[-1])  # Residual
        return laplacian_pyr

    def _reconstruct(self, laplacian_pyr):
        """Reconstruct image from Laplacian pyramid."""
        img = laplacian_pyr[-1]
        for i in range(len(laplacian_pyr) - 2, -1, -1):
            h, w = laplacian_pyr[i].shape[:2]
            img = cv2.pyrUp(img, dstsize=(w, h))
            img = img + laplacian_pyr[i]
        return np.clip(img, 0, 255).astype(np.uint8)

    def blend(self, imgA, imgB, mask):
        """
        Blend imgA and imgB using the mask with Laplacian pyramid blending.
        
        Args:
            imgA: First image (where mask = 1)
            imgB: Second image (where mask = 0)
            mask: Float mask [0, 1], single channel. 1 = imgA, 0 = imgB
            
        Returns:
            Blended image
        """
        # Generate Gaussian pyramids for images and mask
        gauss_pyr_A = self._get_gaussian_pyramid(imgA, self.levels)
        gauss_pyr_B = self._get_gaussian_pyramid(imgB, self.levels)
        gauss_pyr_mask = self._get_gaussian_pyramid(mask, self.levels)

        # Generate Laplacian pyramids
        lap_pyr_A = self._get_laplacian_pyramid(gauss_pyr_A)
        lap_pyr_B = self._get_laplacian_pyramid(gauss_pyr_B)

        # Blend at each pyramid level
        blend_pyr = []
        for lA, lB, m in zip(lap_pyr_A, lap_pyr_B, gauss_pyr_mask):
            # Expand mask dimensions to match image channels if necessary
            if len(m.shape) == 2 and len(lA.shape) == 3:
                m = m[:, :, np.newaxis]
            
            blended_level = lA * m + lB * (1.0 - m)
            blend_pyr.append(blended_level)

        return self._reconstruct(blend_pyr)


def load_video_frames(video_path, start_frame=0, end_frame=None, step=1):
    """
    Load frames from video file with optional subsampling.
    
    Args:
        video_path: Path to video file
        start_frame: First frame to load
        end_frame: Last frame to load (exclusive), None for all
        step: Take every 'step' frame
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    for i, frame in enumerate(iio.imiter(str(video_path))):
        if i < start_frame:
            continue
        if end_frame is not None and i >= end_frame:
            break
        if (i - start_frame) % step == 0:
            frames.append(frame)

    print(frames[0].shape)
    return frames


def load_frames_from_directory(input_frames_path):
    """
    Load frames from a directory containing frame_00000.jpg, frame_00001.jpg, etc.
    
    Args:
        input_frames_path: Path to directory containing frames
        
    Returns:
        List of frames as numpy arrays (RGB)
    """
    input_path = Path(input_frames_path)
    
    # Find all frame files
    frame_files = sorted(input_path.glob("frame_*.jpg"))
    
    if len(frame_files) == 0:
        raise ValueError(f"No frame files found in {input_frames_path}")
    
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        frame = np.array(img)
        frames.append(frame)
    
    return frames


def accumulate_homographies(H_successive, m):
    """
    Accumulate pairwise transforms to a reference frame.
    
    Args:
        H_successive: List of homographies H[i] mapping frame i to frame i+1
        m: Reference frame index
        
    Returns:
        List of homographies mapping each frame to reference frame m
    """
    mat_lst = [None] * (len(H_successive) + 1)
    
    # Forward: frames 0 to m-1
    forward_H = np.eye(3)
    for i in range(m - 1, -1, -1):
        forward_H = forward_H @ H_successive[i]
        mat_lst[i] = forward_H / forward_H[2, 2]
    
    # Backward: frames m+1 to end
    backward_H = np.eye(3)
    for j in range(m, len(H_successive)):
        backward_H = backward_H @ np.linalg.inv(H_successive[j])
        mat_lst[j + 1] = backward_H / backward_H[2, 2]
    
    mat_lst[m] = np.eye(3)
    return mat_lst


def compute_bounding_box(homography, w, h):
    """
    Compute axis-aligned bounding box of warped image.
    
    Args:
        homography: 3x3 transformation matrix
        w, h: Original image dimensions
        
    Returns:
        2x2 array: [[min_x, min_y], [max_x, max_y]]
    """
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32)
    ones = np.ones((4, 1))
    corners_hom = np.hstack([corners, ones])
    
    transformed = (homography @ corners_hom.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    
    min_x, min_y = transformed.min(axis=0)
    max_x, max_y = transformed.max(axis=0)
    
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(int)


def warp_image(image, homography):
    """
    Warp image using homography with inverse mapping.
    
    Args:
        image: Input image
        homography: 3x3 transformation matrix
        
    Returns:
        Tuple of (warped_image, bounding_box)
    """
    h, w = image.shape[:2]
    box = compute_bounding_box(homography, w, h)
    
    x_out = np.arange(box[0, 0], box[1, 0] + 1)
    y_out = np.arange(box[0, 1], box[1, 1] + 1)
    x_grid, y_grid = np.meshgrid(x_out, y_out)
    
    ones = np.ones_like(x_grid)
    coords_out = np.stack([x_grid, y_grid, ones], axis=-1).reshape(-1, 3)
    
    H_inv = np.linalg.inv(homography)
    coords_in_flat = (H_inv @ coords_out.T).T
    coords_in_flat = coords_in_flat[:, :2] / coords_in_flat[:, 2:3]
    
    map_x = coords_in_flat[:, 0].reshape(y_grid.shape).astype(np.float32)
    map_y = coords_in_flat[:, 1].reshape(y_grid.shape).astype(np.float32)
    
    warped = np.zeros((len(y_out), len(x_out), 3), dtype=image.dtype)
    for c in range(3):
        warped[:, :, c] = cv2.remap(
            image[:, :, c], map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
    
    return warped, box


def preprocess_video(video_path):
    """
    Load video, fix direction, and compute all homographies.
    
    This consolidates the common preprocessing pipeline used by
    both main stereo mosaicing and bonus forward panorama.
    
    Args:
        video_path: Path to input video
        
    Returns:
        Tuple of (frames, homographies) where homographies are 
        accumulated to the middle reference frame
    """
    print("=" * 60)
    print("PREPROCESSING VIDEO")
    print("=" * 60)
    
    frames = load_video_frames(video_path)
    print(f"Loaded {len(frames)} frames")
    
    frames = detect_and_fix_direction(frames)
    
    print("Computing transforms...")
    H_successive = []
    for i in range(len(frames) - 1):
        H = estimate_horizon_homography(frames[i], frames[i + 1])
        H_successive.append(H)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames) - 1} transforms")
    
    reference_idx = len(frames) // 2
    print(f"Accumulating to reference frame {reference_idx}...")
    homographies = accumulate_homographies(H_successive, reference_idx)
    
    return frames, homographies


def generate_panoramas(frames, homographies, num_viewpoints=24, canvas_scale=1.0, 
                       centered=False, use_pyramid_blending=True):
    """
    Generate stereo panoramas using Barcode Blending.
    
    Implements the method from:
    "Real-Time Stereo Mosaicing using Feature Tracking" (Vivet, Peleg, Binefa 2011)
    
    Key idea: Build separate mosaics from odd and even frames with double-width
    strips, then blend them using a barcode mask and pyramid blending.
    
    Args:
        frames: List of input frames
        homographies: List of homographies to reference frame
        num_viewpoints: Number of output panoramas (for stereo effect)
        canvas_scale: Scale factor for canvas
        centered: If True, dynamically track content center
        use_pyramid_blending: If True, use Laplacian pyramid blending
        
    Returns:
        Array of panorama frames
    """
    h, w = frames[0].shape[:2]
    n_frames = len(frames)
    
    # --- 1. Compute Global Boundaries ---
    bounding_boxes = np.zeros((n_frames, 2, 2))
    for i in range(n_frames):
        bounding_boxes[i] = compute_bounding_box(homographies[i], w, h)

    bounding_boxes[:, :, 0] *= canvas_scale
    global_offset = np.min(bounding_boxes, axis=(0, 1))
    bounding_boxes -= global_offset
    
    panorama_size = np.max(bounding_boxes, axis=(0, 1)).astype(int) + 1
    panorama_height = panorama_size[1]
    panorama_width_global = panorama_size[0]
    
    # 10% margin to avoid sampling black pixels at edges
    margin = int(w * 0.1) 
    slice_centers = np.linspace(margin, w - margin, num_viewpoints, endpoint=True, dtype=int)
    
    # Compute warped positions of slice centers for each viewpoint and frame
    warped_slice_centers = np.zeros((num_viewpoints, n_frames))
    for v, center in enumerate(slice_centers):
        center_point = np.array([[center, h // 2, 1]], dtype=np.float32)
        for i, H in enumerate(homographies):
            warped = (H @ center_point.T).T
            x_raw = warped[0, 0] / warped[0, 2]
            x_pos = x_raw * canvas_scale
            warped_slice_centers[v, i] = x_pos - global_offset[0]

    # Compute strip boundaries from midpoints
    midpoints = (warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2
    first_boundary = warped_slice_centers[:, 0:1] - (midpoints[:, 0:1] - warped_slice_centers[:, 0:1])
    last_boundary = warped_slice_centers[:, -1:] + (warped_slice_centers[:, -1:] - midpoints[:, -1:])
    boundaries = np.hstack([first_boundary, midpoints, last_boundary]).round().astype(int)
    
    print(f"\n[Panorama Generation] {num_viewpoints} viewpoints, {panorama_width_global}x{panorama_height}")
    
    # --- 2. Warp Frames ---
    print("Warping frames...")
    warped_frames = []
    for i, frame in enumerate(frames):
        warped_image, box = warp_image(frame, homographies[i])
        warped_frames.append(warped_image)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_frames} frames")


    # Initialize pyramid blender
    blender = PyramidBlender(levels=4) if use_pyramid_blending else None
    
    raw_panoramas = []
    content_bounds = []

    print("Compositing panoramas...")
    
    for pano_idx in range(num_viewpoints):
        # Accumulators for odd and even frames (double-width strips)
        acc_odd = np.zeros((panorama_height, panorama_width_global, 3), dtype=np.float32)
        w_odd = np.zeros((panorama_height, panorama_width_global, 1), dtype=np.float32)
        acc_even = np.zeros((panorama_height, panorama_width_global, 3), dtype=np.float32)
        w_even = np.zeros((panorama_height, panorama_width_global, 1), dtype=np.float32)
        
        min_x_content = panorama_width_global
        max_x_content = 0
        
        for i in range(n_frames):
            warped_image = warped_frames[i]
            warped_h, warped_w = warped_image.shape[:2]
            y_offset = int(bounding_boxes[i][0, 1])
            y_bottom = y_offset + warped_h
            x_offset = int(bounding_boxes[i][0, 0])

            # Double-width barcode logic: each strip covers 2 frame widths
            start_idx = i
            end_idx = min(i + 2, n_frames) 
            global_start = boundaries[pano_idx, start_idx] - 1
            global_end = boundaries[pano_idx, end_idx] + 1
            
            if global_end <= global_start:
                continue

            local_start = int((global_start - x_offset) / canvas_scale)
            local_end = int((global_end - x_offset) / canvas_scale)

            if local_end <= 0 or local_start >= warped_w:
                continue
            valid_start = max(0, local_start)
            valid_end = min(warped_w, local_end)
            if valid_start >= valid_end:
                continue
            
            strip = warped_image[:, valid_start:valid_end].astype(np.float32)
            
            paste_start = max(0, global_start)
            paste_end = min(panorama_width_global, global_end)
            
            # Track content bounds
            if paste_start < min_x_content:
                min_x_content = paste_start
            if paste_end > max_x_content:
                max_x_content = paste_end
            
            target_w = paste_end - paste_start
            if strip.shape[1] != target_w:
                if target_w <= 0:
                    continue
                strip = cv2.resize(strip, (target_w, strip.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Accumulate to odd or even mosaic
            if i % 2 != 0:  # Odd frame
                acc_odd[y_offset:y_bottom, paste_start:paste_end] += strip
                w_odd[y_offset:y_bottom, paste_start:paste_end] += 1.0
            else:  # Even frame
                acc_even[y_offset:y_bottom, paste_start:paste_end] += strip
                w_even[y_offset:y_bottom, paste_start:paste_end] += 1.0

        # --- 3. Normalize and Blend ---
        # Normalize odd mosaic
        valid_odd = w_odd > 0
        mosaic_odd = np.zeros_like(acc_odd)
        mosaic_odd[np.repeat(valid_odd, 3, axis=2)] = (acc_odd / np.maximum(w_odd, 1))[np.repeat(valid_odd, 3, axis=2)]
        
        # Normalize even mosaic
        valid_even = w_even > 0
        mosaic_even = np.zeros_like(acc_even)
        mosaic_even[np.repeat(valid_even, 3, axis=2)] = (acc_even / np.maximum(w_even, 1))[np.repeat(valid_even, 3, axis=2)]
        
        if use_pyramid_blending and blender is not None:
            # Create barcode mask
            barcode_mask = np.zeros((panorama_height, panorama_width_global), dtype=np.float32)
            
            for i in range(n_frames):
                strip_start = boundaries[pano_idx, i]
                strip_end = boundaries[pano_idx, i + 1] if i + 1 <= n_frames else boundaries[pano_idx, -1]
                
                strip_start = max(0, min(strip_start, panorama_width_global))
                strip_end = max(0, min(strip_end, panorama_width_global))
                
                if strip_end <= strip_start:
                    continue
                    
                strip_mid = (strip_start + strip_end) // 2
                is_odd = (i % 2 != 0)
                
                # Barcode pattern: alternating halves
                if is_odd:
                    barcode_mask[:, strip_start:strip_mid] = 0.0  # Use even mosaic
                    barcode_mask[:, strip_mid:strip_end] = 1.0    # Use odd mosaic
                else:
                    barcode_mask[:, strip_start:strip_mid] = 1.0  # Use odd mosaic (inverted logic for blending)
                    barcode_mask[:, strip_mid:strip_end] = 0.0    # Use even mosaic
            
            # Handle regions with only one mosaic having content
            only_odd = valid_odd & ~valid_even
            only_even = valid_even & ~valid_odd
            barcode_mask[only_odd[:, :, 0]] = 1.0
            barcode_mask[only_even[:, :, 0]] = 0.0
            
            # Pyramid blend
            result = blender.blend(mosaic_odd, mosaic_even, barcode_mask)
        else:
            # Simple averaging fallback
            canvas_acc = acc_odd + acc_even
            weight_acc = w_odd + w_even
            valid_mask = weight_acc > 0
            safe_weights = np.maximum(weight_acc, 1.0)
            result = canvas_acc / safe_weights
            result = np.clip(result, 0, 255)
            result[~np.repeat(valid_mask, 3, axis=2)] = 0
            result = result.astype(np.uint8)
        
        raw_panoramas.append(result)
        content_bounds.append((min_x_content, max_x_content))

        if (pano_idx + 1) % 10 == 0:
            print(f"  {pano_idx+1}/{num_viewpoints} panoramas")

    # --- 4. Dynamic Centering (optional) ---
    if centered:
        print("Applying center tracking...")
        max_content_width = max(mx - mn for mn, mx in content_bounds)
        output_width = int(max_content_width)
        
        final_panoramas = np.zeros((num_viewpoints, panorama_height, output_width, 3), dtype=np.uint8)
        
        for idx, raw_pano in enumerate(raw_panoramas):
            mn, mx = content_bounds[idx]
            content_center = (mn + mx) // 2
            
            start_x = content_center - (output_width // 2)
            end_x = start_x + output_width
            
            src_start = max(0, start_x)
            src_end = min(panorama_width_global, end_x)
            
            dst_start = src_start - start_x
            dst_end = dst_start + (src_end - src_start)
            
            if dst_end > output_width:
                dst_end = output_width
            
            final_panoramas[idx, :, dst_start:dst_end, :] = raw_pano[:, src_start:src_end, :]
            
        return final_panoramas

    return np.array(raw_panoramas)


def pad_to_macro_block(frame, macro_block_size=16):
    """Pad frame dimensions to be divisible by macro block size (for video encoding)."""
    h, w = frame.shape[:2]
    pad_h = (macro_block_size - (h % macro_block_size)) % macro_block_size
    pad_w = (macro_block_size - (w % macro_block_size)) % macro_block_size
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)


# =============================================================================
# API Function (Required by Exercise)
# =============================================================================

def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4.
    
    :param input_frames_path: path to a dir with input video frames.
        We will test your code with a dir that has K frames, each in the format
        "frame_i:05d.jpg" (e.g., frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, ...).
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorama frames (of size n_out_frames),
        each list item should be a PIL image of a generated panorama.
    """
    print("=" * 60)
    print("STEREO MOSAIC GENERATION")
    print("=" * 60)
    
    # Load frames from directory
    print(f"Loading frames from {input_frames_path}...")
    frames = load_frames_from_directory(input_frames_path)
    print(f"  Loaded {len(frames)} frames")
    
    # Detect and fix direction
    print("Detecting direction...")
    frames = detect_and_fix_direction(frames)
    
    # Compute homographies
    print("Computing transforms...")
    H_successive = []
    for i in range(len(frames) - 1):
        H = estimate_horizon_homography(frames[i], frames[i + 1])
        H_successive.append(H)
        if (i + 1) % 50 == 0:
            print(f"  Computed {i + 1}/{len(frames) - 1} transforms")
    
    reference_idx = len(frames) // 2
    print(f"Accumulating to reference frame {reference_idx}...")
    homographies = accumulate_homographies(H_successive, reference_idx)
    
    # Generate panoramas
    print(f"Generating {n_out_frames} panoramas...")
    panoramas = generate_panoramas(frames, homographies, num_viewpoints=n_out_frames)
    
    # Convert to PIL Images
    print("Converting to PIL images...")
    pil_images = []
    for pano in panoramas:
        pil_img = Image.fromarray(pano)
        pil_images.append(pil_img)
    
    print("Done!")
    return pil_images


# =============================================================================
# Video-based Entry Point (for direct use)
# =============================================================================

def generate_dynamic_panorama(video_path, num_viewpoints=24, canvas_scale=1.0, 
                              trim_start=0, trim_end=0, reverse_output=False, 
                              centered=False, use_pyramid_blending=True):
    """
    Main entry point: Process video into stereo mosaic.
    
    Args:
        video_path: Path to input video
        num_viewpoints: Number of output panorama frames
        canvas_scale: Scale factor for output
        trim_start: Remove this many frames from start of output
        trim_end: Remove this many frames from end of output
        reverse_output: Reverse output frame order
        centered: Apply dynamic center tracking
        use_pyramid_blending: Use Laplacian pyramid blending (vs simple averaging)
        
    Returns:
        List of panorama frames ready for video encoding
    """
    print("=" * 60)
    print("STEREO MOSAIC GENERATION")
    print("=" * 60)
    
    frames, homographies = preprocess_video(video_path)
    
    panoramas = generate_panoramas(
        frames, homographies, num_viewpoints, canvas_scale, 
        centered, use_pyramid_blending
    )
    
    # Trim output
    if trim_end > 0:
        panoramas = panoramas[trim_start:-trim_end]
    elif trim_start > 0:
        panoramas = panoramas[trim_start:]
    
    # Reverse if requested
    if reverse_output:
        panoramas = panoramas[::-1]
    
    # Pad for video encoding
    output_frames = [pad_to_macro_block(pano) for pano in panoramas]
    
    print(f"Done! Generated {len(output_frames)} panoramas")
    return output_frames