"""Export per-scene camera poses + frame thumbnails for the Unity frame-comparison
overlay ("show the real video frame nearest the current fly-around view").

For a reconstruction tag it writes, next to the scene's unity_export assets:
    <out-dir>/cameras.json
    <out-dir>/frames/######.jpg      (one per REGISTERED frame, downscaled)

Only the registered/reference frames actually used in the reconstruction are
exported (not every video frame). Poses are in the SAME coordinate frame as the
exported pointcloud_scene.ply -- the raw reconstruction world frame, BEFORE any
Unity handedness flip -- so a frustum placed at `position` looking along
`forward` lines up with the point cloud directly.

Camera convention (OpenCV, as used throughout the pipeline): world->camera is
x_cam = R @ x_world + t, camera looks along +Z_cam, +X_cam right, +Y_cam DOWN.
Hence:
    position = -R^T t                (camera centre in world coords)
    forward  = R^T @ [0, 0, 1]       (optical axis, points INTO the scene)
    up       = R^T @ [0,-1, 0]       (camera up = -Y because +Y is down)
    quat     = camera->world rotation R^T as [x, y, z, w]

Usage:
    python export_poses_for_unity.py [--tag TAG] [--out-dir DIR]
                                     [--max-edge 640] [--jpg-quality 90]
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).parent))
from extract_frames import extract_frames
import calibration

OUT_DIR = Path(__file__).parent.parent / "output"
UNITY_DIR = Path(__file__).parent.parent / "unity_export"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="longer_walk_s8",
                   help="reconstruction tag -> reads output/<tag>_reconstruction.pkl")
    p.add_argument("--out-dir", default=str(UNITY_DIR),
                   help="scene's unity_export folder (write cameras.json + frames/ here)")
    p.add_argument("--max-edge", type=int, default=640,
                   help="downscale each thumbnail so its long edge is at most this (px)")
    p.add_argument("--jpg-quality", type=int, default=90)
    args = p.parse_args()

    with open(OUT_DIR / f"{args.tag}_reconstruction.pkl", "rb") as f:
        b = pickle.load(f)
    cameras = b["cameras"]
    frame_indices = b["frame_indices"]   # camera c -> index into the extracted-frame list
    K = np.asarray(b["K"], float)
    stride = int(b["stride"])
    video_path = b["video_path"]

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Re-extract exactly the images the reconstruction saw (same stride), then
    # undistort with the same K (K stays valid: undistort uses it as the new
    # camera matrix, no crop/recenter) so poses + intrinsics are self-consistent.
    print(f"[{args.tag}] extracting frames (stride {stride}) from {video_path} ...", flush=True)
    frames = extract_frames(Path(video_path), stride=stride)
    frames = calibration.undistort_frames(frames, K)
    H, W = frames[0].shape[:2]
    print(f"  {len(frames)} frames at {W}x{H}; exporting {len(cameras)} registered cameras", flush=True)

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    cam_list = []
    for c, cam in enumerate(cameras):
        R = np.asarray(cam["R"], float)
        t = np.asarray(cam["t"], float).reshape(3)
        C = -R.T @ t                       # camera centre in world coords
        forward = R.T @ np.array([0.0, 0.0, 1.0])   # optical axis, into scene
        up = R.T @ np.array([0.0, -1.0, 0.0])       # camera up (-Y_cam)
        forward /= np.linalg.norm(forward)
        up /= np.linalg.norm(up)
        quat = Rotation.from_matrix(R.T).as_quat()   # camera->world, [x,y,z,w]

        e = int(frame_indices[c])          # extracted-list index for this camera
        video_frame = e * stride           # frame number in the SOURCE video
        img_name = f"{video_frame:06d}.jpg"

        # write a downscaled thumbnail of the exact undistorted image
        img = frames[e]
        scale = args.max_edge / max(W, H)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, round(W * scale)), max(1, round(H * scale))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frames_dir / img_name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])

        cam_list.append({
            "frame": video_frame,
            "image": f"frames/{img_name}",
            "position": [float(C[0]), float(C[1]), float(C[2])],
            "forward": [float(forward[0]), float(forward[1]), float(forward[2])],
            "up": [float(up[0]), float(up[1]), float(up[2])],
            "quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        })

    data = {
        "scene": args.tag,
        "source_video": Path(video_path).name,
        "frame_size": [W, H],
        "coordinate_frame": ("identical to pointcloud_scene.ply (raw reconstruction "
                             "world coords, BEFORE any Unity handedness flip)"),
        "camera_convention": ("OpenCV: world->cam x_cam = R x_world + t; camera looks "
                              "+Z, +X right, +Y down. position=-R^T t (camera centre); "
                              "forward=optical axis into scene; up=-Y_cam; "
                              "quat=camera->world rotation [x,y,z,w]."),
        "intrinsics_note": ("fx,fy,cx,cy are for frame_size (full undistorted res); the "
                            "JPGs are thumbnails -- scale intrinsics by thumb_w/W for an "
                            "overlay at thumbnail resolution."),
        "cameras": cam_list,
    }
    (out_dir / "cameras.json").write_text(json.dumps(data, indent=2))
    print(f"  -> {out_dir/'cameras.json'} ({len(cam_list)} cameras) "
          f"+ {len(cam_list)} thumbnails in {frames_dir}", flush=True)


if __name__ == "__main__":
    main()
