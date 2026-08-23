"""Crop a reconstruction (or any cloud from it) to the ORBITED OBJECT, using
only the camera geometry (Part B; reuses Part A read-only, edits nothing).

An orbit points every camera INWARD at the object, which gives two automatic,
label-free handles on where the object is and how big the keep-region should be:

  * CENTER = the convergence point of the camera viewing rays (the least-squares
    point closest to every optical axis). This is what all the cameras are
    looking at -- the object -- and it is robust to a TILTED orbit (filming from
    above/below), unlike the camera-centroid which then sits above the object.
  * RADIUS = median camera->center distance = the orbit radius. The object is
    INSIDE the ring; the background (far walls, a brief far-view, stray floaters)
    is beyond it. So keeping points within frac*radius of the center isolates the
    object and discards the rest.

This is the inverse of run_combine's core_crop and dissolves the "need an
isolated statue" problem: any orbit-able object works because the background is
cropped by construction. Use before Part C E-Init so the covariance ellipsoid is
the OBJECT's, not the background's.

    # crop the sparse cloud of a reconstruction:
    python Final/src/partB/crop_to_orbit_center.py bin_1_loop --frac 0.8
    # crop a dense cloud with the SAME object sphere (from that recon's cameras):
    python Final/src/partB/crop_to_orbit_center.py bin_1_loop --ply output/bin_1_loop_dense.ply --frac 0.8
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "output"


def camera_centers_dirs(cameras):
    """World camera centers C = -R^T t and forward optical axes d = R^T[0,0,1]
    (OpenCV convention: +Z points into the scene)."""
    C, D = [], []
    for c in cameras:
        R = np.asarray(c["R"], float)
        t = np.asarray(c["t"], float).reshape(3)
        C.append(-R.T @ t)
        D.append(R.T @ np.array([0.0, 0.0, 1.0]))
    C = np.array(C)
    D = np.array(D)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return C, D


def orbit_center_radius(cameras):
    """Least-squares convergence point of the camera viewing rays + orbit radius.

    center solves  (Σ (I - d_i d_i^T)) x = Σ (I - d_i d_i^T) C_i  -- the point of
    minimum total squared distance to every optical axis. radius = median
    |C_i - center|. Returns (center, radius, cam_centroid, tilt_offset)."""
    C, D = camera_centers_dirs(cameras)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for Ci, di in zip(C, D):
        M = np.eye(3) - np.outer(di, di)
        A += M
        b += M @ Ci
    center = np.linalg.solve(A, b)
    radius = float(np.median(np.linalg.norm(C - center, axis=1)))
    centroid = C.mean(axis=0)
    tilt = float(np.linalg.norm(centroid - center))  # how far the ring plane sits off the look-at point
    return center, radius, centroid, tilt


def _load_cloud(ply_path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pcd, pts, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="reconstruction tag (reads <tag>_reconstruction.pkl for the cameras)")
    ap.add_argument("--ply", default=None,
                    help="cloud to crop (default: the sparse points from the bundle, "
                         "coloured by time). Pass a dense PLY to crop it with the same sphere.")
    ap.add_argument("--frac", type=float, default=0.8,
                    help="keep radius as a fraction of the orbit radius (default 0.8; "
                         "your 'circle radius + a bit' is ~1.0-1.1, tighter isolates more)")
    ap.add_argument("--out", default=None, help="output PLY path")
    args = ap.parse_args()

    b = pickle.load(open(OUT / f"{args.tag}_reconstruction.pkl", "rb"))
    center, radius, centroid, tilt = orbit_center_radius(b["cameras"])
    keep_r = args.frac * radius
    print(f"{args.tag}: {len(b['cameras'])} cameras")
    print(f"  ray-convergence center = {np.round(center,2)}  (camera-centroid off by "
          f"{tilt:.2f} = orbit tilt)")
    print(f"  orbit radius (median cam->center) = {radius:.2f}")
    print(f"  keep sphere radius = frac {args.frac} x {radius:.2f} = {keep_r:.2f}")

    import open3d as o3d
    if args.ply:
        pcd, pts, cols = _load_cloud(args.ply)
        src_name = Path(args.ply).stem
    else:
        import matplotlib.cm as cm
        pts = np.asarray(b["points_world"], float)
        pf = np.asarray(b["point_frame"], float)
        t = (pf - pf.min()) / max(1e-9, (pf.max() - pf.min()))
        cols = cm.viridis(t)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        src_name = f"{args.tag}_sparse"

    d = np.linalg.norm(pts - center, axis=1)
    keep = d <= keep_r
    print(f"  kept {keep.sum()}/{len(pts)} points ({keep.mean():.1%}); "
          f"discarded {(~keep).sum()} beyond the sphere")
    # frac sweep so the tradeoff is visible without re-running
    print("  keep-fraction at other radii: " + ", ".join(
        f"{f:g}x={ (d <= f*radius).mean():.0%}" for f in (0.4, 0.6, 0.8, 1.0, 1.2)))

    out = Path(args.out) if args.out else OUT / f"{src_name}_objcrop.ply"
    cropped = pcd.select_by_index(np.where(keep)[0])
    o3d.io.write_point_cloud(str(out), cropped)
    ext = pts[keep].max(0) - pts[keep].min(0)
    print(f"  extent after crop {np.round(ext,2)} -> {out}")

    meta = {"tag": args.tag, "center": center.tolist(), "orbit_radius": radius,
            "frac": args.frac, "keep_radius": keep_r, "tilt_offset": tilt,
            "kept": int(keep.sum()), "total": int(len(pts))}
    json.dump(meta, open(OUT / f"{args.tag}_objcrop.json", "w"), indent=2)


if __name__ == "__main__":
    main()
