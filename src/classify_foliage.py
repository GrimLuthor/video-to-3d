"""Foliage vs. surface segmentation of the dense cloud (classical, no CNN).

Discriminator = LOCAL GEOMETRY: for each point, PCA over its neighbourhood gives
eigenvalues l0<=l1<=l2. A wall/ground is locally planar (small l0 -> low "surface
variation"); a tree crown is a fuzzy volume (l0 comparable -> high surface
variation / scattering). Optionally AND-ed with a green-ness colour cue.

Outputs a colour-coded classification preview (foliage = red overlay) + the split
clouds `*_surface.ply` / `*_foliage.ply`, so the threshold can be eyeballed and
tuned BEFORE meshing (mesh the surface, keep foliage as points).

Usage:
    python classify_foliage.py [cloud.ply] [--var-thresh T] [--use-green] [--radius R]
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cloud", nargs="?", default=str(OUT_DIR / "longer_walk_s8_dense_showcase.ply"))
    p.add_argument("--var-thresh", type=float, default=0.10,
                   help="surface-variation threshold; above = scattered = foliage")
    p.add_argument("--radius", type=float, default=0.0,
                   help="neighbourhood radius for local PCA (0 = auto from scene size)")
    p.add_argument("--use-green", action="store_true",
                   help="also require green-ness (excess green) to call a point foliage")
    p.add_argument("--green-thresh", type=float, default=0.02,
                   help="excess-green threshold (higher = only clearly-green counts)")
    args = p.parse_args()

    cloud_path = Path(args.cloud)
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    n = len(pcd.points)
    print(f"loaded {n:,} points")

    bbox = pcd.get_axis_aligned_bounding_box()
    radius = args.radius or float(np.linalg.norm(bbox.get_extent()) * 0.006)
    print(f"local-PCA radius = {radius:.3f}; computing covariances...", flush=True)
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    cov = np.asarray(pcd.covariances)
    evals = np.linalg.eigvalsh(cov)                 # ascending: l0<=l1<=l2
    s = evals.sum(axis=1) + 1e-12
    surface_variation = evals[:, 0] / s             # low=planar, high=scattered/foliage

    foliage = surface_variation > args.var_thresh
    if args.use_green and pcd.has_colors():
        rgb = np.asarray(pcd.colors)
        exg = 2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]  # excess green
        foliage &= (exg > args.green_thresh)
    print(f"classified foliage: {foliage.mean():.1%} of points "
          f"(var-thresh {args.var_thresh}{', +green' if args.use_green else ''})")

    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.full((n, 3), 0.6)

    # split clouds for the next (meshing) step
    tag = cloud_path.stem.split("_dense")[0]
    for name, mask in [("surface", ~foliage), ("foliage", foliage)]:
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(xyz[mask])
        sub.colors = o3d.utility.Vector3dVector(rgb[mask])
        out = OUT_DIR / f"{tag}_{name}.ply"
        o3d.io.write_point_cloud(str(out), sub)
        print(f"  wrote {out.name}: {mask.sum():,} pts")

    # classification preview: surface keeps colour, foliage overlaid red
    vis_rgb = rgb.copy()
    vis_rgb[foliage] = [0.9, 0.05, 0.05]
    vpcd = o3d.geometry.PointCloud()
    vpcd.points = o3d.utility.Vector3dVector(xyz)
    vpcd.colors = o3d.utility.Vector3dVector(vis_rgb)
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1600, height=900)
        vis.add_geometry(vpcd)
        vc = vis.get_view_control()
        vc.set_lookat(bbox.get_center())
        vc.set_front([0.0, -0.25, -1.0]); vc.set_up([0.0, -1.0, 0.0]); vc.set_zoom(0.46)
        vis.poll_events(); vis.update_renderer()
        prev = OUT_DIR / f"{tag}_foliage_classification.png"
        vis.capture_screen_image(str(prev), do_render=True)
        vis.destroy_window()
        print(f"Saved {prev} (foliage = red)")
    except Exception as e:
        print(f"(preview skipped: {e})")


if __name__ == "__main__":
    main()
