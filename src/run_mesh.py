"""Stage 4b: surface meshing from a dense point cloud (Open3D, classical).

First-pass approach: estimate normals -> Poisson surface reconstruction ->
density-trim (drop the low-support vertices Poisson bridges across gaps/foliage)
-> crop to the cloud's extent -> save a coloured mesh (PLY). Poisson fits a single
surface through the point band, which is what collapses the "forward-back stretch"
on walls/windows into clean sheets. Foliage (not a real surface) will mesh roughly
-- that's expected; a later pass can segment it out.

Usage:
    python run_mesh.py [ply] [--depth D] [--voxel V] [--density-quantile Q]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ply", nargs="?", default=str(OUT_DIR / "longer_walk_s8_dense_showcase.ply"))
    p.add_argument("--depth", type=int, default=10,
                   help="Poisson octree depth (higher = finer + slower/more memory)")
    p.add_argument("--voxel", type=float, default=0.0,
                   help="optional pre-downsample (helps memory on big clouds; 0 = none)")
    p.add_argument("--density-quantile", type=float, default=0.03,
                   help="drop vertices below this density quantile (removes bridged blobs)")
    p.add_argument("--smooth", type=int, default=15,
                   help="Taubin smoothing iterations (reduces the foamy/bumpy look from "
                        "point noise; feature-preserving. 0 = off)")
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    ply_path = Path(args.ply)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"loaded {len(pcd.points):,} points from {ply_path.name}")
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"  downsampled to {len(pcd.points):,} points (voxel {args.voxel})")

    # normals (required for Poisson), oriented toward the camera trajectory so the
    # surface faces the viewer consistently.
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag * 0.01, max_nn=30))
    tag = ply_path.stem.split("_dense")[0]
    centers_path = OUT_DIR / f"{tag}_camera_centers.npy"
    if centers_path.exists():
        cam_mean = np.load(centers_path).mean(axis=0)
        pcd.orient_normals_towards_camera_location(cam_mean)
        print(f"  normals oriented toward camera trajectory ({centers_path.name})")
    else:
        pcd.orient_normals_consistent_tangent_plane(30)

    print(f"Poisson reconstruction (depth={args.depth})...", flush=True)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth)
    densities = np.asarray(densities)
    print(f"  raw mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # trim low-density vertices (Poisson bridges empty space -> low support there)
    keep = densities >= np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(~keep)
    # Poisson inflates a closed "bubble" past the data; crop back to the cloud extent
    mesh = mesh.crop(bbox)
    if args.smooth > 0:
        # Taubin (lambda/mu) smoothing: reduces the foamy surface noise Poisson
        # picks up from the thick/noisy point band, without the shrinkage a plain
        # Laplacian causes -> walls flatten out but windows/edges survive.
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth)
        print(f"  Taubin-smoothed x{args.smooth}")
    mesh.compute_vertex_normals()
    print(f"  after density-trim + crop + smooth: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    out = OUT_DIR / f"{tag}_mesh{args.out_suffix}.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"Saved {out}")

    # self-documenting build params (read by export_for_unity -> scene_info.json)
    import json
    meta = {"smooth": args.smooth, "poisson_depth": args.depth,
            "density_quantile": args.density_quantile,
            "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
            "source_cloud": ply_path.name}
    (OUT_DIR / f"{tag}_mesh{args.out_suffix}_meta.json").write_text(json.dumps(meta, indent=2))

    # best-effort offscreen preview (legacy GL; skip if the context can't render)
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1600, height=900)
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.light_on = False              # flat vertex colours (no dark shading)
        opt.mesh_show_back_face = True
        vc = vis.get_view_control()       # look into the scene from the camera side
        vc.set_lookat(bbox.get_center())
        vc.set_front([0.0, -0.25, -1.0]); vc.set_up([0.0, -1.0, 0.0]); vc.set_zoom(0.46)
        vis.poll_events(); vis.update_renderer()
        prev = OUT_DIR / f"{tag}_mesh{args.out_suffix}_preview.png"
        vis.capture_screen_image(str(prev), do_render=True)
        vis.destroy_window()
        print(f"Saved {prev}")
    except Exception as e:
        print(f"(offscreen preview skipped: {e}; view the mesh with view_ply.py)")


if __name__ == "__main__":
    main()
