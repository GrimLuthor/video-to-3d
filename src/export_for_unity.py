"""Stage 5 prep: export the reconstruction as Unity-ready assets for a
side-by-side MESH vs POINT-CLOUD fly-through.

- mesh  -> decimated, vertex-coloured GLB (glTF binary; Unity imports it with
  colours via built-in glTF / the glTFast package).
- cloud -> voxel-thinned PLY for the Pcx point-cloud importer (Keijiro).

Writes into Final/unity_export/ with a README describing the Unity setup.

Usage:
    python export_for_unity.py [--mesh PLY] [--cloud PLY] [--tris N] [--cloud-voxel V]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

OUT_DIR = Path(__file__).parent.parent / "output"
UNITY_DIR = Path(__file__).parent.parent / "unity_export"

README = """# Unity assets — side-by-side mesh vs. point cloud

Generated from the classical monocular reconstruction pipeline.

## Files
- `mesh_scene.glb` — decimated, vertex-coloured surface mesh (~{tris} triangles).
- `pointcloud_scene.ply` — thinned dense point cloud (~{pts} points) for Pcx.
- `scene_info.json` — machine-readable scene stats for the UI (point/triangle
  counts, visibility-cull and OUTLIER-removed counts, registered cameras, source
  video). Read this for any scene-description / info panel.

## Unity setup (side-by-side fly-through)
1. **Point cloud**: install **Pcx** (Keijiro's point-cloud importer;
   github.com/keijiro/Pcx) via Package Manager (git URL) or drop it in `Assets/`.
   Drag `pointcloud_scene.ply` into `Assets/` — Pcx imports it as a mesh with a
   point material. Add it to the scene.
2. **Mesh**: import `mesh_scene.glb` (Unity 2020+ supports glTF via the
   **glTFast** package; install it from Package Manager). Drag it into the scene.
   Use an **Unlit / Vertex Color** shader so the baked colours show without
   lighting darkening them (the reconstruction has no real surface normals for
   lighting) — a 2-line unlit shader that outputs the vertex colour, or Shader
   Graph "Unlit" with Vertex Color node.
3. **Side-by-side**: place the mesh and the point cloud at the same scale, offset
   along X (e.g. mesh at x=0, cloud at x=+30 — the scene is ~40 units wide), or
   put them on a toggle so you can flip between them from the same viewpoint.
4. **Fly-through**: add a camera and animate it along the reconstructed path, or
   use Cinemachine. The recovered camera trajectory (see
   `output/longer_walk_s8_camera_centers.npy`) can drive a matching path if you
   want the virtual camera to retrace the original walk.

## Notes
- Coordinates are in the reconstruction's arbitrary (up-to-scale) units; scale to
  taste in Unity. Y is roughly "down" in this data — rotate 180 about X if it
  imports upside-down.
- Colours are per-vertex (from the source pixels); keep everything **unlit**.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", default=str(OUT_DIR / "longer_walk_s8_mesh.ply"))
    p.add_argument("--cloud", default=str(OUT_DIR / "longer_walk_s8_dense_showcase.ply"))
    p.add_argument("--tris", type=int, default=800000, help="decimation target triangle count")
    p.add_argument("--cloud-voxel", type=float, default=0.03, help="cloud thinning voxel (0 = keep all)")
    p.add_argument("--out-dir", default=str(UNITY_DIR),
                   help="output folder (use a subfolder per scene to avoid overwriting)")
    args = p.parse_args()

    unity_dir = Path(args.out_dir)
    unity_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    print(f"mesh in: {len(mesh.triangles):,} tris")
    if len(mesh.triangles) > args.tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.tris)
    mesh.compute_vertex_normals()
    glb = unity_dir / "mesh_scene.glb"
    o3d.io.write_triangle_mesh(str(glb), mesh)
    print(f"  -> {glb.name}: {len(mesh.triangles):,} tris, colours={mesh.has_vertex_colors()}")

    cloud = o3d.io.read_point_cloud(args.cloud)
    print(f"cloud in: {len(cloud.points):,} pts")
    if args.cloud_voxel > 0:
        cloud = cloud.voxel_down_sample(args.cloud_voxel)
    ply = unity_dir / "pointcloud_scene.ply"
    o3d.io.write_point_cloud(str(ply), cloud)
    print(f"  -> {ply.name}: {len(cloud.points):,} pts")

    # machine-readable scene metadata for the Unity UI (counts + reconstruction
    # stats incl. the OUTLIER-removed count the UI asked for).
    info = {
        "scene": Path(args.cloud).stem.split("_dense")[0],
        "source_video": None,
        "mesh": {"file": "mesh_scene.glb",
                 "triangles": len(mesh.triangles), "vertices": len(mesh.vertices)},
        "pointcloud": {"file": "pointcloud_scene.ply", "points": len(cloud.points)},
        "coordinate_notes": "up-to-scale; Y roughly down; keep UNLIT; "
                            "glTF->Unity handedness flip on import",
    }
    stats_path = Path(args.cloud).with_name(Path(args.cloud).stem + "_stats.json")
    if stats_path.exists():
        info["reconstruction_stats"] = json.load(open(stats_path))  # raw_fused, after_downsample,
        #   visibility_cull_removed, outlier_removed, final_points, reference_views, cameras_used
    recon_path = OUT_DIR / f"{info['scene']}_reconstruction.pkl"
    if recon_path.exists():
        import pickle
        info["source_video"] = Path(pickle.load(open(recon_path, "rb")).get("video_path", "")).name or None
    (unity_dir / "scene_info.json").write_text(json.dumps(info, indent=2))
    print(f"  -> scene_info.json (incl. reconstruction_stats.outlier_removed)")

    (unity_dir / "README.md").write_text(
        README.format(tris=f"{len(mesh.triangles):,}", pts=f"{len(cloud.points):,}"))
    print(f"Wrote {unity_dir/'README.md'}")


if __name__ == "__main__":
    main()
