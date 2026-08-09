"""Stage 5 prep: export the reconstruction as Unity-ready assets for a
side-by-side MESH vs POINT-CLOUD fly-through.

- mesh  -> vertex-coloured GLB (glTF binary; Unity imports it with colours via
  the glTFast package), decimated unless --no-decimate.
- cloud -> voxel-thinned PLY for the Pcx point-cloud importer (Keijiro).

Writes into Final/unity_export/ with a README describing the Unity setup. Both
the README and scene_info.json are rendered from the same `info` dict, so their
counts cannot drift apart.

Usage:
    python export_for_unity.py [--mesh PLY] [--cloud PLY] [--tris N] [--cloud-voxel V]
    python export_for_unity.py --out-dir DIR --refresh-docs
"""

import argparse
import json
from pathlib import Path

# open3d is imported lazily in main(): --refresh-docs only rewrites prose and
# must stay runnable without the geometry stack installed.

OUT_DIR = Path(__file__).parent.parent / "output"
UNITY_DIR = Path(__file__).parent.parent / "unity_export"

# Single source of truth: the viewer shows this verbatim, so it must describe
# what this folder actually contains.
COORD_NOTES = ("up-to-scale; Y roughly down; keep UNLIT; "
               "glTF->Unity handedness flip on import. The mesh, pointcloud and "
               "cameras.json all share this coordinate frame.")


def build_readme(info, unity_dir):
    """Render README.md from the same facts that go into scene_info.json."""
    scene = info.get("scene") or unity_dir.name
    m = info.get("mesh") or {}
    pc = info.get("pointcloud") or {}
    stats = info.get("reconstruction_stats") or {}

    # Mesh line: describe the mesh that was actually written, not the intent.
    res = "decimated" if m.get("decimated") else "full-resolution"
    detail = [f"{m.get('triangles', 0):,} tris", f"{m.get('vertices', 0):,} verts"]
    if m.get("smoothing_iterations"):
        detail.append(f"Taubin smoothing {m['smoothing_iterations']}")
    if m.get("poisson_depth"):
        detail.append(f"Poisson depth {m['poisson_depth']}")

    lines = [f"# {scene} — Unity assets", ""]
    if info.get("source_video"):
        lines += [f"Generated from `{info['source_video']}` by the classical monocular "
                  f"reconstruction pipeline.", ""]
    else:
        lines += ["Generated from the classical monocular reconstruction pipeline.", ""]

    lines += [
        "## Files",
        f"- `{m.get('file', 'mesh_scene.glb')}` — {res}, vertex-coloured surface mesh "
        f"({', '.join(detail)}). Import via glTFast; use an **unlit vertex-colour** shader.",
        f"- `{pc.get('file', 'pointcloud_scene.ply')}` — {pc.get('points', 0):,} points "
        f"(Pcx, Point mode).",
    ]

    cams_path = unity_dir / "cameras.json"
    if cams_path.exists():
        try:
            n_cams = len(json.loads(cams_path.read_text(encoding="utf-8"))["cameras"])
            n_cams = f"{n_cams} "
        except Exception:
            n_cams = ""
        lines.append(f"- `cameras.json` + `frames/` — {n_cams}registered camera poses and "
                     f"thumbnails for the frame-comparison overlay. Same coordinate frame as "
                     f"the mesh/cloud; apply the SAME glTFast handedness flip to the poses.")
    lines += [
        "- `scene_info.json` — machine-readable counts, smoothing and reconstruction",
        "  stats. Read this for any scene-description / info panel.",
        "",
    ]

    if stats:
        lines += [
            "## Reconstruction",
            f"- {stats.get('cameras_used', 0)} cameras registered "
            f"({stats.get('reference_views', 0)} reference views).",
            f"- {stats.get('raw_fused', 0):,} raw fused points "
            f"-> {stats.get('after_downsample', 0):,} after downsample.",
            f"- {stats.get('visibility_cull_removed', 0):,} visibility-culled "
            f"(tol {stats.get('visibility_cull_tol', 0)}), "
            f"{stats.get('outlier_removed', 0):,} outliers removed.",
            f"- {stats.get('final_points', 0):,} final points.",
            "",
        ]

    lines += [
        "## Unity setup (side-by-side fly-through)",
        "1. **Point cloud**: install **Pcx** (Keijiro's point-cloud importer;",
        "   github.com/keijiro/Pcx) via Package Manager (git URL) or drop it in `Assets/`,",
        "   or read the PLY at runtime. Either way use a **Point** topology material.",
        "2. **Mesh**: import the `.glb` with the **glTFast** package. Use an",
        "   **Unlit / Vertex Color** shader so the baked colours show without lighting",
        "   darkening them — the reconstruction has no real surface normals for lighting.",
        "3. **Side-by-side**: place the mesh and the point cloud at the same scale and",
        "   offset them along X, or put them on a toggle so you can flip between them",
        "   from the same viewpoint.",
        "4. **Fly-through**: add a camera and animate it along the reconstructed path,",
        f"   or use Cinemachine. The recovered trajectory in",
        f"   `output/{scene}_camera_centers.npy` can drive a matching path if you want",
        "   the virtual camera to retrace the original walk.",
        "",
        "## Notes",
        "- Coordinates are in the reconstruction's arbitrary (up-to-scale) units; scale to",
        "  taste in Unity. Y is roughly \"down\" in this data — rotate 180 about X if it",
        "  imports upside-down.",
        "- Colours are per-vertex (from the source pixels); keep everything **unlit**.",
        "",
    ]
    return "\n".join(lines)


def write_docs(info, unity_dir):
    """Write scene_info.json + README.md. UTF-8 explicitly: the default on Windows
    is the ANSI codepage, which turns every em dash in the README into mojibake."""
    (unity_dir / "scene_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (unity_dir / "README.md").write_text(build_readme(info, unity_dir), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", default=str(OUT_DIR / "longer_walk_s8_mesh.ply"))
    p.add_argument("--cloud", default=str(OUT_DIR / "longer_walk_s8_dense_showcase.ply"))
    p.add_argument("--tris", type=int, default=800000, help="decimation target triangle count")
    p.add_argument("--no-decimate", action="store_true",
                   help="export the mesh at full resolution (skip triangle decimation)")
    p.add_argument("--cloud-voxel", type=float, default=0.03, help="cloud thinning voxel (0 = keep all)")
    p.add_argument("--mesh-smooth", type=int, default=None,
                   help="Taubin smoothing iterations the mesh was built with, recorded "
                        "into scene_info (for legacy meshes with no _meta.json sidecar)")
    p.add_argument("--out-dir", default=str(UNITY_DIR),
                   help="output folder (use a subfolder per scene to avoid overwriting)")
    p.add_argument("--refresh-docs", action="store_true",
                   help="rewrite scene_info.json's coordinate_notes and regenerate README.md "
                        "from the existing scene_info.json, without re-exporting any geometry")
    args = p.parse_args()

    unity_dir = Path(args.out_dir)

    # Docs-only repair path: the geometry in this folder is already correct, only
    # the prose around it has drifted.
    if args.refresh_docs:
        info_path = unity_dir / "scene_info.json"
        if not info_path.exists():
            raise SystemExit(f"--refresh-docs: no scene_info.json in {unity_dir}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["coordinate_notes"] = COORD_NOTES
        # Drop the dead A/B-variant keys older exports left behind.
        for stale in ("mesh_variants", "recommended_smoothing"):
            info.pop(stale, None)
        if isinstance(info.get("mesh", {}).get("smoothing_iterations"), float):
            info["mesh"]["smoothing_iterations"] = int(info["mesh"]["smoothing_iterations"])
        write_docs(info, unity_dir)
        m = info.get("mesh", {})
        print(f"{unity_dir.name}: refreshed docs "
              f"({m.get('triangles', 0):,} tris, decimated={m.get('decimated')})")
        return

    import open3d as o3d

    unity_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    raw_tris = len(mesh.triangles)
    print(f"mesh in: {raw_tris:,} tris")
    decimated = (not args.no_decimate) and raw_tris > args.tris
    if decimated:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.tris)
    else:
        print("  (full resolution -- no decimation)")
    mesh.compute_vertex_normals()
    glb = unity_dir / "mesh_scene.glb"
    o3d.io.write_triangle_mesh(str(glb), mesh)
    print(f"  -> {glb.name}: {len(mesh.triangles):,} tris, colours={mesh.has_vertex_colors()}")

    # smoothing / build params: prefer an explicit flag, else a run_mesh sidecar
    mesh_meta = {}
    meta_path = Path(args.mesh).with_name(Path(args.mesh).stem + "_meta.json")
    if meta_path.exists():
        mesh_meta = json.load(open(meta_path))
    smoothing = args.mesh_smooth if args.mesh_smooth is not None else mesh_meta.get("smooth")
    # Keep this an int: Unity's JsonUtility refuses a fractional literal (1.0) in
    # an int field and drops the whole scene_info object when it sees one.
    if smoothing is not None:
        smoothing = int(smoothing)

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
                 "triangles": len(mesh.triangles), "vertices": len(mesh.vertices),
                 "decimated": bool(decimated),
                 "smoothing_iterations": smoothing,
                 "poisson_depth": mesh_meta.get("poisson_depth")},
        "pointcloud": {"file": "pointcloud_scene.ply", "points": len(cloud.points)},
        "coordinate_notes": COORD_NOTES,
    }
    stats_path = Path(args.cloud).with_name(Path(args.cloud).stem + "_stats.json")
    if stats_path.exists():
        info["reconstruction_stats"] = json.load(open(stats_path))  # raw_fused, after_downsample,
        #   visibility_cull_removed, outlier_removed, final_points, reference_views, cameras_used
    recon_path = OUT_DIR / f"{info['scene']}_reconstruction.pkl"
    if recon_path.exists():
        import pickle
        info["source_video"] = Path(pickle.load(open(recon_path, "rb")).get("video_path", "")).name or None

    write_docs(info, unity_dir)
    print(f"  -> scene_info.json (incl. reconstruction_stats.outlier_removed)")
    print(f"Wrote {unity_dir/'README.md'}")


if __name__ == "__main__":
    main()
