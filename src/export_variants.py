"""Export several Taubin-smoothing variants of a scene's mesh (+ one shared point
cloud) for side-by-side A/B comparison in Unity. Poisson is assumed already run
into a crisp (smooth-0) base mesh; each variant just applies Taubin smoothing to
it, so the expensive step happens once per scene.

Writes into <out-dir>:
    mesh_scene_s{N}.glb   (one per smoothing level, full resolution)
    pointcloud_scene.ply  (shared, full)
    scene_info.json       (lists every mesh variant + its smoothing level)
    README.md

Usage:
    python export_variants.py --tag TAG --crisp CRISP.ply --cloud CLOUD.ply
                              --out-dir DIR [--smooths 1 5 10] [--cloud-voxel 0]
"""

import argparse
import json
import pickle
from pathlib import Path

import open3d as o3d

from export_for_unity import COORD_NOTES

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--crisp", required=True, help="crisp (smooth-0) base mesh .ply")
    p.add_argument("--cloud", required=True, help="dense point cloud .ply")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--smooths", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--cloud-voxel", type=float, default=0.0, help="0 = full cloud")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = o3d.io.read_triangle_mesh(args.crisp)
    print(f"crisp base: {len(base.triangles):,} tris from {Path(args.crisp).name}", flush=True)

    variants = []
    for s in args.smooths:
        m = o3d.io.read_triangle_mesh(args.crisp)   # fresh copy per level
        if s > 0:
            m = m.filter_smooth_taubin(number_of_iterations=s)
        m.compute_vertex_normals()
        fn = f"mesh_scene_s{s}.glb"
        o3d.io.write_triangle_mesh(str(out_dir / fn), m)
        variants.append({"file": fn, "smoothing_iterations": s,
                         "triangles": len(m.triangles), "vertices": len(m.vertices)})
        print(f"  -> {fn}: {len(m.triangles):,} tris (smooth {s}), colours={m.has_vertex_colors()}",
              flush=True)

    cloud = o3d.io.read_point_cloud(args.cloud)
    if args.cloud_voxel > 0:
        cloud = cloud.voxel_down_sample(args.cloud_voxel)
    o3d.io.write_point_cloud(str(out_dir / "pointcloud_scene.ply"), cloud)
    print(f"  -> pointcloud_scene.ply: {len(cloud.points):,} pts", flush=True)

    info = {
        "scene": args.tag,
        "source_video": None,
        "mesh_variants": variants,
        "mesh": variants[0],  # primary/back-compat = first (lowest) smoothing level
        "recommended_smoothing": variants[0]["smoothing_iterations"],
        "pointcloud": {"file": "pointcloud_scene.ply", "points": len(cloud.points)},
        "coordinate_notes": COORD_NOTES,
    }
    stats_path = Path(args.cloud).with_name(Path(args.cloud).stem + "_stats.json")
    if stats_path.exists():
        info["reconstruction_stats"] = json.load(open(stats_path))
    recon = OUT_DIR / f"{args.tag}_reconstruction.pkl"
    if recon.exists():
        info["source_video"] = Path(pickle.load(open(recon, "rb")).get("video_path", "")).name or None
    (out_dir / "scene_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  -> scene_info.json ({len(variants)} mesh variants: "
          f"{[v['smoothing_iterations'] for v in variants]})", flush=True)

    lines = "\n".join(f"- `{v['file']}` — Taubin smoothing {v['smoothing_iterations']} "
                      f"({v['triangles']:,} tris)" for v in variants)
    (out_dir / "README.md").write_text(
        f"# {args.tag} — Unity assets (smoothing A/B)\n\n"
        f"Mesh variants (import whichever you prefer; all full resolution, share the\n"
        f"same `pointcloud_scene.ply` and `cameras.json` coordinate frame):\n\n{lines}\n\n"
        f"- `pointcloud_scene.ply` — {len(cloud.points):,} pts (Pcx, Point mode).\n"
        f"- Use an **unlit vertex-colour** shader. glTFast flips handedness on import;\n"
        f"  apply the same flip to `cameras.json` poses for the frame-comparison overlay.\n",
        encoding="utf-8")
    print(f"Wrote {out_dir/'README.md'}", flush=True)


if __name__ == "__main__":
    main()
