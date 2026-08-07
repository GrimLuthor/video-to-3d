"""Render a clean offscreen view (PNG) of a point cloud or mesh, for report figures.

Uses the legacy Open3D renderer (works headless on this machine, unlike filament),
flat vertex colours (no lighting-darkening), a tunable point size, and a
scene-facing camera. Works for either a point-cloud .ply or a mesh .ply.

Usage:
    python render_view.py <ply> [--out PNG] [--point-size S] [--zoom Z]
                           [--front X Y Z] [--up X Y Z] [--width W] [--height H]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ply")
    p.add_argument("--out", default=None, help="output PNG (default: <ply>_view.png)")
    p.add_argument("--point-size", type=float, default=2.5)
    p.add_argument("--zoom", type=float, default=0.5)
    p.add_argument("--front", type=float, nargs=3, default=[0.0, -0.3, -1.0])
    p.add_argument("--up", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--from-camera", type=int, default=None,
                   help="render from reconstruction camera N's actual pose (use -1 for "
                        "the middle camera). Best for viewing sparse/fragmentary scenes "
                        "as the real frame saw them; overrides --front/--up/--zoom.")
    args = p.parse_args()

    ply = Path(args.ply)
    mesh = o3d.io.read_triangle_mesh(str(ply))
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        geom = mesh
        bbox = mesh.get_axis_aligned_bounding_box()
        print(f"mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    else:
        geom = o3d.io.read_point_cloud(str(ply))
        bbox = geom.get_axis_aligned_bounding_box()
        print(f"point cloud: {len(geom.points):,} points")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=args.width, height=args.height)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.light_on = False
    opt.mesh_show_back_face = True
    opt.point_size = args.point_size
    opt.background_color = np.array([1.0, 1.0, 1.0])
    vc = vis.get_view_control()
    if args.from_camera is not None:
        tag = ply.stem.split("_dense")[0]
        b = pickle.load(open(OUT_DIR / f"{tag}_reconstruction.pkl", "rb"))
        cams, K = b["cameras"], b["K"]
        i = args.from_camera if args.from_camera >= 0 else len(cams) // 2
        R, t = cams[i]["R"], cams[i]["t"].reshape(3)
        # principal point forced to image centre -- Open3D's view control rejects a
        # strongly off-centre one; near enough for a viewing render.
        s = args.width / (2 * K[0, 2])   # K's native width ~ 2*cx; scale fx/fy to window
        intr = o3d.camera.PinholeCameraIntrinsic(
            args.width, args.height, K[0, 0] * s, K[1, 1] * s,
            args.width / 2 - 0.5, args.height / 2 - 0.5)
        ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = intr; params.extrinsic = ext
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        print(f"  rendered from reconstruction camera {i}"
              + (f" (video frame {b['frame_indices'][i]})" if 'frame_indices' in b else ""))
    else:
        vc.set_lookat(bbox.get_center())
        vc.set_front(args.front)
        vc.set_up(args.up)
        vc.set_zoom(args.zoom)
    vis.poll_events(); vis.update_renderer()
    out = Path(args.out) if args.out else ply.with_name(ply.stem + "_view.png")
    vis.capture_screen_image(str(out), do_render=True)
    vis.destroy_window()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
