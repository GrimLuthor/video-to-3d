"""Interactive viewer for the dense reconstruction (Open3D).

Default: a GUI window showing the coloured cloud + camera trajectory (red), with
a stats panel in the top-left corner (point counts / how much was filtered, read
from the fusion's <name>_stats.json sidecar) and a live DEPTH-axis (Z) slider to
stretch/squash depth for inspecting depth smear.

--simple falls back to the lightweight viewer (depth keys K/J/H instead of a
slider) in case the GUI backend misbehaves.

Usage:
    python view_ply.py [path_to_ply] [--clean] [--voxel V] [--no-trajectory] [--simple]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

OUT_DIR = Path(__file__).parent.parent / "output"


def load_scene(args):
    ply_path = Path(args.ply)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_loaded = len(pcd.points)
    print(f"loaded {n_loaded:,} points from {ply_path.name}")
    if n_loaded == 0:
        raise SystemExit("empty / unreadable point cloud")
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"  downsampled to {len(pcd.points):,} points (voxel {args.voxel})")
    if args.clean:
        pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  after outlier removal: {len(keep):,} kept")

    traj = None
    tag = ply_path.stem.split("_dense")[0]  # strip _dense<suffix>
    centers_path = OUT_DIR / f"{tag}_camera_centers.npy"
    if not args.no_trajectory and centers_path.exists():
        centers = np.load(centers_path)
        traj = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(centers),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centers) - 1)]))
        traj.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * (len(centers) - 1))
        print(f"  overlaid trajectory ({len(centers)} cameras)")

    stats = {}
    stats_path = ply_path.with_name(ply_path.stem + "_stats.json")
    if stats_path.exists():
        stats = json.load(open(stats_path))
    return ply_path, pcd, traj, stats


def stats_text(stats, shown):
    lines = [f"points shown: {shown:,}"]
    order = [("reference_views", "reference views"), ("cameras_used", "cameras used"),
             ("raw_fused", "raw fused"), ("after_downsample", "after downsample"),
             ("visibility_cull_removed", "cull removed"),
             ("outlier_removed", "outliers removed"), ("final_points", "final points")]
    for k, label in order:
        if k in stats:
            v = stats[k]
            lines.append(f"{label}: {v:,}" if isinstance(v, int) else f"{label}: {v}")
    return "\n".join(lines)


def run_gui(ply_path, pcd, traj, stats):
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    gui.Application.instance.initialize()
    w = gui.Application.instance.create_window(ply_path.name, 1500, 950)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    w.add_child(scene)

    mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.0
    scene.scene.add_geometry("pcd", pcd, mat)
    if traj is not None:
        lmat = rendering.MaterialRecord(); lmat.shader = "unlitLine"; lmat.line_width = 2.0
        scene.scene.add_geometry("traj", traj, lmat)
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    cz = float(np.asarray(pcd.points)[:, 2].mean())

    # corner HUD
    em = w.theme.font_size
    panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
    try:
        panel.background_color = gui.Color(0.0, 0.0, 0.0, 0.6)
    except Exception:
        pass
    info = gui.Label(stats_text(stats, len(pcd.points)))
    panel.add_child(info)
    panel.add_child(gui.Label("depth (Z) scale:"))
    slider = gui.Slider(gui.Slider.DOUBLE)
    slider.set_limits(0.2, 5.0); slider.double_value = 1.0

    def on_z(val):
        T = np.eye(4); T[2, 2] = val; T[2, 3] = cz * (1.0 - val)
        scene.scene.set_geometry_transform("pcd", T)
        if traj is not None:
            scene.scene.set_geometry_transform("traj", T)

    slider.set_on_value_changed(on_z)
    panel.add_child(slider)
    w.add_child(panel)

    def on_layout(ctx):
        r = w.content_rect
        scene.frame = r
        pref = panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        panel.frame = gui.Rect(r.x + em, r.y + em, max(pref.width, 12 * em), pref.height)

    w.set_on_layout(on_layout)
    print("GUI: drag rotate, scroll zoom, depth slider top-left, close window to quit")
    gui.Application.instance.run()


def _title(ply_path, stats, shown):
    """Compact one-line stats for the window title bar (the legacy viewer can't
    overlay text inside the view, but the title bar is visible and reliable)."""
    bits = [f"{shown:,} pts"]
    if "visibility_cull_removed" in stats:
        bits.append(f"cull -{stats['visibility_cull_removed']:,}")
    if "outlier_removed" in stats:
        bits.append(f"outliers -{stats['outlier_removed']:,}")
    if "cameras_used" in stats:
        bits.append(f"{stats['cameras_used']} cams")
    return f"{ply_path.name}  |  " + "  ".join(bits)


def run_simple(ply_path, pcd, traj, stats):
    print(stats_text(stats, len(pcd.points)))
    base = np.asarray(pcd.points).copy()
    cz = float(base[:, 2].mean())
    tb = np.asarray(traj.points).copy() if traj is not None else None
    st = {"z": 1.0}

    def apply(vis=None):
        p = base.copy(); p[:, 2] = (base[:, 2] - cz) * st["z"] + cz
        pcd.points = o3d.utility.Vector3dVector(p)
        if traj is not None:
            t = tb.copy(); t[:, 2] = (tb[:, 2] - cz) * st["z"] + cz
            traj.points = o3d.utility.Vector3dVector(t)
        if vis is not None:
            vis.update_geometry(pcd)
            if traj is not None:
                vis.update_geometry(traj)
        return False

    def rescale(f):
        def cb(vis):
            st["z"] = float(np.clip(st["z"] * f, 0.05, 30.0)); print(f"  Z scale = {st['z']:.2f}")
            return apply(vis)
        return cb

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=_title(ply_path, stats, len(pcd.points)),
                      width=1400, height=900)
    vis.add_geometry(pcd)
    if traj is not None:
        vis.add_geometry(traj)
    vis.register_key_callback(ord("K"), rescale(1.25))
    vis.register_key_callback(ord("J"), rescale(0.8))
    vis.register_key_callback(ord("H"), lambda v: (st.update(z=1.0), apply(v))[-1])
    print("depth-axis: K stretch, J squash, H reset | drag rotate, scroll zoom, q quit")
    vis.run(); vis.destroy_window()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ply", nargs="?", default=str(OUT_DIR / "longer_walk_incr_dense.ply"))
    p.add_argument("--clean", action="store_true", help="preview statistical outlier removal")
    p.add_argument("--voxel", type=float, default=0.0, help="extra voxel downsample")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--gui", action="store_true",
                   help="use the filament GUI (corner stats panel + depth slider). "
                        "Needs a working OpenGL/filament context -- some Windows GPUs "
                        "give a black screen / wglMakeCurrent error; use the default instead.")
    p.add_argument("--simple", action="store_true", help=argparse.SUPPRESS)  # back-compat alias
    args = p.parse_args()

    # a mesh PLY (has triangles) -> show it directly with the legacy renderer
    mesh = o3d.io.read_triangle_mesh(args.ply)
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        print(f"mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=Path(args.ply).name, width=1400, height=900)
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.light_on = False            # show flat vertex (pixel) colours, no shading-darkening
        opt.mesh_show_back_face = True   # don't see through single-sided triangles
        print("mesh view (flat colours): drag rotate, scroll zoom, q quit")
        vis.run(); vis.destroy_window()
        return

    ply_path, pcd, traj, stats = load_scene(args)
    # Default = the legacy viewer: reliable across GPUs (the filament GUI fails on
    # some Windows setups with a wglMakeCurrent/black-screen error). Stats go in
    # the window title + console; depth-axis via K/J/H keys. --gui opts into the
    # fancier panel+slider where the GPU supports it.
    if args.gui and not args.simple:
        try:
            run_gui(ply_path, pcd, traj, stats)
            return
        except Exception as e:
            print(f"GUI viewer failed ({e}); falling back to the legacy viewer")
    run_simple(ply_path, pcd, traj, stats)


if __name__ == "__main__":
    main()
