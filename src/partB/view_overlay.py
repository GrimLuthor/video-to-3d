"""
Interactive viewer for a Part B aligned pair, with toggles:

    1  : show/hide scene 1  (source, red in two-tone)
    2  : show/hide scene 2  (target, blue in two-tone)
    C  : cycle colour mode  (native pixel/time colour  <->  two-tone red/blue)
    [ ]: decrease / increase point size
    W  : white / black background
    H  : reprint this help   (Q or Esc to quit)

Uses the legacy Open3D viewer (works on this GPU; the filament GUI does not).

    python Final/src/partB/view_overlay.py <tag>
    # loads output/<tag>_src_aligned.ply and output/<tag>_tgt_aligned.ply
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

OUT = Path(__file__).resolve().parents[2] / "output"

RED = [0.85, 0.12, 0.12]
BLUE = [0.12, 0.35, 0.90]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "meonot_sparse"
    src = o3d.io.read_point_cloud(str(OUT / f"{tag}_src_aligned.ply"))
    tgt = o3d.io.read_point_cloud(str(OUT / f"{tag}_tgt_aligned.ply"))

    # remember native colours (fall back to a flat grey if a cloud has none)
    def native(pcd):
        if pcd.has_colors():
            return np.asarray(pcd.colors).copy()
        return np.tile([0.6, 0.6, 0.6], (len(pcd.points), 1))

    nat = {"src": native(src), "tgt": native(tgt)}
    two = {"src": np.tile(RED, (len(src.points), 1)),
           "tgt": np.tile(BLUE, (len(tgt.points), 1))}

    state = {"src": True, "tgt": True, "mode": "native", "psize": 2.0, "white": True}
    geoms = {"src": src, "tgt": tgt}
    present = {"src": True, "tgt": True}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Part B overlay: {tag}", width=1600, height=900)
    for g in geoms.values():
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.light_on = False
    opt.point_size = state["psize"]
    opt.background_color = np.array([1.0, 1.0, 1.0])

    def apply_colors():
        colmap = nat if state["mode"] == "native" else two
        for k, g in geoms.items():
            g.colors = o3d.utility.Vector3dVector(colmap[k])
            vis.update_geometry(g)

    def sync_visibility():
        for k, g in geoms.items():
            want = state[k]
            if want and not present[k]:
                vis.add_geometry(g, reset_bounding_box=False); present[k] = True
            elif not want and present[k]:
                vis.remove_geometry(g, reset_bounding_box=False); present[k] = False

    def toggle(k):
        def cb(v):
            state[k] = not state[k]; sync_visibility()
            print(f"  scene {k}: {'ON' if state[k] else 'off'}")
            return False
        return cb

    def cycle_color(v):
        state["mode"] = "two-tone" if state["mode"] == "native" else "native"
        apply_colors()
        print(f"  colour: {state['mode']}")
        return False

    def psize(delta):
        def cb(v):
            state["psize"] = float(np.clip(state["psize"] + delta, 1.0, 12.0))
            vis.get_render_option().point_size = state["psize"]
            return False
        return cb

    def bg(v):
        state["white"] = not state["white"]
        c = 1.0 if state["white"] else 0.0
        vis.get_render_option().background_color = np.array([c, c, c])
        return False

    def help_(v):
        print(__doc__); return False

    vis.register_key_callback(ord("1"), toggle("src"))
    vis.register_key_callback(ord("2"), toggle("tgt"))
    vis.register_key_callback(ord("C"), cycle_color)
    vis.register_key_callback(ord("["), psize(-1.0))
    vis.register_key_callback(ord("]"), psize(+1.0))
    vis.register_key_callback(ord("W"), bg)
    vis.register_key_callback(ord("H"), help_)

    apply_colors()
    print(__doc__)
    print(f"scene 1 (red)  = {tag}_src_aligned.ply   {len(src.points):,} pts")
    print(f"scene 2 (blue) = {tag}_tgt_aligned.ply   {len(tgt.points):,} pts")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
