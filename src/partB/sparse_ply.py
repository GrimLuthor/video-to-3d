"""
Export a Part A sparse reconstruction to a viewable PLY, coloured by time
(the frame each point was first seen), so it can be opened in view_ply.py to
judge registration health / courtyard structure before paying for dense fusion.

    python Final/src/partB/sparse_ply.py <tag>
    -> output/<tag>_sparse.ply   (view with:  python Final/src/view_ply.py output/<tag>_sparse.ply)
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.cm as cm

OUT = Path(__file__).resolve().parents[2] / "output"


def main():
    tag = sys.argv[1]
    b = pickle.load(open(OUT / f"{tag}_reconstruction.pkl", "rb"))
    pts = np.asarray(b["points_world"], dtype=np.float64)
    pf = np.asarray(b["point_frame"], dtype=np.float64)
    t = (pf - pf.min()) / max(1e-9, (pf.max() - pf.min()))
    col = cm.viridis(t)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(col)
    out = OUT / f"{tag}_sparse.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    ext = pts.max(0) - pts.min(0)
    print(f"{tag}: {len(pts)} sparse pts, extent {np.round(ext,2)} -> {out}")


if __name__ == "__main__":
    main()
