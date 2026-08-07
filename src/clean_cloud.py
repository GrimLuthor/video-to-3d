"""Remove floater outliers from an existing dense .ply (statistical outlier
removal), without re-running fusion. Writes <name>_clean.ply.

Usage:
    python clean_cloud.py [path_to_ply] [--std S] [--nb N]
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ply", nargs="?", default=str(OUT_DIR / "longer_walk_incr_dense.ply"))
    p.add_argument("--std", type=float, default=2.0, help="lower = more aggressive")
    p.add_argument("--nb", type=int, default=20, help="neighbours for the distance stat")
    args = p.parse_args()

    src = Path(args.ply)
    pcd = o3d.io.read_point_cloud(str(src))
    n0 = len(pcd.points)
    print(f"loaded {n0:,} points")
    pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=args.nb, std_ratio=args.std)
    print(f"kept {len(keep):,} ({n0 - len(keep):,} floaters removed, {100*(n0-len(keep))/n0:.1f}%)")

    out = src.with_name(src.stem + "_clean.ply")
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
