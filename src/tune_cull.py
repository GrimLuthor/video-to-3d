"""Re-tune the free-space/visibility cull on a saved fusion WITHOUT re-fusing.

Needs a run done with `run_fusion.py ... --save-depths` (writes the pre-cull
cloud <tag>_dense<suffix>_nocull.ply and the depth maps <tag>_depths<suffix>.pkl).
Loads those + the reconstruction poses, applies plane_sweep.free_space_cull with
the given params, and writes a culled ply + preview so tol / min-violations can
be swept in seconds.

Usage:
    python tune_cull.py [tag] [--suffix S] [--tol T] [--min-viol M]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))
import plane_sweep as ps
from run_fusion import scale_K  # noqa: F401  (kept for parity)

OUT_DIR = Path(__file__).parent.parent / "output"


def scale_K_half(K):
    Ks = K.copy(); Ks[0, 0] *= 0.5; Ks[1, 1] *= 0.5; Ks[0, 2] *= 0.5; Ks[1, 2] *= 0.5
    return Ks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag", nargs="?", default="longer_walk_incr")
    p.add_argument("--suffix", default="_cullsub", help="the run's --out-suffix")
    p.add_argument("--tol", type=float, default=0.06)
    p.add_argument("--min-viol", type=int, default=3)
    p.add_argument("--scale", type=float, default=0.5, help="depth-map working scale used at fusion")
    args = p.parse_args()

    bundle = pickle.load(open(OUT_DIR / f"{args.tag}_reconstruction.pkl", "rb"))
    cameras, K_full = bundle["cameras"], bundle["K"]
    K = scale_K_half(K_full) if args.scale == 0.5 else K_full.copy()

    depths = pickle.load(open(OUT_DIR / f"{args.tag}_depths{args.suffix}.pkl", "rb"))
    nocull = OUT_DIR / f"{args.tag}_dense{args.suffix}_nocull.ply"
    pcd = o3d.io.read_point_cloud(str(nocull))
    xyz = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    print(f"{len(xyz):,} pre-cull points, {len(depths)} depth maps; "
          f"cull tol={args.tol} min_viol={args.min_viol}")

    keep, viol, supp = ps.free_space_cull(xyz, cameras, depths, K,
                                          tol=args.tol, min_violations=args.min_viol)
    print(f"kept {keep.sum():,} / {len(xyz):,} ({100*(1-keep.mean()):.1f}% removed); "
          f"mean violations={viol.mean():.2f}, mean support={supp.mean():.2f}")

    out = OUT_DIR / f"{args.tag}_dense{args.suffix}_culltuned.ply"
    culled = o3d.geometry.PointCloud()
    culled.points = o3d.utility.Vector3dVector(xyz[keep])
    culled.colors = o3d.utility.Vector3dVector(rgb[keep] / 255.0)
    o3d.io.write_point_cloud(str(out), culled)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
