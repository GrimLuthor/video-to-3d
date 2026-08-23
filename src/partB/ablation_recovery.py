"""
Part B quantitative ablation (mirrors the paper's success-rate experiment):
does E-Init's ellipsoid initialization actually matter for registering two
independent-frame reconstructions of the same scene?

For N random similarity transforms applied to a real reconstruction, we try to
recover each with ICP started from three inits and report the SUCCESS RATE +
median error per init:

  einit    : the paper's ellipsoid init (rotation from principal axes,
             scale from eigenvalue ratio) + ICP
  centroid : barycenter + eigenvalue-scale, but IDENTITY rotation, + ICP
             (isolates the value of the ellipsoid ROTATION)
  identity : no init at all (+ ICP)  -- the paper's "ICP without initialization"

A trial counts as success if rot_err < 3 deg and scale_err < 3%. As the applied
rotation grows past ICP's convergence basin, centroid/identity collapse while
einit stays flat -- that gap is the ellipsoid's contribution.

    python Final/src/partB/ablation_recovery.py columns_p0 --trials 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from ellipsoid_init import e_init_similarity, rotation_angle_deg
from similarity_icp import refine_similarity

OUT = Path(__file__).resolve().parents[2] / "output"


def as_pcd(xyz):
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(xyz); return p


def recover(src_pcd, tgt_pcd, src_pts, tgt_pts, mode, voxel, s_eig):
    if mode == "identity":
        T0 = np.eye(4)
    elif mode == "centroid":
        bs, bt = src_pts.mean(0), tgt_pts.mean(0)
        T0 = np.eye(4); T0[:3, :3] = s_eig * np.eye(3); T0[:3, 3] = bt - s_eig * bs
    else:  # einit: pick best of the sign candidates by refined fitness
        init = e_init_similarity(src_pts, tgt_pts, seed=0)
        best = None
        for c in init["candidates"]:
            r = refine_similarity(src_pcd, tgt_pcd, c["T"], voxel=voxel)
            if best is None or r["fitness"] > best["fitness"]:
                best = r
        return best
    return refine_similarity(src_pcd, tgt_pcd, T0, voxel=voxel)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="reconstruction tag (uses output/<tag>_sparse.ply)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(OUT / f"{args.tag}_sparse.ply"))
    base = np.asarray(pcd.points)
    diag = float(np.linalg.norm(base.max(0) - base.min(0)))
    voxel = diag / 200.0
    rng = np.random.default_rng(args.seed)
    modes = ["einit", "centroid", "identity"]
    # bucket trials by applied-rotation magnitude to expose the basin effect
    buckets = [(0, 45), (45, 90), (90, 180)]
    stats = {m: {b: [] for b in buckets} for m in modes}

    for _ in range(args.trials):
        ang = rng.uniform(5, 175)                       # applied rotation magnitude (deg)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        R = Rotation.from_rotvec(np.radians(ang) * axis).as_matrix()
        s = float(rng.uniform(0.5, 2.0))
        t = rng.uniform(-0.3, 0.3, size=3) * diag
        src_xyz = (s * R @ base.T).T + t               # transformed copy
        gt_s, gt_R = 1.0 / s, R.T                       # src -> tgt (inverse)

        src_pcd, tgt_pcd = as_pcd(src_xyz), as_pcd(base)
        s_eig = float(np.median(np.sqrt(
            np.linalg.eigvalsh(np.cov(base.T)) / np.linalg.eigvalsh(np.cov(src_xyz.T)))))
        b = next(b for b in buckets if b[0] <= ang < b[1])
        for m in modes:
            r = recover(src_pcd, tgt_pcd, src_xyz, base, m, voxel, s_eig)
            R_est = r["T"][:3, :3] / r["scale"]
            ok = (rotation_angle_deg(R_est, gt_R) < 3.0 and
                  abs(r["scale"] - gt_s) / gt_s < 0.03)
            stats[m][b].append(ok)

    print(f"\nSuccess rate over {args.trials} random similarities "
          f"(scale 0.5-2x), by applied-rotation bucket:\n")
    print(f"{'init':<10} " + "  ".join(f"{lo}-{hi}deg" for lo, hi in buckets) + "   overall")
    for m in modes:
        cells, allv = [], []
        for b in buckets:
            v = stats[m][b]; allv += v
            cells.append(f"{(100*np.mean(v) if v else 0):5.0f}%({len(v):2d})")
        print(f"{m:<10} " + "  ".join(cells) + f"   {100*np.mean(allv):5.0f}%")


if __name__ == "__main__":
    main()
