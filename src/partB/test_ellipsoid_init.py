"""
Synthetic ground-truth validation of e_init_similarity, in the spirit of the
paper's own noise/occlusion experiments (Sec. IV-V). We take a real dense cloud,
apply a KNOWN similarity (scale, rotation, translation) plus additive noise and
a partial crop (occlusion / partial overlap), and check that E-Init recovers the
transform.

Run (venv python):
    python Final/src/partB/test_ellipsoid_init.py Final/output/einstein_s8_dense_showcase.ply
"""

from __future__ import annotations

import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from ellipsoid_init import e_init_similarity, rotation_angle_deg


def _random_rotation(rng):
    return Rotation.from_rotvec(rng.normal(size=3)).as_matrix()


def run_case(pts, *, s_true, deg_noise_frac, crop_frac, rng, label):
    R_true = _random_rotation(rng)
    t_true = rng.uniform(-5, 5, size=3)

    # target = the original cloud; source = a transformed, noised, cropped copy.
    # We ask E-Init for the similarity mapping source -> target, and compare to
    # the exact inverse of what we applied.
    moved = (s_true * R_true @ pts.T).T + t_true

    # additive noise, scaled to the cloud size so deg_noise_frac is meaningful
    extent = np.linalg.norm(pts.max(0) - pts.min(0))
    moved = moved + rng.normal(scale=deg_noise_frac * extent, size=moved.shape)

    # partial overlap: keep a random spatial slab of each cloud
    def crop(X):
        axis = 0
        lo = np.quantile(X[:, axis], (1 - crop_frac) * rng.uniform(0, 1))
        hi = lo + crop_frac * (X[:, axis].max() - X[:, axis].min())
        return X[(X[:, axis] >= lo) & (X[:, axis] <= hi)]

    src = crop(moved) if crop_frac < 1.0 else moved
    tgt = crop(pts) if crop_frac < 1.0 else pts

    out = e_init_similarity(src, tgt, seed=int(rng.integers(1 << 30)))

    # ground-truth inverse similarity mapping source(=moved) -> target(=pts):
    #   pts = (1/s) R_true^T (moved - t_true)
    s_gt = 1.0 / s_true
    R_gt = R_true.T
    s_err = abs(out["s"] - s_gt) / s_gt
    r_err = rotation_angle_deg(out["R"], R_gt)

    print(f"[{label}]  s_true(src->tgt)={s_gt:.4f}  recovered={out['s']:.4f}  "
          f"scale_err={100*s_err:5.2f}%   rot_err={r_err:6.2f} deg   "
          f"axis_scales={np.round(out['scale_per_axis'],3)}  "
          f"NN={out['score']:.4f}")
    return s_err, r_err


def main():
    ply = sys.argv[1] if len(sys.argv) > 1 else \
        "Final/output/einstein_s8_dense_showcase.ply"
    pcd = o3d.io.read_point_cloud(ply)
    pts = np.asarray(pcd.points)
    # voxel-downsample for speed of the test
    pcd = pcd.voxel_down_sample(0.05)
    pts = np.asarray(pcd.points)
    print(f"Loaded {ply}: {len(pts)} points (downsampled)\n")

    rng = np.random.default_rng(42)
    results = []
    print("--- clean, full overlap, varying scale ---")
    for s in (0.5, 2.0, 5.0):
        results.append(run_case(pts, s_true=s, deg_noise_frac=0.0,
                                 crop_frac=1.0, rng=rng, label=f"scale x{s}"))
    print("\n--- additive noise, full overlap ---")
    for nf in (0.005, 0.02, 0.05):
        results.append(run_case(pts, s_true=1.7, deg_noise_frac=nf,
                                 crop_frac=1.0, rng=rng, label=f"noise {nf}"))
    print("\n--- partial overlap (crop), light noise ---")
    for cf in (0.9, 0.75, 0.6):
        results.append(run_case(pts, s_true=1.7, deg_noise_frac=0.01,
                                 crop_frac=cf, rng=rng, label=f"overlap {cf}"))

    s_errs = [r[0] for r in results]
    r_errs = [r[1] for r in results]
    print(f"\nSUMMARY  median scale_err={100*np.median(s_errs):.2f}%  "
          f"median rot_err={np.median(r_errs):.2f} deg")
    # The clean full-overlap cases (first 3) must be near-exact -- that is the
    # core claim (scale + rotation recovered from the ellipsoids alone). Noise
    # and partial-overlap cases are EXPECTED to degrade (the ellipsoid inflates
    # under noise; partial overlap biases the moments) and are refined by ICP.
    ok = all(r[0] < 0.01 and r[1] < 1.0 for r in results[:3])
    print("CLEAN FULL-OVERLAP RECOVERY (exact):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
