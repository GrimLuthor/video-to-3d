"""
Part B runner: combine two independent SfM reconstructions into one point cloud.

Pipeline:  E-Init(+scale)  ->  scaled ICP refine  ->  transform + merge  ->  PLY
Optional:  --baseline  runs FPFH+RANSAC (scale-normalized) for comparison.

Real use (two clouds of the same place, each from Part A):
    python Final/src/partB/run_combine.py SOURCE.ply TARGET.ply --tag mypair

Dry run with NO new capture (splits one cloud into two overlapping halves with a
known random similarity, then re-merges -- validates the whole pipeline):
    python Final/src/partB/run_combine.py Final/output/einstein_s8_dense_showcase.ply \
        --dry-run-split --tag einstein_split
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ellipsoid_init import (e_init_similarity, rotation_angle_deg,
                            gravity_yaw_candidates)
from similarity_icp import refine_similarity, fpfh_ransac_init
from piecewise_refine import piecewise_refine

OUT_DIR = Path(__file__).resolve().parents[2] / "output"


def cam_centers(recon_pkl):
    b = pickle.load(open(recon_pkl, "rb"))
    return np.array([(-c["R"].T @ c["t"]).ravel() for c in b["cameras"]])


def core_crop(pcd, centers, pct):
    """Keep only points near the camera path (the region this walk actually
    observed up close). Points far from every camera are the far, non-shared
    structure that biases global registration -- drop them. pct = quantile of
    point->nearest-camera distance to keep."""
    pts = np.asarray(pcd.points)
    d, _ = cKDTree(centers).query(pts, k=1, workers=-1)
    keep = d <= np.quantile(d, pct)
    return pcd.select_by_index(np.where(keep)[0])


def gravity_up(recon_pkl):
    """Signed world up-vector of a reconstruction, from its camera poses.

    The phone was held roughly upright throughout the walk, so the camera's
    image-up direction (-Y in OpenCV camera coords) maps to world as R^T[0,-1,0];
    averaged over all cameras this points along gravity-up. Returns a unit 3-vec.
    """
    b = pickle.load(open(recon_pkl, "rb"))
    ups = [c["R"].T @ np.array([0.0, -1.0, 0.0]) for c in b["cameras"]]
    u = np.mean(ups, axis=0)
    return u / (np.linalg.norm(u) + 1e-12)


# ---------------------------------------------------------------------------
def render(pcd, out_png, point_size=2.0, zoom=0.6):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.light_on = False
    opt.point_size = point_size
    opt.background_color = np.array([1.0, 1.0, 1.0])
    vc = vis.get_view_control()
    vc.set_lookat(pcd.get_axis_aligned_bounding_box().get_center())
    vc.set_front([0.0, -0.3, -1.0])
    vc.set_up([0.0, -1.0, 0.0])
    vc.set_zoom(zoom)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(str(out_png), do_render=True)
    vis.destroy_window()
    print(f"  saved {out_png}")


def two_tone(src, tgt):
    """Copy of src (red) + tgt (blue) to visualize alignment/overlap."""
    a = o3d.geometry.PointCloud(src); a.paint_uniform_color([0.85, 0.12, 0.12])
    b = o3d.geometry.PointCloud(tgt); b.paint_uniform_color([0.12, 0.35, 0.90])
    return a + b


def make_split(pcd, rng, overlap=0.6, s=None, noise_frac=0.004):
    """Split one cloud into two spatially-overlapping halves; apply a known
    random similarity to the 'source' half. Returns (src, tgt, gt) where gt is
    the ground-truth similarity mapping src -> tgt."""
    pts = np.asarray(pcd.points)
    col = np.asarray(pcd.colors) if pcd.has_colors() else None
    axis = 0
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    span = hi - lo
    mid = lo + span * 0.5
    half = span * (0.5 + overlap / 2.0)
    left = pts[:, axis] <= lo + half           # target half
    right = pts[:, axis] >= hi - half          # source half

    def sub(mask):
        q = o3d.geometry.PointCloud()
        q.points = o3d.utility.Vector3dVector(pts[mask])
        if col is not None:
            q.colors = o3d.utility.Vector3dVector(col[mask])
        return q

    tgt = sub(left)
    src = sub(right)

    s = s if s is not None else float(rng.uniform(0.4, 2.5))
    R = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    t = rng.uniform(-8, 8, size=3)
    P = np.asarray(src.points)
    P2 = (s * R @ P.T).T + t
    P2 += rng.normal(scale=noise_frac * span, size=P2.shape)
    src.points = o3d.utility.Vector3dVector(P2)
    # gt similarity src->tgt is the inverse of what we applied
    gt = dict(s=1.0 / s, R=R.T, t=-(1.0 / s) * (R.T @ t))
    print(f"  dry-run split: tgt={len(tgt.points)} src={len(src.points)} "
          f"pts, applied scale x{s:.3f}")
    return src, tgt, gt


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="source cloud .ply (moved into target frame)")
    ap.add_argument("target", nargs="?", default=None, help="target cloud .ply")
    ap.add_argument("--tag", default="combined")
    ap.add_argument("--init-voxel", type=float, default=0.0,
                    help="voxel downsample before E-Init scoring (0 = raw)")
    ap.add_argument("--icp-voxel", type=float, default=0.0,
                    help="voxel size for ICP refinement (0 = auto: target_diag/200)")
    ap.add_argument("--merge-voxel", type=float, default=0.0,
                    help="voxel for the final merged cloud (0 = auto: target_diag/300)")
    ap.add_argument("--allow-reflection", action="store_true")
    ap.add_argument("--select", choices=["icp", "nn"], default="icp",
                    help="candidate selection: 'icp' (refine all, pick best "
                         "overlap -- robust to flips) or 'nn' (paper's raw NN)")
    ap.add_argument("--init-mode", choices=["einit", "identity", "centroid"],
                    default="einit",
                    help="what ICP starts from -- for the ablation: 'einit' (the "
                         "paper's ellipsoid init), 'identity' (no init at all), or "
                         "'centroid' (barycenter+scale only, no ellipsoid rotation)")
    ap.add_argument("--src-recon", default=None,
                    help="source reconstruction .pkl (enables gravity up-gate)")
    ap.add_argument("--tgt-recon", default=None,
                    help="target reconstruction .pkl (enables gravity up-gate)")
    ap.add_argument("--yaw-search", type=int, default=0,
                    help="N: with gravity, brute-force N yaw samples about up "
                         "(for degenerate/near-symmetric footprints). 0 = off")
    ap.add_argument("--core-crop", action="store_true",
                    help="register only the near-camera CORE of each cloud "
                         "(drops far, non-shared structure); needs recon pkls")
    ap.add_argument("--core-pct", type=float, default=0.6,
                    help="keep points within this quantile of point->nearest-camera dist")
    ap.add_argument("--piecewise", type=int, default=0,
                    help="N>0: after global ICP, apply an N-segment piecewise-"
                         "similarity refinement (smoothly blended) to correct "
                         "residual scale DRIFT along the path. 0 = off")
    ap.add_argument("--baseline", action="store_true",
                    help="also run FPFH+RANSAC (scale-normalized) for comparison")
    ap.add_argument("--dry-run-split", action="store_true",
                    help="split SOURCE into two overlapping halves w/ known transform")
    ap.add_argument("--overlap", type=float, default=0.6,
                    help="dry-run overlap fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(exist_ok=True)
    metrics = {"tag": args.tag}

    # ---- load inputs -----------------------------------------------------
    gt = None
    if args.dry_run_split:
        base = o3d.io.read_point_cloud(args.source)
        src_pcd, tgt_pcd, gt = make_split(base, rng, overlap=args.overlap)
    else:
        if args.target is None:
            ap.error("need a TARGET cloud unless --dry-run-split is given")
        src_pcd = o3d.io.read_point_cloud(args.source)
        tgt_pcd = o3d.io.read_point_cloud(args.target)
    print(f"source: {len(src_pcd.points):,} pts   target: {len(tgt_pcd.points):,} pts")

    # Clouds used for REGISTRATION. With --core-crop we register only the near-
    # camera core of each (dropping far, non-shared structure that biases global
    # alignment); the FULL clouds are still used for the final merge/overlay.
    src_reg, tgt_reg = src_pcd, tgt_pcd
    if args.core_crop:
        if not (args.src_recon and args.tgt_recon):
            ap.error("--core-crop needs --src-recon and --tgt-recon")
        src_reg = core_crop(src_pcd, cam_centers(args.src_recon), args.core_pct)
        tgt_reg = core_crop(tgt_pcd, cam_centers(args.tgt_recon), args.core_pct)
        print(f"core-crop @ pct {args.core_pct}: "
              f"src {len(src_pcd.points):,}->{len(src_reg.points):,}   "
              f"tgt {len(tgt_pcd.points):,}->{len(tgt_reg.points):,}")

    # scale-relative voxels from the registration target's size
    tgt_diag = float(np.linalg.norm(tgt_reg.get_axis_aligned_bounding_box().get_extent()))
    icp_voxel = args.icp_voxel if args.icp_voxel > 0 else tgt_diag / 200.0
    merge_voxel = args.merge_voxel if args.merge_voxel > 0 else tgt_diag / 300.0
    print(f"reg-target diag {tgt_diag:.2f}  ->  icp_voxel {icp_voxel:.3f}  merge_voxel {merge_voxel:.3f}")

    src_pts = np.asarray(src_reg.points)
    tgt_pts = np.asarray(tgt_reg.points)
    if args.init_voxel > 0:
        src_pts = np.asarray(src_reg.voxel_down_sample(args.init_voxel).points)
        tgt_pts = np.asarray(tgt_reg.voxel_down_sample(args.init_voxel).points)

    # ---- 1) E-Init(+scale) ----------------------------------------------
    print("\n[1] E-Init (ellipsoid + eigenvalue-ratio scale)")
    up_s = up_t = None
    if args.src_recon and args.tgt_recon:
        up_s, up_t = gravity_up(args.src_recon), gravity_up(args.tgt_recon)
        print(f"    gravity up-gate ON  (src up {np.round(up_s,2)}, tgt up {np.round(up_t,2)})")
    init = e_init_similarity(src_pts, tgt_pts,
                             allow_reflection=args.allow_reflection,
                             seed=args.seed, up_source=up_s, up_target=up_t)
    print(f"    recovered scale s = {init['s']:.4f}   "
          f"per-axis {np.round(init['scale_per_axis'], 3)}   "
          f"(spread => how similarity-like the two clouds are)")
    print(f"    tested {init['n_candidates']} sign-candidates, NN score {init['score']:.4f}")
    metrics["einit"] = dict(scale=init["s"],
                            scale_per_axis=init["scale_per_axis"].tolist(),
                            nn_score=init["score"])
    if gt is not None:
        e_s = abs(init["s"] - gt["s"]) / gt["s"]
        e_r = rotation_angle_deg(init["R"], gt["R"])
        print(f"    vs ground truth: scale_err {100*e_s:.2f}%   rot_err {e_r:.2f} deg")
        metrics["einit"].update(gt_scale_err=e_s, gt_rot_err_deg=e_r)

    # ---- 2) scaled-ICP refinement ---------------------------------------
    # Raw NN distance is a weak selector among the sign-flip candidates when the
    # footprint is near-symmetric / overlap is partial -- it can pick an
    # upside-down flip. So (default) we ICP-refine EVERY candidate and keep the
    # one whose refined overlap (fitness) is best. A wrong flip refines to a much
    # lower fitness than the true orientation.
    print(f"\n[2] ICP refinement  (init-mode: {args.init_mode})")
    if args.init_mode == "identity":                      # ablation: no init
        cand_list = [dict(T=np.eye(4))]
        labeler = lambda c: "identity (no init)"
    elif args.init_mode == "centroid":                    # ablation: no ellipsoid rot
        bs, bt = src_pts.mean(0), tgt_pts.mean(0)
        T = np.eye(4); T[:3, :3] = init["s"] * np.eye(3); T[:3, 3] = bt - init["s"] * bs
        cand_list = [dict(T=T)]
        labeler = lambda c: "centroid+scale"
    elif args.yaw_search > 0 and up_s is not None:
        cand_list = gravity_yaw_candidates(src_pts, tgt_pts, up_s, up_t,
                                           init["s"], n_yaw=args.yaw_search)
        print(f"    gravity yaw-search: {len(cand_list)} samples about up-axis")
        labeler = lambda c: f"yaw {c['yaw_deg']:5.0f}deg"
    else:
        cand_list = init["candidates"]
        labeler = lambda c: f"NN {c['score']:.3f}"

    cand_reports = []
    for i, c in enumerate(cand_list):
        r = refine_similarity(src_reg, tgt_reg, c["T"], voxel=icp_voxel)
        cand_reports.append(r)
        print(f"    cand {i:2d} [{labeler(c)}]  ->  ICP fitness {r['fitness']:.3f}"
              f"  rmse {r['inlier_rmse']:.4f}")
        if args.select == "nn" and args.yaw_search == 0:
            break
    ref = (cand_reports[0] if args.select == "nn" and args.yaw_search == 0
           else max(cand_reports, key=lambda r: r["fitness"]))
    print(f"    selected fitness {ref['fitness']:.3f}   inlier_rmse {ref['inlier_rmse']:.4f}   "
          f"scale {ref['scale']:.4f}")
    metrics["icp"] = dict(fitness=ref["fitness"], inlier_rmse=ref["inlier_rmse"],
                          scale=ref["scale"], seconds=ref["seconds"])
    if gt is not None:
        R_ref = ref["T"][:3, :3] / ref["scale"]
        e_s = abs(ref["scale"] - gt["s"]) / gt["s"]
        e_r = rotation_angle_deg(R_ref, gt["R"])
        print(f"    vs ground truth: scale_err {100*e_s:.2f}%   rot_err {e_r:.2f} deg")
        metrics["icp"].update(gt_scale_err=e_s, gt_rot_err_deg=e_r)

    # ---- optional baseline ----------------------------------------------
    if args.baseline:
        print("\n[*] baseline FPFH+RANSAC (scale-normalized by E-Init scale)")
        src_norm = o3d.geometry.PointCloud(src_pcd)
        src_norm.scale(init["s"], center=src_norm.get_center())
        base = fpfh_ransac_init(src_norm, tgt_pcd, voxel=icp_voxel)
        base_ref = refine_similarity(src_norm, tgt_pcd, base["T"],
                                     voxel=icp_voxel)
        print(f"    RANSAC fitness {base['fitness']:.3f} ({base['seconds']:.1f}s) "
              f"-> +ICP fitness {base_ref['fitness']:.3f} "
              f"rmse {base_ref['inlier_rmse']:.4f}")
        metrics["baseline_fpfh_ransac"] = dict(
            ransac_fitness=base["fitness"], ransac_seconds=base["seconds"],
            icp_fitness=base_ref["fitness"], icp_inlier_rmse=base_ref["inlier_rmse"])

    # ---- 3) transform + merge -------------------------------------------
    print("\n[3] merge")
    src_moved = o3d.geometry.PointCloud(src_pcd).transform(ref["T"])

    if args.piecewise > 0:
        # principal (path) axis = larger-spread horizontal axis of the target
        ext = tgt_pcd.get_axis_aligned_bounding_box().get_extent()
        up_axis = int(np.argmin(ext))                    # up ~ smallest extent
        horiz = [a for a in range(3) if a != up_axis]
        axis = horiz[int(np.argmax([ext[a] for a in horiz]))]
        pw_voxel = tgt_diag / 200.0
        tgt_ds = tgt_pcd.voxel_down_sample(pw_voxel)
        warped, info = piecewise_refine(np.asarray(src_moved.points), tgt_ds,
                                        axis=axis, n_seg=args.piecewise,
                                        max_corr_dist=pw_voxel * 6)
        src_moved.points = o3d.utility.Vector3dVector(warped)
        print(f"    piecewise refine: {args.piecewise} segments on axis "
              f"{'XYZ'[axis]}, local scales {np.round(info['scales'],3)}")
        metrics["piecewise"] = info
    before = two_tone(src_pcd, tgt_pcd)
    after = two_tone(src_moved, tgt_pcd)
    render(before, OUT_DIR / f"{args.tag}_before.png")
    render(after, OUT_DIR / f"{args.tag}_after.png")

    # two-tone aligned overlay (red = source, blue = target) for inspecting overlap
    overlay_ply = OUT_DIR / f"{args.tag}_overlay.ply"
    o3d.io.write_point_cloud(str(overlay_ply), after)
    print(f"    overlay (red=src, blue=tgt) -> {overlay_ply}")

    # each aligned scene in its OWN native colour, for the toggle viewer
    o3d.io.write_point_cloud(str(OUT_DIR / f"{args.tag}_src_aligned.ply"), src_moved)
    o3d.io.write_point_cloud(str(OUT_DIR / f"{args.tag}_tgt_aligned.ply"), tgt_pcd)

    merged = src_moved + tgt_pcd
    if merge_voxel > 0:
        merged = merged.voxel_down_sample(merge_voxel)
    out_ply = OUT_DIR / f"{args.tag}_merged.ply"
    o3d.io.write_point_cloud(str(out_ply), merged)
    print(f"    merged {len(merged.points):,} pts -> {out_ply}")
    render(merged, OUT_DIR / f"{args.tag}_merged.png")
    metrics["merged_points"] = len(merged.points)

    with open(OUT_DIR / f"{args.tag}_partB_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDone. metrics -> {args.tag}_partB_metrics.json")


if __name__ == "__main__":
    main()
