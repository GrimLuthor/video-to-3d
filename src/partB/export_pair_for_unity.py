"""Export a Part-C aligned PAIR for the Unity two-video toggle viewer.

Produces, in <out-dir>/<pair>/:
  cloud_A.ply, cloud_B.ply     -- the two captures, BOTH already in the common
                                  (target 'A') coordinate frame, native colour.
  camerasA.json, framesA/       -- video A's registered camera poses + thumbnails.
  camerasB.json, framesB/       -- video B's poses, TRANSFORMED into the common frame.
  pair_info.json                -- object name, the similarity transform B->A, counts.

Cloud A (the target) is saved by run_combine unchanged, so the common frame IS A's
own reconstruction frame; A's poses need no transform. Cloud B (the source) was
moved by run_combine's similarity T, so B's poses are transformed by the SAME T,
which we recover exactly via Umeyama between the source input cloud and the saved
_src_aligned cloud (run_combine transforms in place, order preserved). Camera
position -> s*R*C + t; forward/up rotate by R only (scale/translation don't rotate
a direction). This matches the pointcloud_scene coordinates so a frustum sits on
the cloud, and the nearest-frame metric (position + forward) works for both videos.

    python export_pair_for_unity.py --pair bin \
        --tgt-recon output/bin_1_loop_reconstruction.pkl --tgt-cloud output/bin_pair_dense_tgt_aligned.ply \
        --src-recon output/bin_2_loop_reconstruction.pkl --src-cloud output/bin_pair_dense_src_aligned.ply \
        --src-input output/bin_2_loop_dense_objcrop.ply
Green (built by manual steps) instead passes --transform output/green_T.npy.
"""
from __future__ import annotations
import argparse, json, shutil
import pickle
from pathlib import Path
import cv2, numpy as np, open3d as o3d
import sys
SRC = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SRC))
from extract_frames import extract_frames
import calibration

OUT = Path(__file__).resolve().parents[2] / "output"
UNITY = Path(__file__).resolve().parents[2] / "unity_export"


def umeyama(P, Q):
    """Similarity (s,R,t) with Q ~= s*R*P + t (Umeyama 1991)."""
    muP, muQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - muP, Q - muQ
    Sigma = (Qc.T @ Pc) / len(P)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    varP = (Pc ** 2).sum() / len(P)
    s = np.trace(np.diag(D) @ S) / varP
    t = muQ - s * R @ muP
    return s, R, t


def recover_transform(src_input_ply, src_aligned_ply):
    P = np.asarray(o3d.io.read_point_cloud(str(src_input_ply)).points)
    Q = np.asarray(o3d.io.read_point_cloud(str(src_aligned_ply)).points)
    n = min(len(P), len(Q))
    if len(P) != len(Q):
        print(f"  WARN input/aligned counts differ ({len(P)} vs {len(Q)}); using first {n}")
    s, R, t = umeyama(P[:n], Q[:n])
    resid = np.linalg.norm((s * (R @ P[:n].T).T + t) - Q[:n], axis=1)
    print(f"  recovered T: scale {s:.4f}, residual median {np.median(resid):.5f} (0 = exact)")
    return s, R, t


def export_poses(recon_pkl, out_dir, name, s, R, t, max_edge=640, jpg_q=88):
    b = pickle.load(open(recon_pkl, "rb"))
    cams, fidx = b["cameras"], b["frame_indices"]
    K = np.asarray(b["K"], float); stride = int(b["stride"]); vp = b["video_path"]
    frames_dir = out_dir / f"frames{name}"; frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{name}] extracting frames (stride {stride}) from {Path(vp).name} ...", flush=True)
    frames = calibration.undistort_frames(extract_frames(Path(vp), stride=stride), K)
    H, W = frames[0].shape[:2]
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    out = []
    for c, cam in enumerate(cams):
        Rc = np.asarray(cam["R"], float); tc = np.asarray(cam["t"], float).reshape(3)
        C = -Rc.T @ tc
        fwd = Rc.T @ np.array([0., 0., 1.]); upv = Rc.T @ np.array([0., -1., 0.])
        # into the common frame: position by full similarity, directions rotate only
        Cw = s * (R @ C) + t
        fwdw = R @ fwd; upw = R @ upv
        fwdw /= np.linalg.norm(fwdw); upw /= np.linalg.norm(upw)
        vframe = int(fidx[c]) * stride; img_name = f"{vframe:06d}.jpg"
        img = frames[int(fidx[c])]
        sc = max_edge / max(W, H)
        if sc < 1.0:
            img = cv2.resize(img, (max(1,round(W*sc)), max(1,round(H*sc))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frames_dir / img_name), img, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
        out.append({"frame": vframe, "image": f"frames{name}/{img_name}",
                    "position": Cw.tolist(), "forward": fwdw.tolist(), "up": upw.tolist(),
                    "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}})
    data = {"video": Path(vp).name, "frame_size": [W, H], "n_cameras": len(out),
            "coordinate_frame": "common (target A) reconstruction frame, matches cloud_*.ply",
            "camera_convention": "OpenCV +Z fwd/+X right/+Y down; position=camera centre; "
                                 "quat omitted (use forward+up). Apply the same glTFast flip as the clouds.",
            "cameras": out}
    (out_dir / f"cameras{name}.json").write_text(json.dumps(data, indent=2))
    print(f"  [{name}] -> cameras{name}.json ({len(out)} cams) + thumbnails")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True, help="object name (bin/tree/green) -> unity_export/pair_<name>/")
    ap.add_argument("--tgt-recon", required=True); ap.add_argument("--tgt-cloud", required=True)
    ap.add_argument("--src-recon", required=True); ap.add_argument("--src-cloud", required=True)
    ap.add_argument("--src-input", default=None, help="source cloud fed to run_combine (for T recovery)")
    ap.add_argument("--transform", default=None, help="npy dict {s,R,t} instead of recovering from clouds")
    args = ap.parse_args()

    out_dir = UNITY / f"pair_{args.pair}"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== pair '{args.pair}' -> {out_dir} ==")
    # clouds (already in common frame)
    shutil.copy(args.tgt_cloud, out_dir / "cloud_A.ply")
    shutil.copy(args.src_cloud, out_dir / "cloud_B.ply")
    nA = len(o3d.io.read_point_cloud(str(out_dir/"cloud_A.ply")).points)
    nB = len(o3d.io.read_point_cloud(str(out_dir/"cloud_B.ply")).points)
    print(f"  cloud_A {nA:,} pts, cloud_B {nB:,} pts")

    if args.transform:
        T = np.load(args.transform, allow_pickle=True).item()
        s, R, t = T["s"], np.asarray(T["R"]), np.asarray(T["t"])
        print(f"  loaded T from {args.transform}: scale {s:.4f}")
    else:
        s, R, t = recover_transform(args.src_input, args.src_cloud)

    export_poses(args.tgt_recon, out_dir, "A", 1.0, np.eye(3), np.zeros(3))   # A: identity
    export_poses(args.src_recon, out_dir, "B", s, R, t)                        # B: into common frame

    info = {"object": args.pair, "cloud_A": "target capture", "cloud_B": "source capture (aligned to A)",
            "transform_B_to_A": {"scale": float(s), "R": R.tolist(), "t": t.tolist()},
            "points": {"A": nA, "B": nB},
            "note": "toggle A/B/both; tint A=blue B=red or show native colour; nearest frame from either video."}
    (out_dir / "pair_info.json").write_text(json.dumps(info, indent=2))
    print(f"  wrote pair_info.json  DONE")


if __name__ == "__main__":
    main()
