# Monocular video → 3D scene (classical, no CNNs)

Reconstructs a 3D scene from a single handheld phone video using classical
computer vision only (SIFT + windowed multi-view incremental SfM + bundle
adjustment → plane-sweep MVS → fusion → Poisson mesh). No neural networks.

This repo is the **Python pipeline only** — the Unity viewer, input videos,
generated outputs, and report images are intentionally not tracked (see
`.gitignore`). It's meant to be cloned on a second machine to process videos in
parallel, then the results zipped back.

## Demo

▶️ **[Video walkthroughs (playlist)](https://www.youtube.com/playlist?list=PLQ1XqiA4sMEs)**
— every reconstructed scene, the orbited objects, and the two-capture merging, all in the
interactive Unity viewer (mesh vs. point cloud, with the nearest real frame shown alongside
the view). A still gallery is in [`media/GALLERY.md`](media/GALLERY.md).

## Downloads

Prebuilt viewer, exported 3D assets, and the source videos are on Google Drive —
no build step required:

**[⬇ Download folder](https://drive.google.com/drive/folders/122Sn61TLV2FvIv6QcKYfbAaIMkctk2dS?usp=sharing)**

- `viewer_win.zip` — standalone Windows build of the interactive viewer (unzip and run `VIdeoTo3D.exe`).
- `unity_assets.zip` — every scene's mesh + point cloud (+ camera poses and frame thumbnails).
- `videos/` — the source phone videos.

## Setup (fresh machine)

Use **Python 3.11 or 3.12** (this was built on 3.12; open3d needs a matching wheel).

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/mac: source venv/bin/activate
pip install -r requirements.txt
```

Reference versions known to work: numpy 2.4, opencv-python 4.13, open3d 0.19,
scipy 1.18.

## Calibration (read this before running someone else's video)

Camera intrinsics `K` and distortion are **hardcoded** in
[`src/calibration.py`](src/calibration.py) for a **Samsung S22 Ultra, video mode,
1920×1080**. If your footage is from the same phone/mode it just works. For a
different camera or resolution you must recalibrate:
- `src/calibrate_from_video.py` (record `src/../calibration/checkerboard_9x6.png`
  shown on a screen), then paste the resulting `K` / dist into `calibration.py`.
- The aspect ratio must match 16:9, or `scale_K` will refuse (a guard against a
  wrong-crop K silently biasing the geometry).

## Run the pipeline on a video

Run all commands **from the repo root** (scripts resolve `output/` relative to
themselves, so cwd doesn't matter for outputs, but the video path is relative to
cwd). Put your video anywhere, e.g. `capture/myclip.mp4`.

```bash
# 1) SfM: poses + sparse cloud  (stride 8; --window 10 = wider tracks)
python src/run_incremental_sfm.py capture/myclip.mp4 8 myclip_s8 --window 10

# 2) Dense MVS + fusion  (--save-depths lets the cull be re-tuned offline)
python src/run_fusion.py myclip_s8 --save-depths --out-suffix _showcase

# 3) (optional) Poisson mesh  (smoothing scales INVERSELY with density:
#    ~0 for very dense clouds, ~15 for sparse; 5 is a safe middle)
python src/run_mesh.py output/myclip_s8_dense_showcase.ply --smooth 5
```

Tuning knobs worth knowing:
- **`stride`** (arg 2): larger = fewer frames = faster but sparser. For a *long*
  video, bump it (e.g. 12–15) so SfM/dense stay tractable.
- **`--pose-jump-guard`** (default 12): rejects a PnP pose that jumps
  implausibly far from its neighbour — the guard against glass/reflection
  mismatches corrupting the reconstruction. `0` disables. Self-scaling; inert on
  clean walks.
- **`run_fusion.py --scale`** (default 0.5): MVS resolution. `1.0` = full-res,
  ~4× slower, crisper. `--planes`, `--min-support`, `--rel-thresh` tune density
  vs. cleanliness.

Progress prints live to the console (SfM registration + BA RMSE; dense stage has a
per-depth-map ETA). Runtime is dominated by the dense stage (~15–20s per frame).

## Part B & C — object loops and merging two captures

Part A reconstructs a *scene* from a walk-past video. Part B/C instead **orbit a single
object** in a loop, so the loop can be closed and the object cropped out of its
background, and then **merge two separate loops of the same object** into one model.
Everything is in `src/partB/` and does not touch the Part A pipeline; see
[`PART_B_SUMMARY.md`](PART_B_SUMMARY.md) for the full writeup and the method background.

```bash
# 1) Loop-closed SfM of an orbit (like run_incremental_sfm, + loop closure)
python src/partB/run_loop_sfm.py capture/myobject.mp4 5 myobject_loop --window 10
#    --no-loop  runs the same pipeline WITHOUT loop closure (the ablation)

# 2) Dense MVS + fusion (same runner as Part A; the loop bundle is phase-0)
python src/run_fusion.py myobject_loop --save-depths
#    run_fusion's voxel is ABSOLUTE, so a small-scale reconstruction over-merges;
#    re-fuse finer from the saved depths:
#    python src/run_fusion.py myobject_loop --depths-from "" --voxel <~0.002*orbit_radius> --out-suffix _fine

# 3) Crop the object out of its background (automatic from the camera geometry)
python src/partB/crop_to_orbit_center.py myobject_loop --ply output/myobject_loop_dense.ply --frac 0.8
```

Then merge two captures of the same object with ellipsoid ICP (Kolpakov & Werman,
extended to also recover the unknown scale):

```bash
# register SOURCE onto TARGET  (--src/--tgt-recon enable the gravity up-gate)
python src/partB/run_combine.py output/objB_crop.ply output/objA_crop.ply --tag obj_pair \
    --src-recon output/objB_loop_reconstruction.pkl --tgt-recon output/objA_loop_reconstruction.pkl
#    --init-mode {einit,identity,centroid} = the ablation; --yaw-search N helps near-symmetric objects
python src/partB/view_overlay.py obj_pair          # interactive red/blue toggle overlay

# export the aligned pair for the Unity two-video viewer (poses for BOTH captures)
python src/partB/export_pair_for_unity.py --pair obj \
    --tgt-recon output/objA_loop_reconstruction.pkl --tgt-cloud output/obj_pair_tgt_aligned.ply \
    --src-recon output/objB_loop_reconstruction.pkl --src-cloud output/obj_pair_src_aligned.ply \
    --src-input output/objB_crop.ply
```

Two things worth knowing:
- This works only for a **complete, orbited object** — both loops must see it from all
  sides. It does *not* merge two ordinary scene walks; `PART_B_SUMMARY.md` explains why
  (the covariance ellipsoid is sampling-dependent).
- It works best on an **asymmetric, textured** object. A round or symmetric object leaves
  a residual rotation that geometry alone cannot fix, and colour has to break the tie.

## Transferring results back to the main machine

After a run, zip the `output/` files for your tag and copy them over. The ones
that matter:

- `myclip_s8_reconstruction.pkl` — poses (R,t), K, sparse cloud, frame indices **(required)**
- `myclip_s8_dense_showcase.ply` — the dense coloured point cloud **(the main result)**
- `myclip_s8_dense_showcase_stats.json` — point/cull/outlier counts
- `myclip_s8_camera_centers.npy` — trajectory
- `myclip_s8_mesh.ply` — the mesh, if you ran step 3
- `myclip_s8_depths_showcase.pkl` — saved depth maps *(optional, large; only if
  you want to re-tune the cull/fusion offline via `--depths-from _showcase`)*

On the main machine, drop them into `output/`, then run the Unity export
(`src/export_for_unity.py`, `src/export_poses_for_unity.py`) or inspect with
`src/view_ply.py`.

## Layout

- `src/` — Part A pipeline code (SfM, MVS, fusion, mesh, exporters, viewers, utils).
- `src/partB/` — Part B/C: loop-closed object SfM, orbit crop, and ellipsoid-ICP merging.
- `calibration/` — checkerboard image for `calibrate_from_video.py`.
- `capture/`, `output/` — created locally; git-ignored (drop videos in / results out).
- `PROJECT_SUMMARY.md`, `PART_B_SUMMARY.md`, `plan.md` — design notes and roadmap (background context).
