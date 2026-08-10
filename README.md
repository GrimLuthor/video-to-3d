# Monocular video → 3D scene (classical, no CNNs)

Reconstructs a 3D scene from a single handheld phone video using classical
computer vision only (SIFT + windowed multi-view incremental SfM + bundle
adjustment → plane-sweep MVS → fusion → Poisson mesh). No neural networks.

This repo is the **Python pipeline only** — the Unity viewer, input videos,
generated outputs, and report images are intentionally not tracked (see
`.gitignore`). It's meant to be cloned on a second machine to process videos in
parallel, then the results zipped back.

## Demo

▶️ **[Video walkthrough](https://youtu.be/Eb8VIMXfWzk)** — every reconstructed scene
and the interactive Unity viewer (mesh vs. point cloud, with the nearest real frame
shown alongside the view). A still gallery is in [`media/GALLERY.md`](media/GALLERY.md).

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

- `src/` — all pipeline code (SfM, MVS, fusion, mesh, exporters, viewers, utils).
- `calibration/` — checkerboard image for `calibrate_from_video.py`.
- `capture/`, `output/` — created locally; git-ignored (drop videos in / results out).
- `PROJECT_SUMMARY.md`, `plan.md` — design notes and roadmap (background context).
