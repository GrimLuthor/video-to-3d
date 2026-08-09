# Project summary — 3D reconstruction from a monocular video (classical) + Unity viewer

Portable handoff / context bridge. Point Claude Code at this file from the Unity
(Rider) session so it has full context — the detailed running notes live in the
Python folder's Claude memory, which is directory-scoped and won't auto-load elsewhere.

## Goal
Reconstruct a real 3D scene from ONE handheld phone video (monocular; no depth
sensor, **no CNNs** — classical computer-vision only) and view it as an
interactive fly-through in Unity. Course project.

## Two codebases (kept separate, bridged by exported files)
- **Python pipeline**: `c:\Users\Daniel\Desktop\Tasks now\Computer Vision\Final`
  (source in `Final/src`, outputs in `Final/output`).
- **Unity viewer**: `Final/VideoTo3D/` (the Unity project). Consumes the assets in
  `Final/unity_export/`. Edit its C# in Rider (set Rider as Unity's External Script
  Editor). **When starting a Unity/Rider session, read `Final/VideoTo3D/CLAUDE_START.md`
  first** (it points back here + to the unity_export README).

## Source footage
`Final/capture/longer_walk.mp4` — ~48 s handheld walk (Samsung S22 Ultra, video
mode, EIS off) past a stone building/courtyard with trees, grass, a parked car,
and a bin. Calibration in `src/calibration.py` (video-mode K, checkerboard).

## Pipeline stages + key scripts (all in `Final/src`)
1. **SfM (poses + sparse cloud)** — `run_incremental_sfm.py` (entry point:
   `python run_incremental_sfm.py <video> <stride>`).
   - `feature_tracks.py`: SIFT + **RootSIFT** + **mutual cross-check** + windowed
     matching + **MAGSAC++** essential-matrix verification + union-find tracks.
   - `incremental_sfm.py`: seed pair -> PnP registration -> multi-view triangulation
     -> recovery pass -> `bundle_adjustment.py` (sparse robust BA).
   - Solved the original **~12x monocular scale drift** by using windowed
     multi-view tracks (wide baselines) instead of a chain of consecutive pairs.
2. **Dense depth (MVS)** + **fusion** — `run_fusion.py`.
   - `plane_sweep.py`: plane-sweep depth (ZNCC, multi-baseline, best-K), cross-view
     `geometric_consistency`, `free_space_cull` (support-aware ghost removal), and
     an experimental `patchmatch_depth` (kept OFF — marginal).
   - Fusion default = "B" setting (min_support 3, rel_thresh 0.015). Offline
     re-fuse from saved depth maps via `--depths-from`.
3. **Meshing** — `run_mesh.py`: normals -> Poisson (depth 10) -> density trim ->
   crop -> **Taubin smoothing** -> coloured mesh + offscreen preview.
4. **Unity export** — `export_for_unity.py`: decimated vertex-coloured **GLB** mesh
   + thinned **PLY** point cloud + a Unity README, into `Final/unity_export/`.
5. **Viewers/tools**: `view_ply.py` (legacy Open3D viewer; auto-detects meshes;
   `K/J/H` depth-axis; flat vertex colours), `clean_cloud.py`, `tune_cull.py`,
   `classify_foliage.py`.

## Current best outputs (`Final/output`)
- `longer_walk_s8_dense_showcase.ply` — **4.6M-pt** coloured cloud (stride 8 +
  cleaned front-end). The best point cloud.
- `longer_walk_s8_mesh.ply` — smoothed coloured surface mesh (1.77M verts).
- `longer_walk_s8_depths_showcase.pkl` — saved depth maps (offline fusion tuning).
- `longer_walk_incr_dense_b.ply` — stride-15 reference cloud.

## Unity assets (`Final/unity_export/`) — for the SIDE-BY-SIDE fly-through
- `mesh_scene.glb` — **full-resolution** vertex-coloured mesh, Taubin smoothing 1
  (import via **glTFast**). Not decimated (user runs on an RTX 3070). Per-scene tri
  counts in `scene_info.json` (`mesh.triangles`, `mesh.smoothing_iterations`).
- `pointcloud_scene.ply` — dense cloud (import via **Pcx**; use "Point" mode).
- `scene_info.json` — machine-readable scene stats for any info/description panel:
  `scene`, `source_video`, `mesh{triangles,vertices}`, `pointcloud{points}`,
  `coordinate_notes`, and `reconstruction_stats{reference_views, cameras_used,
  raw_fused, after_downsample, visibility_cull_removed, outlier_removed,
  final_points}`. **Read this for counts / outlier count / camera count.**
- `README.md` — full Unity setup (unlit vertex-colour shader, layout, fly-through).
- Per-scene variants in subfolders, e.g. `unity_export/walk/`.
- Camera path for the fly-through: `output/<scene>_camera_centers.npy`; full poses
  (for the frame-comparison overlay) in `output/<scene>_reconstruction.pkl`.
- `cameras.json` + `frames/######.jpg` — per registered frame: `position`
  (camera centre), `forward`/`up`/`quat` (OpenCV +Z fwd/+Y down, camera->world),
  `intrinsics`, and a downscaled thumbnail, all in the SAME frame as
  `pointcloud_scene.ply`. Drives the **frame-comparison overlay** (nearest real
  frame to the fly-cam). Regenerate via `src/export_poses_for_unity.py --tag <tag>
  --out-dir unity_export/<scene>` (`--max-edge` thumbnail size).

## Key decisions & honest findings (so we don't re-litigate)
- **Scale drift SOLVED** (windowed multi-view SfM), not a monocular limitation.
- **Front-end cleanup** (RootSIFT + mutual cross-check + MAGSAC++) + **stride 8**
  = the big quality jump (less wall doubling, recovered a distant bin & a tree
  trunk). Registered 149/178 frames.
- **PatchMatch depth**: marginal (trades wall-thinness for foreground/edge errors,
  fronto-parallel ceiling) -> kept plane-sweep + B fusion.
- **Foliage segmentation** (classical geometry+colour): INSUFFICIENT on this noisy
  cloud (speckled) -> keep the FULL mesh; the side-by-side's point-cloud half
  showcases foliage instead. (Legit report finding.)
- **Gaussian Splatting** would represent foliage / novel views far better and
  REUSES our SfM poses+cloud as input, but it's a modern optimization method
  (outside the classical scope; vanilla 3DGS has no neural net though) and needs a
  CUDA GPU. Framed as "future work our pipeline sets up." User has an RTX 3070.
- **Open3D filament GUI** black-screens on this machine (Windows laptop Optimus /
  driver quirk, NOT GPU power) -> `view_ply.py` defaults to the legacy renderer;
  `--gui` is opt-in only.
- Point-cloud "lag" concern was hardware-conservative; on the RTX 3070 the full
  4.6M cloud is fine. If it ever stutters it's Pcx disk/billboard OVERDRAW -> use
  Pcx "Point" mode.

## Report assets
`Final/report_assets/` — informatively-named stage images `00`..`16` + `MANIFEST.md`
documenting the development story (SfM/scale-drift -> depth -> consistency ->
fusion -> near-fixes -> cull -> tuning -> PatchMatch -> mesh -> foliage finding).

## How to resume
- Rebuild from scratch on any video: `run_incremental_sfm.py <video> <stride>`
  then `run_fusion.py <tag> [--save-depths]` then `run_mesh.py` then
  `export_for_unity.py`. Stride is a user/prof-tunable arg.
- **Next up = Unity side-by-side**: create the Unity project, import the two
  `unity_export/` assets (Pcx + glTFast), apply an unlit vertex-colour shader,
  place mesh and cloud side-by-side, animate a camera along the trajectory path.
