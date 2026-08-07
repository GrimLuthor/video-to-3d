# Final Project — Street 3D Reconstruction from Video

## Goal
Reconstruct a real 3D representation of a street (captured from a laterally
translating camera — walking, car, or boat) so it can be viewed from novel
viewpoints (zoomed in/out, moved along/into the scene), tolerating missing
data. Later: import the result into Unity for interactive viewing.

This is the multi-view generalization of `ex2` (two-view epipolar geometry:
SIFT matching, Hartley normalization, RANSAC 5-point E, pose recovery,
triangulation). `ex4` (last semester, in this same folder) did *2D* stereo
mosaicing via optical-flow homographies and barcode blending — it faked a
parallax/stereo effect by sampling different x-offsets per frame, but never
recovered real 3D structure. This project replaces that with actual
calibrated multi-view geometry.

## Overall Roadmap
- **Stage 1 (this doc, active now):** capture protocol + N-view pipeline
  scaffold + validation on a short pilot clip. No bundle adjustment yet —
  just prove the chain of pairwise poses is sound and scale-consistent.
- **Stage 2:** full street capture + global bundle adjustment (least-squares
  refinement over all cameras + points) → accurate trajectory + sparse cloud
  for the whole street.
- **Stage 3 (the "B" plane-sweep half of the plan):** dense reconstruction —
  sweep depth-hypothesis planes and warp neighboring frames via homography
  (directly reusing the `ex4` warping code/experience), score photo-consistency,
  keep the best depth per pixel. Produces dense depth instead of a sparse cloud.
- **Stage 4:** fuse per-view depth into a unified point cloud / mesh, export
  (e.g. PLY/OBJ).
- **Stage 5:** Unity import + novel-view / fly-through rendering.

## Camera
Reusing the Samsung S22 Ultra calibration from `ex2` (Zhang's method,
checkerboard), main camera, 1x zoom:

```
K = [[1333.5,    0.0, 1000.0],
     [   0.0, 1333.5,  750.0],
     [   0.0,    0.0,    1.0]]
```
at effective resolution 2000x1500 (4:3). Distortion coefficients = 0 (S22
Ultra's ISP pre-corrects JPEG stills). **Caveat carried into Stage 1 Step 0
below: this K was derived from still photos — phone video mode may use a
different sensor crop/aspect ratio than stills, which would invalidate the
scaling assumption.** We reuse it as a bootstrap assumption (same philosophy
as `ex2_plan.md`: assume it's good, validate with checkpoints, only redo
calibration if a checkpoint actually fails) rather than blindly recalibrating.

---

## Stage 1: Capture Protocol + Pipeline Scaffold + Pilot Validation

### Step 1.0 — Recording settings checkpoint (do this first, before any real capture)
- Set the S22 Ultra to record video at **4:3 aspect** if the camera app
  offers it (Pro video / "3:4" mode), matching the aspect ratio the K matrix
  above was calibrated for.
- Turn **off** video stabilization (EIS). Stabilization warps frames in ways
  a static pinhole model doesn't capture.
- No zoom, main lens only (same physical camera as `ex2`).

**Checkpoint:** From the same tripod position, take one still photo and one
short (~1s) video clip. Compare the framing/content at the edges of both —
they should show the same scene extent. If the video is visibly more
cropped/zoomed or a different aspect ratio, **stop** — the `ex2` K does not
transfer as-is, and we need a quick dedicated calibration pass in the actual
video recording mode before continuing.

### Step 1.1 — Pilot clip capture
Doesn't need to be the real street yet — just enough to validate the code.
- 10-20 seconds, smooth **lateral translation** (like the `ex2` stereo pair,
  but continuous instead of 2 shots), minimal rotation.
- Scene with real depth variation and texture (same guidance as `ex2`: avoid
  blank walls / featureless surfaces).

### Step 1.2 — Frame extraction
Sample frames from the pilot video at a stride chosen so consecutive sampled
frames have a baseline similar in spirit to `ex2`'s ~10cm stereo pair.
Start with a guess (e.g. every 5th frame at 30fps) and adjust based on the
Step 1.3 checkpoint (inlier ratio / rotation too small → increase stride;
matching starts failing → decrease it).

### Step 1.3 — Pairwise pose chain (extends `ex2` steps 2–5 to N frames)
For every consecutive pair of sampled frames, reuse the exact `ex2` pipeline:
SIFT + Lowe ratio test → Hartley normalization → RANSAC 5-point essential
matrix (using K) → `recoverPose`. Collect per-pair inlier ratio, rotation
angle, translation direction.

**Checkpoint:** median inlier ratio > 30% (same bar as `ex2`), rotation angle
per consecutive pair small (a few degrees — much smaller than the single
`ex2` pair since consecutive video frames move less), translation directions
roughly consistent frame-to-frame (no wild flips). Fails → adjust stride or
recapture more smoothly.

### Step 1.4 — Naive pose chaining + scale propagation
This is the actual new geometric content in Stage 1 (full bundle adjustment
is Stage 2). Composing relative poses sequentially works for rotation, but
**translation scale is not shared across pairs** — `recoverPose` only
returns a unit-norm translation direction per pair, so pair (i, i+1)'s
translation has no inherent relationship to pair (i-1, i)'s scale. We fix
this with the classic incremental-SfM trick: for each new triple
(i-1, i, i+1), take 3D points already triangulated from (i-1, i) that are
also matched in frame i+1, and solve (least squares) for the scale factor
that makes pose (i, i+1)'s triangulated points consistent with the existing
point cloud, instead of assuming unit baseline for every pair.

Output: a chained camera trajectory + merged sparse 3D point cloud from the
pilot clip.

### Step 1.5 — Go / no-go
- Trajectory shape roughly matches the physical path walked (straight line
  if you walked straight).
- Point cloud shows recognizable structure, not random scatter.
- Reprojection error of chained cameras stays low (a few px).
- Fail → recheck Step 1.0 (K/FOV mismatch is the most likely root cause),
  re-tune stride, or recapture more smoothly.

Once Stage 1 passes on the pilot clip, we move to Stage 2 (real street
footage + bundle adjustment) and Stage 3 (dense plane-sweep MVS).

---

## Progress so far

- `src/calibration.py` — shared K/dist_coeffs, plus `scale_K` /
  `scale_K_rotated` helpers that scale the reference K to a different
  resolution and **raise if the aspect ratio doesn't match** (catches a
  wrong camera mode or portrait/landscape mismatch before it silently
  corrupts the geometry).
- `src/pairwise_pose.py` — `estimate_pairwise_pose()`, a refactor of `ex2`
  steps 2-5 (SIFT+ratio match → Hartley normalize → RANSAC 5-point E →
  `recoverPose`) into a function usable on any image pair, not just the one
  `ex2` stereo pair.
- `src/test_pairwise_regression.py` — **verified**: re-running this
  refactored pipeline on `ex2`'s own `stereo/1.jpeg`/`2.jpeg` reproduces the
  exact published numbers in `ex2/report.tex` (13934/11488 keypoints, 1714
  matches, 1144 RANSAC inliers, 66.74%). Confirms the refactor is faithful
  before building on top of it.
- **Finding during that test:** `ex2`'s stereo photos load via `cv2.imread`
  as 2000 (h) x 1500 (w) — portrait — while the calibrated K's principal
  point (cx=1000, cy=750) assumes 2000 (w) x 1500 (h) — landscape. Likely an
  EXIF-rotation-vs-calibration mismatch. It didn't fail `ex2`'s checkpoints
  (RANSAC tolerates a somewhat-off principal point), but it's a real bias on
  the recovered geometry. `scale_K`'s aspect-ratio check now catches this
  class of bug automatically; `scale_K_rotated` handles a clean 90-degree
  case. **Action for new capture: hold the phone landscape for the street
  video** (natural for a wide street shot anyway), and still check
  `frame.shape` after extraction as a Step 1.0/1.2 checkpoint.
- `src/extract_frames.py` — frame sampling from a video at a fixed stride
  (`cv2.VideoCapture`-based; note in the module that phone rotation
  metadata isn't reliably auto-applied here, another reason to shoot
  landscape and verify `frame.shape`).
- `src/chain_poses.py` — the actual new geometric contribution: sequential
  pose chaining with 3-view common-point scale propagation (step 1.4),
  since naive chaining of unit-norm `recoverPose` translations has no shared
  scale across pairs. Matches shared features between consecutive pairwise
  results by pixel-coordinate proximity (same SIFT detection on the shared
  frame), triangulates them locally, and least-squares-fits the scale that
  reconciles the new pair with the already-reconstructed map.
- `src/run_stage1_pilot.py` — end-to-end runner: video → frames → chained
  trajectory + point cloud → prints the step 1.3/1.5 checkpoints → saves a
  3D plot to `output/stage1_pilot_reconstruction.png`.

**Not yet done (blocked on you):** Step 1.0 (recording-settings checkpoint)
and Step 1.1 (pilot clip capture) require physically shooting footage with
the phone. Once you have a pilot clip, run:
```
python Final/src/run_stage1_pilot.py <path_to_pilot_video> [stride]
```
and we'll read the checkpoint output together and tune stride/capture style
from there.
