# Part B — Scene/Cloud Registration via Werman's Ellipsoid ICP-Init

Companion to Part A (`PROJECT_SUMMARY.md` / video→3D pipeline). This documents the
Part B investigation: registering/merging independent 3D reconstructions using
**Kolpakov & Werman, "An approach to robust ICP initialization"** (arXiv:2212.05332,
paper PDF at `Final/An Approach to Robust ICP Initialization.pdf`). Session date:
2026-08-20/21. All Part B code is isolated in `Final/src/partB/`; **Part A is untouched.**

---
## 1. The method and our extension

**E-Init (the paper):** aligns two point clouds that are the same shape up to a
RIGID motion (Q = O·P·S, rotation O, permutation S) by matching their covariance
**ellipsoids**: eigenvectors give the rotation (up to 2^d sign flips, tested by
nearest-neighbour distance); then run ICP. Provably robust to noise / occlusion /
cardinality. Designed for and tested on **complete single objects** (Caerbannog
teapot/bunny/cow; scanned statues), with **large overlap**.

**Our extension (scale):** monocular SfM clouds have ARBITRARY relative scale, so
they're related by a SIMILARITY (s·O·P·S), not a rigid motion. Since E_Q = s²·O·E_P·Oᵀ,
scale falls entirely in the eigenvalues: **s = median over axes of sqrt(λ_tgt/λ_src)**
(median, not geomean — robust to one axis inflating under partial overlap). This turns
E-Init into a similarity initializer. Validated exactly on synthetic ground truth.

---
## 2. Code (`Final/src/partB/`)

- `ellipsoid_init.py` — `e_init_similarity` (covariance→eig→scale + sign candidates,
  optional gravity up-gate), `gravity_yaw_candidates`, `rotation_angle_deg`.
- `similarity_icp.py` — `refine_similarity`: TWO-STAGE = rigid point-to-plane ICP at
  the E-Init scale (robust R,t; NEVER free-scale from a rough start — Umeyama collapses
  under partial overlap), then a GUARDED with_scaling pass (generous corr dist) to
  correct residual UNIFORM scale. Plus `fpfh_ransac_init` baseline.
- `run_combine.py` — the runner. Flags: `--src-recon/--tgt-recon` (gravity up-gate),
  `--yaw-search N`, `--core-crop/--core-pct`, `--init-mode {einit,identity,centroid}`
  (ablation), `--piecewise N` (drift correction), `--baseline`, `--dry-run-split/--overlap`
  (known-transform self-test), `--select {icp,nn}`. Saves `<tag>_overlay.ply` (two-tone
  red=src/blue=tgt), `<tag>_{src,tgt}_aligned.ply` (native colour, for the toggle viewer),
  `<tag>_merged.ply`, `<tag>_partB_metrics.json`.
- `piecewise_refine.py` — smoothly-blended per-segment similarity to correct residual
  scale DRIFT along the path (segment along principal axis, fit local similarity per
  segment, interpolate scale/rot/trans). Optimal segment count is data-dependent — pick
  by cross-validation (16 for columns).
- `view_overlay.py <tag>` — interactive toggle viewer (1/2 show-hide each scene, C
  native↔two-tone colour, [ ] point size, W bg). Legacy GL (filament dead on this GPU).
- `sparse_ply.py <tag>` — export a Part A sparse reconstruction to a time-coloured PLY.
- `reconstruct_phase.py <video> <stride> <phase> --tag` — reconstruct a PHASE-OFFSET
  frame subset (idx%stride==phase) to get two independent same-scene clouds w/o new capture.
- `ablation_recovery.py <tag> --trials N` — success-rate of recovering a known random
  similarity per init mode, bucketed by rotation (reproduces the paper's Fig-3 story).

Interpreter (as Part A): `Computer Vision\venv\Scripts\python.exe` (open3d 0.19, cv2 4.13).

---
## 3. THE CENTRAL FINDING: three regimes

E-Init assumes the two clouds are the same **point set** up to a transform (true when
you scan one OBJECT twice). It is silent/wrong when they are the same **scene** captured
by different trajectories, because **the covariance ellipsoid is sampling-dependent** —
different path/pace/coverage → different second moments even for identical geometry. Real
data lands in three regimes:

| Regime | Example | E-Init | Evidence |
|---|---|---|---|
| **Refinement** (full overlap, same sampling) | columns phase-0 vs phase-4 (interleaved frames of ONE video) | **WORKS**, 0.995 fitness | per-axis scale tight [1.05,0.97,1.02]; ablation einit 100% |
| **Extension** (partial overlap) | — | E-Init global scale FAILS; needs scale-search + FPFH | walk cross-day |
| **No co-visibility / different sampling** | meonot (orthogonal), walk morning vs evening (different route) | **FAILS** | mismatched normalized eigenvalue spectra |

**Quantitative ablation** (`ablation_recovery.py` on columns, 30 known random similarities):
- **einit 100%** success at ALL rotation magnitudes (0-180°).
- **centroid** (scale+translation, no ellipsoid rotation) 100%@0-45° / 78%@45-90° / 0%@90-180° (= ICP's convergence basin).
- **identity** (no init) 0% everywhere (unknown ×0.5-2 scale sinks plain ICP).
→ The ellipsoid ROTATION is what buys pose-invariance; the eigenvalue-SCALE is what buys scale-invariance. This is Part B's headline positive result.

**Eigenvalue spectra (why scenes fail), normalized [1, λ2, λ3] and λ1/λ2 gap:**
- columns (works): [1, 0.20, 0.01] gap 4.9  — distinct spectrum, E-Init's ideal.
- walk_s8 dense (fails): [1, 0.88, 0.17] gap **1.14** — near-degenerate (λ1≈λ2 → yaw undefined).
- walk_pm dense (fails): [1, 0.50, 0.07] gap 2.0.
- walk_s8 vs walk_pm normalized spectra DIFFER → NOT the same shape → not related by any
  similarity → E-Init premise violated. Density (millions of culled points) did NOT fix
  this — it was never floaters, it's sampling/coverage.

---
## 4. The journey (each step forced by a measured failure — good "story" material)

1. **Scale extension** validated on synthetic (exact recovery; noise inflates the
   ellipsoid → scale bias, motivating ICP scale-refine).
2. **meonot** (square courtyard, two ~90° walks): raw NN picks an upside-down FLIP →
   ICP-refined fitness still can't break it (symmetry) → **gravity up-gate** (mean of
   R^T[0,-1,0] over cameras ~ world up; reject up→down candidates) → upright but ~45° yaw
   off (degenerate horizontal eigenvalues) → **gravity yaw-search** (fix scale+up,
   brute-force the 1 remaining DOF). Ultimately meonot FAILS anyway: orthogonal views share
   the SCENE but almost no co-VISIBLE SURFACE (different walls; grid trees look different
   from 90°). Both E-Init AND FPFH land on the same wrong pose. INFORMATION limit.
3. **columns** (elongated, distinct spectrum — E-Init's ideal): phase-0/phase-4 refinement
   SUCCEEDS. Sparse 0.965 → dense 0.995. Then user-caught residuals, each fixed:
   - "columns diverge outward" = residual UNIFORM scale → **scale-refinement** (generous-dist
     Umeyama after rigid ICP) → 0.933→0.957, fitness 0.969→0.995.
   - "worse to the left" = residual scale DRIFT (differential between the two monocular
     recons) → **piecewise** drift correction → overall residual -33%; CROSS-VALIDATION
     showed genuine optimum ~16 segments (held-out left 0.18), more = overfitting (32 worse
     than 8 on held-out); left floor ~0.15 = intrinsic NON-similar drift (a similarity, even
     piecewise, can absorb scale/offset but not bending). Root cause is UPSTREAM (one-way-walk
     drift, no loop closure).
4. **walk cross-day** (morning walk_s8 vs evening walk_pm=walk_evening.mp4, different day+time,
   different route): the EXTENSION/partial-overlap + degenerate case.
   - LIGHTING is INVISIBLE to the method (E-Init/ICP/FPFH are geometric, not photometric) —
     a real positive; failures are geometry/overlap, never morning-vs-evening.
   - E-Init failed (tail-inflated covariance → wrong scale 2.96, fitness 0.24). A floater
     "trail" (3% of points, extent 139→32 after cleaning) confirmed the covariance is
     dominated by far outliers → **always clean before aligning**.
   - Scale-search + FPFH looked promising on sparse (0.64 = ~90% of shared stretch) BUT on
     DENSE the high FPFH fitness (0.87-0.96) was a **DENSITY ARTIFACT**: evening's 4M points
     sit inside morning's 18M-point volume and score "matched" without surfaces aligning.
     NORMAL-AGREEMENT check = 0.52 (random; a real match is ~0.8+). User confirmed visually:
     "way off in rotation and scale and translation." So scenes captured by different routes
     do NOT register by ANY method — another information limit.

---
## 5. KEY LESSONS (report-worthy)

- **E-Init is a same-POINT-SET method** (object scanned twice), not a same-SCENE method.
  Covariance is sampling-dependent; the paper's object-scan experiments structurally hide this.
- **Always clean (outlier/floater removal) before ellipsoid alignment** — far outliers have
  outsized leverage on the covariance (rotation + scale).
- **Fitness is unreliable when densities differ** (small cloud in big dense cloud) — verify
  with surface NORMAL agreement, not just correspondence count.
- **Geometric registration is lighting/time-of-day invariant** — a genuine strength vs
  photometric/image-feature methods.
- **Piecewise/non-rigid corrections must be cross-validated** (held-out residual) — training
  residual always drops with more segments (fits noise); held-out reveals the real optimum.

---
## 6. CURRENT DECISION POINT (drawing board)

Direct scene-to-scene registration via E-Init is a proven dead end (shape/sampling mismatch).
Three resolutions on the table (user deciding; user writing a proposition with their own idea):
1. **Loop closure** — use E-Init to align a walk's revisit-region to its start-region (same
   structure → same shape) → measures/corrects Part A's drift. Scenes ✓ + Werman ✓ + fixes A ✓.
   (Assistant's recommendation.)
2. **Object / multi-video fusion** — orbit a compact ASYMMETRIC object in 2-3 videos, register
   with E-Init(+scale) → one fused model. Werman's native regime; clean success; feels detached
   from A's scene focus.
3. **Feature-based scene extension** — pose-graph + FPFH on HIGH-overlap same-route pairs.
   Natural Part A continuation but does NOT use the paper.

---
## 7. FILE CATALOG (output/ unless noted)

**Reconstructions (Part B, tag → what):**
- `columns_p0`, `columns_p4` — columns stride-8 phase-0 / phase-4 (independent same-scene, 132 cams each).
- `columns_s16` — columns stride-16 (wide-baseline; only 53/66 cams = stride-tradeoff demo).
- `columns_p4d` — phase-4 REBUNDLED to stride-4 phase-0-equivalent so run_fusion works (dense).
- `walk_pm` — evening walk (`walk_evening.mp4`, 32s, 119 cams) — the reconstruction the user LIKED and asked to export.
- `walk_s8` (Part A, morning walk) / `meonot_1`,`meonot_2` (courtyard).

**Key clouds:**
- `columns_dense_pw16_merged.ply` — BEST columns dense refined merge (16-segment piecewise; the SUCCESS deliverable). Also pw20/pw32 (overfit).
- `columns_dense_refine_merged.ply` — dense scale-corrected (no piecewise).
- `walk_pm_dense_dense.ply` — evening dense cloud (4.0M pts) + `_nocull`. → being exported.
- `walk_dense_fpfh_overlay.ply` — the FAILED cross-day dense alignment (density artifact; verified wrong).
- `meonot_sparse_merged.ply` / `walk_crossday_*` — the failure-case overlays.

**Figures — `Final/report_assets_partB/`** (see its MANIFEST): 01-05 columns success chain,
10 meonot fail, 20-22 walk cross-day (E-Init poor / FPFH scale-search / dense density-artifact).

**Unity export (this session):** `unity_export/walk_evening/` — evening reconstruction (mesh + cloud), being generated.

---
## 8. Memory
Dir-scoped auto-memory: `partB_scene_registration.md` (+ index line in MEMORY.md). This file
is the portable, report-oriented version.
