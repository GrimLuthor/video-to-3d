# Results gallery

Reconstructions from single handheld phone videos (classical pipeline, no neural networks).

- **Source videos:** _add YouTube playlist link_
- **Unity viewer demo:** _add video link_

## Reconstructed scenes

Each scene: a close-up (with the nearest real video frame overlaid), a wider overview, and the surface mesh.

### Walk (garden courtyard)

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/walk_closeup.jpg) | ![](scenes/walk_macro.jpg) | ![](scenes/walk_mesh.jpg) |

### Longer walk (stone building)

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/longer_walk_closeup.jpg) | ![](scenes/longer_walk_macro.jpg) | ![](scenes/longer_walk_mesh.jpg) |

### Dean's office

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/dean_office_closeup.jpg) | ![](scenes/dean_office_macro.jpg) | ![](scenes/dean_office_mesh.jpg) |

### Residential building

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/residential_closeup.jpg) | ![](scenes/residential_macro.jpg) | ![](scenes/residential_mesh.jpg) |

### Einstein

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/einstein_closeup.jpg) | ![](scenes/einstein_macro.jpg) | ![](scenes/einstein_mesh.jpg) |

### Columns

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/columns_closeup.jpg) | ![](scenes/columns_macro.jpg) | ![](scenes/columns_mesh.jpg) |

### Library (indoor)

| Close-up (+ real frame) | Overview | Mesh |
|---|---|---|
| ![](scenes/library_closeup.jpg) | ![](scenes/library_macro.jpg) | ![](scenes/library_mesh.jpg) |

## How the pipeline works

| Feature matches | Dense depth | Cross-view consistency | Camera trajectory |
|---|---|---|---|
| ![](pipeline/features_matches.jpg) | ![](pipeline/depth.jpg) | ![](pipeline/consistency.jpg) | ![](pipeline/sfm_trajectory.jpg) |

## Limitations

| Glass (prep_academy) | Open scene real / reconstructed (tayelet) | Sky halo |
|---|---|---|
| ![](limitations/glass_prep_academy.jpg) | ![](limitations/tayelet_real.jpg)<br>![](limitations/tayelet_sparse.jpg) | ![](limitations/sky_halo.jpg) |

## Unity viewer

![](viewer/closest_frame.jpg)

_Mesh vs. point-cloud fly-through with the nearest real frame shown alongside the view._
