# Multi-View 3D Reconstruction: Mathematical Formulations & Architecture

This document provides the full mathematical foundations, algorithmic stages, and architectural implementation details of the multi-view 3D computer vision pipeline.

---

## 📐 1. Camera Projection & Epipolar Geometry

For an arbitrary 3D world landmark $\mathbf{X}_j = [X_j, Y_j, Z_j, 1]^\top \in \mathbb{P}^3$ observed by camera $i$, its projection onto the 2D image plane $\mathbf{x}_{ij} = [u_{ij}, v_{ij}, 1]^\top \in \mathbb{P}^2$ is governed by the pinhole camera projection matrix:

$$\lambda \mathbf{x}_{ij} = \mathbf{K}_i \left[ \mathbf{R}_i \;\middle|\; \mathbf{t}_i \right] \mathbf{X}_j$$

Where:
- $\lambda \in \mathbb{R}^+$ is the projective depth scalar.
- $\mathbf{K}_i$ is the $3 \times 3$ intrinsic camera calibration matrix:
  $$\mathbf{K}_i = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
  where $f_x, f_y$ are focal lengths in pixel units, and $(c_x, c_y)$ is the principal point.
- $\mathbf{R}_i \in \mathrm{SO}(3)$ is the $3 \times 3$ orthonormal rotation matrix.
- $\mathbf{t}_i \in \mathbb{R}^3$ is the translation vector.
- The optical camera center in world coordinates is $\mathbf{C}_i = -\mathbf{R}_i^\top \mathbf{t}_i$.

### Epipolar Constraint & Fundamental Matrix
Between two corresponding image points $\mathbf{x}_1$ and $\mathbf{x}_2$ across views 1 and 2, the epipolar coplanarity constraint requires:

$$\mathbf{x}_2^\top \mathbf{F} \mathbf{x}_1 = 0$$

Where $\mathbf{F} = \mathbf{K}_2^{-\top} [\mathbf{t}]_\times \mathbf{R} \mathbf{K}_1^{-1}$ is the fundamental matrix of rank 2. Inlier correspondences are estimated robustly using RANSAC with symmetric epipolar distance filtering.

---

## 🔍 2. Structure from Motion (SfM) Pipeline

The sparse reconstruction executes through COLMAP's incremental SfM engine:

1. **SIFT Feature Extraction**:
   - Detects scale-space extrema across Difference-of-Gaussians (DoG) pyramids.
   - Computes 128-dimensional orientation-invariant gradient histograms.
   - Accelerated via NVIDIA CUDA on supported hardware.
2. **Exhaustive Matching & Epipolar Verification**:
   - Matches descriptors across all image pairs $\binom{N}{2}$.
   - Applies Lowe's ratio test ($d_1 / d_2 < 0.8$) to eliminate ambiguous matches.
   - Verifies homography or epipolar geometry using RANSAC.
3. **Two-View Initialization**:
   - Selects an optimal initial image pair maximizing baseline parallax and verified inlier correspondences.
4. **Incremental Pose Estimation & Triangulation**:
   - Solves the Perspective-n-Point (PnP) problem using EPnP + RANSAC for newly registered cameras.
   - Triangulates new 3D points $\mathbf{X}_j$ by linear least squares (DLT).
5. **Global Levenberg-Marquardt Bundle Adjustment**:
   - Jointly minimizes the total Huber-weighted reprojection error over all visible 3D points and registered cameras:
     $$\min_{\{\mathbf{R}_i, \mathbf{t}_i, \mathbf{K}_i\}, \{\mathbf{X}_j\}} \sum_{i=1}^M \sum_{j \in \mathcal{V}_i} \rho\left( \left\| \mathbf{x}_{ij} - \pi(\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i, \mathbf{X}_j) \right\|_2^2 \right)$$
     where $\rho(s)$ is the Cauchy/Huber robust loss function to suppress outlying tracks.

---

## ⚡ 3. Dense Multi-View Stereo (PatchMatch MVS)

Dense reconstruction infers a depth scalar $d(p)$ and surface normal vector $\mathbf{n}(p)$ for every pixel $p$:

1. **Image Undistortion**:
   - Resamples original distorted photos into pinhole coordinates using estimated lens distortion coefficients ($k_1, k_2$).
2. **CUDA PatchMatch Stereo**:
   - Defines a local planar hypothesis $\theta = (\mathbf{n}, d)$ at each pixel.
   - Evaluates photometric consistency across neighboring source views using Normalized Cross-Correlation (NCC) over slanted $11 \times 11$ pixel patches:
     $$\mathcal{C}_{\mathrm{NCC}}(p, \theta) = \frac{\sum (I_{\mathrm{ref}} - \bar{I}_{\mathrm{ref}})(I_{\mathrm{src}}(\mathbf{H}_\theta p) - \bar{I}_{\mathrm{src}})}{\sqrt{\sum (I_{\mathrm{ref}} - \bar{I}_{\mathrm{ref}})^2 \sum (I_{\mathrm{src}}(\mathbf{H}_\theta p) - \bar{I}_{\mathrm{src}})^2}}$$
     where $\mathbf{H}_\theta = \mathbf{K}_{\mathrm{src}} (\mathbf{R} - \frac{\mathbf{t} \mathbf{n}^\top}{d}) \mathbf{K}_{\mathrm{ref}}^{-1}$ is the induced plane-induced homography.
   - Propagates best-matching spatial hypotheses horizontally and vertically across CUDA warps.
3. **Geometric & Photometric Consistency Filtering**:
   - Enforces depth agreement across at least 3 distinct views ($\ge 3$ consistent rays).
   - Rejects depth hypotheses with photometric variance $> 0.4$ or normal disagreement $> 30^\circ$.
4. **Stereo Point Fusion**:
   - Merges consistent per-view depth maps into an unorganized 3D point cloud $\mathcal{P}_{\mathrm{dense}} = \{\mathbf{p}_k, \mathbf{n}_k, \mathbf{c}_k\}$.

---

## 🔺 4. Screened Poisson Surface Reconstruction

To extract a continuous 2D polygonal manifold $\mathcal{M} = (V, F)$ from oriented points $\mathcal{P}_{\mathrm{dense}}$:

1. Formulates the surface reconstruction as finding a 3D indicator function $\chi : \mathbb{R}^3 \to \mathbb{R}$ ($\chi = 1$ inside object, $\chi = 0$ outside) whose gradient matches the oriented normal field:
   $$\nabla \chi \approx \mathbf{V}$$
2. Solves the screened Poisson equation with point interpolation constraints:
   $$\Delta \chi - \alpha \sum_{k} \delta(\mathbf{x} - \mathbf{p}_k) \chi(\mathbf{x}) = \nabla \cdot \mathbf{V}$$
3. Discretizes the domain using an adaptive spatial octree (depth 8–10).
4. Extracts the continuous polygonal surface via the Marching Cubes isosurface extraction on the zero-level set:
   $$\mathcal{M} = \left\{ \mathbf{x} \in \mathbb{R}^3 \;\middle|\; \chi(\mathbf{x}) = \tau \right\}$$
5. Trims low-density surface artifacts using local point support thresholds.

---

## 🧊 5. Volumetric Voxel Grid Discretization

To produce a discrete regular spatial occupancy model $\mathcal{G} \subset \mathbb{Z}^3$:

1. Computes the bounding box diagonal $D = \|\mathbf{p}_{\max} - \mathbf{p}_{\min}\|_2$.
2. Discretizes continuous space into uniform cubes of edge width $s = \alpha D$ (default $\alpha = 0.01$):
   $$\mathbf{i} = \left\lfloor \frac{\mathbf{x} - \mathbf{p}_{\min}}{s} \right\rfloor \in \mathbb{Z}^3$$
3. Occupied grid coordinates are stored in compressed integer numpy format (`voxels.npz`) containing:
   - `indices`: $N_v \times 3$ matrix of integer grid coordinates $(i_x, i_y, i_z) \in \mathbb{N}^3$.
   - `origin`: $1 \times 3$ float vector of bounding box origin $\mathbf{p}_{\min}$.
   - `voxel_size`: Float scalar grid cell resolution $s$.
4. Renders interactive 3D instanced cubes inside the Three.js WebGL viewer with constant $O(1)$ memory queries.

---

## 🎭 6. Automated Neural Foreground Segmentation (`rembg`)

For tabletop and turntable objects where a static background or cloth could compromise camera poses:

1. Uses the pre-trained **IS-Net (Iterative Structured Network)** dichotomous image segmentation model.
2. Extracts high-resolution alpha mattes $\alpha(u, v) \in [0, 255]$ without painting images or altering original RGB data.
3. Normalizes EXIF orientation tags losslessly to prevent orientation mismatch between COLMAP camera readers and PIL image decoders.
4. Generates binary exclusion masks ($\alpha \ge 240$, eroded by 2 pixels to preserve clean object silhouettes) fed into COLMAP's `ImageReader.mask_path`.
