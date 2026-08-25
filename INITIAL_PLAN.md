# Project Portpan: Requirements & Implementation Specification

## 1. Problem Description

The goal is to automatically superimpose high-magnification digiscope observations onto an ultra-wide panoramic base image of the same geographical scene.

### Core Challenges

* **Extreme Scale Disparity:** Digiscope shots capture a minute sub-region ($<1\%$) of the panorama field of view, making standard global feature matchers fail across scale space.
* **Optical Distortions & Vignetting:** Digiscope images contain circular black barrel housings, radial sharpness falloff, and strong chromatic aberration near the aperture boundary.
* **Transient & Repetitive Natural Features:** Moving ocean foam, uniform foliage, and repetitive rock surfaces lack distinct keypoints compared to structured architectural scenes.
* **Temporal Media Blending:** Live Photo assets contain short video streams with differing frame counts, motion jitter, and loop seams that must align precisely with static landscape coordinates.

---

## 2. Directory Structure Specification

The pipeline operates on structured location directories in `data/locations/`:

```text
data/locations/
└── <location_name>/
    ├── base/
    │   └── PXL_<date>_<time>.jpg          # Substrate canvas image
    ├── images/
    │   ├── PXL_<date>_<time>.jpg          # Static keyframes / digiscope captures
    │   └── PXL_<date>_<time>.jpg
    └── videos/
        ├── PXL_<date>_<time>_exported_stabilized_<ID>.mp4          # Live Photo embedded video streams
        └── PXL_<date>_<time>_exported_stabilized_<ID>.mp4

```

---

## 3. Phased Implementation Plan

### Phase 1: Core Spatial Matching & Alignment Engine (Milestone 1)

*Objective: Establish a robust, hands-off coordinate registration pipeline that maps any static digiscope image onto the base panorama.*

* **1.1 Aperture Segmentation & Inscribed Pre-Crop**
* Auto-detect optical circle $(c_x, c_y, r)$ via binary thresholding and contour fitting (`cv2.minEnclosingCircle`).
* Separate the image into two internal components:
* **Matching Target:** Inscribed square patch (free of vignette and outer blur).
* **Render Mask:** Full circular aperture retaining the chromatic fringe ring with a $2–4\text{ px}$ anti-aliasing edge.




* **1.2 Semantic Coarse Localization (Gemini Multi-Modal API)**
* Pass `base/panorama.jpg` and the pre-cropped digiscope shot to the Gemini API.
* Retrieve normalized bounding box coordinates $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ representing the estimated region in the panorama.
* Pad bounding box coordinates by $+25\%$ to account for regional approximation tolerances.


* **1.3 Dense Sub-Pixel Registration (Dedicated CV Matcher)**
* Extract the designated high-resolution sub-patch from `base/panorama.jpg`.
* Execute dense feature regression using **RoMa** (or **LoFTR** / **SuperPoint+LightGlue**).
* Estimate homography matrix $H_{\text{local}}$ via `cv2.findHomography` with `USAC_MAGSAC`.


* **1.4 Validation & Global Mapping**
* Evaluate geometric sanity: check RANSAC inlier ratio ($>0.25$), minimum inliers ($N \ge 30$), and positive determinant ($\det(H) > 0$).
* Fallback: If verification fails, prompt Gemini for secondary candidate bounding boxes.
* Compute global affine/homography transform $H_{\text{global}}$ mapping the digiscope asset directly to base panorama pixel space.
* Cache $H_{\text{global}}$ mappings to `data/locations/<location_name>/transforms.json`.



---

### Phase 2: Static Canvas Rendering & Overlay Verification (Milestone 2)

*Objective: Composite all static digiscope shots onto the substrate panorama using high-quality alpha blending and chromatic framing.*

* **2.1 Layer Warping & Projection**
* Load `base/panorama.jpg` into memory as the master canvas.
* Apply cached $H_{\text{global}}$ matrices to warp all masked digiscope shots (`images/*.jpg`) into the shared coordinate frame.


* **2.2 Optical Blending & Vignette Styling**
* Apply the preserved chromatic aberration ring around each circular boundary as a distinct porthole viewport.
* Export the verified composite to `output/<location_name>_composite_static.png`.



---

### Phase 3: Dynamic Live Photo Video Integration & Looping (Milestone 3)

*Objective: Ingest dynamic Live Photo video streams and build a temporally synchronized multi-window video composition.*

* **3.1 Video Ingestion & Stream Extraction**
* Ingest `.mov`/`.mp4` streams from `videos/`.
* Standardize all video streams to a unified frame rate ($30\text{ fps}$).


* **3.2 Seamless Loop Synchronization**
* Determine shared temporal loop duration $T$ (e.g., exactly 60 or 90 frames).
* Build forward-backward ping-pong buffers or alpha crossfades per clip to eliminate loop seam jumps (especially for breaking ocean waves and water ripples).


* **3.3 Dynamic Canvas Frame Synthesis**
* Frame-by-frame, apply the static $H_{\text{global}}$ to warp corresponding video frames.
* Composite the animated apertures over the static base panorama canvas.
* Encode final synchronized video master (`output/<location_name>_loop.mp4` / `WebM`) using H.264 / H.265.



---

### Phase 4: Stylization & Vectorization [Decoupled Extension]

*Objective: Transform the static composite or animated canvas into a unified vintage cartographic parchment/sketch aesthetic.*

* **4.1 Global Raster Stylization**
* Restyle the canvas using diffusion-based line art / sketch models (e.g., Nano Banana / ControlNet Lineart) to homogenize resolution mismatches into stylized cross-hatching and ink paths.


* **4.2 Vector SVG Trace**
* Process the stylized static canvas through `vtracer` / `potrace` to generate an infinite-resolution SVG parchment base.
* (Optional dynamic mode) Embed looping video apertures directly within SVG porthole windows for web deployment.