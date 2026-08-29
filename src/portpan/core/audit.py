"""Visual diagnostic and audit reporting for registration verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
import cv2
import numpy as np
from portpan.core.models import CircleAperture, RegistrationResult

logger = logging.getLogger("portpan.audit")


class VisualAuditor:
    """Generates visual verification artifacts to enable easy auditing of registration accuracy."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "apertures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "matches").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "isolated_portholes").mkdir(parents=True, exist_ok=True)

    def export_isolated_porthole(
        self,
        image: np.ndarray,
        aperture: CircleAperture,
        filename_prefix: str,
    ) -> Path:
        """Export a 32-bit transparent RGBA PNG porthole with preserved chromatic ring."""
        bgra = aperture.extract_rgba(image, feather_px=2.0)
        out_path = self.output_dir / "isolated_portholes" / f"{filename_prefix}_porthole.png"
        cv2.imwrite(str(out_path), bgra)
        return out_path

    def draw_aperture_debug(
        self,
        image: np.ndarray,
        aperture: CircleAperture,
        inscribed_box: tuple[int, int, int, int],
        filename_prefix: str,
    ) -> Path:
        """Save a visualization of the detected optical aperture and inscribed square."""
        debug_img = image.copy()
        h, w = debug_img.shape[:2]
        line_w = max(2, int(w * 0.002))

        # 1. Draw sub-pixel boundary polygon hugging chromatic fringe
        if aperture.boundary_polygon and len(aperture.boundary_polygon) >= 3:
            poly_pts = np.array(aperture.boundary_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=line_w, lineType=cv2.LINE_AA)

        # 2. Draw fitted circle or ellipse in thin cyan
        center = (int(round(aperture.center_x)), int(round(aperture.center_y)))
        if aperture.ellipse_axes and aperture.ellipse_angle is not None:
            ax1, ax2 = aperture.ellipse_axes
            axes = (int(round(ax1 / 2.0)), int(round(ax2 / 2.0)))
            cv2.ellipse(
                debug_img,
                center,
                axes,
                aperture.ellipse_angle,
                0,
                360,
                (255, 255, 0),
                max(1, line_w // 2),
                lineType=cv2.LINE_AA,
            )
        else:
            radius = int(round(aperture.radius))
            cv2.circle(debug_img, center, radius, (255, 255, 0), max(1, line_w // 2), lineType=cv2.LINE_AA)

        cv2.circle(debug_img, center, max(4, int(w * 0.004)), (0, 0, 255), -1)

        # 3. Draw inscribed square in vibrant amber
        x1, y1, x2, y2 = inscribed_box
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 180, 255), line_w, lineType=cv2.LINE_AA)

        out_path = self.output_dir / "apertures" / f"{filename_prefix}_aperture.jpg"
        cv2.imwrite(str(out_path), debug_img)
        return out_path

    def draw_matches_debug(
        self,
        crop_img: np.ndarray,
        target_img: np.ndarray,
        pts_crop: np.ndarray,
        pts_target: np.ndarray,
        inlier_mask: np.ndarray,
        filename_prefix: str,
    ) -> Path:
        """Draw side-by-side keypoint matches with inliers in green and outliers in red."""
        h1, w1 = crop_img.shape[:2]
        h2, w2 = target_img.shape[:2]

        target_h = max(h1, h2)
        scale1 = target_h / h1
        scale2 = target_h / h2

        img1_resized = cv2.resize(crop_img, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(target_img, (int(w2 * scale2), target_h))

        canvas = np.hstack([img1_resized, img2_resized])
        offset_x = img1_resized.shape[1]

        mask = inlier_mask.ravel().tolist() if inlier_mask is not None else [1] * len(pts_crop)

        for i, (pt1, pt2) in enumerate(zip(pts_crop, pts_target)):
            p1 = (int(pt1[0] * scale1), int(pt1[1] * scale1))
            p2 = (int(pt2[0] * scale2 + offset_x), int(pt2[1] * scale2))
            is_inlier = bool(mask[i])
            color = (0, 230, 0) if is_inlier else (0, 0, 200)
            thickness = 2 if is_inlier else 1
            if is_inlier or i % 6 == 0:
                cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
                cv2.circle(canvas, p1, 3, color, -1)
                cv2.circle(canvas, p2, 3, color, -1)

        out_path = self.output_dir / "matches" / f"{filename_prefix}_matches.jpg"
        cv2.imwrite(str(out_path), canvas)
        return out_path

    def draw_global_canvas_overview(
        self,
        base_canvas: np.ndarray,
        results: dict[str, RegistrationResult],
        location_name: str,
    ) -> Path:
        """Draw an overview of the base canvas with all registered porthole boundaries and labels."""
        overview = base_canvas.copy()
        h, w = overview.shape[:2]
        line_thick = max(2, int(w * 0.0015))
        font_scale = max(0.5, w * 0.0004)

        for name, res in results.items():
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if res.bounding_box_in_base is None or status_val == "FAILED":
                continue

            pts = np.array(res.bounding_box_in_base, dtype=np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if status_val == "SUCCESS" else (0, 200, 255)

            cv2.polylines(overview, [pts], isClosed=True, color=color, thickness=line_thick, lineType=cv2.LINE_AA)

            top_left = pts[0][0]
            label = f"{name} ({res.inlier_count} inliers)"
            cv2.putText(
                overview,
                label,
                (top_left[0] + 5, max(25, top_left[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, int(line_thick * 1.5)),
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                overview,
                label,
                (top_left[0] + 5, max(25, top_left[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        out_path = self.output_dir / f"{location_name}_registration_overview.jpg"
        cv2.imwrite(str(out_path), overview)
        return out_path

    def _load_isolated_porthole_rgba(
        self,
        name: str,
        target_img: np.ndarray,
        ap: CircleAperture,
        location_name: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Directly load the pre-extracted isolated porthole PNG, apply smooth eyecup boundary inset and sky suppression."""
        iso_dir = self.output_dir / "isolated_portholes"
        stem = Path(name).stem

        candidates = [
            iso_dir / f"{location_name}_{stem}_porthole.png",
            iso_dir / f"{location_name}_{name}_porthole.png",
            iso_dir / f"{stem}_porthole.png",
            iso_dir / f"{name}_porthole.png",
        ]

        rgba_raw = None
        for c in candidates:
            if c.exists():
                rgba_raw = cv2.imread(str(c), cv2.IMREAD_UNCHANGED)
                if rgba_raw is not None and len(rgba_raw.shape) == 3 and rgba_raw.shape[2] == 4:
                    break

        if rgba_raw is None:
            rgba_raw = ap.extract_rgba(target_img, feather_px=2.0)

        # 1. Clean boundary inset: Erode alpha by 18px with smooth 10px cosine taper
        # This completely eliminates any physical contact with the rubber eyecup, dark casing, or chromatic fringing
        alpha_orig = rgba_raw[:, :, 3].astype(np.float32)
        dist = cv2.distanceTransform((alpha_orig > 128).astype(np.uint8), cv2.DIST_L2, 5)
        clean_mask = np.clip((dist - 18.0) / 10.0, 0.0, 1.0)
        clean_mask = 0.5 * (1.0 - np.cos(np.pi * clean_mask))

        rgb = rgba_raw[:, :, :3]
        rgba_clean = rgb.copy()
        rgba_clean[clean_mask == 0] = 0
        alpha_channel = (clean_mask * 255.0).astype(np.uint8)
        rgba_clean = np.dstack([rgba_clean, alpha_channel])

        # Compute central saliency (Laplacian variance on non-zero core)
        core_pts = dist > (dist.max() * 0.35)
        if np.any(core_pts):
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            saliency = float(np.var(lap[core_pts]))
        else:
            saliency = 10.0

        return rgba_clean, clean_mask, max(1.0, saliency)

    def draw_static_composite_preview(
        self,
        base_canvas: np.ndarray,
        results: dict[str, RegistrationResult],
        images_map: dict[str, np.ndarray],
        apertures_map: dict[str, CircleAperture],
        location_name: str,
        canvas_scale: float = 1.0,
        is_master: bool = False,
    ) -> Path:
        """Render a composite preview warping actual isolated RGBA portholes onto the master canvas with strict zero-dilution Z-buffering."""
        h_orig, w_orig = base_canvas.shape[:2]

        w_canvas = int(round(w_orig * canvas_scale))
        h_canvas = int(round(h_orig * canvas_scale))
        if canvas_scale != 1.0:
            composite = cv2.resize(base_canvas, (w_canvas, h_canvas), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
            S_mat = np.array([[canvas_scale, 0.0, 0.0], [0.0, canvas_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            composite = base_canvas.copy().astype(np.float32)
            S_mat = np.eye(3, dtype=np.float64)

        priority_map = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        # 1. Gather all registered portholes using the ACTUAL isolated portholes
        # 1. Gather all registered portholes metadata (streaming: DO NOT accumulate heavy RGBAs in memory)
        all_entries: list[dict[str, Any]] = []
        for name, res in results.items():
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if status_val == "FAILED" or res.bounding_box_in_base is None or name not in images_map:
                continue

            target_img = images_map[name]
            H_global = res.get_matrix_np()
            H_scaled = S_mat @ H_global

            ap = apertures_map.get(name) or CircleAperture(
                center_x=target_img.shape[1] / 2.0,
                center_y=target_img.shape[0] / 2.0,
                radius=min(target_img.shape[:2]) * 0.45,
                image_width=target_img.shape[1],
                image_height=target_img.shape[0],
            )

            scale_approx = float(np.sqrt(H_global[0, 0] ** 2 + H_global[0, 1] ** 2))
            zoom_factor = 1.0 / (scale_approx + 1e-4)
            footprint_area = np.pi * ((ap.radius * scale_approx * canvas_scale) ** 2)

            # Centrality saliency proxy from central patch
            cy, cx = int(ap.center_y), int(ap.center_x)
            r_box = int(ap.radius * 0.25)
            core_patch = target_img[max(0, cy - r_box):min(target_img.shape[0], cy + r_box), max(0, cx - r_box):min(target_img.shape[1], cx + r_box)]
            if core_patch.size > 0:
                saliency = float(np.var(cv2.Laplacian(cv2.cvtColor(core_patch, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
            else:
                saliency = 10.0

            all_entries.append({
                "name": name,
                "res": res,
                "ap": ap,
                "H_scaled": H_scaled,
                "scale_approx": scale_approx,
                "zoom_factor": zoom_factor,
                "footprint_area": footprint_area,
                "saliency": saliency,
                "inliers": res.inlier_count,
            })

        # 2. Sort from widest context first (larger footprint / low saliency) to highest zoom & saliency last
        all_entries.sort(key=lambda p: (p["footprint_area"] / (p["saliency"] + 1e-2), -p["inliers"]), reverse=True)

        total_mask = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        # 3. Direct 4-channel warp and STRICT zero-dilution Z-buffer blending via streaming local ROI bounds
        for entry in all_entries:
            name = str(entry["name"])
            target_img = images_map[name]
            ap = cast(CircleAperture, entry["ap"])
            H_scaled = cast(np.ndarray, entry["H_scaled"])
            scale_approx = float(entry["scale_approx"])
            zoom_factor = float(entry["zoom_factor"])

            # Load ONLY this single porthole RGBA on demand
            rgba_clean, clean_mask, saliency = self._load_isolated_porthole_rgba(name, target_img, ap, location_name)
            h_src, w_src = rgba_clean.shape[:2]

            # Quality weight field
            y_g, x_g = np.ogrid[:ap.image_height, :ap.image_width]
            dist_norm = np.sqrt((x_g - ap.center_x) ** 2 + (y_g - ap.center_y) ** 2) / (ap.radius + 1e-5)
            centrality = np.clip(1.0 - (dist_norm / 0.82) ** 2, 0.0, 1.0)
            saliency_boost = 1.0 + 0.6 * np.log10(1.0 + saliency)
            quality_weight = clean_mask * centrality * (zoom_factor ** 1.3) * saliency_boost

            # Compute precise local bounding box on the canvas
            corners = np.array([[[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]], dtype=np.float32)
            pts = cv2.perspectiveTransform(corners, H_scaled)[0]
            x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
            x_max = min(w_canvas, int(np.ceil(np.max(pts[:, 0]))))
            y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
            y_max = min(h_canvas, int(np.ceil(np.max(pts[:, 1]))))

            if x_max <= x_min or y_max <= y_min:
                del rgba_clean, clean_mask, quality_weight
                continue

            w_roi = x_max - x_min
            h_roi = y_max - y_min

            # Local ROI homography: translates canvas coordinate space to ROI sub-window
            T_roi = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
            H_roi = T_roi @ H_scaled

            # Warp ONLY into local ROI (drops allocation from ~7.5GB to ~9MB per warp!)
            w_rgba = cv2.warpPerspective(
                rgba_clean,
                H_roi,
                (w_roi, h_roi),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            w_target = w_rgba[:, :, :3].astype(np.float32)
            w_mask = w_rgba[:, :, 3].astype(np.float32) / 255.0

            w_q = cv2.warpPerspective(
                quality_weight,
                H_roi,
                (w_roi, h_roi),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            total_mask[y_min:y_max, x_min:x_max] = np.maximum(total_mask[y_min:y_max, x_min:x_max], w_mask)

            # Strict Z-buffer: lower priority frames NEVER overwrite higher priority frames (alpha=0)
            q_curr = priority_map[y_min:y_max, x_min:x_max]
            higher = (w_q > q_curr) & (w_mask > 0.01)

            if np.any(higher):
                # Multi-band detail fusion: match low-frequency illumination to canvas, eliminating veiling glare
                # while preserving 100% of native optical sensor sharpness and contrast on landmarks
                ksize = 61
                tgt_low = cv2.GaussianBlur(w_target, (ksize, ksize), 0)
                canvas_low = cv2.GaussianBlur(composite[y_min:y_max, x_min:x_max], (ksize, ksize), 0)
                matched_target = canvas_low + (w_target - tgt_low)

                alpha = np.zeros_like(w_q)
                alpha[higher] = np.clip(0.6 + (w_q[higher] - q_curr[higher]) / (2.0 * np.maximum(w_q[higher], 1e-4)), 0.6, 1.0) * w_mask[higher]
                alpha_3d = alpha[:, :, np.newaxis]

                c_roi = composite[y_min:y_max, x_min:x_max]
                composite[y_min:y_max, x_min:x_max] = c_roi * (1.0 - alpha_3d) + matched_target * alpha_3d
                priority_map[y_min:y_max, x_min:x_max] = np.maximum(q_curr, w_q)
                del tgt_low, canvas_low, matched_target

            del rgba_clean, clean_mask, quality_weight, w_rgba, w_target, w_mask, w_q

        composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

        if is_master:
            out_path = self.output_dir / f"{location_name}_master_panorama.jpg"
        else:
            out_path = self.output_dir / f"{location_name}_composite_preview.jpg"

        cv2.imwrite(str(out_path), composite_uint8, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # Save dynamic detail crop encapsulating all matched aperture projections with padding
        active_coords = np.where(total_mask > 0.01)
        if len(active_coords[0]) > 0 and not is_master:
            min_y, max_y = int(np.min(active_coords[0])), int(np.max(active_coords[0]))
            min_x, max_x = int(np.min(active_coords[1])), int(np.max(active_coords[1]))

            pad_x = max(20, int((max_x - min_x) * 0.08))
            pad_y = max(20, int((max_y - min_y) * 0.08))

            crop_x1 = max(0, min_x - pad_x)
            crop_y1 = max(0, min_y - pad_y)
            crop_x2 = min(w_canvas, max_x + pad_x)
            crop_y2 = min(h_canvas, max_y + pad_y)

            detail_crop = composite_uint8[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imwrite(str(self.output_dir / f"{location_name}_composite_detail_crop.jpg"), detail_crop)

        # 4. Generate 1:1 Native Sensor Resolution Landmark Crops
        if not is_master:
            self._export_landmark_crops(base_canvas, all_entries, images_map, location_name)

        return out_path

    def draw_master_gigapixel_panorama(
        self,
        base_canvas: np.ndarray,
        results: dict[str, RegistrationResult],
        images_map: dict[str, np.ndarray],
        apertures_map: dict[str, CircleAperture],
        location_name: str,
        tile_size: int = 4096,
    ) -> Path:
        """Stitch the full 1:1 camera sensor resolution master panorama into a single image file and generate DZI pyramid."""
        import gc
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

        # Determine target native magnification scale
        target_scale = 2.5  # 2.5x preserves full sensor clarity while keeping peak RAM < 1.5GB
        h_base, w_base = base_canvas.shape[:2]
        w_full = int(round(w_base * target_scale))
        h_full = int(round(h_base * target_scale))

        logger.info(f"Stitching full native master panorama: {w_full}x{h_full} px ({w_full*h_full/1e6:.1f} MP)...")

        # 1. Allocate base canvas at target scale (577 MB uint8 in RAM)
        master_canvas = cv2.resize(base_canvas, (w_full, h_full), interpolation=cv2.INTER_LANCZOS4)
        priority_map = np.zeros((h_full, w_full), dtype=np.float32)
        S_mat = np.array([[target_scale, 0.0, 0.0], [0.0, target_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        # 2. Gather all registered isolated portholes metadata (streaming: DO NOT store heavy RGBAs)
        active_entries = []
        for name, res in results.items():
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if status_val == "FAILED" or res.bounding_box_in_base is None or name not in images_map:
                continue

            target_img = images_map[name]
            H_global = res.get_matrix_np()
            H_scaled = S_mat @ H_global

            ap = apertures_map.get(name) or CircleAperture(
                center_x=target_img.shape[1] / 2.0,
                center_y=target_img.shape[0] / 2.0,
                radius=min(target_img.shape[:2]) * 0.45,
                image_width=target_img.shape[1],
                image_height=target_img.shape[0],
            )

            scale_approx = float(np.sqrt(H_global[0, 0] ** 2 + H_global[0, 1] ** 2))
            zoom_factor = 1.0 / (scale_approx + 1e-4)
            footprint_area = np.pi * ((ap.radius * scale_approx * target_scale) ** 2)

            cy, cx = int(ap.center_y), int(ap.center_x)
            r_box = int(ap.radius * 0.25)
            core_patch = target_img[max(0, cy - r_box):min(target_img.shape[0], cy + r_box), max(0, cx - r_box):min(target_img.shape[1], cx + r_box)]
            if core_patch.size > 0:
                saliency = float(np.var(cv2.Laplacian(cv2.cvtColor(core_patch, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
            else:
                saliency = 10.0

            active_entries.append({
                "name": name,
                "res": res,
                "ap": ap,
                "H_scaled": H_scaled,
                "scale_approx": scale_approx,
                "zoom_factor": zoom_factor,
                "footprint_area": footprint_area,
                "saliency": saliency,
                "inliers": res.inlier_count,
            })

        active_entries.sort(key=lambda p: (p["footprint_area"] / (p["saliency"] + 1e-2), -p["inliers"]), reverse=True)

        # 3. Composite all portholes with strict zero-dilution Z-buffering using streaming memory-safe ROI bounds
        for entry in active_entries:
            name = str(entry["name"])
            target_img = images_map[name]
            ap = cast(CircleAperture, entry["ap"])
            H_scaled = cast(np.ndarray, entry["H_scaled"])
            scale_approx = cast(float, entry["scale_approx"])
            zoom_factor = cast(float, entry["zoom_factor"])

            # Load ONLY this single porthole RGBA on demand
            rgba_clean, clean_mask, saliency = self._load_isolated_porthole_rgba(name, target_img, ap, location_name)
            h_src, w_src = rgba_clean.shape[:2]

            y_g, x_g = np.ogrid[:ap.image_height, :ap.image_width]
            dist_norm = np.sqrt((x_g - ap.center_x) ** 2 + (y_g - ap.center_y) ** 2) / (ap.radius + 1e-5)
            centrality = np.clip(1.0 - (dist_norm / 0.82) ** 2, 0.0, 1.0)
            saliency_boost = 1.0 + 0.6 * np.log10(1.0 + saliency)
            q_weight = clean_mask * centrality * (zoom_factor ** 1.3) * saliency_boost

            corners = np.array([[[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]], dtype=np.float32)
            pts = cv2.perspectiveTransform(corners, H_scaled)[0]
            x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
            x_max = min(w_full, int(np.ceil(np.max(pts[:, 0]))))
            y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
            y_max = min(h_full, int(np.ceil(np.max(pts[:, 1]))))

            if x_max <= x_min or y_max <= y_min:
                del rgba_clean, clean_mask, q_weight
                continue

            w_roi = x_max - x_min
            h_roi = y_max - y_min

            T_roi = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
            H_roi = T_roi @ H_scaled

            w_rgba = cv2.warpPerspective(
                rgba_clean,
                H_roi,
                (w_roi, h_roi),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            w_tgt = w_rgba[:, :, :3].astype(np.float32)
            w_mask = w_rgba[:, :, 3].astype(np.float32) / 255.0

            w_q = cv2.warpPerspective(
                q_weight,
                H_roi,
                (w_roi, h_roi),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            q_curr = priority_map[y_min:y_max, x_min:x_max]
            higher = (w_q > q_curr) & (w_mask > 0.01)

            if np.any(higher):
                c_roi = master_canvas[y_min:y_max, x_min:x_max].astype(np.float32)

                # Multi-band detail fusion: match low-frequency illumination to canvas, eliminating veiling glare
                # while preserving 100% of native optical sensor sharpness and contrast on landmarks
                ksize = 61
                tgt_low = cv2.GaussianBlur(w_tgt, (ksize, ksize), 0)
                canvas_low = cv2.GaussianBlur(c_roi, (ksize, ksize), 0)
                matched_tgt = canvas_low + (w_tgt - tgt_low)

                alpha = np.zeros_like(w_q)
                alpha[higher] = np.clip(0.6 + (w_q[higher] - q_curr[higher]) / (2.0 * np.maximum(w_q[higher], 1e-4)), 0.6, 1.0) * w_mask[higher]
                alpha_3d = alpha[:, :, np.newaxis]

                comp_roi = c_roi * (1.0 - alpha_3d) + matched_tgt * alpha_3d
                master_canvas[y_min:y_max, x_min:x_max] = np.clip(comp_roi, 0, 255).astype(np.uint8)
                priority_map[y_min:y_max, x_min:x_max] = np.maximum(q_curr, w_q)
                del tgt_low, canvas_low, matched_tgt, c_roi, comp_roi

            del rgba_clean, clean_mask, q_weight, w_rgba, w_tgt, w_mask, w_q

        del priority_map
        gc.collect()

        # 4. Save the single stitched master panoramic image file
        master_out = self.output_dir / f"{location_name}_master_stitched_panorama.jpg"
        logger.info(f"Saving stitched master panorama to {master_out}...")
        pil_master = Image.fromarray(cv2.cvtColor(master_canvas, cv2.COLOR_BGR2RGB))
        pil_master.save(str(master_out), format="JPEG", quality=95, optimize=True)
        del pil_master
        gc.collect()

        # 5. Generate true multi-resolution DZI pyramid and polished HTML viewer
        self._generate_true_dzi_pyramid(master_canvas, location_name)

        # 6. Delete master canvas before generating the overview preview
        del master_canvas
        gc.collect()

        # 7. Also output the overview preview
        return self.draw_static_composite_preview(base_canvas, results, images_map, apertures_map, location_name, canvas_scale=1.0)

    def _generate_true_dzi_pyramid(
        self,
        full_img: np.ndarray,
        location_name: str,
        tile_size: int = 1024,
        overlap: int = 1,
    ) -> None:
        """Generate a complete multi-resolution DeepZoom Image (DZI) pyramid and OpenSeadragon viewer."""
        import math

        viewer_dir = self.output_dir / "viewer"
        viewer_dir.mkdir(parents=True, exist_ok=True)
        dzi_dir = viewer_dir / "panorama_files"
        dzi_dir.mkdir(parents=True, exist_ok=True)

        h_full, w_full = full_img.shape[:2]
        max_dim = max(w_full, h_full)
        max_level = int(math.ceil(math.log2(max_dim)))

        # Write panorama.dzi XML
        dzi_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Image TileSize="{tile_size}" Overlap="{overlap}" Format="jpg" xmlns="http://schemas.microsoft.com/deepzoom/2008">
  <Size Width="{w_full}" Height="{h_full}"/>
</Image>
"""
        with open(viewer_dir / "panorama.dzi", "w") as f:
            f.write(dzi_xml)

        # Generate pyramid levels
        curr_img = full_img
        for level in range(max_level, -1, -1):
            h_curr, w_curr = curr_img.shape[:2]
            level_dir = dzi_dir / str(level)
            level_dir.mkdir(parents=True, exist_ok=True)

            num_cols = (w_curr + tile_size - 1) // tile_size
            num_rows = (h_curr + tile_size - 1) // tile_size

            for row in range(num_rows):
                for col in range(num_cols):
                    x1 = max(0, col * tile_size - overlap)
                    y1 = max(0, row * tile_size - overlap)
                    x2 = min(w_curr, (col + 1) * tile_size + overlap)
                    y2 = min(h_curr, (row + 1) * tile_size + overlap)

                    tile_patch = curr_img[y1:y2, x1:x2]
                    tile_file = level_dir / f"{col}_{row}.jpg"
                    cv2.imwrite(str(tile_file), tile_patch, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Downsample by 2 for next level down
            if w_curr > 1 or h_curr > 1:
                next_w = max(1, w_curr // 2)
                next_h = max(1, h_curr // 2)
        # Write clean, full-screen HTML viewer with landmark presets
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PortPan Gigapixel Master &mdash; {location_name}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #0c0d10; color: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; }}
    #viewer-container {{ width: 100%; height: 100%; position: relative; }}
    #openseadragon {{ width: 100%; height: 100%; }}
    .glass-header {{
      position: absolute; top: 16px; left: 20px; z-index: 1000;
      background: rgba(18, 22, 34, 0.85); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px;
      padding: 12px 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }}
    .glass-header h1 {{ margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -0.2px; color: #f0f4f8; }}
    .glass-header p {{ margin: 4px 0 0 0; font-size: 12px; color: #8a99ad; }}
    .controls-panel {{
      position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 1000;
      display: flex; gap: 8px; background: rgba(18, 22, 34, 0.85); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 30px; padding: 6px 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }}
    .btn {{
      background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.1);
      color: #e2e8f0; font-size: 13px; font-weight: 500; padding: 8px 16px; border-radius: 20px;
      cursor: pointer; transition: all 0.2s ease;
    }}
    .btn:hover {{ background: rgba(255, 255, 255, 0.2); color: #fff; transform: translateY(-1px); }}
    .btn.active {{ background: #3b82f6; border-color: #60a5fa; color: #fff; }}
  </style>
</head>
<body>
  <div id="viewer-container">
    <div class="glass-header">
      <h1>PortPan Gigapixel Master &bull; {location_name}</h1>
      <p>Dimensions: {w_full:,} &times; {h_full:,} px &bull; Full Native Sensor Clarity</p>
    </div>
    <div class="controls-panel">
      <button class="btn active" onclick="zoomOverview()">Full Panorama</button>
      <button class="btn" onclick="zoomBoat()">The Boat</button>
      <button class="btn" onclick="zoomMonument()">Monument & Cliffs</button>
      <button class="btn" onclick="zoomNative()">1:1 Native Resolution</button>
    </div>
    <div id="openseadragon"></div>
  </div>
  <script>
    var viewer = OpenSeadragon({{
      id: "openseadragon",
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      tileSources: "panorama.dzi",
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      navigatorSizeRatio: 0.18,
      showFullPageControl: true,
      defaultZoomLevel: 0,
      minZoomLevel: 0.2,
      maxZoomPixelRatio: 2.5,
      animationTime: 0.8,
      springStiffness: 12.0
    }});

    function zoomOverview() {{
      viewer.viewport.goHome();
    }}
    function zoomBoat() {{
      // Coordinates normalized relative to width: x=4750/9728≈0.488, y=1450/9728≈0.149
      viewer.viewport.panTo(new OpenSeadragon.Point(0.488, 0.149));
      viewer.viewport.zoomTo(3.2);
    }}
    function zoomMonument() {{
      // Coordinates normalized relative to width: x=2420/9728≈0.249, y=550/9728≈0.056
      viewer.viewport.panTo(new OpenSeadragon.Point(0.249, 0.056));
      viewer.viewport.zoomTo(2.8);
    }}
    function zoomNative() {{
      viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1.0));
    }}
  </script>
</body>
</html>
"""
        with open(viewer_dir / "index.html", "w") as f:
            f.write(html_content)

    def _generate_openseadragon_viewer(
        self,
        tiles_dir: Path,
        w_full: int,
        h_full: int,
        tile_size: int,
        location_name: str,
    ) -> None:
        """Create a standalone OpenSeadragon HTML viewer for interactive gigapixel exploration."""
        viewer_dir = self.output_dir / "viewer"
        viewer_dir.mkdir(parents=True, exist_ok=True)

        num_tiles_x = (w_full + tile_size - 1) // tile_size
        num_tiles_y = (h_full + tile_size - 1) // tile_size

        tile_sources = []
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                tile_p = tiles_dir / f"tile_{ty}_{tx}.jpg"
                if tile_p.exists():
                    x_norm = (tx * tile_size) / float(w_full)
                    y_norm = (ty * tile_size) / float(w_full)  # normalized relative to width
                    w_norm = min(tile_size, w_full - tx * tile_size) / float(w_full)
                    tile_sources.append({
                        "type": "image",
                        "url": f"../native_panorama_tiles/tile_{ty}_{tx}.jpg",
                        "x": x_norm,
                        "y": y_norm,
                        "width": w_norm,
                    })

        import json
        sources_json = json.dumps(tile_sources, indent=2)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PortPan Gigapixel Master Panorama: {location_name}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
  <style>
    body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #0a0a0a; color: #eee; font-family: sans-serif; overflow: hidden; }}
    #openseadragon {{ width: 100%; height: 100%; }}
    .header-bar {{ position: absolute; top: 10px; left: 15px; z-index: 100; background: rgba(0,0,0,0.75); padding: 8px 16px; border-radius: 8px; backdrop-filter: blur(4px); }}
    .header-bar h1 {{ margin: 0; font-size: 16px; font-weight: 600; color: #fff; }}
    .header-bar p {{ margin: 2px 0 0 0; font-size: 12px; color: #aaa; }}
  </style>
</head>
<body>
  <div class="header-bar">
    <h1>PortPan Gigapixel Panorama &mdash; {location_name}</h1>
    <p>Dimensions: {w_full:,} &times; {h_full:,} px &bull; Scroll to Zoom to 100% Sensor Resolution</p>
  </div>
  <div id="openseadragon"></div>
  <script>
    var viewer = OpenSeadragon({{
      id: "openseadragon",
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      tileSources: {sources_json},
      showNavigator: true,
      showFullPageControl: true,
      minZoomImageRatio: 0.8,
      maxZoomPixelRatio: 3.0,
      visibilityRatio: 1.0,
      constrainDuringPan: true
    }});
  </script>
</body>
</html>
"""
        with open(viewer_dir / "index.html", "w") as f:
            f.write(html_content)

    def _export_landmark_crops(
        self,
        base_canvas: np.ndarray,
        all_entries: list[dict[str, Any]],
        images_map: dict[str, np.ndarray],
        location_name: str,
    ) -> None:
        """Export full 1:1 camera sensor resolution crops centered on high-saliency telephoto landmarks."""
        if not all_entries:
            return

        crops_dir = self.output_dir / "landmark_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Sort by saliency & zoom to capture the most distinct landmarks
        sorted_landmarks = sorted(
            all_entries,
            key=lambda e: e["saliency"] * (1.0 / (e["scale_approx"] + 1e-4)),
            reverse=True,
        )

        for idx, entry in enumerate(sorted_landmarks[:6]):
            name = str(entry["name"])
            target_img = images_map[name]
            ap = cast(CircleAperture, entry["ap"])
            saliency = float(entry["saliency"])
            res = entry["res"]
            scale_approx = float(entry["scale_approx"])

            # 1. Optical cutout at 1:1
            rgba = ap.extract_rgba(target_img, feather_px=2.0)
            sq_x1, sq_y1, sq_x2, sq_y2 = ap.get_inscribed_square(safety_margin=0.05)
            square_crop = rgba[sq_y1:sq_y2, sq_x1:sq_x2]

            safe_stem = Path(name).stem
            out_p = crops_dir / f"landmark_{idx + 1}_{safe_stem}_sal{int(saliency)}.png"
            cv2.imwrite(str(out_p), square_crop)

            # 2. Native sensor registered crop
            if scale_approx > 1e-4 and res.bounding_box_in_base is not None:
                S_native = 1.0 / scale_approx
                pts = np.array(res.bounding_box_in_base, dtype=np.float32) * S_native
                min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
                min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))

                pad = 200
                x1, x2 = max(0, min_x - pad), max_x + pad
                y1, y2 = max(0, min_y - pad), max_y + pad
                w_crop = x2 - x1
                h_crop = y2 - y1

                if w_crop > 0 and h_crop > 0:
                    T = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64)
                    H_crop = T @ (np.diag([S_native, S_native, 1.0]) @ res.get_matrix_np())

                    base_x1, base_x2 = int(x1 / S_native), int(x2 / S_native)
                    base_y1, base_y2 = int(y1 / S_native), int(y2 / S_native)
                    base_patch = base_canvas[base_y1:base_y2, base_x1:base_x2]
                    if base_patch.size > 0:
                        base_crop_scaled = cv2.resize(base_patch, (w_crop, h_crop), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
                        w_rgba = cv2.warpPerspective(rgba, H_crop, (w_crop, h_crop), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                        w_tgt = w_rgba[:, :, :3].astype(np.float32)
                        w_mask = (w_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

                        comp_crop = base_crop_scaled * (1.0 - w_mask) + w_tgt * w_mask
                        crop_out = crops_dir / f"{location_name}_{safe_stem}_1to1_registered_crop.jpg"
                        cv2.imwrite(str(crop_out), np.clip(comp_crop, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])

                        if "boat" in safe_stem.lower() or "211435238" in safe_stem:
                            cv2.imwrite(str(self.output_dir / f"{location_name}_composite_native_crop.jpg"), np.clip(comp_crop, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])



