"""Visual diagnostic and audit reporting for registration verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2
import numpy as np
from portpan.core.models import CircleAperture, RegistrationResult


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

    def draw_static_composite_preview(
        self,
        base_canvas: np.ndarray,
        results: dict[str, RegistrationResult],
        images_map: dict[str, np.ndarray],
        apertures_map: dict[str, CircleAperture],
        location_name: str,
        canvas_scale: float = 1.0,
    ) -> Path:
        """Render a composite preview warping all registered portholes onto the master canvas with super-resolution detail."""
        h_orig, w_orig = base_canvas.shape[:2]

        w_canvas = int(round(w_orig * canvas_scale))
        h_canvas = int(round(h_orig * canvas_scale))
        if canvas_scale != 1.0:
            composite = cv2.resize(base_canvas, (w_canvas, h_canvas), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
            S_mat = np.array([[canvas_scale, 0.0, 0.0], [0.0, canvas_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            composite = base_canvas.copy().astype(np.float32)
            S_mat = np.eye(3, dtype=np.float64)

        # 1. Gather all registered portholes (preserving 100% of successful registrations)
        all_entries: list[dict[str, Any]] = []
        for name, res in results.items():
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if status_val == "FAILED" or res.bounding_box_in_base is None or name not in images_map:
                continue

            target_img = images_map[name]
            H_global = res.get_matrix_np()
            H_scaled = S_mat @ H_global

            if apertures_map and name in apertures_map:
                ap = apertures_map[name]
            else:
                from portpan.segmentation.circle_segmenter import RobustCircleSegmenter
                ap = RobustCircleSegmenter().segment(target_img)

            # Generate smooth optical mask with soft S-curve outer glass transition
            y_grid, x_grid = np.ogrid[:ap.image_height, :ap.image_width]
            dist_from_center = np.sqrt((x_grid - ap.center_x) ** 2 + (y_grid - ap.center_y) ** 2)
            r_core = ap.radius * 0.96
            r_inner, r_outer = r_core * 0.88, r_core
            mask_s = np.clip((r_outer - dist_from_center) / (r_outer - r_inner + 1e-5), 0.0, 1.0)
            mask_s = 0.5 - 0.5 * np.cos(mask_s * np.pi)  # smooth S-curve
            mask_native = ap.apply_luma_modulation(mask_s.astype(np.float32), target_img, rim_fraction=0.20)

            # Canvas scale and footprint area
            scale_approx = float(np.sqrt(H_scaled[0, 0] ** 2 + H_scaled[0, 1] ** 2))
            r_canvas = float(ap.radius) * scale_approx
            footprint_area = np.pi * (r_canvas ** 2)

            all_entries.append({
                "name": name,
                "res": res,
                "H_scaled": H_scaled,
                "mask_native": mask_native,
                "footprint_area": footprint_area,
                "inliers": res.inlier_count,
            })

        # 2. Strict Z-sort by canvas footprint area descending (widest context in back, highest zoom on top)
        all_entries.sort(key=lambda p: (p["footprint_area"], -p["inliers"]), reverse=True)

        total_mask = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        # 3. Warp and blend in-place one porthole at a time with ROI bounding boxes
        for entry in all_entries:
            name = entry["name"]
            target_img = images_map[name]
            H_scaled = entry["H_scaled"]
            mask_native = entry["mask_native"]

            # Warp mask and target to canvas
            w_mask = cv2.warpPerspective(
                mask_native,
                H_scaled,
                (w_canvas, h_canvas),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            w_target = cv2.warpPerspective(
                target_img,
                H_scaled,
                (w_canvas, h_canvas),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            total_mask = np.maximum(total_mask, w_mask)

            # Blend strictly within active ROI bounding box
            nz_rows, nz_cols = np.where(w_mask > 0.001)
            if len(nz_rows) > 0:
                ymin, ymax = int(np.min(nz_rows)), int(np.max(nz_rows)) + 1
                xmin, xmax = int(np.min(nz_cols)), int(np.max(nz_cols)) + 1

                m_roi = w_mask[ymin:ymax, xmin:xmax, np.newaxis]
                c_roi = composite[ymin:ymax, xmin:xmax]
                b_roi = w_target[ymin:ymax, xmin:xmax].astype(np.float32)

                composite[ymin:ymax, xmin:xmax] = c_roi * (1.0 - m_roi) + b_roi * m_roi

            del w_mask, w_target

        composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

        # Save main composite preview with maximum JPEG quality (100)
        out_path = self.output_dir / f"{location_name}_composite_preview.jpg"
        cv2.imwrite(str(out_path), composite_uint8, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # Save dynamic detail crop encapsulating all matched aperture projections with padding
        active_coords = np.where(total_mask > 0.01)
        if len(active_coords[0]) > 0:
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

        # 4. Generate 1:1 Native Sensor Resolution Crops for Highest-Zoom / Telephoto Portholes
        self._export_native_sensor_crops(base_canvas, all_entries, images_map, apertures_map, location_name)

        return out_path

    def _export_native_sensor_crops(
        self,
        base_canvas: np.ndarray,
        all_entries: list[dict[str, Any]],
        images_map: dict[str, np.ndarray],
        apertures_map: dict[str, CircleAperture],
        location_name: str,
    ) -> None:
        """Export 1:1 camera sensor resolution crops without downsampling for key zoom regions."""
        if not all_entries:
            return

        crops_dir = self.output_dir / "native_sensor_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        h_base, w_base = base_canvas.shape[:2]

        # Pick top 5 most zoomed-in portholes (smallest scale / highest zoom)
        sorted_by_zoom = sorted(
            all_entries,
            key=lambda e: np.sqrt(e["res"].get_matrix_np()[0, 0] ** 2 + e["res"].get_matrix_np()[0, 1] ** 2),
            reverse=True,
        )

        for entry in sorted_by_zoom[:5]:
            name = entry["name"]
            res = entry["res"]
            target_img = images_map[name]
            ap = apertures_map.get(name) or CircleAperture(
                center_x=target_img.shape[1] / 2.0,
                center_y=target_img.shape[0] / 2.0,
                radius=min(target_img.shape[:2]) * 0.45,
                image_width=target_img.shape[1],
                image_height=target_img.shape[0],
            )

            H = res.get_matrix_np()
            s = float(np.sqrt(H[0, 0] ** 2 + H[0, 1] ** 2))
            if s <= 1e-4:
                continue

            S_native = 1.0 / s
            pts = np.array(res.bounding_box_in_base, dtype=np.float32) * S_native
            min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))

            pad = 200
            x1, x2 = max(0, min_x - pad), max_x + pad
            y1, y2 = max(0, min_y - pad), max_y + pad
            w_crop = x2 - x1
            h_crop = y2 - y1

            if w_crop <= 0 or h_crop <= 0:
                continue

            T = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64)
            H_crop = T @ (np.diag([S_native, S_native, 1.0]) @ H)

            base_x1, base_x2 = int(x1 / S_native), int(x2 / S_native)
            base_y1, base_y2 = int(y1 / S_native), int(y2 / S_native)
            base_patch = base_canvas[base_y1:base_y2, base_x1:base_x2]
            if base_patch.size == 0:
                continue

            base_crop_scaled = cv2.resize(base_patch, (w_crop, h_crop), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

            y_grid, x_grid = np.ogrid[:ap.image_height, :ap.image_width]
            dist_from_center = np.sqrt((x_grid - ap.center_x) ** 2 + (y_grid - ap.center_y) ** 2)
            r_core = ap.radius * 0.96
            r_inner, r_outer = r_core * 0.88, r_core
            mask_s = np.clip((r_outer - dist_from_center) / (r_outer - r_inner + 1e-5), 0.0, 1.0)
            mask_s = 0.5 - 0.5 * np.cos(mask_s * np.pi)
            mask_native = ap.apply_luma_modulation(mask_s.astype(np.float32), target_img, rim_fraction=0.20)

            w_mask = cv2.warpPerspective(mask_native, H_crop, (w_crop, h_crop), flags=cv2.INTER_LINEAR)
            w_tgt = cv2.warpPerspective(target_img, H_crop, (w_crop, h_crop), flags=cv2.INTER_LANCZOS4).astype(np.float32)

            m = w_mask[:, :, np.newaxis]
            comp_crop = base_crop_scaled * (1.0 - m) + w_tgt * m

            stem = Path(name).stem
            crop_out = crops_dir / f"{location_name}_{stem}_1to1_sensor_crop.jpg"
            cv2.imwrite(str(crop_out), np.clip(comp_crop, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])

            # Also write the primary native detail crop
            if "boat" in stem.lower() or "211435238" in stem:
                cv2.imwrite(str(self.output_dir / f"{location_name}_composite_native_crop.jpg"), np.clip(comp_crop, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])

