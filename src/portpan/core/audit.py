"""Visual diagnostic and audit reporting for registration verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
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
        is_master: bool = False,
    ) -> Path:
        """Render a composite preview warping all registered portholes onto the master canvas with priority Z-buffer blending."""
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

            # Exact optical boundary mask with adaptive luma suppression
            mask_native = (ap.generate_mask(image=target_img).astype(np.float32) / 255.0)

            # Object centrality weight: centered subjects (r < 0.6) have maximum weight
            y_g, x_g = np.ogrid[:ap.image_height, :ap.image_width]
            dist_norm = np.sqrt((x_g - ap.center_x) ** 2 + (y_g - ap.center_y) ** 2) / (ap.radius + 1e-5)
            centrality = np.clip(1.0 - (dist_norm / 0.85) ** 2, 0.0, 1.0)

            scale_approx = float(np.sqrt(H_global[0, 0] ** 2 + H_global[0, 1] ** 2))
            zoom_factor = 1.0 / (scale_approx + 1e-4)

            # Combined quality weight for object-centered z-ordering and blending
            quality_weight = mask_native * centrality * (zoom_factor ** 1.5)
            footprint_area = np.pi * ((ap.radius * scale_approx * canvas_scale) ** 2)

            all_entries.append({
                "name": name,
                "res": res,
                "H_scaled": H_scaled,
                "mask_native": mask_native,
                "quality_weight": quality_weight,
                "scale_approx": scale_approx,
                "footprint_area": footprint_area,
                "inliers": res.inlier_count,
            })

        # 2. Sort from widest context first (larger footprint) to highest zoom last (smaller footprint)
        all_entries.sort(key=lambda p: (p["footprint_area"], -p["inliers"]), reverse=True)

        total_mask = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        # 3. Warp and blend using priority Z-buffer weighting
        for entry in all_entries:
            name = str(entry["name"])
            target_img = images_map[name]
            H_scaled = cast(np.ndarray, entry["H_scaled"])
            mask_native = cast(np.ndarray, entry["mask_native"])
            quality_weight = cast(np.ndarray, entry["quality_weight"])

            # Warp mask, quality weight, and target to canvas
            w_mask = cv2.warpPerspective(
                mask_native,
                H_scaled,
                (w_canvas, h_canvas),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            w_q = cv2.warpPerspective(
                quality_weight,
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

            # Priority Z-buffer blend strictly within active ROI bounding box
            nz_rows, nz_cols = np.where(w_mask > 0.001)
            if len(nz_rows) > 0:
                ymin, ymax = int(np.min(nz_rows)), int(np.max(nz_rows)) + 1
                xmin, xmax = int(np.min(nz_cols)), int(np.max(nz_cols)) + 1

                m_roi = w_mask[ymin:ymax, xmin:xmax]
                q_new = w_q[ymin:ymax, xmin:xmax]
                q_curr = priority_map[ymin:ymax, xmin:xmax]

                alpha = np.zeros_like(q_new)
                higher = q_new >= q_curr
                lower = ~higher

                alpha[higher] = np.clip(0.5 + (q_new[higher] - q_curr[higher]) / (2.0 * np.maximum(q_new[higher], 1e-4)), 0.5, 1.0) * m_roi[higher]
                diff = q_curr[lower] - q_new[lower]
                alpha[lower] = np.clip(1.0 - diff / (q_new[lower] + 1e-4), 0.0, 0.5) * m_roi[lower]

                alpha_3d = alpha[:, :, np.newaxis]
                c_roi = composite[ymin:ymax, xmin:xmax]
                b_roi = w_target[ymin:ymax, xmin:xmax].astype(np.float32)

                composite[ymin:ymax, xmin:xmax] = c_roi * (1.0 - alpha_3d) + b_roi * alpha_3d
                priority_map[ymin:ymax, xmin:xmax] = np.maximum(q_curr, q_new)

            del w_mask, w_q, w_target

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

        # 4. Generate 1:1 Native Sensor Resolution Crops for Highest-Zoom / Telephoto Portholes
        if not is_master:
            self._export_native_sensor_crops(base_canvas, all_entries, images_map, apertures_map, location_name)

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
        """Render the full 1:1 camera sensor resolution master panorama in memory-safe tiles (<300MB RAM)."""
        scales = []
        for name, res in results.items():
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if status_val != "FAILED" and res.bounding_box_in_base is not None and name in images_map:
                H = res.get_matrix_np()
                s = float(np.sqrt(H[0, 0] ** 2 + H[0, 1] ** 2))
                if s > 1e-4:
                    scales.append(s)

        if not scales:
            return self.draw_static_composite_preview(base_canvas, results, images_map, apertures_map, location_name, canvas_scale=1.0)

        s_min = min(scales)
        S_native = min(6.5, 1.0 / s_min)
        h_base, w_base = base_canvas.shape[:2]
        w_full = int(round(w_base * S_native))
        h_full = int(round(h_base * S_native))

        tiles_dir = self.output_dir / "native_panorama_tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        S_mat = np.diag([S_native, S_native, 1.0])

        # Prepare list of registered entries
        active_entries = []
        for name, res in results.items():
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if status_val != "FAILED" and res.bounding_box_in_base is not None and name in images_map:
                ap = apertures_map.get(name) or CircleAperture(
                    center_x=images_map[name].shape[1] / 2.0,
                    center_y=images_map[name].shape[0] / 2.0,
                    radius=min(images_map[name].shape[:2]) * 0.45,
                    image_width=images_map[name].shape[1],
                    image_height=images_map[name].shape[0],
                )
                H_global = res.get_matrix_np()
                scale_approx = float(np.sqrt(H_global[0, 0] ** 2 + H_global[0, 1] ** 2))
                zoom_factor = 1.0 / (scale_approx + 1e-4)

                mask_native = ap.generate_mask(image=images_map[name]).astype(np.float32) / 255.0
                y_g, x_g = np.ogrid[:ap.image_height, :ap.image_width]
                dist_norm = np.sqrt((x_g - ap.center_x) ** 2 + (y_g - ap.center_y) ** 2) / (ap.radius + 1e-5)
                centrality = np.clip(1.0 - (dist_norm / 0.85) ** 2, 0.0, 1.0)
                q_weight = mask_native * centrality * (zoom_factor ** 1.5)

                active_entries.append({
                    "name": name,
                    "res": res,
                    "ap": ap,
                    "H_global": H_global,
                    "mask_native": mask_native,
                    "q_weight": q_weight,
                    "scale_approx": scale_approx,
                    "footprint_area": np.pi * ((ap.radius * scale_approx * S_native) ** 2),
                    "inliers": res.inlier_count,
                })

        active_entries.sort(key=lambda p: (p["footprint_area"], -p["inliers"]), reverse=True)

        # Tile across full native space
        num_tiles_x = (w_full + tile_size - 1) // tile_size
        num_tiles_y = (h_full + tile_size - 1) // tile_size

        for ty_idx in range(num_tiles_y):
            for tx_idx in range(num_tiles_x):
                tile_x = tx_idx * tile_size
                tile_y = ty_idx * tile_size
                tw = min(tile_size, w_full - tile_x)
                th = min(tile_size, h_full - tile_y)

                # Base patch
                b_x1, b_x2 = int(tile_x / S_native), int(min(w_full, tile_x + tw) / S_native)
                b_y1, b_y2 = int(tile_y / S_native), int(min(h_full, tile_y + th) / S_native)
                base_patch = base_canvas[b_y1:b_y2, b_x1:b_x2]
                if base_patch.size == 0:
                    continue

                tile_base = cv2.resize(base_patch, (tw, th), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
                tile_comp = tile_base.copy()
                tile_priority = np.zeros((th, tw), dtype=np.float32)

                T_tile = np.array([[1, 0, -tile_x], [0, 1, -tile_y], [0, 0, 1]], dtype=np.float64)

                has_portholes = False
                for entry in active_entries:
                    name_str = str(entry["name"])
                    target_img = images_map[name_str]
                    ap_entry = cast(CircleAperture, entry["ap"])
                    H_global_entry = cast(np.ndarray, entry["H_global"])
                    H_tile = T_tile @ (S_mat @ H_global_entry)

                    # Quick intersection check
                    pts = cv2.perspectiveTransform(
                        np.array([[[0, 0], [ap_entry.image_width, 0], [ap_entry.image_width, ap_entry.image_height], [0, ap_entry.image_height]]], dtype=np.float32),
                        H_tile,
                    )[0]
                    if np.max(pts[:, 0]) < 0 or np.min(pts[:, 0]) >= tw or np.max(pts[:, 1]) < 0 or np.min(pts[:, 1]) >= th:
                        continue

                    has_portholes = True
                    mask_native_entry = cast(np.ndarray, entry["mask_native"])
                    q_weight_entry = cast(np.ndarray, entry["q_weight"])

                    w_mask = cv2.warpPerspective(mask_native_entry, H_tile, (tw, th), flags=cv2.INTER_LINEAR)
                    w_q = cv2.warpPerspective(q_weight_entry, H_tile, (tw, th), flags=cv2.INTER_LINEAR)
                    w_tgt = cv2.warpPerspective(target_img, H_tile, (tw, th), flags=cv2.INTER_LANCZOS4).astype(np.float32)

                    nz = np.where(w_mask > 0.001)
                    if len(nz[0]) > 0:
                        y1, y2 = int(np.min(nz[0])), int(np.max(nz[0])) + 1
                        x1, x2 = int(np.min(nz[1])), int(np.max(nz[1])) + 1

                        m_roi = w_mask[y1:y2, x1:x2]
                        q_new = w_q[y1:y2, x1:x2]
                        q_curr = tile_priority[y1:y2, x1:x2]

                        alpha = np.zeros_like(q_new)
                        higher = q_new >= q_curr
                        lower = ~higher

                        alpha[higher] = np.clip(0.5 + (q_new[higher] - q_curr[higher]) / (2.0 * np.maximum(q_new[higher], 1e-4)), 0.5, 1.0) * m_roi[higher]
                        diff = q_curr[lower] - q_new[lower]
                        alpha[lower] = np.clip(1.0 - diff / (q_new[lower] + 1e-4), 0.0, 0.5) * m_roi[lower]

                        alpha_3d = alpha[:, :, np.newaxis]
                        tile_comp[y1:y2, x1:x2] = tile_comp[y1:y2, x1:x2] * (1.0 - alpha_3d) + w_tgt[y1:y2, x1:x2] * alpha_3d
                        tile_priority[y1:y2, x1:x2] = np.maximum(q_curr, q_new)

                    del w_mask, w_q, w_tgt

                if has_portholes:
                    tile_out = tiles_dir / f"tile_{ty_idx}_{tx_idx}.jpg"
                    cv2.imwrite(str(tile_out), np.clip(tile_comp, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])

                del tile_base, tile_comp, tile_priority

        # Also output the full-scene overview preview
        return self.draw_static_composite_preview(base_canvas, results, images_map, apertures_map, location_name, canvas_scale=1.0)

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

        # Pick top 5 most zoomed-in portholes (smallest scale / highest zoom)
        sorted_by_zoom = sorted(
            all_entries,
            key=lambda e: e["scale_approx"],
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

            mask_native = (ap.generate_mask(feather_px=4.0).astype(np.float32) / 255.0)

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



