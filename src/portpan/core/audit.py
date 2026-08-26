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
            if res.bounding_box_in_base is None or res.status == "FAILED":
                continue

            pts = np.array(res.bounding_box_in_base, dtype=np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if res.status == "SUCCESS" else (0, 200, 255)

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
        canvas_scale: float = 2.0,
    ) -> Path:
        """Render a composite preview warping all registered portholes onto the master canvas with super-resolution detail."""
        h_orig, w_orig = base_canvas.shape[:2]

        # Scale canvas up to super-resolution to retain full optical detail of digiscope telephoto shots
        w_canvas = int(round(w_orig * canvas_scale))
        h_canvas = int(round(h_orig * canvas_scale))
        if canvas_scale != 1.0:
            composite = cv2.resize(base_canvas, (w_canvas, h_canvas), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
            S_mat = np.array([[canvas_scale, 0.0, 0.0], [0.0, canvas_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            composite = base_canvas.copy().astype(np.float32)
            S_mat = np.eye(3, dtype=np.float64)

        portholes: list[dict[str, Any]] = []
        total_mask = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        for name, res in results.items():
            if res.status == "FAILED" or name not in images_map or name not in apertures_map:
                continue

            target_img = images_map[name]
            aperture = apertures_map[name]
            H_global = res.get_matrix_np()
            H_scaled = S_mat @ H_global

            # Generate exact chroma-preserving feathered mask with luma modulation to prevent dark bevel blocking
            mask = aperture.generate_mask(feather_px=2.0, image=target_img).astype(np.float32) / 255.0

            # Apply mask to target image
            masked_target = target_img.astype(np.float32) * mask[:, :, np.newaxis]

            # Warp masked target and mask into base canvas space
            warped_target = cv2.warpPerspective(
                masked_target,
                H_scaled,
                (w_canvas, h_canvas),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped_mask = cv2.warpPerspective(
                mask,
                H_scaled,
                (w_canvas, h_canvas),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            total_mask = np.maximum(total_mask, warped_mask)
            footprint_area = float(np.sum(warped_mask > 0.05))

            portholes.append({
                "name": name,
                "warped_target": warped_target,
                "warped_mask": warped_mask,
                "area": footprint_area,
            })

        # Z-sort: largest footprint first (background), smallest/highest zoom on top
        portholes.sort(key=lambda p: p["area"], reverse=True)

        for p in portholes:
            m3 = np.clip(p["warped_mask"][:, :, np.newaxis], 0.0, 1.0)
            composite = composite * (1.0 - m3) + p["warped_target"] * m3

        composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

        # Save main composite preview
        out_path = self.output_dir / f"{location_name}_composite_preview.jpg"
        cv2.imwrite(str(out_path), composite_uint8)

        # Save dynamic detail crop encapsulating all matched aperture projections with padding
        active_coords = np.where(total_mask > 0.01)
        if len(active_coords[0]) > 0:
            min_y, max_y = int(np.min(active_coords[0])), int(np.max(active_coords[0]))
            min_x, max_x = int(np.min(active_coords[1])), int(np.max(active_coords[1]))

            # Add 8% padding margin around the bounding box (at least 20px)
            pad_x = max(20, int((max_x - min_x) * 0.08))
            pad_y = max(20, int((max_y - min_y) * 0.08))

            crop_x1 = max(0, min_x - pad_x)
            crop_y1 = max(0, min_y - pad_y)
            crop_x2 = min(w_canvas, max_x + pad_x)
            crop_y2 = min(h_canvas, max_y + pad_y)

            detail_crop = composite_uint8[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imwrite(str(self.output_dir / f"{location_name}_composite_detail_crop.jpg"), detail_crop)

        return out_path
