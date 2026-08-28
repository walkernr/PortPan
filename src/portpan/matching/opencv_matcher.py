"""High-precision OpenCV SIFT/AKAZE matcher with FLANN acceleration and USAC_MAGSAC estimation."""

from __future__ import annotations

import cv2
import numpy as np
from portpan.matching.base import BaseDenseMatcher, MatchResult


class OpenCVMatcher(BaseDenseMatcher):
    """Dense feature matcher using multi-scale SIFT, FLANN indexing, and USAC_MAGSAC homography estimation."""

    def __init__(
        self,
        algorithm: str = "SIFT",
        n_features: int = 0,  # 0 means unrestricted SIFT features
        ratio_threshold: float = 0.78,
        ransac_reproj_thresh: float = 4.5,
        scales: list[float] | None = None,
    ) -> None:
        self.algorithm = algorithm.upper()
        self.n_features = n_features
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.scales = scales or [1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125, 1.33]

    def _create_detector(self) -> cv2.Feature2D:
        if self.algorithm == "AKAZE":
            return cv2.AKAZE_create()
        return cv2.SIFT_create(
            nfeatures=self.n_features,
            contrastThreshold=0.03,
            edgeThreshold=12,
        )

    def _create_matcher(self) -> cv2.DescriptorMatcher:
        if self.algorithm == "AKAZE":
            return cv2.BFMatcher(cv2.NORM_HAMMING)
        # FLANN with KD-Trees
        index_params = dict(algorithm=1, trees=4)
        search_params = dict(checks=32)
        return cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, base_crop: np.ndarray, target_patch: np.ndarray) -> MatchResult:
        """Match features between base_crop and target_patch with multi-scale pyramid search."""
        detector = self._create_detector()

        gray_base = cv2.cvtColor(base_crop, cv2.COLOR_BGR2GRAY) if len(base_crop.shape) == 3 else base_crop
        gray_target = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY) if len(target_patch.shape) == 3 else target_patch

        kp_base, des_base = detector.detectAndCompute(gray_base, None)
        if des_base is None or len(kp_base) < 8:
            return MatchResult(
                H_local=None,
                pts_base=np.empty((0, 2), dtype=np.float32),
                pts_target=np.empty((0, 2), dtype=np.float32),
                inlier_mask=None,
                inlier_count=0,
                inlier_ratio=0.0,
            )

        matcher = self._create_matcher()
        best_result: MatchResult | None = None
        max_inliers = -1

        h_t, w_t = gray_target.shape[:2]

        for scale in self.scales:
            scaled_target = (
                cv2.resize(gray_target, (int(w_t * scale), int(h_t * scale)), interpolation=cv2.INTER_LINEAR)
                if abs(scale - 1.0) > 1e-3
                else gray_target
            )

            kp_tgt, des_tgt = detector.detectAndCompute(scaled_target, None)
            if des_tgt is None or len(kp_tgt) < 8:
                continue

            try:
                matches = matcher.knnMatch(des_tgt, des_base, k=2)
            except Exception:
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des_tgt, des_base, k=2)

            good_matches: list[cv2.DMatch] = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 6:
                continue

            pts_tgt_arr = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]) / scale
            pts_base_arr = np.float32([kp_base[m.trainIdx].pt for m in good_matches])

            # 1. Estimate Similarity (Scale + Rotation + Translation)
            M_sim, mask_sim = cv2.estimateAffinePartial2D(
                pts_tgt_arr,
                pts_base_arr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thresh,
            )
            inliers_sim_count = int(mask_sim.ravel().tolist().count(1)) if mask_sim is not None else 0

            # 2. Estimate Homography using USAC_MAGSAC
            H_cand, mask_h = cv2.findHomography(
                pts_tgt_arr,
                pts_base_arr,
                cv2.USAC_MAGSAC,
                self.ransac_reproj_thresh,
                maxIters=1200,
                confidence=0.99,
            )
            inliers_h_count = int(mask_h.ravel().tolist().count(1)) if mask_h is not None else 0

            # Check if Homography has perspective distortion or aspect ratio squash
            use_homography = False
            if H_cand is not None and inliers_h_count >= 15:
                # Check singular values of 2x2 affine part to prevent squashing/distortion
                try:
                    u, s, vh = np.linalg.svd(H_cand[:2, :2] / (H_cand[2, 2] or 1.0))
                    cond = s[0] / (s[1] or 1e-6)
                    # For telephoto digiscoping, condition number should be near 1.0 (isotropic)
                    if cond <= 1.20 and inliers_h_count > inliers_sim_count + 5:
                        use_homography = True
                except Exception:
                    use_homography = False

            if use_homography:
                H_chosen = H_cand
                mask_chosen = mask_h
                inlier_count = inliers_h_count
                is_aff = False
            elif M_sim is not None and inliers_sim_count >= 10:
                H_chosen = np.vstack([M_sim, [0.0, 0.0, 1.0]])
                mask_chosen = mask_sim
                inlier_count = inliers_sim_count
                is_aff = True
            elif H_cand is not None:
                H_chosen = H_cand
                mask_chosen = mask_h
                inlier_count = inliers_h_count
                is_aff = False
            else:
                continue

            inlier_ratio = inlier_count / float(len(good_matches)) if good_matches else 0.0

            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_result = MatchResult(
                    H_local=H_chosen,
                    pts_base=pts_base_arr,
                    pts_target=pts_tgt_arr,
                    inlier_mask=mask_chosen,
                    inlier_count=inlier_count,
                    inlier_ratio=inlier_ratio,
                    is_affine_fallback=is_aff,
                )

                # Early exit if we found a very strong match
                if inlier_count >= 35 and inlier_ratio >= 0.12:
                    return best_result

        if best_result is not None and best_result.inlier_count >= 12:
            return best_result

        return best_result or MatchResult(
            H_local=None,
            pts_base=np.empty((0, 2), dtype=np.float32),
            pts_target=np.empty((0, 2), dtype=np.float32),
            inlier_mask=None,
            inlier_count=0,
            inlier_ratio=0.0,
        )
