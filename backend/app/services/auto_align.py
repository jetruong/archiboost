"""
Auto Alignment - Architecture-optimized automatic alignment pipeline.

This pipeline prioritizes geometry-first, constrained, explainable alignment
optimized for architectural line drawings.

Pipeline overview:
1. PREPROCESSING: Build linework representations (edges + text suppression)
2. ROTATION CANDIDATES: Evaluate fixed rotations {0°, 90°, 180°, 270°}
3. TRANSLATION: Use phase correlation for each rotation candidate
4. SCORING: Score candidates by correlation response + overlap
5. REFINEMENT: Optionally refine with AKAZE/SIFT/ORB if confidence is low
6. GUARDRAILS: Reject alignments that violate bounds

COORDINATE FRAME NOTES:
- All operations happen in A's coordinate frame (output size = A's size)
- Rotation is about B's center, then translated to align with A
- The composed affine matrix M maps points from B to A: p_A = M @ [p_B; 1]
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from app.config import settings
from app.services.align import SimilarityTransform, Point2D, RotationConstraint
from app.services.preprocess import preprocess_service, PreprocessResult

logger = logging.getLogger(__name__)


class RefinementMethod(str, Enum):
    """Feature detector used for refinement."""
    NONE = "NONE"      # No refinement (phase correlation only)
    AKAZE = "AKAZE"    # AKAZE features
    SIFT = "SIFT"      # SIFT features
    ORB = "ORB"        # ORB features (fallback)


@dataclass
class RotationCandidate:
    """A candidate rotation with its translation estimate.
    
    The affine_matrix is the authoritative transform (2x3 matrix).
    rotation_deg/rad, tx, ty are derived for convenience but the matrix
    is what actually gets used for warping.
    """
    rotation_deg: float
    rotation_rad: float
    tx: float               # Translation in A's coordinate frame
    ty: float               # Translation in A's coordinate frame
    phase_response: float   # Phase correlation response (0-1)
    overlap_ratio: float    # Bbox overlap after warping (0-1)
    score: float            # Combined score
    affine_matrix: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float64))
    
    @property
    def transform(self) -> SimilarityTransform:
        """Get the transform for this candidate (scale=1.0).
        
        Note: This reconstructs a SimilarityTransform from the affine matrix
        for compatibility, but the affine_matrix is the source of truth.
        """
        return SimilarityTransform(
            scale=1.0,
            rotation_rad=self.rotation_rad,
            tx=self.tx,
            ty=self.ty,
        )


@dataclass
class AutoAlignDebug:
    """Debug metadata for auto alignment."""
    # Pipeline info
    rotation_candidates_evaluated: List[float] = field(default_factory=list)
    rotation_candidate_scores: Dict[float, float] = field(default_factory=dict)
    rotation_candidate_used: float = 0.0
    fine_rotation_applied: bool = False  # Whether fine rotation refinement was used
    
    # Phase correlation
    phase_response: float = 0.0
    phase_translation: Tuple[float, float] = (0.0, 0.0)
    
    # Overlap
    overlap_ratio: float = 0.0
    bbox_a: Optional[Tuple[int, int, int, int]] = None
    bbox_b_warped: Optional[Tuple[int, int, int, int]] = None
    
    # Refinement
    refinement_method_used: str = "NONE"
    refinement_matches: int = 0
    refinement_inliers: int = 0
    
    # Scale estimation
    scale_source: str = "default"  # "default", "feature", "manual"
    
    # Final metrics
    final_confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Rejection (if failed)
    rejection_reason: Optional[str] = None
    guardrail_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Helper to convert numpy types to Python natives
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val
        
        def to_native_tuple(t):
            if t is None:
                return None
            return [to_native(x) for x in t]
        
        return {
            "rotation_candidates_evaluated": self.rotation_candidates_evaluated,
            "rotation_candidate_scores": {k: to_native(v) for k, v in self.rotation_candidate_scores.items()},
            "rotation_candidate_used": to_native(self.rotation_candidate_used),
            "fine_rotation_applied": self.fine_rotation_applied,
            "phase_response": round(to_native(self.phase_response), 4),
            "phase_translation": [round(to_native(t), 2) for t in self.phase_translation],
            "overlap_ratio": round(to_native(self.overlap_ratio), 4),
            "bbox_a": to_native_tuple(self.bbox_a),
            "bbox_b_warped": to_native_tuple(self.bbox_b_warped),
            "refinement_method_used": self.refinement_method_used,
            "refinement_matches": to_native(self.refinement_matches),
            "refinement_inliers": to_native(self.refinement_inliers),
            "scale_source": self.scale_source,
            "final_confidence": round(to_native(self.final_confidence), 4),
            "confidence_breakdown": {k: round(to_native(v), 4) for k, v in self.confidence_breakdown.items()},
            "rejection_reason": self.rejection_reason,
            "guardrail_violations": self.guardrail_violations,
        }


@dataclass
class AutoAlignResult:
    """Result of auto alignment v2."""
    transform: SimilarityTransform
    confidence: float
    residual_error: float
    debug: AutoAlignDebug
    matched_points_a: List[Point2D]  # Representative points used
    matched_points_b: List[Point2D]
    affine_matrix: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float64))


class AutoAlignError(Exception):
    """Error during auto alignment v2."""
    def __init__(self, code: str, message: str, details: dict = None, debug: AutoAlignDebug = None):
        self.code = code
        self.message = message
        self.details = details or {}
        self.debug = debug or AutoAlignDebug()
        super().__init__(message)


class AutoAlignService:
    """
    Auto alignment service v2 - optimized for architectural drawings.
    
    COORDINATE FRAME CONVENTION:
    - All transformations produce a matrix that maps B -> A
    - Rotation is computed about B's center, then composed with translation
    - The output frame is always A's dimensions (h_a, w_a)
    - Phase correlation: phaseCorrelate(B, A) returns shift to align B to A
    """
    
    def __init__(self):
        self.config = settings
    
    # ============================================================
    # ROTATION + TRANSLATION ESTIMATION (PRIMARY)
    # ============================================================
    
    def _get_rotation_candidates(
        self,
        rotation_constraint: RotationConstraint,
    ) -> List[float]:
        """
        Get coarse rotation candidates based on constraint.
        
        Returns list of rotation angles in degrees to evaluate.
        """
        if rotation_constraint == RotationConstraint.NONE:
            # Only evaluate 0°
            return [0.0]
        elif rotation_constraint == RotationConstraint.SNAP_90:
            # Evaluate all 90° candidates
            return [0.0, 90.0, 180.0, 270.0]
        else:  # FREE
            # Start with 90° candidates, may refine later
            return [0.0, 90.0, 180.0, 270.0]
    
    def _get_fine_rotation_candidates(
        self,
        base_angle_deg: float,
    ) -> List[float]:
        """
        Get fine rotation candidates around a base angle.
        
        For example, if base_angle_deg=90 and range=15, step=3:
        Returns [75, 78, 81, 84, 87, 93, 96, 99, 102, 105]
        (excludes base_angle since it was already evaluated)
        """
        candidates = []
        range_deg = self.config.fine_rotation_range_deg
        step_deg = self.config.fine_rotation_step_deg
        
        # Generate angles from (base - range) to (base + range) in steps
        angle = base_angle_deg - range_deg
        while angle <= base_angle_deg + range_deg:
            # Skip the base angle (already evaluated in coarse search)
            if abs(angle - base_angle_deg) > 0.1:
                # Normalize to 0-360 range
                normalized = angle % 360
                if normalized not in candidates:
                    candidates.append(normalized)
            angle += step_deg
        
        return sorted(candidates)
    
    def _build_rotation_matrix(
        self,
        angle_deg: float,
        center_x: float,
        center_y: float,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Build a 2x3 rotation matrix about a given center.
        
        This is equivalent to:
        1. Translate so center is at origin
        2. Rotate by angle
        3. Scale by scale factor
        4. Translate back
        
        Args:
            angle_deg: Rotation angle in degrees (positive = counter-clockwise)
            center_x, center_y: Center of rotation
            scale: Scale factor (default 1.0)
        
        Returns:
            2x3 affine matrix
        """
        return cv2.getRotationMatrix2D((center_x, center_y), angle_deg, scale)
    
    def _warp_to_frame(
        self,
        image: np.ndarray,
        matrix: np.ndarray,
        output_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Warp an image using an affine matrix to a fixed output size.
        
        This is the ONLY warp function used throughout the pipeline to ensure
        coordinate frame consistency.
        
        Args:
            image: Source image
            matrix: 2x3 affine transformation matrix
            output_size: (width, height) of output
            interpolation: cv2 interpolation flag
        
        Returns:
            Warped image of size output_size
        """
        return cv2.warpAffine(
            image,
            matrix.astype(np.float32),
            output_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    
    def _apply_window(self, image: np.ndarray, window_type: str) -> np.ndarray:
        """Apply window function to image for phase correlation."""
        if window_type == "none":
            return image.astype(np.float32)
        
        h, w = image.shape[:2]
        
        if window_type == "hann":
            win_y = np.hanning(h)
            win_x = np.hanning(w)
        elif window_type == "hamming":
            win_y = np.hamming(h)
            win_x = np.hamming(w)
        else:
            return image.astype(np.float32)
        
        window = np.outer(win_y, win_x).astype(np.float32)
        return image.astype(np.float32) * window
    
    def _estimate_translation_phase_correlation(
        self,
        image_a: np.ndarray,
        image_b_warped: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimate translation using phase correlation.
        
        IMPORTANT: Both images must be the same size (warped to A's frame).
        
        cv2.phaseCorrelate(src1, src2) returns the shift (dx, dy) such that
        shifting src1 by (dx, dy) aligns it with src2.
        
        So phaseCorrelate(B, A) returns shift to move B toward A.
        
        Args:
            image_a: Reference image (linework A)
            image_b_warped: Image B already warped to A's frame (with rotation only)
        
        Returns:
            Tuple of (dx, dy, response) where (dx, dy) is the shift to add
            to make B align with A.
        """
        assert image_a.shape == image_b_warped.shape, \
            f"Images must be same size: A={image_a.shape}, B={image_b_warped.shape}"
        
        # Convert to float32
        a_f32 = image_a.astype(np.float32)
        b_f32 = image_b_warped.astype(np.float32)
        
        # Optional: light blur to stabilize correlation peak
        blur_ksize = 3
        a_f32 = cv2.GaussianBlur(a_f32, (blur_ksize, blur_ksize), 0)
        b_f32 = cv2.GaussianBlur(b_f32, (blur_ksize, blur_ksize), 0)
        
        # Apply window function
        windowed_a = self._apply_window(a_f32, self.config.phase_correlation_window)
        windowed_b = self._apply_window(b_f32, self.config.phase_correlation_window)
        
        # Phase correlation: phaseCorrelate(src1, src2) returns shift to align src1 to src2
        # We want shift to align B to A, so: phaseCorrelate(B, A)
        (dx, dy), response = cv2.phaseCorrelate(windowed_b, windowed_a)
        
        logger.debug(f"Phase correlation: dx={dx:.2f}, dy={dy:.2f}, response={response:.4f}")
        
        return dx, dy, response
    
    def _compose_rotation_translation_matrix(
        self,
        rotation_matrix: np.ndarray,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        """
        Compose a rotation matrix with a translation.
        
        The final matrix M performs: rotate first, then translate.
        M = T @ R (but stored as 2x3 affine)
        
        For a 2x3 matrix [a b tx; c d ty], adding translation means:
        M[0,2] += dx
        M[1,2] += dy
        
        Args:
            rotation_matrix: 2x3 rotation matrix
            dx, dy: Translation to add
        
        Returns:
            2x3 composed affine matrix
        """
        M = rotation_matrix.copy()
        M[0, 2] += dx
        M[1, 2] += dy
        return M
    
    def _compute_overlap_from_warped(
        self,
        linework_a: np.ndarray,
        linework_b_warped: np.ndarray,
    ) -> Tuple[float, Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Compute overlap ratio between A and already-warped B.
        
        Both images must be the same size (in A's coordinate frame).
        
        Returns:
            Tuple of (overlap_ratio, bbox_a, bbox_b_warped)
        """
        # Get bbox of A
        bbox_a = preprocess_service.compute_linework_bbox(linework_a)
        if bbox_a is None:
            return 0.0, (0, 0, 0, 0), None
        
        # Get bbox of warped B
        bbox_b = preprocess_service.compute_linework_bbox(linework_b_warped)
        if bbox_b is None:
            return 0.0, bbox_a, None
        
        # Compute intersection
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2])
        y2 = min(bbox_a[1] + bbox_a[3], bbox_b[1] + bbox_b[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, bbox_a, bbox_b
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Compute union
        area_a = bbox_a[2] * bbox_a[3]
        area_b = bbox_b[2] * bbox_b[3]
        union_area = area_a + area_b - intersection_area
        
        # IoU-like overlap ratio
        overlap_ratio = intersection_area / union_area if union_area > 0 else 0.0
        
        return overlap_ratio, bbox_a, bbox_b
    
    def _evaluate_rotation_candidates(
        self,
        linework_a: np.ndarray,
        linework_b: np.ndarray,
        candidates_deg: List[float],
        debug: AutoAlignDebug,
    ) -> List[RotationCandidate]:
        """
        Evaluate all rotation candidates and return scored list.
        
        For each candidate angle:
        1. Build rotation matrix about B's center
        2. Warp B into A's frame using the rotation matrix
        3. Estimate translation via phase correlation
        4. Compose final transform: rotation + translation
        5. Warp B with full transform to compute overlap
        6. Score = weighted(phase_response, overlap)
        
        All operations happen in A's coordinate frame.
        """
        results = []
        debug.rotation_candidates_evaluated = candidates_deg
        
        h_a, w_a = linework_a.shape[:2]
        h_b, w_b = linework_b.shape[:2]
        output_size = (w_a, h_a)  # (width, height) for cv2
        
        # B's center (rotation pivot)
        center_b_x = w_b / 2.0
        center_b_y = h_b / 2.0
        
        for angle_deg in candidates_deg:
            angle_rad = math.radians(angle_deg)
            
            # Step 1: Build rotation matrix about B's center
            # This rotates B about its center
            M_rot = self._build_rotation_matrix(angle_deg, center_b_x, center_b_y)
            
            # Step 2: Warp B linework to A's frame using rotation only
            b_rotated = self._warp_to_frame(linework_b, M_rot, output_size)
            
            # Step 3: Estimate translation via phase correlation
            # This returns (dx, dy) to shift rotated B to align with A
            dx, dy, phase_response = self._estimate_translation_phase_correlation(
                linework_a, b_rotated
            )
            
            # Step 4: Compose final transform: rotation + translation
            M_full = self._compose_rotation_translation_matrix(M_rot, dx, dy)
            
            # Step 5: Warp B with full transform for overlap scoring
            b_fully_warped = self._warp_to_frame(linework_b, M_full, output_size)
            
            # Step 6: Compute overlap using the fully warped image
            overlap_ratio, bbox_a, bbox_b = self._compute_overlap_from_warped(
                linework_a, b_fully_warped
            )
            
            # Compute score
            score = (
                self.config.confidence_weight_phase * phase_response +
                self.config.confidence_weight_overlap * overlap_ratio
            )
            
            # Extract tx, ty from the composed matrix for reporting
            # M_full = [[cos, -sin, tx], [sin, cos, ty]]
            final_tx = M_full[0, 2]
            final_ty = M_full[1, 2]
            
            candidate = RotationCandidate(
                rotation_deg=angle_deg,
                rotation_rad=angle_rad,
                tx=final_tx,
                ty=final_ty,
                phase_response=phase_response,
                overlap_ratio=overlap_ratio,
                score=score,
                affine_matrix=M_full,
            )
            
            debug.rotation_candidate_scores[angle_deg] = score
            
            logger.debug(
                f"Rotation {angle_deg}°: phase={phase_response:.4f}, "
                f"overlap={overlap_ratio:.4f}, score={score:.4f}, "
                f"tx={final_tx:.2f}, ty={final_ty:.2f}"
            )
            
            results.append(candidate)
        
        # Sort by score (descending)
        results.sort(key=lambda c: c.score, reverse=True)
        
        return results
    
    # ============================================================
    # FEATURE-BASED REFINEMENT (SECONDARY)
    # ============================================================
    
    def _create_feature_detector(self, method: str) -> Optional[cv2.Feature2D]:
        """Create a feature detector by name."""
        if method == "AKAZE":
            return cv2.AKAZE_create()
        elif method == "SIFT":
            return cv2.SIFT_create(nfeatures=self.config.feature_max_features)
        elif method == "ORB":
            return cv2.ORB_create(nfeatures=self.config.feature_max_features)
        else:
            return None
    
    def _get_norm_type(self, method: str) -> int:
        """Get the appropriate norm type for matching."""
        if method in ["AKAZE", "ORB"]:
            return cv2.NORM_HAMMING
        else:  # SIFT and others use L2
            return cv2.NORM_L2
    
    def _refine_with_features(
        self,
        linework_a: np.ndarray,
        linework_b: np.ndarray,
        initial_matrix: np.ndarray,
        rotation_constraint: RotationConstraint,
        debug: AutoAlignDebug,
    ) -> Optional[Tuple[np.ndarray, float, List[Point2D], List[Point2D]]]:
        """
        Refine alignment using feature detection and matching.
        
        Tries detectors in order: AKAZE → SIFT → ORB
        
        Returns:
            Tuple of (refined_matrix, confidence, points_a, points_b) or None
        """
        h_a, w_a = linework_a.shape[:2]
        output_size = (w_a, h_a)
        
        # Convert linework to uint8 if needed
        img_a = linework_a.astype(np.uint8) if linework_a.dtype != np.uint8 else linework_a
        img_b = linework_b.astype(np.uint8) if linework_b.dtype != np.uint8 else linework_b
        
        for method in self.config.feature_detectors:
            logger.debug(f"Trying feature refinement with {method}")
            
            detector = self._create_feature_detector(method)
            if detector is None:
                continue
            
            try:
                # Detect keypoints and compute descriptors
                kp_a, desc_a = detector.detectAndCompute(img_a, None)
                kp_b, desc_b = detector.detectAndCompute(img_b, None)
                
                if desc_a is None or desc_b is None:
                    logger.debug(f"{method}: No descriptors found")
                    continue
                
                if len(kp_a) < self.config.feature_min_matches or len(kp_b) < self.config.feature_min_matches:
                    logger.debug(f"{method}: Insufficient keypoints (A={len(kp_a)}, B={len(kp_b)})")
                    continue
                
                # Match features
                norm_type = self._get_norm_type(method)
                bf = cv2.BFMatcher(norm_type, crossCheck=False)
                
                matches = bf.knnMatch(desc_a, desc_b, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) < 2:
                        continue
                    m, n = match_pair
                    if m.distance < self.config.feature_match_ratio * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < self.config.feature_min_matches:
                    logger.debug(f"{method}: Insufficient matches ({len(good_matches)})")
                    continue
                
                debug.refinement_matches = len(good_matches)
                
                # Get matched points
                pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Estimate similarity transform with RANSAC
                matrix, inliers = cv2.estimateAffinePartial2D(
                    pts_b, pts_a,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.config.ransac_reproj_threshold,
                    maxIters=self.config.ransac_max_iters,
                    confidence=self.config.ransac_confidence,
                )
                
                if matrix is None:
                    logger.debug(f"{method}: RANSAC failed")
                    continue
                
                num_inliers = int(np.sum(inliers)) if inliers is not None else 0
                debug.refinement_inliers = num_inliers
                debug.refinement_method_used = method
                
                if num_inliers < self.config.feature_min_matches:
                    logger.debug(f"{method}: Insufficient inliers ({num_inliers})")
                    continue
                
                # Extract transform parameters from matrix
                a, b, tx = matrix[0]
                c, d, ty = matrix[1]
                
                scale = math.sqrt(a * a + c * c)
                rotation_rad = math.atan2(c, a)
                
                # Apply rotation constraint
                if rotation_constraint == RotationConstraint.NONE:
                    rotation_rad = 0.0
                elif rotation_constraint == RotationConstraint.SNAP_90:
                    rotation_deg = math.degrees(rotation_rad)
                    snapped_deg = round(rotation_deg / 90) * 90
                    
                    # If fine rotation is enabled, allow small deviations from 90° snaps
                    if self.config.fine_rotation_enabled:
                        # Keep actual rotation if within fine rotation range of a 90° snap
                        deviation = abs(rotation_deg - snapped_deg)
                        if deviation <= self.config.fine_rotation_range_deg:
                            # Keep the actual rotation from feature matching
                            logger.debug(
                                f"Fine rotation: keeping {rotation_deg:.2f}° "
                                f"(within {deviation:.2f}° of {snapped_deg}°)"
                            )
                            pass  # rotation_rad stays as-is
                        else:
                            rotation_rad = math.radians(snapped_deg)
                    else:
                        rotation_rad = math.radians(snapped_deg)
                # FREE mode: keep actual rotation from feature matching
                # (no snapping - allows any angle)
                
                # Rebuild matrix with constrained rotation but keep translation
                cos_t = scale * math.cos(rotation_rad)
                sin_t = scale * math.sin(rotation_rad)
                
                # Recompute translation using inlier centroids
                if inliers is not None:
                    inlier_mask = inliers.ravel().astype(bool)
                    pts_a_inlier = pts_a[inlier_mask].reshape(-1, 2)
                    pts_b_inlier = pts_b[inlier_mask].reshape(-1, 2)
                else:
                    pts_a_inlier = pts_a.reshape(-1, 2)
                    pts_b_inlier = pts_b.reshape(-1, 2)
                
                centroid_a = np.mean(pts_a_inlier, axis=0)
                centroid_b = np.mean(pts_b_inlier, axis=0)
                
                # Transform centroid_b with rotation+scale
                centroid_b_transformed = np.array([
                    cos_t * centroid_b[0] - sin_t * centroid_b[1],
                    sin_t * centroid_b[0] + cos_t * centroid_b[1]
                ])
                
                tx = centroid_a[0] - centroid_b_transformed[0]
                ty = centroid_a[1] - centroid_b_transformed[1]
                
                # Build refined matrix
                refined_matrix = np.array([
                    [cos_t, -sin_t, tx],
                    [sin_t, cos_t, ty]
                ], dtype=np.float64)
                
                # Confidence from inlier ratio
                inlier_ratio = num_inliers / len(good_matches)
                confidence = inlier_ratio
                
                debug.scale_source = "feature"
                
                # Get representative points
                points_a = [Point2D(x=float(pts_a_inlier[0, 0]), y=float(pts_a_inlier[0, 1])),
                           Point2D(x=float(pts_a_inlier[1, 0]), y=float(pts_a_inlier[1, 1]))]
                points_b = [Point2D(x=float(pts_b_inlier[0, 0]), y=float(pts_b_inlier[0, 1])),
                           Point2D(x=float(pts_b_inlier[1, 0]), y=float(pts_b_inlier[1, 1]))]
                
                logger.info(
                    f"Feature refinement ({method}): scale={scale:.4f}, "
                    f"rotation={math.degrees(rotation_rad):.2f}°, "
                    f"inliers={num_inliers}/{len(good_matches)}"
                )
                
                return refined_matrix, confidence, points_a, points_b
            
            except Exception as e:
                logger.warning(f"{method} refinement failed: {e}")
                continue
        
        logger.debug("All feature refinement methods failed")
        return None
    
    # ============================================================
    # GUARDRAILS
    # ============================================================
    
    def _extract_transform_params(
        self,
        matrix: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Extract scale, rotation, tx, ty from a 2x3 affine matrix.
        
        For a similarity transform: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        """
        a, b, tx = matrix[0]
        c, d, ty = matrix[1]
        
        scale = math.sqrt(a * a + c * c)
        rotation_rad = math.atan2(c, a)
        
        return scale, rotation_rad, tx, ty
    
    def _check_guardrails(
        self,
        matrix: np.ndarray,
        image_a_shape: Tuple[int, int],
        image_b_shape: Tuple[int, int],
        overlap_ratio: float,
        rotation_constraint: RotationConstraint,
        debug: AutoAlignDebug,
    ) -> List[str]:
        """
        Check alignment guardrails and return list of violations.
        
        Guardrails:
        - Scale within bounds
        - Rotation respects constraint
        - Translation within bounds
        - Sufficient overlap
        """
        violations = []
        
        h_a, w_a = image_a_shape
        h_b, w_b = image_b_shape
        
        scale, rotation_rad, tx, ty = self._extract_transform_params(matrix)
        rotation_deg = math.degrees(rotation_rad)
        
        # Scale bounds
        if scale < self.config.scale_min:
            violations.append(f"Scale {scale:.3f} < min {self.config.scale_min}")
        elif scale > self.config.scale_max:
            violations.append(f"Scale {scale:.3f} > max {self.config.scale_max}")
        
        # Rotation constraint
        if rotation_constraint == RotationConstraint.NONE and abs(rotation_deg) > 1.0:
            violations.append(f"Rotation {rotation_deg:.2f}° but constraint is NONE")
        elif rotation_constraint == RotationConstraint.SNAP_90:
            expected = round(rotation_deg / 90) * 90
            deviation = abs(rotation_deg - expected)
            
            # Allow deviation if fine rotation is enabled and within range
            if self.config.fine_rotation_enabled:
                allowed_deviation = self.config.fine_rotation_range_deg
            else:
                allowed_deviation = 1.0
            
            if deviation > allowed_deviation:
                violations.append(f"Rotation {rotation_deg:.2f}° not snapped to 90°")
        
        # Translation bounds
        max_tx = self.config.translation_max_fraction * max(w_a, w_b)
        max_ty = self.config.translation_max_fraction * max(h_a, h_b)
        
        if abs(tx) > max_tx:
            violations.append(f"Translation X {tx:.1f} > max {max_tx:.1f}")
        if abs(ty) > max_ty:
            violations.append(f"Translation Y {ty:.1f} > max {max_ty:.1f}")
        
        # Overlap
        if overlap_ratio < self.config.min_overlap_ratio:
            violations.append(f"Overlap {overlap_ratio:.2%} < min {self.config.min_overlap_ratio:.2%}")
        
        debug.guardrail_violations = violations
        return violations
    
    # ============================================================
    # CONFIDENCE SCORING
    # ============================================================
    
    def _compute_final_confidence(
        self,
        phase_response: float,
        overlap_ratio: float,
        inlier_ratio: Optional[float],
        scale: float,
        debug: AutoAlignDebug,
    ) -> float:
        """
        Compute final confidence score.
        
        Weighted combination of:
        - Phase correlation response
        - Bounding box overlap
        - Inlier ratio (if refinement ran)
        
        Also penalizes extreme scale values.
        """
        breakdown = {}
        
        # Base components
        if inlier_ratio is not None and inlier_ratio > 0:
            # With refinement
            confidence = (
                self.config.confidence_weight_phase * phase_response +
                self.config.confidence_weight_overlap * overlap_ratio +
                self.config.confidence_weight_inlier * inlier_ratio
            )
            breakdown["phase"] = phase_response
            breakdown["overlap"] = overlap_ratio
            breakdown["inlier"] = inlier_ratio
        else:
            # Without refinement - redistribute inlier weight
            total_weight = self.config.confidence_weight_phase + self.config.confidence_weight_overlap
            confidence = (
                (self.config.confidence_weight_phase / total_weight) * phase_response +
                (self.config.confidence_weight_overlap / total_weight) * overlap_ratio
            )
            breakdown["phase"] = phase_response
            breakdown["overlap"] = overlap_ratio
        
        # Scale penalty
        if scale < self.config.scale_warn_min or scale > self.config.scale_warn_max:
            scale_penalty = 0.9
            confidence *= scale_penalty
            breakdown["scale_penalty"] = scale_penalty
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        debug.confidence_breakdown = breakdown
        debug.final_confidence = confidence
        
        return confidence
    
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    
    def auto_align(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        rotation_constraint: RotationConstraint = RotationConstraint.NONE,
    ) -> AutoAlignResult:
        """
        Automatically align two architectural drawings.
        
        Pipeline:
        1. Preprocess both images to extract linework
        2. Evaluate rotation candidates with phase correlation
        3. Select best rotation + translation
        4. Optionally refine with features if confidence is low
        5. Apply guardrails and compute final confidence
        6. Return result or raise error if alignment fails
        
        Args:
            image_a: Reference image (BGR)
            image_b: Image to align to A (BGR)
            rotation_constraint: How to handle rotation
        
        Returns:
            AutoAlignResult with transform and metadata
        
        Raises:
            AutoAlignError: If alignment fails guardrails or confidence check
        """
        debug = AutoAlignDebug()
        
        logger.info(
            f"Auto align v2 starting: A={image_a.shape[:2]}, B={image_b.shape[:2]}, "
            f"constraint={rotation_constraint.value}"
        )
        
        # Step 1: Preprocess images
        logger.debug("Step 1: Preprocessing images")
        prep_a = preprocess_service.build_linework(image_a)
        prep_b = preprocess_service.build_linework(image_b)
        
        if prep_a.edge_count < 100:
            debug.rejection_reason = "Insufficient linework in image A"
            raise AutoAlignError(
                code="INSUFFICIENT_CONTENT",
                message="Image A has insufficient linework for alignment",
                details={"edge_count_a": prep_a.edge_count, "minimum": 100},
                debug=debug,
            )
        
        if prep_b.edge_count < 100:
            debug.rejection_reason = "Insufficient linework in image B"
            raise AutoAlignError(
                code="INSUFFICIENT_CONTENT",
                message="Image B has insufficient linework for alignment",
                details={"edge_count_b": prep_b.edge_count, "minimum": 100},
                debug=debug,
            )
        
        # Step 2: Evaluate rotation candidates (coarse: 0°, 90°, 180°, 270°)
        logger.debug("Step 2: Evaluating coarse rotation candidates")
        rotation_candidates_deg = self._get_rotation_candidates(rotation_constraint)
        
        candidates = self._evaluate_rotation_candidates(
            prep_a.linework, prep_b.linework, rotation_candidates_deg, debug
        )
        
        if not candidates:
            debug.rejection_reason = "No valid rotation candidates"
            raise AutoAlignError(
                code="NO_VALID_CANDIDATES",
                message="Could not find any valid rotation candidates",
                debug=debug,
            )
        
        # Step 2b: Fine rotation refinement (if enabled and confidence is low)
        best_coarse_candidate = candidates[0]
        
        if (
            self.config.fine_rotation_enabled
            and rotation_constraint != RotationConstraint.NONE
            and best_coarse_candidate.score < self.config.fine_rotation_threshold
        ):
            logger.debug(
                f"Step 2b: Fine rotation refinement around {best_coarse_candidate.rotation_deg}° "
                f"(coarse score={best_coarse_candidate.score:.4f} < threshold={self.config.fine_rotation_threshold})"
            )
            
            # Get fine rotation candidates around best coarse angle
            fine_candidates_deg = self._get_fine_rotation_candidates(best_coarse_candidate.rotation_deg)
            
            if fine_candidates_deg:
                logger.debug(f"Evaluating {len(fine_candidates_deg)} fine angles: {fine_candidates_deg}")
                
                fine_candidates = self._evaluate_rotation_candidates(
                    prep_a.linework, prep_b.linework, fine_candidates_deg, debug
                )
                
                # Mark that fine rotation was applied
                debug.fine_rotation_applied = True
                
                # Merge with coarse candidates and re-sort
                all_candidates = candidates + fine_candidates
                all_candidates.sort(key=lambda c: c.score, reverse=True)
                candidates = all_candidates
                
                if candidates[0].rotation_deg != best_coarse_candidate.rotation_deg:
                    logger.info(
                        f"Fine rotation improved: {best_coarse_candidate.rotation_deg}° "
                        f"(score={best_coarse_candidate.score:.4f}) -> "
                        f"{candidates[0].rotation_deg}° (score={candidates[0].score:.4f})"
                    )
        
        # Step 3: Select best candidate
        best_candidate = candidates[0]
        current_matrix = best_candidate.affine_matrix.copy()
        
        debug.rotation_candidate_used = best_candidate.rotation_deg
        debug.phase_response = best_candidate.phase_response
        debug.phase_translation = (best_candidate.tx, best_candidate.ty)
        debug.overlap_ratio = best_candidate.overlap_ratio
        
        logger.info(
            f"Best candidate: {best_candidate.rotation_deg}° with score={best_candidate.score:.4f}, "
            f"tx={best_candidate.tx:.2f}, ty={best_candidate.ty:.2f}"
        )
        
        # Extract current scale (should be 1.0 from phase correlation)
        scale, rotation_rad, tx, ty = self._extract_transform_params(current_matrix)
        inlier_ratio = None
        
        # Representative points (use corners transformed to estimate)
        h_a, w_a = prep_a.original_shape
        h_b, w_b = prep_b.original_shape
        
        # Default representative points (corners)
        points_a = [Point2D(x=w_a * 0.25, y=h_a * 0.25), Point2D(x=w_a * 0.75, y=h_a * 0.75)]
        points_b = [Point2D(x=w_b * 0.25, y=h_b * 0.25), Point2D(x=w_b * 0.75, y=h_b * 0.75)]
        
        # Step 4: Check if refinement is needed
        initial_confidence = best_candidate.score
        
        if initial_confidence < self.config.feature_refinement_threshold:
            logger.debug("Step 4: Attempting feature refinement")
            
            refinement_result = self._refine_with_features(
                prep_a.linework, prep_b.linework,
                current_matrix, rotation_constraint, debug
            )
            
            if refinement_result is not None:
                refined_matrix, refined_confidence, ref_points_a, ref_points_b = refinement_result
                
                # Warp B with refined matrix and compute overlap
                output_size = (w_a, h_a)
                b_refined_warped = self._warp_to_frame(
                    prep_b.linework, refined_matrix, output_size
                )
                refined_overlap, _, _ = self._compute_overlap_from_warped(
                    prep_a.linework, b_refined_warped
                )
                
                # Accept refinement if it improves things
                if refined_overlap > best_candidate.overlap_ratio:
                    current_matrix = refined_matrix
                    scale, rotation_rad, tx, ty = self._extract_transform_params(current_matrix)
                    points_a = ref_points_a
                    points_b = ref_points_b
                    inlier_ratio = debug.refinement_inliers / debug.refinement_matches if debug.refinement_matches > 0 else None
                    debug.overlap_ratio = refined_overlap
                    
                    logger.info(
                        f"Refinement accepted: scale={scale:.4f}, "
                        f"rotation={math.degrees(rotation_rad):.2f}°, "
                        f"overlap improved {best_candidate.overlap_ratio:.2%} -> {refined_overlap:.2%}"
                    )
                else:
                    logger.debug("Refinement rejected - did not improve overlap")
                    debug.refinement_method_used = "NONE"
        else:
            logger.debug("Step 4: Skipping refinement (high confidence)")
            debug.refinement_method_used = "NONE"
        
        # Step 5: Compute final overlap with current matrix
        output_size = (w_a, h_a)
        b_final_warped = self._warp_to_frame(prep_b.linework, current_matrix, output_size)
        overlap_ratio, bbox_a, bbox_b = self._compute_overlap_from_warped(
            prep_a.linework, b_final_warped
        )
        debug.overlap_ratio = overlap_ratio
        debug.bbox_a = bbox_a
        debug.bbox_b_warped = bbox_b
        
        # Step 6: Check guardrails
        logger.debug("Step 5: Checking guardrails")
        violations = self._check_guardrails(
            current_matrix,
            prep_a.original_shape,
            prep_b.original_shape,
            overlap_ratio,
            rotation_constraint,
            debug,
        )
        
        if violations:
            debug.rejection_reason = f"Guardrail violations: {', '.join(violations)}"
            raise AutoAlignError(
                code="GUARDRAIL_VIOLATION",
                message=f"Alignment violates guardrails: {'; '.join(violations)}",
                details={"violations": violations},
                debug=debug,
            )
        
        # Step 7: Compute final confidence
        logger.debug("Step 6: Computing final confidence")
        final_confidence = self._compute_final_confidence(
            debug.phase_response, overlap_ratio, inlier_ratio, scale, debug
        )
        
        # Step 8: Final check
        if final_confidence < self.config.auto_align_min_confidence:
            debug.rejection_reason = f"Confidence {final_confidence:.2%} < threshold {self.config.auto_align_min_confidence:.2%}"
            raise AutoAlignError(
                code="AUTO_ALIGNMENT_FAILED",
                message=(
                    f"Alignment confidence too low ({final_confidence:.1%}). "
                    "The drawings may be too different for automatic alignment. "
                    "Consider using manual anchor points."
                ),
                details={
                    "confidence": final_confidence,
                    "threshold": self.config.auto_align_min_confidence,
                    "phase_response": debug.phase_response,
                    "overlap_ratio": overlap_ratio,
                },
                debug=debug,
            )
        
        # Build SimilarityTransform for compatibility
        scale, rotation_rad, tx, ty = self._extract_transform_params(current_matrix)
        final_transform = SimilarityTransform(
            scale=scale,
            rotation_rad=rotation_rad,
            tx=tx,
            ty=ty,
        )
        
        # Compute residual error using representative points
        # Transform points from B to A using the matrix
        def transform_point(p: Point2D, M: np.ndarray) -> Point2D:
            x_new = M[0, 0] * p.x + M[0, 1] * p.y + M[0, 2]
            y_new = M[1, 0] * p.x + M[1, 1] * p.y + M[1, 2]
            return Point2D(x=x_new, y=y_new)
        
        mapped_p0 = transform_point(points_b[0], current_matrix)
        mapped_p1 = transform_point(points_b[1], current_matrix)
        
        error_0 = math.sqrt((mapped_p0.x - points_a[0].x) ** 2 + (mapped_p0.y - points_a[0].y) ** 2)
        error_1 = math.sqrt((mapped_p1.x - points_a[1].x) ** 2 + (mapped_p1.y - points_a[1].y) ** 2)
        residual_error = math.sqrt((error_0 ** 2 + error_1 ** 2) / 2)
        
        logger.info(
            f"Auto align v2 complete: scale={scale:.4f}, "
            f"rotation={math.degrees(rotation_rad):.2f}°, "
            f"translation=({tx:.2f}, {ty:.2f}), "
            f"confidence={final_confidence:.2%}"
        )
        
        return AutoAlignResult(
            transform=final_transform,
            confidence=final_confidence,
            residual_error=residual_error,
            debug=debug,
            matched_points_a=points_a,
            matched_points_b=points_b,
            affine_matrix=current_matrix,
        )


# Global service instance
auto_align_service = AutoAlignService()
