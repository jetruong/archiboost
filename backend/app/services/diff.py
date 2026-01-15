"""
Difference detection service for comparing architectural drawings.

Computes XOR-based difference masks between aligned linework and extracts
labeled regions for annotation and summarization.

Key features:
- Position tolerance: Absorbs small translation/alignment errors by dilating
  linework masks before comparison. This prevents false positives from
  slightly shifted but otherwise identical content.
- Morphological cleanup: Removes noise and closes small gaps.
- Region filtering: Ignores small regions below a configurable threshold.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DiffRegion:
    """A detected difference region."""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area: int
    centroid: Tuple[float, float]  # x, y
    

@dataclass
class DiffResult:
    """Result of difference detection."""
    diff_mask: np.ndarray          # Binary difference mask
    annotated_overlay: np.ndarray  # Overlay image with labeled regions
    regions: List[DiffRegion]      # List of detected difference regions
    stats: dict                    # Processing statistics


def compute_contrasting_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Compute a contrasting color that stands out from two given BGR colors.
    
    Uses a simple algorithm: finds a color from candidates that maximizes
    distance from both input colors.
    
    Args:
        color1: First BGR color tuple
        color2: Second BGR color tuple
    
    Returns:
        BGR color tuple that contrasts with both inputs
    """
    import math
    
    # Candidate colors (BGR format)
    candidates = [
        (0, 255, 255),    # Yellow
        (0, 255, 0),      # Green
        (0, 165, 255),    # Orange
        (255, 0, 255),    # Magenta
        (128, 0, 128),    # Purple
        (0, 128, 0),      # Dark Green
        (255, 255, 255),  # White
        (0, 0, 0),        # Black
    ]
    
    def color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    best_candidate = candidates[0]
    best_score = 0
    
    for candidate in candidates:
        dist1 = color_distance(candidate, color1)
        dist2 = color_distance(candidate, color2)
        # Score is the minimum distance to either color (we want this high)
        score = min(dist1, dist2)
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    return best_candidate


class DiffDetectionService:
    """Service for detecting and labeling differences between drawings."""
    
    def __init__(
        self,
        min_region_area: int = 100,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        position_tolerance: int = 5,
    ):
        """
        Initialize the difference detection service.
        
        Args:
            min_region_area: Minimum pixel area to consider a difference region
            morph_kernel_size: Kernel size for morphological cleanup
            morph_iterations: Number of morphological operations
            position_tolerance: Pixel tolerance for positional differences.
                               Both linework masks are dilated by this amount before
                               comparison, absorbing small translation/alignment errors.
                               Set to 0 for pixel-perfect comparison.
        """
        self.min_region_area = min_region_area
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.position_tolerance = position_tolerance
    
    def compute_diff_mask(
        self,
        linework_a: np.ndarray,
        linework_b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute XOR difference mask between two linework images.
        
        Uses a tolerance-based approach: both linework masks are dilated by
        `position_tolerance` pixels before comparison. This absorbs small
        translation or alignment errors, preventing false positives from
        slightly shifted but otherwise identical content.
        
        Args:
            linework_a: Binary linework mask from drawing A
            linework_b: Binary linework mask from drawing B (should be aligned)
        
        Returns:
            Binary difference mask where 255 = difference, 0 = same
        """
        # Ensure both are same size
        h_a, w_a = linework_a.shape[:2]
        h_b, w_b = linework_b.shape[:2]
        
        # Use larger dimensions
        h = max(h_a, h_b)
        w = max(w_a, w_b)
        
        # Create padded versions
        padded_a = np.zeros((h, w), dtype=np.uint8)
        padded_b = np.zeros((h, w), dtype=np.uint8)
        
        padded_a[:h_a, :w_a] = linework_a
        padded_b[:h_b, :w_b] = linework_b
        
        # Apply position tolerance by dilating both masks
        # This makes the comparison tolerant of small positional shifts
        if self.position_tolerance > 0:
            tolerance_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.position_tolerance * 2 + 1, self.position_tolerance * 2 + 1)
            )
            dilated_a = cv2.dilate(padded_a, tolerance_kernel, iterations=1)
            dilated_b = cv2.dilate(padded_b, tolerance_kernel, iterations=1)
            
            # A difference is detected when:
            # - Content exists in A but not in the tolerance zone of B, OR
            # - Content exists in B but not in the tolerance zone of A
            # This means: (A AND NOT dilated_B) OR (B AND NOT dilated_A)
            diff_a_only = cv2.bitwise_and(padded_a, cv2.bitwise_not(dilated_b))
            diff_b_only = cv2.bitwise_and(padded_b, cv2.bitwise_not(dilated_a))
            diff_mask = cv2.bitwise_or(diff_a_only, diff_b_only)
            
            logger.debug(
                f"Position tolerance applied: {self.position_tolerance}px. "
                f"A-only: {np.sum(diff_a_only > 0)} px, B-only: {np.sum(diff_b_only > 0)} px"
            )
        else:
            # No tolerance - use strict XOR
            diff_mask = cv2.bitwise_xor(padded_a, padded_b)
        
        # Apply morphological cleanup to reduce noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Close small gaps
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        
        # Remove small isolated noise
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return diff_mask
    
    def extract_regions(self, diff_mask: np.ndarray) -> List[DiffRegion]:
        """
        Extract labeled regions from the difference mask.
        
        Args:
            diff_mask: Binary difference mask
        
        Returns:
            List of DiffRegion objects
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            diff_mask, connectivity=8
        )
        
        regions = []
        region_id = 1  # Start from 1 for user-friendly labeling
        
        # Process each component (skip background label 0)
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            
            # Skip small regions
            if area < self.min_region_area:
                continue
            
            centroid = (float(centroids[label, 0]), float(centroids[label, 1]))
            
            regions.append(DiffRegion(
                id=region_id,
                bbox=(x, y, w, h),
                area=area,
                centroid=centroid,
            ))
            region_id += 1
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r.area, reverse=True)
        
        # Re-assign IDs based on sorted order
        for i, region in enumerate(regions):
            region.id = i + 1
        
        logger.info(f"Extracted {len(regions)} difference regions")
        return regions
    
    def colorize_linework(
        self,
        linework: np.ndarray,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Colorize a binary linework mask.
        
        Args:
            linework: Binary mask (255 = line, 0 = background)
            color: BGR color tuple
        
        Returns:
            BGRA image with colored lines on transparent background
        """
        h, w = linework.shape[:2]
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set color where there's linework
        bgra[linework > 0] = [color[0], color[1], color[2], 255]
        
        return bgra
    
    def create_composite_overlay(
        self,
        linework_a: np.ndarray,
        linework_b: np.ndarray,
        color_a: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
        color_b: Tuple[int, int, int] = (255, 255, 0),  # Cyan in BGR
        background_color: Tuple[int, int, int] = (255, 255, 255),  # White
    ) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Create a composite overlay showing both drawings with different colors.
        
        This creates the same visualization as the frontend overlay editor:
        Drawing A in red, Drawing B in cyan, overlaid on white background.
        
        Args:
            linework_a: Binary linework mask from drawing A
            linework_b: Binary linework mask from drawing B (should be aligned)
            color_a: BGR color for drawing A (default red)
            color_b: BGR color for drawing B (default cyan)
            background_color: BGR background color (default white)
        
        Returns:
            Tuple of (BGR composite image, color_a used, color_b used)
        """
        # Ensure both are same size
        h_a, w_a = linework_a.shape[:2]
        h_b, w_b = linework_b.shape[:2]
        h = max(h_a, h_b)
        w = max(w_a, w_b)
        
        # Create padded versions
        padded_a = np.zeros((h, w), dtype=np.uint8)
        padded_b = np.zeros((h, w), dtype=np.uint8)
        padded_a[:h_a, :w_a] = linework_a
        padded_b[:h_b, :w_b] = linework_b
        
        # Create white background
        composite = np.ones((h, w, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
        
        # Draw A first (red), then B on top (cyan)
        # This shows: where only A = red, where only B = cyan, where both = darker mix
        composite[padded_a > 0] = color_a
        composite[padded_b > 0] = color_b
        
        # For overlapping areas, blend the colors to show both
        overlap = (padded_a > 0) & (padded_b > 0)
        if np.any(overlap):
            # Overlapping areas get a mix (appears darker/purple-ish)
            composite[overlap] = (
                (np.array(color_a) * 0.5 + np.array(color_b) * 0.5).astype(np.uint8)
            )
        
        return composite, color_a, color_b
    
    def create_annotated_overlay(
        self,
        base_image: np.ndarray,
        diff_mask: np.ndarray,
        regions: List[DiffRegion],
        color_a: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR (for contrast calculation)
        color_b: Tuple[int, int, int] = (255, 255, 0),  # Cyan in BGR (for contrast calculation)
        highlight_color: Optional[Tuple[int, int, int]] = None,  # Auto-computed if None
        box_color: Optional[Tuple[int, int, int]] = None,  # Auto-computed if None
        label_color: Tuple[int, int, int] = (255, 255, 255),  # White
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Create an annotated overlay image with labeled difference regions.
        
        Args:
            base_image: Base image to overlay on (should be composite of both drawings)
            diff_mask: Binary difference mask
            regions: List of extracted regions
            color_a: BGR color used for drawing A (for computing contrast)
            color_b: BGR color used for drawing B (for computing contrast)
            highlight_color: Color to highlight differences (auto-computed if None)
            box_color: Color for bounding boxes (auto-computed if None)
            label_color: Color for region labels
            alpha: Transparency for difference highlighting
        
        Returns:
            Annotated overlay image (BGR)
        """
        # Compute contrasting colors if not provided
        if box_color is None or highlight_color is None:
            contrasting = compute_contrasting_color(color_a, color_b)
            if box_color is None:
                box_color = contrasting
            if highlight_color is None:
                highlight_color = contrasting
            logger.info(f"Computed contrasting annotation color: BGR{contrasting}")
        # Ensure base image is BGR
        if len(base_image.shape) == 2:
            overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_image.copy()
        
        # Handle size mismatch between base image and diff mask
        # Pad the overlay to match diff_mask dimensions if needed
        h_overlay, w_overlay = overlay.shape[:2]
        h_mask, w_mask = diff_mask.shape[:2]
        
        if h_overlay != h_mask or w_overlay != w_mask:
            # Create a canvas the size of the diff_mask
            h_target = max(h_overlay, h_mask)
            w_target = max(w_overlay, w_mask)
            
            padded_overlay = np.ones((h_target, w_target, 3), dtype=np.uint8) * 255  # White background
            padded_overlay[:h_overlay, :w_overlay] = overlay
            overlay = padded_overlay
            
            # Also pad the diff_mask if needed
            if h_mask != h_target or w_mask != w_target:
                padded_mask = np.zeros((h_target, w_target), dtype=np.uint8)
                padded_mask[:h_mask, :w_mask] = diff_mask
                diff_mask = padded_mask
        
        # Create highlighted version (yellow highlight on difference areas)
        highlight_layer = np.zeros_like(overlay)
        highlight_layer[diff_mask > 0] = highlight_color
        
        # Blend highlight with base
        overlay = cv2.addWeighted(overlay, 1.0, highlight_layer, alpha, 0)
        
        # Draw bounding boxes and labels for each region
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        for region in regions:
            x, y, w, h = region.bbox
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
            
            # Create label text
            label = f"R{region.id}"
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label above the box
            label_x = x
            label_y = max(y - 5, text_h + 5)
            
            # Draw background rectangle for label
            cv2.rectangle(
                overlay,
                (label_x - 2, label_y - text_h - 2),
                (label_x + text_w + 2, label_y + 2),
                box_color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                overlay,
                label,
                (label_x, label_y),
                font,
                font_scale,
                label_color,
                font_thickness,
            )
        
        return overlay
    
    def detect_differences(
        self,
        linework_a: np.ndarray,
        linework_b: np.ndarray,
        base_image: Optional[np.ndarray] = None,
        use_composite_overlay: bool = True,
        color_a: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
        color_b: Tuple[int, int, int] = (255, 255, 0),  # Cyan in BGR
    ) -> DiffResult:
        """
        Full difference detection pipeline.
        
        Args:
            linework_a: Binary linework mask from drawing A
            linework_b: Binary linework mask from drawing B (aligned)
            base_image: Optional base image for annotation (uses composite if None)
            use_composite_overlay: If True, create composite showing both drawings
                                   (A in red, B in cyan) for better AI analysis
            color_a: BGR color for drawing A in composite
            color_b: BGR color for drawing B in composite
        
        Returns:
            DiffResult with diff mask, annotated overlay, and regions
        """
        logger.info("Starting difference detection")
        
        # Step 1: Compute XOR difference
        diff_mask = self.compute_diff_mask(linework_a, linework_b)
        
        # Step 2: Extract regions
        regions = self.extract_regions(diff_mask)
        
        # Step 3: Create annotated overlay
        used_color_a = color_a
        used_color_b = color_b
        
        if base_image is None:
            if use_composite_overlay:
                # Create composite overlay showing both drawings (A=red, B=cyan)
                # This gives AI models better context to understand differences
                base_image, used_color_a, used_color_b = self.create_composite_overlay(
                    linework_a, linework_b, color_a, color_b
                )
                logger.info(f"Created composite overlay (A={used_color_a}, B={used_color_b}) for annotation")
            else:
                # Fallback: Convert linework to grayscale image for annotation
                base_image = 255 - linework_a  # Invert so lines are dark on white
        
        # Pass colors to annotated overlay for contrasting bbox color computation
        annotated = self.create_annotated_overlay(
            base_image, diff_mask, regions,
            color_a=used_color_a, color_b=used_color_b
        )
        
        # Compute statistics
        total_diff_pixels = int(np.sum(diff_mask > 0))
        total_pixels = diff_mask.shape[0] * diff_mask.shape[1]
        
        stats = {
            'total_regions': len(regions),
            'total_diff_pixels': total_diff_pixels,
            'diff_percentage': total_diff_pixels / total_pixels * 100 if total_pixels > 0 else 0,
            'largest_region_area': regions[0].area if regions else 0,
        }
        
        logger.info(
            f"Difference detection complete: {len(regions)} regions, "
            f"{stats['diff_percentage']:.2f}% difference"
        )
        
        return DiffResult(
            diff_mask=diff_mask,
            annotated_overlay=annotated,
            regions=regions,
            stats=stats,
        )


# Global service instance - uses settings from config
diff_detection_service = DiffDetectionService(
    min_region_area=settings.diff_min_region_area,
    morph_kernel_size=settings.diff_morph_kernel_size,
    morph_iterations=settings.diff_morph_iterations,
    position_tolerance=settings.diff_position_tolerance,
)

logger.info(
    f"DiffDetectionService initialized: position_tolerance={settings.diff_position_tolerance}px, "
    f"min_region_area={settings.diff_min_region_area}px, "
    f"morph_kernel={settings.diff_morph_kernel_size}, morph_iter={settings.diff_morph_iterations}"
)
