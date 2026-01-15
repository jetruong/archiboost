"""
Overlay service for generating composite visualizations of aligned drawings.

Creates a single overlay image where:
- Drawing A linework is tinted RED
- Drawing B linework (after alignment warp) is tinted CYAN/BLUE
- Overlapping lines appear dark/black
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.services.align import SimilarityTransform

logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for overlay generation."""
    color_a: Tuple[int, int, int] = (255, 0, 0)      # Red (BGR)
    color_b: Tuple[int, int, int] = (255, 255, 0)    # Cyan (BGR) 
    alpha_a: float = 0.7
    alpha_b: float = 0.7
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White (BGR)
    line_threshold: int = 240  # Grayscale threshold for line detection
    invert_mask: bool = True   # True for dark lines on light background


@dataclass
class OverlayResult:
    """Result of overlay generation."""
    overlay_image: np.ndarray
    width: int
    height: int
    config: OverlayConfig
    processing_time_ms: int
    diff_mask: Optional[np.ndarray] = None  # XOR mask for differences


class OverlayService:
    """Service for generating overlay visualizations."""
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
    
    def extract_line_mask(
        self,
        image: np.ndarray,
        threshold: int = None,
    ) -> np.ndarray:
        """
        Extract a binary mask of linework from an image.
        
        For architectural drawings with dark lines on white background,
        this converts to grayscale and applies thresholding.
        
        Args:
            image: BGR or RGB image
            threshold: Grayscale threshold (pixels darker than this are lines)
        
        Returns:
            Binary mask where 255 = line, 0 = background
        """
        threshold = threshold or self.config.line_threshold
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold: pixels below threshold are lines (dark on light)
        # We want lines = 255 (white in mask)
        if self.config.invert_mask:
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def warp_image(
        self,
        image: np.ndarray,
        transform: SimilarityTransform,
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp an image using a similarity transform.
        
        Args:
            image: Source image to warp
            transform: Similarity transform (maps source -> destination)
            output_size: (width, height) of output image
        
        Returns:
            Tuple of (warped image, validity mask)
            - warped image: the transformed image with white fill for empty areas
            - validity mask: 255 where original image content exists, 0 for filled areas
        """
        matrix = transform.matrix_2x3.astype(np.float32)
        
        # Create validity mask by warping a solid mask
        h, w = image.shape[:2]
        valid_mask_src = np.ones((h, w), dtype=np.uint8) * 255
        
        # Warp the validity mask - this tells us where actual B content is after warping
        valid_mask = cv2.warpAffine(
            valid_mask_src,
            matrix,
            output_size,
            flags=cv2.INTER_NEAREST,  # Use nearest neighbor for crisp edges
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,  # Areas outside original image become 0
        )
        
        # cv2.warpAffine expects (width, height) for dsize
        warped = cv2.warpAffine(
            image,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255,
        )
        
        return warped, valid_mask
    
    def create_tinted_layer(
        self,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float,
        validity_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create a tinted BGRA layer from a binary mask.
        
        Args:
            mask: Binary mask (255 = line, 0 = background)
            color: BGR color tuple for the tint
            alpha: Opacity (0.0-1.0)
            validity_mask: Optional mask (255 = valid region, 0 = out of bounds)
                           Only applies tint where validity_mask is non-zero
        
        Returns:
            BGRA image with tinted lines and transparent background
        """
        height, width = mask.shape
        
        # Create BGRA layer
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Apply validity mask if provided (only tint where content actually exists)
        if validity_mask is not None:
            effective_mask = cv2.bitwise_and(mask, validity_mask)
        else:
            effective_mask = mask
        
        # Set color where effective mask is non-zero
        layer[effective_mask > 0, 0] = color[0]  # B
        layer[effective_mask > 0, 1] = color[1]  # G
        layer[effective_mask > 0, 2] = color[2]  # R
        layer[effective_mask > 0, 3] = int(alpha * 255)  # A
        
        return layer
    
    def composite_layers(
        self,
        layer_a: np.ndarray,
        layer_b: np.ndarray,
        background_color: Tuple[int, int, int] = None,
    ) -> np.ndarray:
        """
        Composite two BGRA layers onto a background.
        
        Uses standard alpha blending with multiply-like effect for overlapping lines.
        
        Args:
            layer_a: BGRA layer for drawing A (red)
            layer_b: BGRA layer for drawing B (cyan)
            background_color: BGR background color
        
        Returns:
            BGR composite image
        """
        background_color = background_color or self.config.background_color
        height, width = layer_a.shape[:2]
        
        # Start with background
        result = np.zeros((height, width, 3), dtype=np.uint8)
        result[:, :] = background_color
        
        # Convert layers to float for blending
        layer_a_float = layer_a.astype(np.float32) / 255.0
        layer_b_float = layer_b.astype(np.float32) / 255.0
        result_float = result.astype(np.float32) / 255.0
        
        # Extract alpha channels
        alpha_a = layer_a_float[:, :, 3:4]
        alpha_b = layer_b_float[:, :, 3:4]
        
        # Extract RGB (actually BGR)
        rgb_a = layer_a_float[:, :, :3]
        rgb_b = layer_b_float[:, :, :3]
        
        # Blend A onto background
        result_float = result_float * (1 - alpha_a) + rgb_a * alpha_a
        
        # Blend B onto result (multiply where both have content for darker overlap)
        # For overlapping lines, we want them to appear darker
        overlap_mask = (alpha_a[:, :, 0] > 0) & (alpha_b[:, :, 0] > 0)
        
        # Standard alpha blend for B
        result_float = result_float * (1 - alpha_b) + rgb_b * alpha_b
        
        # Darken overlapping areas (multiply blend)
        if np.any(overlap_mask):
            # Where both layers have content, darken the result
            darkening_factor = 0.3
            result_float[overlap_mask] = result_float[overlap_mask] * darkening_factor
        
        # Convert back to uint8
        result = (np.clip(result_float, 0, 1) * 255).astype(np.uint8)
        
        return result
    
    def generate_overlay(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        transform: SimilarityTransform,
        config: Optional[OverlayConfig] = None,
    ) -> OverlayResult:
        """
        Generate an overlay visualization of two aligned drawings.
        
        Pipeline:
        1. Extract line masks from both images
        2. Warp image B's mask using the transform
        3. Create tinted BGRA layers for each
        4. Composite onto white background
        5. Optionally generate diff mask (XOR)
        
        Args:
            image_a: Drawing A image (BGR)
            image_b: Drawing B image (BGR)
            transform: Similarity transform mapping B -> A
            config: Optional override configuration
        
        Returns:
            OverlayResult with composite image and metadata
        """
        start_time = time.time()
        config = config or self.config
        
        # Get dimensions (use A as reference)
        height_a, width_a = image_a.shape[:2]
        output_size = (width_a, height_a)
        
        logger.info(
            f"Generating overlay: A={width_a}x{height_a}, "
            f"transform scale={transform.scale:.3f}, rotation={transform.rotation_deg:.2f}°"
        )
        
        # Step 1: Extract line masks
        mask_a = self.extract_line_mask(image_a, config.line_threshold)
        mask_b = self.extract_line_mask(image_b, config.line_threshold)
        
        logger.debug(f"Mask A: {np.sum(mask_a > 0)} line pixels")
        logger.debug(f"Mask B: {np.sum(mask_b > 0)} line pixels")
        
        # Step 2: Warp mask B to A's coordinate space
        # Also get validity mask showing where B actually has content after warping
        mask_b_warped, validity_mask_b = self.warp_image(mask_b, transform, output_size)
        
        logger.debug(f"Warped mask B: {np.sum(mask_b_warped > 0)} line pixels")
        logger.debug(f"Valid B region: {np.sum(validity_mask_b > 0)} pixels")
        
        # Step 3: Create tinted layers
        # For A: no validity mask needed (it's the reference)
        layer_a = self.create_tinted_layer(mask_a, config.color_a, config.alpha_a)
        # For B: use validity mask so we don't tint out-of-bounds areas
        layer_b = self.create_tinted_layer(
            mask_b_warped, config.color_b, config.alpha_b, 
            validity_mask=validity_mask_b
        )
        
        # Step 4: Composite
        overlay = self.composite_layers(layer_a, layer_b, config.background_color)
        
        # Step 5: Generate diff mask (XOR - pixels present in one but not both)
        # Apply validity mask to warped B so we only compare in the valid region
        mask_b_warped_valid = cv2.bitwise_and(mask_b_warped, validity_mask_b)
        diff_mask = cv2.bitwise_xor(mask_a, mask_b_warped_valid)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Overlay generated in {processing_time}ms")
        
        return OverlayResult(
            overlay_image=overlay,
            width=width_a,
            height=height_a,
            config=config,
            processing_time_ms=processing_time,
            diff_mask=diff_mask,
        )
    
    def save_overlay(
        self,
        result: OverlayResult,
        output_path: Path,
        save_diff_mask: bool = False,
    ) -> None:
        """Save overlay image and optionally diff mask."""
        # Save main overlay
        cv2.imwrite(str(output_path), result.overlay_image)
        logger.info(f"Saved overlay to {output_path}")
        
        # Optionally save diff mask
        if save_diff_mask and result.diff_mask is not None:
            diff_path = output_path.parent / f"{output_path.stem}_diff{output_path.suffix}"
            cv2.imwrite(str(diff_path), result.diff_mask)
            logger.info(f"Saved diff mask to {diff_path}")


# Global service instance
overlay_service = OverlayService()
