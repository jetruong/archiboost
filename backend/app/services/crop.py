"""
Image cropping service for whitespace removal.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from app.config import settings
from app.models.session import CropMetadata

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        """Create from top-left and bottom-right corners."""
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@dataclass
class CropResult:
    """Result of cropping operation."""
    image: np.ndarray
    metadata: CropMetadata


class CropService:
    """Service for detecting and cropping whitespace from images."""
    
    def __init__(
        self,
        threshold: int = None,
        padding: int = None,
        min_content_area: int = None,
    ):
        self.threshold = threshold or settings.crop_threshold
        self.padding = padding or settings.crop_padding
        self.min_content_area = min_content_area or settings.min_content_area
    
    def detect_content_bbox(
        self,
        image: np.ndarray,
        threshold: int = None,
    ) -> BoundingBox:
        """
        Detect the bounding box of non-white content in an image.
        
        Algorithm:
        1. Convert to grayscale
        2. Apply binary threshold (pixels < threshold are "content")
        3. Find the bounding rectangle of all content pixels
        
        Args:
            image: RGB or BGR image as numpy array
            threshold: Grayscale threshold (0-255), pixels below this are "content"
        
        Returns:
            BoundingBox of detected content
        
        Raises:
            ValueError: If no content is detected
        """
        threshold = threshold or self.threshold
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold: content pixels are those BELOW threshold (darker than white)
        # We invert so content becomes white (255) and background becomes black (0)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find all non-zero (content) pixel coordinates
        coords = cv2.findNonZero(binary)
        
        if coords is None or len(coords) < self.min_content_area:
            raise ValueError(
                f"No significant content detected (threshold={threshold}, "
                f"min_area={self.min_content_area})"
            )
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)
        
        return BoundingBox(x=x, y=y, width=w, height=h)
    
    def apply_padding(
        self,
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
        padding: int = None,
    ) -> BoundingBox:
        """
        Apply padding to a bounding box, clamping to image boundaries.
        
        Args:
            bbox: Original bounding box
            image_width: Width of the source image
            image_height: Height of the source image
            padding: Pixels of padding to add on all sides
        
        Returns:
            Padded bounding box clamped to image boundaries
        """
        padding = padding if padding is not None else self.padding
        
        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(image_width, bbox.x + bbox.width + padding)
        y2 = min(image_height, bbox.y + bbox.height + padding)
        
        return BoundingBox.from_xyxy(x1, y1, x2, y2)
    
    def crop_whitespace(
        self,
        image: np.ndarray,
        threshold: int = None,
        padding: int = None,
    ) -> CropResult:
        """
        Detect and crop whitespace from an image.
        
        Args:
            image: RGB image as numpy array
            threshold: Grayscale threshold for content detection
            padding: Pixels of padding around content
        
        Returns:
            CropResult with cropped image and metadata
        """
        threshold = threshold if threshold is not None else self.threshold
        padding = padding if padding is not None else self.padding
        
        original_height, original_width = image.shape[:2]
        
        logger.info(
            f"Cropping whitespace from {original_width}x{original_height} image "
            f"(threshold={threshold}, padding={padding})"
        )
        
        # Detect content bounding box
        content_bbox = self.detect_content_bbox(image, threshold)
        
        # Apply padding
        final_bbox = self.apply_padding(
            content_bbox, original_width, original_height, padding
        )
        
        # Crop the image
        cropped = image[
            final_bbox.y : final_bbox.y + final_bbox.height,
            final_bbox.x : final_bbox.x + final_bbox.width,
        ].copy()
        
        # Calculate whitespace ratio
        original_area = original_width * original_height
        content_area = content_bbox.width * content_bbox.height
        whitespace_ratio = 1.0 - (content_area / original_area)
        
        metadata = CropMetadata(
            original_width=original_width,
            original_height=original_height,
            content_bbox=content_bbox.to_dict(),
            final_crop_bbox=final_bbox.to_dict(),
            cropped_width=final_bbox.width,
            cropped_height=final_bbox.height,
            threshold_used=threshold,
            padding_applied=padding,
            whitespace_ratio=round(whitespace_ratio, 4),
        )
        
        logger.info(
            f"Cropped to {final_bbox.width}x{final_bbox.height} "
            f"(removed {whitespace_ratio:.1%} whitespace)"
        )
        
        return CropResult(image=cropped, metadata=metadata)
    
    def compute_bbox_for_thresholds(
        self,
        image: np.ndarray,
        thresholds: list[int] = None,
    ) -> list[Tuple[int, BoundingBox]]:
        """
        Compute content bounding boxes for multiple threshold values.
        Useful for debugging or finding optimal threshold.
        
        Args:
            image: RGB image as numpy array
            thresholds: List of threshold values to test
        
        Returns:
            List of (threshold, bbox) tuples for successful detections
        """
        if thresholds is None:
            thresholds = [230, 240, 250, 252, 254]
        
        results = []
        for thresh in thresholds:
            try:
                bbox = self.detect_content_bbox(image, thresh)
                results.append((thresh, bbox))
            except ValueError:
                continue
        
        return results


# Global service instance
crop_service = CropService()
