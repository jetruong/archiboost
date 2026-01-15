"""
Preprocessing service for architectural line drawings.

Provides geometry-first preprocessing to extract clean linework representations
optimized for alignment. Key operations:

1. Contrast normalization (CLAHE)
2. Edge detection (Canny)
3. Text/small component suppression
4. Binary linework mask generation
5. Separate text mask extraction for annotations

This preprocessing makes alignment more robust by:
- Removing text labels that vary between drawings
- Extracting only structural linework (walls, fixtures, etc.)
- Producing deterministic, reproducible results
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    # Canny edge detection
    canny_threshold1: int = settings.canny_threshold1
    canny_threshold2: int = settings.canny_threshold2
    canny_aperture: int = settings.canny_aperture
    
    # Morphological operations
    dilate_kernel: int = settings.linework_dilate_kernel
    dilate_iterations: int = settings.linework_dilate_iterations
    
    # Text suppression
    suppress_text: bool = settings.text_suppress_enabled
    min_area: int = settings.text_suppress_min_area
    min_width: int = settings.text_suppress_min_width
    min_height: int = settings.text_suppress_min_height
    max_aspect_ratio: float = settings.text_suppress_max_aspect_ratio
    
    # Contrast normalization
    normalize_contrast: bool = settings.normalize_contrast
    clahe_clip_limit: float = settings.clahe_clip_limit
    clahe_tile_size: int = settings.clahe_tile_size
    
    # Enhanced text/annotation suppression
    # These parameters help identify annotation bubbles and callout pointers
    annotation_bubble_max_area: int = 5000  # Max area for annotation bubbles
    annotation_bubble_min_circularity: float = 0.5  # Min circularity for bubbles
    callout_pointer_max_length: int = 200  # Max length for callout lines
    line_continuity_threshold: int = 50  # Min pixels for structural lines
    text_char_max_area: int = 400  # Max area for individual text characters


@dataclass
class PreprocessResult:
    """Result of preprocessing an image."""
    linework: np.ndarray      # Binary linework mask (uint8, 0 or 255)
    grayscale: np.ndarray     # Normalized grayscale image
    edge_count: int           # Number of edge pixels
    components_removed: int   # Number of text-like components removed
    original_shape: Tuple[int, int]


@dataclass 
class EnhancedPreprocessResult:
    """Enhanced result with separate linework and text masks."""
    linework_mask: np.ndarray    # Structural lines only (walls, fixtures)
    text_mask: np.ndarray        # Text + small annotation glyphs
    annotation_mask: np.ndarray  # Annotation bubbles and callout pointers
    grayscale: np.ndarray        # Normalized grayscale image
    original_shape: Tuple[int, int]
    stats: dict                  # Processing statistics


class PreprocessService:
    """Service for preprocessing architectural drawings."""
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 2:
            return image.copy()
        elif len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    
    def normalize_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Normalize contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        This helps with drawings that have varying line weights or scanner artifacts.
        CLAHE is deterministic and produces consistent results.
        """
        if not self.config.normalize_contrast:
            return gray
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size)
        )
        normalized = clahe.apply(gray)
        
        logger.debug(f"Contrast normalized with CLAHE (clip={self.config.clahe_clip_limit})")
        return normalized
    
    def extract_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Extract edges using Canny edge detector.
        
        Canny is preferred for architectural drawings because:
        - Produces thin, single-pixel edges
        - Handles varying line weights well
        - Deterministic with fixed parameters
        """
        edges = cv2.Canny(
            gray,
            threshold1=self.config.canny_threshold1,
            threshold2=self.config.canny_threshold2,
            apertureSize=self.config.canny_aperture,
        )
        
        edge_count = np.sum(edges > 0)
        logger.debug(f"Canny edge detection: {edge_count} edge pixels")
        
        return edges
    
    def dilate_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Dilate edges slightly to connect nearby edge pixels.
        
        This helps with:
        - Broken lines from scanner noise
        - Thin lines that might be missed in matching
        """
        if self.config.dilate_kernel <= 0:
            return edges
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.dilate_kernel, self.config.dilate_kernel)
        )
        dilated = cv2.dilate(edges, kernel, iterations=self.config.dilate_iterations)
        
        return dilated
    
    def _compute_component_features(
        self, 
        binary: np.ndarray, 
        label: int, 
        labels: np.ndarray, 
        stats: np.ndarray
    ) -> dict:
        """Compute features for a connected component to classify it."""
        x, y, w, h, area = stats[label]
        
        # Extract component mask
        component_mask = (labels == label).astype(np.uint8) * 255
        
        # Compute circularity (how circular the shape is)
        perimeter = cv2.arcLength(
            cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0] 
            if cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] else np.array([]), 
            True
        )
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Compute solidity (area / convex hull area)
        contours = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours and len(contours[0]) >= 3:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
        else:
            solidity = 0
        
        # Compute aspect ratio
        aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 1
        
        # Compute extent (area / bounding rect area)
        extent = area / (w * h) if w * h > 0 else 0
        
        # Compute skeleton length (proxy for line length)
        skeleton = cv2.ximgproc.thinning(component_mask) if hasattr(cv2, 'ximgproc') else component_mask
        skeleton_length = np.sum(skeleton > 0)
        
        return {
            'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
            'circularity': circularity,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'skeleton_length': skeleton_length,
        }
    
    def _classify_component(self, features: dict) -> str:
        """
        Classify a component as 'linework', 'text', or 'annotation'.
        
        Uses heuristics based on component features:
        - Text: small area, medium aspect ratio, low solidity
        - Annotation bubbles: circular, medium area
        - Callout pointers: thin, elongated lines
        - Linework: larger, connected structures
        """
        area = features['area']
        aspect_ratio = features['aspect_ratio']
        circularity = features['circularity']
        solidity = features['solidity']
        w, h = features['w'], features['h']
        extent = features['extent']
        
        # Individual text characters: small area, compact shape
        if area < self.config.text_char_max_area:
            return 'text'
        
        # Text strings: elongated, low solidity (gaps between characters)
        if (area < self.config.min_area * 15 and 
            aspect_ratio > 3 and 
            solidity < 0.4):
            return 'text'
        
        # Annotation bubbles: circular, bounded size
        if (circularity > self.config.annotation_bubble_min_circularity and
            area < self.config.annotation_bubble_max_area and
            area > 200):  # Not too small
            return 'annotation'
        
        # Callout pointers: thin, elongated
        if (aspect_ratio > 8 and 
            min(w, h) < 20 and 
            max(w, h) < self.config.callout_pointer_max_length):
            return 'annotation'
        
        # Small isolated components: likely text or debris
        if area < self.config.min_area:
            return 'text'
        
        # Very thin components (likely text strokes or annotations)
        if min(w, h) < self.config.min_width:
            return 'text'
        
        # High aspect ratio without sufficient area: text
        if aspect_ratio > self.config.max_aspect_ratio and area < 2000:
            return 'text'
        
        # Default to linework (structural)
        return 'linework'
    
    def extract_enhanced_masks(
        self, 
        image: np.ndarray,
        config: Optional[PreprocessConfig] = None,
    ) -> EnhancedPreprocessResult:
        """
        Extract separate masks for linework, text, and annotations.
        
        This enhanced extraction provides better separation between:
        - Structural linework (walls, fixtures, etc.)
        - Text and small glyphs (labels, dimensions)
        - Annotation bubbles and callout pointers
        
        Args:
            image: Input image (BGR or grayscale)
            config: Optional preprocessing configuration override
        
        Returns:
            EnhancedPreprocessResult with separate masks
        """
        if config is not None:
            self.config = config
        
        original_shape = image.shape[:2]
        logger.info(f"Enhanced mask extraction for image {original_shape[1]}x{original_shape[0]}")
        
        # Convert to grayscale and normalize
        gray = self.to_grayscale(image)
        normalized = self.normalize_contrast(gray)
        
        # Get binary mask of all dark content (lines on white background)
        _, binary = cv2.threshold(normalized, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Also extract edges for thin lines
        edges = self.extract_edges(normalized)
        
        # Combine threshold and edges
        all_content = cv2.bitwise_or(binary, edges)
        
        # Dilate slightly to connect broken lines
        all_content = self.dilate_edges(all_content)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            all_content, connectivity=8
        )
        
        # Initialize output masks
        linework_mask = np.zeros_like(all_content)
        text_mask = np.zeros_like(all_content)
        annotation_mask = np.zeros_like(all_content)
        
        # Statistics
        counts = {'linework': 0, 'text': 0, 'annotation': 0}
        
        # Classify each component (skip background label 0)
        for label in range(1, num_labels):
            # Simple feature extraction (avoid expensive operations)
            x, y, w, h, area = stats[label]
            
            # Quick classification based on simple features
            classification = self._classify_component_simple(area, w, h)
            
            # Add to appropriate mask
            if classification == 'linework':
                linework_mask[labels == label] = 255
            elif classification == 'text':
                text_mask[labels == label] = 255
            else:  # annotation
                annotation_mask[labels == label] = 255
            
            counts[classification] += 1
        
        # Post-process linework mask to connect fragmented lines
        # Use morphological closing to bridge small gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        linework_mask = cv2.morphologyEx(linework_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # Remove any remaining small isolated components from linework
        linework_mask = self._remove_small_components(linework_mask, min_area=150)
        
        logger.info(
            f"Enhanced extraction complete: "
            f"linework={counts['linework']}, text={counts['text']}, annotation={counts['annotation']}"
        )
        
        return EnhancedPreprocessResult(
            linework_mask=linework_mask,
            text_mask=text_mask,
            annotation_mask=annotation_mask,
            grayscale=normalized,
            original_shape=original_shape,
            stats={
                'linework_components': counts['linework'],
                'text_components': counts['text'],
                'annotation_components': counts['annotation'],
                'linework_pixels': int(np.sum(linework_mask > 0)),
                'text_pixels': int(np.sum(text_mask > 0)),
                'annotation_pixels': int(np.sum(annotation_mask > 0)),
            }
        )
    
    def _classify_component_simple(self, area: int, w: int, h: int) -> str:
        """
        Simplified component classification using basic geometry.
        
        This avoids expensive contour operations for better performance.
        """
        aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 1
        extent = area / (w * h) if w * h > 0 else 0
        
        # Individual text characters: very small
        if area < self.config.text_char_max_area:
            return 'text'
        
        # Very thin components: likely text or annotation pointers
        if min(w, h) < self.config.min_width:
            return 'text'
        
        # Small with high aspect ratio: text strings
        if area < self.config.min_area * 10 and aspect_ratio > 5:
            return 'text'
        
        # Medium area, high circularity proxy (extent close to pi/4 for circles)
        # Annotation bubbles tend to have extent around 0.6-0.8
        if (area < self.config.annotation_bubble_max_area and 
            area > 200 and 
            extent > 0.55 and 
            aspect_ratio < 2):
            return 'annotation'
        
        # Long thin shapes: callout pointers
        if (aspect_ratio > 8 and 
            min(w, h) < 20 and 
            max(w, h) < self.config.callout_pointer_max_length):
            return 'annotation'
        
        # Small isolated components
        if area < self.config.min_area:
            return 'text'
        
        # High aspect ratio medium components: text labels
        if aspect_ratio > self.config.max_aspect_ratio and area < 3000:
            return 'text'
        
        # Default: structural linework
        return 'linework'
    
    def _remove_small_components(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Remove small connected components from a mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        result = np.zeros_like(mask)
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                result[labels == label] = 255
        
        return result
    
    def suppress_text_components(self, binary: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Remove text-like connected components from binary image.
        
        Text in architectural drawings causes alignment issues because:
        - Labels may be positioned differently between revisions
        - Annotation text creates spurious feature matches
        - Text components are typically small and high aspect ratio
        
        This function removes components that appear to be text based on:
        - Area (too small)
        - Dimensions (too thin)
        - Aspect ratio (too elongated)
        
        Returns:
            Tuple of (filtered binary image, number of components removed)
        """
        if not self.config.suppress_text:
            return binary, 0
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Create output mask
        result = np.zeros_like(binary)
        components_removed = 0
        
        # Analyze each component (skip background label 0)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            
            # Check if component looks like text
            is_text_like = False
            
            # Too small by area
            if area < self.config.min_area:
                is_text_like = True
            
            # Too thin (likely individual characters)
            elif w < self.config.min_width or h < self.config.min_height:
                is_text_like = True
            
            # High aspect ratio (likely text strings)
            elif w > 0 and h > 0:
                aspect = max(w / h, h / w)
                if aspect > self.config.max_aspect_ratio:
                    is_text_like = True
            
            if is_text_like:
                components_removed += 1
            else:
                # Keep this component
                result[labels == label] = 255
        
        logger.debug(
            f"Text suppression: removed {components_removed} of {num_labels - 1} components"
        )
        
        return result, components_removed
    
    def build_linework(
        self,
        image: np.ndarray,
        config: Optional[PreprocessConfig] = None,
    ) -> PreprocessResult:
        """
        Build a clean linework representation of an architectural drawing.
        
        Pipeline:
        1. Convert to grayscale
        2. Normalize contrast (CLAHE)
        3. Extract edges (Canny)
        4. Dilate edges slightly
        5. Suppress text-like components
        
        The result is a binary mask where 255 = linework, 0 = background.
        This is optimized for phase correlation and feature matching.
        
        Args:
            image: Input image (BGR or grayscale)
            config: Optional preprocessing configuration override
        
        Returns:
            PreprocessResult with linework mask and metadata
        """
        if config is not None:
            self.config = config
        
        original_shape = image.shape[:2]
        logger.info(f"Building linework for image {original_shape[1]}x{original_shape[0]}")
        
        # Step 1: Convert to grayscale
        gray = self.to_grayscale(image)
        
        # Step 2: Normalize contrast
        normalized = self.normalize_contrast(gray)
        
        # Step 3: Extract edges
        edges = self.extract_edges(normalized)
        
        # Step 4: Dilate edges
        dilated = self.dilate_edges(edges)
        
        # Step 5: Suppress text components
        linework, components_removed = self.suppress_text_components(dilated)
        
        edge_count = np.sum(linework > 0)
        
        logger.info(
            f"Linework built: {edge_count} pixels, "
            f"{components_removed} text-like components removed"
        )
        
        return PreprocessResult(
            linework=linework,
            grayscale=normalized,
            edge_count=edge_count,
            components_removed=components_removed,
            original_shape=original_shape,
        )
    
    def compute_linework_bbox(self, linework: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute bounding box of linework content.
        
        Returns:
            Tuple of (x, y, width, height) or None if no content
        """
        if np.sum(linework) == 0:
            return None
        
        # Find rows and columns with content
        rows_with_content = np.any(linework > 0, axis=1)
        cols_with_content = np.any(linework > 0, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            return None
        
        y_indices = np.where(rows_with_content)[0]
        x_indices = np.where(cols_with_content)[0]
        
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


# Global service instance
preprocess_service = PreprocessService()
