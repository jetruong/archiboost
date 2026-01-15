"""
Layer extraction and generation service.

Creates layers from architectural drawings for non-destructive compositing.
Each image becomes a single layer that the frontend can render and transform.
"""

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

from app.models.layers import (
    LayerInfo,
    LayerType,
    LayerSource,
    Transform2D,
    LayerState,
    CanvasState,
    AlignmentState,
    CompositionState,
    CompositionDefaults,
    BlendMode,
)
from app.services.align import SimilarityTransform

logger = logging.getLogger(__name__)


@dataclass
class ExtractedLayer:
    """Result of layer extraction."""
    layer_type: LayerType
    image: np.ndarray  # BGRA
    name: str
    default_color: Optional[str] = None
    default_opacity: float = 1.0


@dataclass
class LayerExtractionResult:
    """Result of extracting layers from an image."""
    layers: List[ExtractedLayer]
    original_shape: Tuple[int, int]  # (height, width)
    processing_time_ms: int


class LayerExtractionService:
    """
    Service for creating layers from architectural drawings.
    
    Each image is converted to a single linework layer (white lines on transparent)
    that can be tinted by the frontend for visual comparison.
    """
    
    def _extract_linework(self, image: np.ndarray) -> np.ndarray:
        """
        Extract linework from an image as white-on-transparent.
        
        This extracts dark lines from the image and converts them to white pixels
        on a transparent background. The frontend will colorize dynamically.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            BGRA image with white linework on transparent background
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold to find dark lines (invert so lines are white in mask)
        # Lower threshold = more sensitive to lighter lines
        _, line_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Create BGRA output: white pixels where there are lines, transparent elsewhere
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[line_mask > 0] = [255, 255, 255, 255]  # White, fully opaque
        # Everywhere else stays [0, 0, 0, 0] - fully transparent
        
        return bgra
    
    def extract_layers(
        self,
        image: np.ndarray,
        source: LayerSource,
        include_text_layer: bool = False,  # Ignored - kept for API compatibility
    ) -> LayerExtractionResult:
        """
        Create a single linework layer from an image.
        
        The linework is extracted as white-on-transparent. The frontend
        will colorize it dynamically (default: red for A, cyan for B).
        
        Args:
            image: Input image (BGR)
            source: Which input pair this image is from
            include_text_layer: Ignored (kept for API compatibility)
        
        Returns:
            LayerExtractionResult with a single white linework layer
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        source_name = "A" if source == LayerSource.PAIR1 else "B"
        default_color = "red" if source == LayerSource.PAIR1 else "cyan"
        
        # Extract linework as white-on-transparent
        linework = self._extract_linework(image)
        
        # Create single layer with white linework (frontend will colorize)
        layer = ExtractedLayer(
            layer_type=LayerType.BASE,
            image=linework,
            name=f"Drawing {source_name}",
            default_color=default_color,  # Frontend will use this for colorization
            default_opacity=1.0,
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Extracted linework from {source.value} ({w}x{h}) in {processing_time}ms"
        )
        
        return LayerExtractionResult(
            layers=[layer],
            original_shape=(h, w),
            processing_time_ms=processing_time,
        )
    
    def layer_to_png_base64(self, image: np.ndarray) -> str:
        """Convert BGRA image to base64-encoded PNG string."""
        # Convert BGRA to RGBA for PIL
        if image.shape[2] == 4:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create PIL Image
        pil_image = Image.fromarray(rgba)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        
        # Encode to base64
        png_bytes = buffer.getvalue()
        base64_str = base64.b64encode(png_bytes).decode("utf-8")
        
        return f"data:image/png;base64,{base64_str}"


class CompositionService:
    """
    Service for building composition state from extracted layers.
    """
    
    def __init__(self):
        self.layer_service = LayerExtractionService()
    
    def build_layer_info(
        self,
        extracted: ExtractedLayer,
        source: LayerSource,
        layer_id: str,
        png_base64: Optional[str] = None,
        png_url: Optional[str] = None,
    ) -> LayerInfo:
        """Build LayerInfo from extracted layer data."""
        h, w = extracted.image.shape[:2]
        
        return LayerInfo(
            id=layer_id,
            name=extracted.name,
            source=source,
            type=extracted.layer_type,
            png_url=png_url,
            png_base64=png_base64,
            width=w,
            height=h,
            default_color=extracted.default_color,
            default_opacity=extracted.default_opacity,
        )
    
    def build_layer_state(
        self,
        layer_info: LayerInfo,
        transform: Optional[Transform2D] = None,
        visible: bool = True,
    ) -> LayerState:
        """Build initial layer state from layer info."""
        return LayerState(
            id=layer_info.id,
            visible=visible,
            opacity=layer_info.default_opacity,
            blend_mode=BlendMode.NORMAL,
            color=layer_info.default_color,
            transform=transform or Transform2D(),
            locked=False,
        )
    
    def build_composition_state(
        self,
        canvas_width: int,
        canvas_height: int,
        layer_states: List[LayerState],
        alignment: Optional[AlignmentState] = None,
        dpi: int = 250,
    ) -> CompositionState:
        """Build complete composition state."""
        return CompositionState(
            canvas=CanvasState(
                width=canvas_width,
                height=canvas_height,
                dpi=dpi,
                background_color="#FFFFFF",
            ),
            layers=layer_states,
            alignment=alignment or AlignmentState(),
            defaults=CompositionDefaults(),
        )
    
    def transform_from_similarity(
        self,
        similarity: SimilarityTransform,
        layer_width: int,
        layer_height: int,
    ) -> Transform2D:
        """
        Convert SimilarityTransform to Transform2D.
        
        The SimilarityTransform computes transforms around the origin (0, 0):
            p_A = s * R(θ) * p_B + t
        
        For CSS to apply this correctly, we need to use pivot (0, 0) because
        a center pivot would require translation adjustment:
            With center pivot: result = R * s * (p - c) + t_css + c
            For this to equal s * R * p + t_orig:
                t_css = t_orig + R * s * c - c
        
        Using (0, 0) pivot keeps the math simple and matches the affine transform directly.
        """
        import math
        
        return Transform2D(
            translate_x=similarity.tx,
            translate_y=similarity.ty,
            scale_x=similarity.scale,
            scale_y=similarity.scale,
            rotation_deg=math.degrees(similarity.rotation_rad),
            pivot_x=0.0,  # Use origin pivot to match affine transform
            pivot_y=0.0,
        )


class BoundsComputationService:
    """
    Service for computing bounding boxes for canvas sizing and export cropping.
    
    Two key bounds are computed:
    - contentBounds: Includes ALL visible content (geometry + text + annotations)
      Used for editor canvas - must NEVER clip any content
    - geometryBounds: Geometry-only bounds (excludes sparse text at edges)
      Used for export cropping to reduce excessive whitespace
    """
    
    def compute_transformed_bounds(
        self,
        width: int,
        height: int,
        transform: Transform2D,
    ) -> Tuple[float, float, float, float]:
        """
        Compute axis-aligned bounding box of a transformed rectangle.
        
        Returns: (min_x, min_y, max_x, max_y)
        """
        import math
        
        # Corner points
        corners = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
        ]
        
        # Apply transform to each corner
        pivot_x = transform.pivot_x * width
        pivot_y = transform.pivot_y * height
        radians = math.radians(transform.rotation_deg)
        cos_r = math.cos(radians)
        sin_r = math.sin(radians)
        
        transformed = []
        for x, y in corners:
            # Translate to pivot
            px = x - pivot_x
            py = y - pivot_y
            
            # Scale
            sx = px * transform.scale_x
            sy = py * transform.scale_y
            
            # Rotate
            rx = sx * cos_r - sy * sin_r
            ry = sx * sin_r + sy * cos_r
            
            # Translate back and apply translation
            tx = rx + pivot_x + transform.translate_x
            ty = ry + pivot_y + transform.translate_y
            
            transformed.append((tx, ty))
        
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        
        return min(xs), min(ys), max(xs), max(ys)
    
    def compute_content_bounds(
        self,
        layers: List[LayerInfo],
        layer_states: List[LayerState],
        canvas_width: int,
        canvas_height: int,
        padding: int = 64,
    ) -> Tuple[float, float, float, float]:
        """
        Compute contentBounds: union of all visible layer bounds + padding.
        
        This is used for editor canvas sizing - must include ALL content.
        
        Returns: (x, y, width, height)
        """
        min_x = 0
        min_y = 0
        max_x = canvas_width
        max_y = canvas_height
        
        for state in layer_states:
            if not state.visible:
                continue
            
            info = next((l for l in layers if l.id == state.id), None)
            if not info:
                continue
            
            x1, y1, x2, y2 = self.compute_transformed_bounds(
                info.width, info.height, state.transform
            )
            
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
        
        # Add padding
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        return min_x, min_y, max_x - min_x, max_y - min_y
    
    def compute_geometry_bounds_from_image(
        self,
        image: np.ndarray,
        margin: int = 24,
    ) -> Tuple[float, float, float, float]:
        """
        Compute geometryBounds from an image by analyzing pixel content.
        
        This uses density-based filtering to find the "geometry core",
        excluding sparse text/annotations that extend far from main content.
        
        Args:
            image: BGR image to analyze
            margin: Margin to add to computed bounds
            
        Returns: (x, y, width, height)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Find non-white pixels (content)
        # White is typically 255, so anything significantly darker is content
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find content bounds
        rows = np.any(binary > 0, axis=1)
        cols = np.any(binary > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return 0, 0, w, h  # Return full image if no content
        
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        content_y1 = row_indices[0]
        content_y2 = row_indices[-1]
        content_x1 = col_indices[0]
        content_x2 = col_indices[-1]
        
        # Compute row/column densities for geometry core detection
        row_density = np.sum(binary > 0, axis=1)
        col_density = np.sum(binary > 0, axis=0)
        
        # Find geometry bounds using density thresholds
        max_row_density = max(np.max(row_density), 1)
        max_col_density = max(np.max(col_density), 1)
        
        min_density = 5
        edge_factor = 0.1
        
        # Trim sparse edges (geometry core detection)
        geo_y1 = content_y1
        geo_y2 = content_y2
        geo_x1 = content_x1
        geo_x2 = content_x2
        
        # Allow trimming up to 10% from each edge
        trim_limit_y = int((content_y2 - content_y1) * 0.1)
        trim_limit_x = int((content_x2 - content_x1) * 0.1)
        
        # Trim top
        for y in range(content_y1, min(content_y1 + trim_limit_y, content_y2)):
            if row_density[y] < min_density or row_density[y] < max_row_density * edge_factor:
                geo_y1 = y + 1
            else:
                break
        
        # Trim bottom
        for y in range(content_y2, max(content_y2 - trim_limit_y, content_y1), -1):
            if row_density[y] < min_density or row_density[y] < max_row_density * edge_factor:
                geo_y2 = y - 1
            else:
                break
        
        # Trim left
        for x in range(content_x1, min(content_x1 + trim_limit_x, content_x2)):
            if col_density[x] < min_density or col_density[x] < max_col_density * edge_factor:
                geo_x1 = x + 1
            else:
                break
        
        # Trim right (important for fire extinguisher example where text extends far right)
        for x in range(content_x2, max(content_x2 - trim_limit_x, content_x1), -1):
            if col_density[x] < min_density or col_density[x] < max_col_density * edge_factor:
                geo_x2 = x - 1
            else:
                break
        
        # Apply margin and clamp
        final_x1 = max(0, geo_x1 - margin)
        final_y1 = max(0, geo_y1 - margin)
        final_x2 = min(w - 1, geo_x2 + margin)
        final_y2 = min(h - 1, geo_y2 + margin)
        
        logger.info(
            f"Geometry bounds: content=({content_x1},{content_y1})-({content_x2},{content_y2}), "
            f"geometry=({geo_x1},{geo_y1})-({geo_x2},{geo_y2}), "
            f"final=({final_x1},{final_y1})-({final_x2},{final_y2})"
        )
        
        return final_x1, final_y1, final_x2 - final_x1 + 1, final_y2 - final_y1 + 1


# Global service instances
layer_extraction_service = LayerExtractionService()
composition_service = CompositionService()
bounds_computation_service = BoundsComputationService()
