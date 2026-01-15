"""
Layer and composition state models for non-destructive overlay system.

These models define the layer-based API output that allows the frontend
to render, edit, and export overlays without backend rendering dependency.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LayerSource(str, Enum):
    """Source input for a layer."""
    PAIR1 = "pair1"
    PAIR2 = "pair2"


class LayerType(str, Enum):
    """Type of layer content."""
    BASE = "base"           # Full rasterized input (fallback)
    LINEWORK = "linework"   # Extracted line/edge content
    TEXT = "text"           # Text and labels
    ANNOTATION = "annotation"  # Dimensions, symbols, etc.


class BlendMode(str, Enum):
    """Compositing blend modes."""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DARKEN = "darken"
    LIGHTEN = "lighten"


class RotationConstraint(str, Enum):
    """Rotation constraint options."""
    NONE = "NONE"
    RIGHT_ANGLES = "RIGHT_ANGLES"
    SMALL_ANGLE = "SMALL_ANGLE"


# ============================================================
# Bounding Box Models
# ============================================================

# Constants for canvas/export bounds
CANVAS_PADDING_PX = 64   # Padding for editor canvas (ensures no content clipping)
EXPORT_MARGIN_PX = 24    # Smaller margin for exported images

class BoundingBox(BaseModel):
    """
    Bounding box representing a rectangular region.
    Used for canvas sizing and export cropping.
    """
    x: float = Field(description="Left edge X coordinate")
    y: float = Field(description="Top edge Y coordinate")
    width: float = Field(description="Width in pixels")
    height: float = Field(description="Height in pixels")


class LayerBoundsInfo(BaseModel):
    """
    Bounding boxes for layer-based canvas sizing and export cropping.
    
    contentBounds: Union of all visible layer bounds + padding.
        - Used for editor canvas sizing
        - Includes ALL content: geometry, text, callouts, dimensions
        - Editor canvas NEVER clips any content
    
    geometryBounds: Tighter bounds focused on geometry/linework only.
        - Used for export cropping
        - Excludes sparse text/annotations at edges
        - Reduces excessive whitespace in exports
    """
    content_bounds: BoundingBox = Field(
        description="Full content bounds for editor (includes text/annotations)"
    )
    geometry_bounds: BoundingBox = Field(
        description="Geometry-only bounds for export cropping"
    )


# ============================================================
# Transform Models
# ============================================================

class Transform2D(BaseModel):
    """2D transformation parameters for a layer."""
    translate_x: float = Field(default=0.0, description="X translation in pixels")
    translate_y: float = Field(default=0.0, description="Y translation in pixels")
    scale_x: float = Field(default=1.0, description="X scale factor")
    scale_y: float = Field(default=1.0, description="Y scale factor")
    rotation_deg: float = Field(default=0.0, description="Rotation in degrees (CCW positive)")
    pivot_x: float = Field(default=0.5, description="Pivot X as fraction of layer width (0-1)")
    pivot_y: float = Field(default=0.5, description="Pivot Y as fraction of layer height (0-1)")
    
    @property
    def is_identity(self) -> bool:
        """Check if transform is identity (no transformation)."""
        return (
            abs(self.translate_x) < 0.001 and
            abs(self.translate_y) < 0.001 and
            abs(self.scale_x - 1.0) < 0.001 and
            abs(self.scale_y - 1.0) < 0.001 and
            abs(self.rotation_deg) < 0.001
        )
    
    def to_matrix_2x3(self) -> List[List[float]]:
        """Convert to 2x3 affine matrix for CSS/canvas transforms."""
        import math
        
        # Build transform: translate -> rotate around pivot -> scale
        angle_rad = math.radians(self.rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Combined matrix: [[a, c, e], [b, d, f]]
        # For scale * rotation * translation
        a = self.scale_x * cos_a
        b = self.scale_x * sin_a
        c = -self.scale_y * sin_a
        d = self.scale_y * cos_a
        e = self.translate_x
        f = self.translate_y
        
        return [[a, c, e], [b, d, f]]


# ============================================================
# Layer Models
# ============================================================

class RGBAColor(BaseModel):
    """RGBA color value."""
    r: int = Field(ge=0, le=255, description="Red channel (0-255)")
    g: int = Field(ge=0, le=255, description="Green channel (0-255)")
    b: int = Field(ge=0, le=255, description="Blue channel (0-255)")
    a: float = Field(default=1.0, ge=0.0, le=1.0, description="Alpha channel (0-1)")
    
    def to_css_rgba(self) -> str:
        """Convert to CSS rgba() string."""
        return f"rgba({self.r}, {self.g}, {self.b}, {self.a})"
    
    def to_hex(self) -> str:
        """Convert to hex string (without alpha)."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class LayerInfo(BaseModel):
    """Information about a single layer in the composition."""
    id: str = Field(description="Unique layer identifier")
    name: str = Field(description="Human-readable layer name")
    source: LayerSource = Field(description="Which input pair this layer comes from")
    type: LayerType = Field(description="Type of layer content")
    
    # Image data - either URL or base64
    png_url: Optional[str] = Field(default=None, description="URL to PNG asset")
    png_base64: Optional[str] = Field(default=None, description="Base64-encoded PNG data")
    
    # Layer dimensions
    width: int = Field(description="Layer width in pixels")
    height: int = Field(description="Layer height in pixels")
    
    # Default styling
    default_color: Optional[str] = Field(
        default=None, 
        description="Default color palette key or hex (e.g., 'red', '#FF0000')"
    )
    default_opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @property
    def has_image(self) -> bool:
        """Check if layer has image data."""
        return self.png_url is not None or self.png_base64 is not None


# ============================================================
# Composition State Models
# ============================================================

class LayerState(BaseModel):
    """Runtime state for a layer in the composition."""
    id: str = Field(description="Layer ID reference")
    visible: bool = Field(default=True, description="Layer visibility")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Layer opacity")
    blend_mode: BlendMode = Field(default=BlendMode.NORMAL, description="Blend mode")
    color: Optional[str] = Field(default=None, description="Tint color (hex or palette key)")
    transform: Transform2D = Field(default_factory=Transform2D, description="Layer transform")
    locked: bool = Field(default=False, description="Layer locked for editing")


class CanvasState(BaseModel):
    """Canvas/document configuration."""
    width: int = Field(description="Canvas width in pixels")
    height: int = Field(description="Canvas height in pixels")
    dpi: int = Field(default=250, description="Document DPI")
    background_color: str = Field(default="#FFFFFF", description="Canvas background color")
    coordinate_system: str = Field(default="top_left_origin", description="Coordinate system")


class AlignmentState(BaseModel):
    """Auto-computed alignment information."""
    method: str = Field(default="auto", description="Alignment method used (auto, manual, identity)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Alignment confidence score")
    
    # Transform from pair2 to pair1 coordinate space
    transform: Transform2D = Field(default_factory=Transform2D, description="Computed alignment transform")
    
    # Diagnostic info
    phase_response: Optional[float] = Field(default=None, description="Phase correlation response")
    overlap_ratio: Optional[float] = Field(default=None, description="Bounding box overlap ratio")
    refinement_method: Optional[str] = Field(default=None, description="Feature refinement method used")
    warnings: List[str] = Field(default_factory=list, description="Alignment warnings")


class ColorPalette(BaseModel):
    """Predefined color palette."""
    name: str
    colors: Dict[str, str] = Field(description="Named colors: {name: hex}")


class CompositionDefaults(BaseModel):
    """Default settings for the composition."""
    palette: ColorPalette = Field(
        default_factory=lambda: ColorPalette(
            name="default",
            colors={
                "red": "#FF0000",
                "cyan": "#00FFFF",
                "blue": "#0000FF",
                "green": "#00FF00",
                "magenta": "#FF00FF",
                "yellow": "#FFFF00",
                "black": "#000000",
                "white": "#FFFFFF",
            }
        )
    )
    snap_angles: List[float] = Field(default=[0, 90, 180, 270], description="Snap rotation angles")
    snap_enabled: bool = Field(default=True, description="Enable snapping")
    grid_size: int = Field(default=10, description="Grid snap size in pixels")


class CompositionState(BaseModel):
    """
    Complete non-destructive composition state.
    
    This JSON structure contains everything needed for the frontend
    to render and edit the overlay without any backend dependency.
    """
    version: str = Field(default="1.0.0", description="State schema version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Canvas configuration
    canvas: CanvasState
    
    # Layer ordering and states (first = bottom, last = top)
    layers: List[LayerState] = Field(default_factory=list, description="Ordered layer states")
    
    # Auto-computed alignment
    alignment: AlignmentState = Field(default_factory=AlignmentState)
    
    # Default settings
    defaults: CompositionDefaults = Field(default_factory=CompositionDefaults)


# ============================================================
# API Request/Response Models
# ============================================================

class ComposeRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{session_id}/compose."""
    auto: bool = Field(default=True, description="Attempt automatic alignment")
    rotation_constraint: RotationConstraint = Field(
        default=RotationConstraint.RIGHT_ANGLES,
        description="Rotation constraint for auto-alignment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (generates internal overlay.png, but never returned)"
    )
    
    # Optional manual transform override
    manual_transform: Optional[Transform2D] = Field(
        default=None,
        description="Manual transform override (used if auto=false)"
    )


class ComposeStatus(str, Enum):
    """Status of compose operation."""
    SUCCESS = "success"
    AUTO_FAILED_FALLBACK_MANUAL = "auto_failed_fallback_manual"


class ComposeResponse(BaseModel):
    """
    Response from POST /api/v1/sessions/{session_id}/compose.
    
    IMPORTANT: This API NEVER returns a flattened overlay image.
    It always returns layers[] + state.json for frontend rendering.
    """
    session_id: str
    status: ComposeStatus
    confidence: float = Field(description="Alignment confidence (0-1)")
    
    # Layer assets (always present)
    layers: List[LayerInfo] = Field(description="Transparent PNG layers for compositing")
    
    # Composition state (always present)
    state: CompositionState = Field(description="Non-destructive composition state")
    
    # Bounds information for canvas sizing and export cropping
    bounds: Optional[LayerBoundsInfo] = Field(
        default=None,
        description="Bounding boxes for canvas (contentBounds) and export (geometryBounds)"
    )
    
    # Warnings (may be present on success or fallback)
    warnings: List[str] = Field(default_factory=list)
    
    # Failure reason (only present when status=auto_failed_fallback_manual)
    reason: Optional[str] = Field(
        default=None,
        description="Reason for auto-alignment failure (if applicable)"
    )
    
    # Processing metadata
    processing_time_ms: int = Field(default=0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Frontend State Persistence
# ============================================================

class SavedEditorState(BaseModel):
    """
    State saved by the frontend editor.
    
    This can be sent back to the backend for re-rendering or
    downloaded as state.json by the user.
    """
    version: str = Field(default="1.0.0")
    session_id: str
    
    # Current composition state
    state: CompositionState
    
    # Edit history (optional)
    history_index: int = Field(default=0, description="Current position in undo/redo stack")
    
    # User preferences
    zoom_level: float = Field(default=1.0)
    pan_x: float = Field(default=0.0)
    pan_y: float = Field(default=0.0)
    
    # Metadata
    saved_at: datetime = Field(default_factory=datetime.utcnow)
