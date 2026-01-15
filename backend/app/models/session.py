"""
Session and internal data models.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import json


class SessionStatus(str, Enum):
    """Session processing status."""
    UPLOADED = "uploaded"
    RASTERIZING = "rasterizing"
    PREVIEW_READY = "preview_ready"
    ALIGNING = "aligning"
    ALIGNED = "aligned"
    RENDERING = "rendering"
    COMPLETE = "complete"
    ERROR = "error"


class FileType(str, Enum):
    """Type of uploaded file."""
    PDF = "pdf"
    PNG = "png"


class FileInfo(BaseModel):
    """Information about an uploaded file."""
    id: str
    original_filename: str
    size_bytes: int
    storage_path: str
    uploaded_at: datetime
    file_type: FileType = FileType.PDF  # Default to PDF for backward compatibility


class CropMetadata(BaseModel):
    """Metadata about the cropping operation."""
    original_width: int
    original_height: int
    content_bbox: dict = Field(description="Bounding box of detected content: {x, y, width, height}")
    final_crop_bbox: dict = Field(description="Final crop box with padding: {x, y, width, height}")
    cropped_width: int
    cropped_height: int
    threshold_used: int
    padding_applied: int
    whitespace_ratio: float = Field(description="Proportion of original that was whitespace")


class PageMetadata(BaseModel):
    """Metadata about PDF page and rasterization."""
    page_number: int = 0
    pdf_width_pt: float
    pdf_height_pt: float
    dpi: int
    raster_width_px: int
    raster_height_px: int
    render_time_ms: int


class PreviewInfo(BaseModel):
    """Information about a generated preview."""
    image_id: str
    storage_path: str
    width_px: int
    height_px: int
    dpi: int
    page_metadata: Optional[PageMetadata] = None  # Only present for PDF files
    crop_metadata: CropMetadata
    generated_at: datetime


class AnchorPoint(BaseModel):
    """A 2D anchor point."""
    x: float
    y: float


class TransformParams(BaseModel):
    """Parameters of a similarity transform."""
    scale: float
    rotation_deg: float
    rotation_rad: float
    tx: float
    ty: float


class AlignmentInfo(BaseModel):
    """Information about computed alignment."""
    transform_type: str = "similarity"
    matrix_2x3: List[List[float]] = Field(description="2x3 affine matrix [[a,b,tx],[c,d,ty]]")
    params: TransformParams
    confidence: float
    residual_error: float
    
    # Input metadata
    coordinate_space: str = "CROPPED_PREVIEW_PIXELS"
    points_a: List[AnchorPoint]
    points_b: List[AnchorPoint]
    
    # Reference dimensions
    reference_width: int = Field(description="Width of image A (reference)")
    reference_height: int = Field(description="Height of image A (reference)")
    target_width: int = Field(description="Width of image B (to be warped)")
    target_height: int = Field(description="Height of image B (to be warped)")
    
    computed_at: datetime


class OverlayRenderSettings(BaseModel):
    """Settings used to render overlay."""
    color_a: str = "red"
    color_b: str = "cyan"
    alpha_a: float = 0.7
    alpha_b: float = 0.7
    line_threshold: int = 240
    background: str = "white"


class OverlayInfo(BaseModel):
    """Information about a generated overlay."""
    image_id: str
    storage_path: str
    width_px: int
    height_px: int
    render_settings: OverlayRenderSettings
    generated_at: datetime
    processing_time_ms: int


class Session(BaseModel):
    """Session state model - persisted as JSON."""
    session_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    
    file_a: Optional[FileInfo] = None
    file_b: Optional[FileInfo] = None
    
    preview_a: Optional[PreviewInfo] = None
    preview_b: Optional[PreviewInfo] = None
    
    # Alignment data (v0.4.0)
    alignment: Optional[AlignmentInfo] = None
    
    # Overlay data (v0.4.0)
    overlay: Optional[OverlayInfo] = None
    
    error_message: Optional[str] = None
    
    def save(self, path: Path) -> None:
        """Save session to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "Session":
        """Load session from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)
