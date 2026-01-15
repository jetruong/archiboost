"""
API response models.
"""

from datetime import datetime
from typing import Optional, Any, List
from pydantic import BaseModel, Field

from app.models.session import (
    SessionStatus, 
    CropMetadata, 
    PageMetadata,
    TransformParams,
    AnchorPoint,
)


class FileUploadInfo(BaseModel):
    """File info returned in upload response."""
    id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    """Response from POST /api/v1/upload."""
    session_id: str
    file_a: FileUploadInfo
    file_b: FileUploadInfo
    status: SessionStatus
    created_at: datetime
    expires_at: datetime


class PreviewResponse(BaseModel):
    """Response from GET /api/v1/sessions/{session_id}/preview."""
    session_id: str
    which: str = Field(description="Which file: 'A' or 'B'")
    image_id: str
    image_url: str
    width_px: int
    height_px: int
    dpi: int
    page_metadata: Optional[PageMetadata] = Field(
        default=None, 
        description="PDF page metadata. Only present for PDF files, null for PNG inputs."
    )
    crop_metadata: CropMetadata
    processing_time_ms: int


class PreviewSummary(BaseModel):
    """Summary of a preview for session response."""
    image_id: str
    image_url: str
    width_px: int
    height_px: int
    generated: bool = True


class AlignmentSummary(BaseModel):
    """Summary of alignment for session response."""
    transform_type: str
    scale: float
    rotation_deg: float
    confidence: float
    computed: bool = True


class OverlaySummary(BaseModel):
    """Summary of overlay for session response."""
    image_id: str
    image_url: str
    width_px: int
    height_px: int
    generated: bool = True


class SessionResponse(BaseModel):
    """Response from GET /api/v1/sessions/{session_id}."""
    session_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    ttl_seconds: int
    
    file_a: Optional[FileUploadInfo] = None
    file_b: Optional[FileUploadInfo] = None
    
    preview_a: Optional[PreviewSummary] = None
    preview_b: Optional[PreviewSummary] = None
    
    alignment: Optional[AlignmentSummary] = None
    overlay: Optional[OverlaySummary] = None
    
    error_message: Optional[str] = None


# ============================================================
# Alignment Request/Response Models
# ============================================================

class AlignRequest(BaseModel):
    """
    Request body for POST /api/v1/sessions/{session_id}/align.
    
    Two modes:
    1. AUTO mode: Set auto=true - automatically detects and matches features
    2. MANUAL mode: Provide points_a and points_b manually
    
    For manual mode, place anchor points on structural drawing features
    (wall corners, line intersections, etc.) NOT on text/labels.
    """
    auto: bool = Field(
        default=False,
        description="If true, automatically detect anchor points using feature matching. "
                    "If false, points_a and points_b must be provided."
    )
    points_a: Optional[List[AnchorPoint]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Two anchor points in image A (cropped preview coordinates). "
                    "Required if auto=false. Place on walls/lines, NOT labels."
    )
    points_b: Optional[List[AnchorPoint]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Two corresponding anchor points in image B (cropped preview coordinates). "
                    "Required if auto=false. Must correspond to the same features as points_a."
    )
    coordinate_space: str = Field(
        default="CROPPED_PREVIEW_PIXELS",
        description="Coordinate space for points. Must be CROPPED_PREVIEW_PIXELS."
    )
    rotation_constraint: str = Field(
        default="SNAP_90",
        description="Rotation constraint: SNAP_90 (try 0°, 90°, 180°, 270° - default), "
                    "NONE (no rotation), FREE (any angle). "
                    "SNAP_90 automatically handles rotated drawings."
    )


class TransformResponse(BaseModel):
    """Transform details in alignment response."""
    transform_type: str = "similarity"
    matrix_2x3: List[List[float]] = Field(description="2x3 affine matrix [[a,b,tx],[c,d,ty]]")
    params: TransformParams


class AlignMetadata(BaseModel):
    """Metadata about images used in alignment."""
    dpi: int
    image_a_size: dict = Field(description="{width, height} of cropped preview A")
    image_b_size: dict = Field(description="{width, height} of cropped preview B")
    crop_a: dict = Field(description="Crop metadata for A")
    crop_b: dict = Field(description="Crop metadata for B")


class AutoAlignDebugResponse(BaseModel):
    """Debug metadata from auto alignment v2 pipeline."""
    rotation_candidates_evaluated: List[float] = Field(description="Rotation angles evaluated (degrees)")
    rotation_candidate_scores: dict = Field(description="Score for each rotation candidate")
    rotation_candidate_used: float = Field(description="Rotation angle selected (degrees)")
    fine_rotation_applied: bool = Field(default=False, description="Whether fine rotation refinement was used")
    phase_response: float = Field(description="Phase correlation response (0-1)")
    phase_translation: List[float] = Field(description="Translation from phase correlation [tx, ty]")
    overlap_ratio: float = Field(description="Bounding box overlap ratio (0-1)")
    bbox_a: Optional[List[int]] = Field(description="Bounding box of A [x, y, w, h]")
    bbox_b_warped: Optional[List[int]] = Field(description="Bounding box of warped B [x, y, w, h]")
    refinement_method_used: str = Field(description="Feature refinement method (NONE, AKAZE, SIFT, ORB)")
    refinement_matches: int = Field(description="Number of feature matches found")
    refinement_inliers: int = Field(description="Number of RANSAC inliers")
    scale_source: str = Field(description="How scale was determined (default, feature)")
    final_confidence: float = Field(description="Final confidence score")
    confidence_breakdown: dict = Field(description="Breakdown of confidence components")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection if failed")
    guardrail_violations: List[str] = Field(default_factory=list, description="List of guardrail violations")


class AlignResponse(BaseModel):
    """Response from POST /api/v1/sessions/{session_id}/align."""
    session_id: str
    transform: TransformResponse
    confidence: float = Field(description="Alignment confidence score (0.0-1.0)")
    residual_error: float = Field(description="RMS error in pixels after mapping")
    coordinate_space: str = "CROPPED_PREVIEW_PIXELS"
    points_a: List[AnchorPoint]
    points_b: List[AnchorPoint]
    metadata: AlignMetadata
    status: SessionStatus
    computed_at: datetime
    auto_debug: Optional[AutoAlignDebugResponse] = Field(
        default=None,
        description="Debug metadata from auto alignment pipeline (only present when auto=true)"
    )


# ============================================================
# Error Models
# ============================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail
