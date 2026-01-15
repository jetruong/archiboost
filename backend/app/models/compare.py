"""
Models for the compare/summarize endpoint.

Defines request and response schemas for difference detection and summarization.
"""

from datetime import datetime
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class DiffRegionModel(BaseModel):
    """A detected difference region."""
    id: int = Field(description="Region ID (1-based)")
    bbox: Tuple[int, int, int, int] = Field(description="Bounding box (x, y, width, height)")
    area: int = Field(description="Area in pixels")
    centroid: Tuple[float, float] = Field(description="Centroid coordinates (x, y)")


class SummarizeRequest(BaseModel):
    """Request body for POST /api/v1/compare/summarize."""
    a_id: str = Field(description="File ID or session ID for Drawing A")
    a_page: int = Field(default=1, description="Page number for Drawing A (1-based)")
    b_id: str = Field(description="File ID or session ID for Drawing B")
    b_page: int = Field(default=1, description="Page number for Drawing B (1-based)")
    
    # Optional settings
    min_region_area: int = Field(default=50, description="Minimum region area to detect")
    include_summary: bool = Field(default=True, description="Whether to include AI summary")
    
    class Config:
        json_schema_extra = {
            "example": {
                "a_id": "abc123",
                "a_page": 1,
                "b_id": "def456",
                "b_page": 1,
                "min_region_area": 50,
                "include_summary": True,
            }
        }


class SummarizeResponse(BaseModel):
    """Response from POST /api/v1/compare/summarize."""
    annotated_overlay_url: str = Field(description="URL to the annotated overlay image")
    diff_mask_url: str = Field(description="URL to the difference mask image")
    regions: List[DiffRegionModel] = Field(description="List of detected difference regions")
    summary: Optional[str] = Field(default=None, description="AI-generated summary of differences")
    
    # Processing metadata
    processing_time_ms: int = Field(default=0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Statistics
    stats: dict = Field(default_factory=dict, description="Processing statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "annotated_overlay_url": "/api/v1/images/overlay_abc123.png",
                "diff_mask_url": "/api/v1/images/diff_abc123.png",
                "regions": [
                    {"id": 1, "bbox": [100, 200, 50, 30], "area": 1500, "centroid": [125.0, 215.0]},
                    {"id": 2, "bbox": [300, 150, 40, 40], "area": 1200, "centroid": [320.0, 170.0]},
                ],
                "summary": "Region 1: Possible addition of a new fixture near the center-left. Region 2: Minor modification to wall segment at top-right. Note: Some differences may be due to line weight variations.",
                "processing_time_ms": 1250,
                "stats": {
                    "total_regions": 2,
                    "total_diff_pixels": 2700,
                    "diff_percentage": 0.15,
                }
            }
        }
