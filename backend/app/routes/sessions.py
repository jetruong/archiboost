"""
Session management endpoints.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, HTTPException, Query, status

from app.config import settings
from app.models.responses import (
    PreviewResponse,
    SessionResponse,
    PreviewSummary,
    FileUploadInfo,
    ErrorResponse,
    AlignmentSummary,
    OverlaySummary,
)
from app.models.session import SessionStatus, PreviewInfo, FileType
from app.services.storage import storage_service
from app.services.rasterize import rasterize_service
from app.services.crop import crop_service

import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def get_session_or_404(session_id: str):
    """Load session or raise 404."""
    session = storage_service.load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": f"Session '{session_id}' does not exist or has expired",
            },
        )
    return session


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get the current status and metadata of a session.
    """
    session = get_session_or_404(session_id)
    
    # Calculate TTL
    now = datetime.now(timezone.utc)
    ttl_seconds = max(0, int((session.expires_at - now).total_seconds()))
    
    # Build preview summaries if available
    preview_a = None
    preview_b = None
    alignment = None
    overlay = None
    
    if session.preview_a:
        preview_a = PreviewSummary(
            image_id=session.preview_a.image_id,
            image_url=f"/api/v1/images/{session.preview_a.image_id}",
            width_px=session.preview_a.width_px,
            height_px=session.preview_a.height_px,
            generated=True,
        )
    
    if session.preview_b:
        preview_b = PreviewSummary(
            image_id=session.preview_b.image_id,
            image_url=f"/api/v1/images/{session.preview_b.image_id}",
            width_px=session.preview_b.width_px,
            height_px=session.preview_b.height_px,
            generated=True,
        )
    
    # Build alignment summary if available
    if session.alignment:
        alignment = AlignmentSummary(
            transform_type=session.alignment.transform_type,
            scale=session.alignment.params.scale,
            rotation_deg=session.alignment.params.rotation_deg,
            confidence=session.alignment.confidence,
            computed=True,
        )
    
    # Build overlay summary if available
    if session.overlay:
        overlay = OverlaySummary(
            image_id=session.overlay.image_id,
            image_url=f"/api/v1/images/{session.overlay.image_id}",
            width_px=session.overlay.width_px,
            height_px=session.overlay.height_px,
            generated=True,
        )
    
    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at,
        expires_at=session.expires_at,
        ttl_seconds=ttl_seconds,
        file_a=FileUploadInfo(
            id=session.file_a.id,
            filename=session.file_a.original_filename,
            size_bytes=session.file_a.size_bytes,
            uploaded_at=session.file_a.uploaded_at,
        ) if session.file_a else None,
        file_b=FileUploadInfo(
            id=session.file_b.id,
            filename=session.file_b.original_filename,
            size_bytes=session.file_b.size_bytes,
            uploaded_at=session.file_b.uploaded_at,
        ) if session.file_b else None,
        preview_a=preview_a,
        preview_b=preview_b,
        alignment=alignment,
        overlay=overlay,
        error_message=session.error_message,
    )


@router.get(
    "/{session_id}/preview",
    response_model=PreviewResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
)
async def get_preview(
    session_id: str,
    which: Literal["A", "B"] = Query(..., description="Which file to preview: 'A' or 'B'"),
    dpi: int = Query(default=None, ge=72, le=600, description="Rasterization DPI"),
) -> PreviewResponse:
    """
    Get a rasterized and cropped preview image for one of the uploaded PDFs.
    
    The preview is generated on first request and cached for subsequent requests.
    """
    start_time = time.time()
    
    session = get_session_or_404(session_id)
    dpi = dpi or settings.default_dpi
    
    which_lower = which.lower()
    which_upper = which.upper()
    
    # Check if preview already exists
    existing_preview = getattr(session, f"preview_{which_lower}")
    if existing_preview and existing_preview.dpi == dpi:
        # Return cached preview
        preview_path = storage_service.get_preview_path(session_id, which_lower)
        if preview_path.exists():
            processing_time = int((time.time() - start_time) * 1000)
            return PreviewResponse(
                session_id=session_id,
                which=which_upper,
                image_id=existing_preview.image_id,
                image_url=f"/api/v1/images/{existing_preview.image_id}",
                width_px=existing_preview.width_px,
                height_px=existing_preview.height_px,
                dpi=existing_preview.dpi,
                page_metadata=existing_preview.page_metadata,
                crop_metadata=existing_preview.crop_metadata,
                processing_time_ms=processing_time,
            )
    
    # Get input file path
    file_info = getattr(session, f"file_{which_lower}")
    if not file_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_FILE",
                "message": f"File {which_upper} has not been uploaded",
            },
        )
    
    input_path = Path(file_info.storage_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "PROCESSING_ERROR",
                "message": "Source file not found",
            },
        )
    
    try:
        page_metadata = None
        
        if file_info.file_type == FileType.PNG:
            # Load PNG directly (already an image)
            logger.info(f"Loading PNG {which_upper} for session {session_id}")
            bgr_image = cv2.imread(str(input_path))
            if bgr_image is None:
                raise ValueError(f"Failed to load PNG image: {input_path}")
            # Convert BGR to RGB for consistency with PDF path
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        else:
            # Rasterize PDF
            logger.info(f"Rasterizing PDF {which_upper} for session {session_id}")
            raster_result = rasterize_service.rasterize_page(input_path, page_number=0, dpi=dpi)
            rgb_image = raster_result.image
            page_metadata = raster_result.page_metadata
        
        # Crop whitespace (applies to both PNG and PDF)
        logger.info(f"Cropping whitespace from {which_upper}")
        crop_result = crop_service.crop_whitespace(rgb_image)
        
        # Save preview image
        preview_path = storage_service.get_preview_path(session_id, which_lower)
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(crop_result.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(preview_path), bgr_image)
        
        logger.info(f"Saved preview to {preview_path}")
        
        # Generate image ID
        image_id = storage_service.generate_image_id(session_id, which_lower)
        
        # Create preview info
        preview_info = PreviewInfo(
            image_id=image_id,
            storage_path=str(preview_path),
            width_px=crop_result.metadata.cropped_width,
            height_px=crop_result.metadata.cropped_height,
            dpi=dpi,
            page_metadata=page_metadata,  # None for PNG files
            crop_metadata=crop_result.metadata,
            generated_at=datetime.now(timezone.utc),
        )
        
        # Update session
        setattr(session, f"preview_{which_lower}", preview_info)
        
        # Update status if both previews are ready
        if session.preview_a and session.preview_b:
            session.status = SessionStatus.PREVIEW_READY
        
        storage_service.save_session(session)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return PreviewResponse(
            session_id=session_id,
            which=which_upper,
            image_id=image_id,
            image_url=f"/api/v1/images/{image_id}",
            width_px=preview_info.width_px,
            height_px=preview_info.height_px,
            dpi=dpi,
            page_metadata=page_metadata,
            crop_metadata=crop_result.metadata,
            processing_time_ms=processing_time,
        )
        
    except ValueError as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "RASTERIZATION_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "PROCESSING_ERROR",
                "message": "Failed to generate preview",
                "details": {"error": str(e)},
            },
        )
