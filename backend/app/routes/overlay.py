"""
Overlay generation endpoint.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models.session import (
    SessionStatus,
    TransformParams,
    OverlayInfo,
    OverlayRenderSettings,
)
from app.models.responses import (
    OverlayResponse,
    TransformResponse,
    OverlayDimensions,
    ErrorResponse,
)
from app.services.storage import storage_service
from app.services.align import SimilarityTransform
from app.services.overlay import overlay_service, OverlayConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["overlay"])


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
    "/{session_id}/overlay",
    response_model=OverlayResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        409: {"model": ErrorResponse, "description": "Alignment required first"},
    },
)
async def get_overlay(session_id: str) -> OverlayResponse:
    """
    Generate and return an overlay visualization.
    
    Creates a composite image where:
    - Drawing A linework is tinted RED
    - Drawing B linework (after alignment warp) is tinted CYAN
    
    Requires alignment to be computed first via POST /align.
    """
    start_time = time.time()
    
    logger.info(f"Overlay request for session {session_id}")
    
    session = get_session_or_404(session_id)
    
    # Check alignment is ready
    if not session.alignment:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "ALIGNMENT_REQUIRED",
                "message": "Alignment must be computed before generating overlay. "
                           "Call POST /api/v1/sessions/{session_id}/align first.",
            },
        )
    
    # Check previews exist
    if not session.preview_a or not session.preview_b:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "PREVIEWS_NOT_READY",
                "message": "Both previews must exist before generating overlay",
            },
        )
    
    # Check if overlay already exists (cache)
    if session.overlay:
        overlay_path = Path(session.overlay.storage_path)
        if overlay_path.exists():
            logger.info(f"Returning cached overlay for session {session_id}")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return OverlayResponse(
                session_id=session_id,
                overlay_image_id=session.overlay.image_id,
                overlay_url=f"/api/v1/images/{session.overlay.image_id}",
                transform=TransformResponse(
                    transform_type=session.alignment.transform_type,
                    matrix_2x3=session.alignment.matrix_2x3,
                    params=session.alignment.params,
                ),
                rendering=session.overlay.render_settings,
                dimensions=OverlayDimensions(
                    width=session.overlay.width_px,
                    height=session.overlay.height_px,
                ),
                status=session.status,
                processing_time_ms=processing_time,
                generated_at=session.overlay.generated_at,
            )
    
    # Load preview images
    preview_a_path = Path(session.preview_a.storage_path)
    preview_b_path = Path(session.preview_b.storage_path)
    
    if not preview_a_path.exists() or not preview_b_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "PREVIEW_FILES_MISSING",
                "message": "Preview image files not found on disk",
            },
        )
    
    image_a = cv2.imread(str(preview_a_path))
    image_b = cv2.imread(str(preview_b_path))
    
    if image_a is None or image_b is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "PREVIEW_LOAD_FAILED",
                "message": "Failed to load preview images",
            },
        )
    
    # Reconstruct transform from stored parameters
    params = session.alignment.params
    transform = SimilarityTransform(
        scale=params.scale,
        rotation_rad=params.rotation_rad,
        tx=params.tx,
        ty=params.ty,
    )
    
    # Configure overlay rendering
    config = OverlayConfig(
        color_a=(0, 0, 255),      # Red in BGR
        color_b=(255, 255, 0),    # Cyan in BGR
        alpha_a=0.7,
        alpha_b=0.7,
        background_color=(255, 255, 255),
        line_threshold=240,
    )
    
    # Generate overlay
    try:
        result = overlay_service.generate_overlay(
            image_a=image_a,
            image_b=image_b,
            transform=transform,
            config=config,
        )
    except Exception as e:
        logger.error(f"Overlay generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "OVERLAY_GENERATION_FAILED",
                "message": f"Failed to generate overlay: {str(e)}",
            },
        )
    
    # Save overlay image
    session_dir = storage_service.get_session_dir(session_id)
    overlay_dir = session_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    overlay_filename = f"overlay_{session_id[:8]}.png"
    overlay_path = overlay_dir / overlay_filename
    
    cv2.imwrite(str(overlay_path), result.overlay_image)
    
    # Also save diff mask for future analysis
    diff_path = overlay_dir / f"diff_{session_id[:8]}.png"
    if result.diff_mask is not None:
        cv2.imwrite(str(diff_path), result.diff_mask)
    
    # Generate image ID
    image_id = f"overlay_{session_id[:8]}"
    
    now = datetime.now(timezone.utc)
    
    # Update session
    render_settings = OverlayRenderSettings(
        color_a="red",
        color_b="cyan",
        alpha_a=config.alpha_a,
        alpha_b=config.alpha_b,
        line_threshold=config.line_threshold,
        background="white",
    )
    
    overlay_info = OverlayInfo(
        image_id=image_id,
        storage_path=str(overlay_path),
        width_px=result.width,
        height_px=result.height,
        render_settings=render_settings,
        generated_at=now,
        processing_time_ms=result.processing_time_ms,
    )
    
    session.overlay = overlay_info
    session.status = SessionStatus.COMPLETE
    storage_service.save_session(session)
    
    total_time = int((time.time() - start_time) * 1000)
    
    logger.info(f"Overlay generated for session {session_id} in {total_time}ms")
    
    return OverlayResponse(
        session_id=session_id,
        overlay_image_id=image_id,
        overlay_url=f"/api/v1/images/{image_id}",
        transform=TransformResponse(
            transform_type=session.alignment.transform_type,
            matrix_2x3=session.alignment.matrix_2x3,
            params=session.alignment.params,
        ),
        rendering=render_settings,
        dimensions=OverlayDimensions(
            width=result.width,
            height=result.height,
        ),
        status=session.status,
        processing_time_ms=total_time,
        generated_at=now,
    )
