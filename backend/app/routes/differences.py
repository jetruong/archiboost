"""
Differences API endpoint.

Generates AI-powered summaries of differences between architectural drawings.
Uses the Gemini service for intelligent analysis of detected changes.

IMPORTANT: This endpoint uses the stored alignment to warp Drawing B into
Drawing A's coordinate space before computing differences. Without alignment,
rotated or offset drawings would show false positives everywhere.
"""

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.storage import storage_service
from app.services.diff import diff_detection_service, DiffRegion
from app.services.gemini import gemini_service
from app.services.align import SimilarityTransform

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["differences"])


# ============================================================
# Response Models
# ============================================================

class RegionInfo(BaseModel):
    """Information about a detected difference region."""
    id: int
    bbox: List[int]  # [x, y, width, height]
    area: int
    centroid: List[float]  # [x, y]


class TransformInput(BaseModel):
    """Transform input for manual alignment."""
    scale: float = 1.0
    rotation_deg: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0


class DifferencesRequest(BaseModel):
    """Request body for differences endpoint."""
    transform: Optional[TransformInput] = None  # Optional manual transform from frontend


class DifferencesResponse(BaseModel):
    """Response from differences endpoint."""
    session_id: str
    summary: str
    regions: List[RegionInfo]
    total_regions: int
    diff_percentage: float
    ai_available: bool
    model_name: Optional[str] = None
    model_display_name: Optional[str] = None
    is_vlm: bool = False
    processing_time_ms: int
    generated_at: datetime
    aligned: bool = True  # Whether alignment was applied
    alignment_source: str = "session"  # "session", "manual", or "identity"
    warnings: List[str] = []  # Any warnings about the diff process


# ============================================================
# Helper Functions
# ============================================================

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


def extract_linework(image: np.ndarray, threshold: int = 240) -> np.ndarray:
    """
    Extract binary linework mask from an image.
    
    Args:
        image: BGR image
        threshold: Pixel value below which is considered linework
    
    Returns:
        Binary mask where 255 = linework, 0 = background
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold to get linework (dark pixels)
    _, linework = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return linework


def rebuild_similarity_transform(matrix_2x3: List[List[float]]) -> SimilarityTransform:
    """
    Rebuild a SimilarityTransform from a stored 2x3 matrix.
    
    The matrix format is:
        [[s*cos(θ), -s*sin(θ), tx],
         [s*sin(θ),  s*cos(θ), ty]]
    
    Args:
        matrix_2x3: 2x3 affine matrix as nested list
    
    Returns:
        SimilarityTransform with extracted parameters
    """
    # Extract components from matrix
    a = matrix_2x3[0][0]  # s*cos(θ)
    b = matrix_2x3[0][1]  # -s*sin(θ)
    tx = matrix_2x3[0][2]
    c = matrix_2x3[1][0]  # s*sin(θ)
    d = matrix_2x3[1][1]  # s*cos(θ)
    ty = matrix_2x3[1][2]
    
    # Recover scale: s = sqrt(a² + c²) = sqrt((s*cos(θ))² + (s*sin(θ))²)
    scale = math.sqrt(a**2 + c**2)
    
    # Recover rotation: θ = atan2(c, a) = atan2(s*sin(θ), s*cos(θ))
    rotation_rad = math.atan2(c, a)
    
    return SimilarityTransform(
        scale=scale,
        rotation_rad=rotation_rad,
        tx=tx,
        ty=ty,
    )


def warp_linework(
    linework: np.ndarray,
    transform: SimilarityTransform,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """
    Warp a linework mask using a similarity transform.
    
    This is the same logic as overlay_service.warp_image but for binary masks.
    
    Args:
        linework: Binary linework mask to warp
        transform: Similarity transform (maps source -> destination)
        output_size: (width, height) of output image
    
    Returns:
        Warped linework mask
    """
    matrix = transform.matrix_2x3.astype(np.float32)
    
    # Warp using nearest neighbor for crisp binary edges
    warped = cv2.warpAffine(
        linework,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,  # Background = no linework
    )
    
    return warped


# ============================================================
# Differences Endpoint
# ============================================================

@router.post(
    "/{session_id}/differences",
    response_model=DifferencesResponse,
    responses={
        404: {"description": "Session not found"},
        409: {"description": "Previews not ready"},
    },
    summary="Generate AI-powered differences summary",
    description="""
Generate an AI-powered summary of differences between the two drawings.

This endpoint:
1. Loads the preview images for both drawings
2. **Warps Drawing B into Drawing A's coordinate space using the provided or stored alignment**
3. Detects difference regions using XOR comparison on the aligned linework
4. Uses Gemini AI to generate a human-readable summary
5. Returns the summary along with region information

**Alignment sources (in order of priority):**
1. `transform` in request body - Use this for manual adjustments from the frontend
2. `session.alignment` - Use stored alignment from compose/align endpoints
3. Identity transform - Fall back with warning if neither is available

If the Gemini API is unavailable, a fallback summary is provided.
""",
)
async def generate_differences(
    session_id: str,
    request: Optional[DifferencesRequest] = None,
) -> DifferencesResponse:
    """
    Generate AI-powered summary of differences between drawings.
    
    Accepts optional transform in request body for manual alignment.
    """
    start_time = time.time()
    warnings: List[str] = []
    aligned = True
    alignment_source = "session"
    
    logger.info(f"Differences request for session {session_id}")
    
    session = get_session_or_404(session_id)
    
    # Check previews are ready
    if not session.preview_a or not session.preview_b:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "PREVIEWS_NOT_READY",
                "message": "Both previews must be generated before analyzing differences",
                "details": {
                    "preview_a_ready": session.preview_a is not None,
                    "preview_b_ready": session.preview_b is not None,
                },
            },
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
    
    # Extract linework from both images
    linework_a = extract_linework(image_a)
    linework_b = extract_linework(image_b)
    
    # Determine transform source (priority: request > session > identity)
    transform: SimilarityTransform
    
    if request and request.transform:
        # Use transform from request (manual adjustment from frontend)
        t = request.transform
        transform = SimilarityTransform(
            scale=t.scale,
            rotation_rad=math.radians(t.rotation_deg),
            tx=t.translate_x,
            ty=t.translate_y,
        )
        alignment_source = "manual"
        logger.info(f"Using manual transform from request")
        
    elif session.alignment:
        # Use stored alignment from compose/align endpoint
        transform = rebuild_similarity_transform(session.alignment.matrix_2x3)
        alignment_source = "session"
        logger.info(f"Using stored alignment from session")
        
    else:
        # Fall back to identity transform with warning
        transform = SimilarityTransform(
            scale=1.0,
            rotation_rad=0.0,
            tx=0.0,
            ty=0.0,
        )
        alignment_source = "identity"
        aligned = False
        warnings.append(
            "No alignment available. Using identity transform. "
            "Results may be inaccurate if drawings are not perfectly aligned. "
            "Consider running compose or align endpoint first, or provide a manual transform."
        )
        logger.warning(f"No alignment available, using identity transform")
    
    logger.info(
        f"Applying alignment transform ({alignment_source}): scale={transform.scale:.4f}, "
        f"rotation={transform.rotation_deg:.2f}°, "
        f"translation=({transform.tx:.1f}, {transform.ty:.1f})"
    )
    
    # Output size is A's dimensions (A is the reference frame)
    height_a, width_a = linework_a.shape[:2]
    output_size = (width_a, height_a)
    
    # Warp linework_b into A's coordinate space
    linework_b_warped = warp_linework(linework_b, transform, output_size)
    
    # Log pixel counts for debugging
    pixels_a = np.sum(linework_a > 0)
    pixels_b_original = np.sum(linework_b > 0)
    pixels_b_warped = np.sum(linework_b_warped > 0)
    
    logger.info(
        f"Linework pixels: A={pixels_a}, B_original={pixels_b_original}, "
        f"B_warped={pixels_b_warped}"
    )
    
    # Check if transform is near-identity (might indicate alignment wasn't meaningful)
    if (abs(transform.scale - 1.0) < 0.001 and 
        abs(transform.rotation_deg) < 0.1 and
        abs(transform.tx) < 1.0 and abs(transform.ty) < 1.0):
        warnings.append(
            "Alignment transform is near-identity. If drawings appear misaligned, "
            "re-run alignment with better anchor points."
        )
    
    # Detect differences using the WARPED linework_b
    # Pass base_image=None to use composite overlay (A=red, B=cyan)
    # This gives Gemini better context to understand the differences
    diff_result = diff_detection_service.detect_differences(
        linework_a=linework_a,
        linework_b=linework_b_warped,  # Use warped B, not original!
        base_image=None,  # Use composite overlay for better AI analysis
        use_composite_overlay=True,
    )
    
    # Convert regions for API response
    region_dicts = [
        {
            "id": r.id,
            "bbox": list(r.bbox),
            "area": r.area,
            "centroid": list(r.centroid),
        }
        for r in diff_result.regions
    ]
    
    # Generate AI summary
    summary = await gemini_service.summarize_differences(
        annotated_overlay=diff_result.annotated_overlay,
        regions=region_dicts,
    )
    
    # Build response
    processing_time = int((time.time() - start_time) * 1000)
    now = datetime.now(timezone.utc)
    
    region_infos = [
        RegionInfo(
            id=r.id,
            bbox=list(r.bbox),
            area=r.area,
            centroid=list(r.centroid),
        )
        for r in diff_result.regions
    ]
    
    logger.info(
        f"Differences analysis complete for session {session_id}: "
        f"{len(region_infos)} regions, "
        f"{diff_result.stats['diff_percentage']:.2f}% difference, "
        f"AI available: {gemini_service.is_available}, "
        f"alignment_source: {alignment_source}"
    )
    
    return DifferencesResponse(
        session_id=session_id,
        summary=summary or "Unable to generate summary",
        regions=region_infos,
        total_regions=len(region_infos),
        diff_percentage=diff_result.stats['diff_percentage'],
        ai_available=gemini_service.is_available,
        model_name=gemini_service.model_name if gemini_service.is_available else None,
        model_display_name=gemini_service.model_display_name if gemini_service.is_available else None,
        is_vlm=gemini_service.is_vlm if gemini_service.is_available else False,
        processing_time_ms=processing_time,
        generated_at=now,
        aligned=aligned,
        alignment_source=alignment_source,
        warnings=warnings,
    )
