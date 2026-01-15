"""
Compose endpoint - Layer-based overlay API.

This endpoint NEVER returns a flattened overlay image.
It ALWAYS returns separated layers[] + state.json for frontend rendering.

The backend may internally generate overlay.png for testing/debugging,
but it is NEVER exposed via this API.
"""

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models.layers import (
    LayerInfo,
    LayerType,
    LayerSource,
    LayerState,
    Transform2D,
    AlignmentState,
    CompositionState,
    ComposeRequest,
    ComposeResponse,
    ComposeStatus,
    BlendMode,
    RotationConstraint as LayerRotationConstraint,
    BoundingBox,
    LayerBoundsInfo,
    CANVAS_PADDING_PX,
    EXPORT_MARGIN_PX,
)
from app.models.responses import ErrorResponse
from app.models.session import SessionStatus, AlignmentInfo, TransformParams, AnchorPoint
from app.services.storage import storage_service
from app.services.align import RotationConstraint as AlignRotationConstraint
from app.services.auto_align import auto_align_service, AutoAlignError
from app.services.layers import (
    layer_extraction_service,
    composition_service,
    bounds_computation_service,
)
from app.services.overlay import overlay_service, OverlayConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["compose"])


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


def _convert_rotation_constraint(constraint: LayerRotationConstraint) -> AlignRotationConstraint:
    """Convert layer rotation constraint to align service constraint."""
    mapping = {
        LayerRotationConstraint.NONE: AlignRotationConstraint.NONE,
        LayerRotationConstraint.RIGHT_ANGLES: AlignRotationConstraint.SNAP_90,
        LayerRotationConstraint.SMALL_ANGLE: AlignRotationConstraint.FREE,
    }
    return mapping.get(constraint, AlignRotationConstraint.SNAP_90)


@router.post(
    "/{session_id}/compose",
    response_model=ComposeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        409: {"model": ErrorResponse, "description": "Previews not ready"},
    },
    summary="Generate layer-based composition",
    description="""
Generate layers and composition state for frontend rendering.

**This API NEVER returns a flattened overlay image.**

It always returns:
- `layers[]`: Array of transparent PNG layers (base64 or URL)
- `state`: Complete non-destructive composition state (state.json)

**Auto-alignment behavior:**
- If `auto=true` (default): Attempts automatic alignment
- If auto-alignment fails: Returns `status="auto_failed_fallback_manual"` with identity transform
- The response is ALWAYS usable - the frontend can render and edit even if alignment failed

**The frontend should:**
1. Render layers using the composition state
2. Allow manual adjustment of transforms
3. Generate flattened output locally when needed
""",
)
async def compose_layers(
    session_id: str,
    request: ComposeRequest,
) -> ComposeResponse:
    """
    Generate layer-based composition for frontend rendering.
    
    Returns separated layers and composition state.
    NEVER returns a flattened overlay image via the API.
    """
    start_time = time.time()
    
    logger.info(
        f"Compose request for session {session_id}, "
        f"auto={request.auto}, rotation={request.rotation_constraint}"
    )
    
    session = get_session_or_404(session_id)
    
    # Check previews are ready
    if not session.preview_a or not session.preview_b:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "PREVIEWS_NOT_READY",
                "message": "Both previews must be generated before composing",
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
    
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    
    # Initialize result variables
    compose_status = ComposeStatus.SUCCESS
    confidence = 0.0
    warnings = []
    reason = None
    alignment_transform = Transform2D()  # Identity by default
    alignment_state = AlignmentState(method="identity", confidence=0.0)
    
    # Attempt auto-alignment if requested
    if request.auto:
        rotation_constraint = _convert_rotation_constraint(request.rotation_constraint)
        
        try:
            auto_result = auto_align_service.auto_align(
                image_a, image_b, rotation_constraint
            )
            
            # Successful auto-alignment
            confidence = auto_result.confidence
            
            # Convert transform
            alignment_transform = composition_service.transform_from_similarity(
                auto_result.transform,
                w_b,
                h_b,
            )
            
            alignment_state = AlignmentState(
                method="auto",
                confidence=confidence,
                transform=alignment_transform,
                phase_response=auto_result.debug.phase_response,
                overlap_ratio=auto_result.debug.overlap_ratio,
                refinement_method=auto_result.debug.refinement_method_used,
                warnings=[],
            )
            
            logger.info(
                f"Auto-alignment succeeded: confidence={confidence:.1%}, "
                f"scale={auto_result.transform.scale:.4f}, "
                f"rotation={auto_result.transform.rotation_deg:.2f}°"
            )
            
            # ============================================================
            # IMPORTANT: Save alignment to session for differences endpoint
            # ============================================================
            session.alignment = AlignmentInfo(
                transform_type="similarity",
                matrix_2x3=auto_result.transform.matrix_2x3_list,
                params=TransformParams(
                    scale=auto_result.transform.scale,
                    rotation_deg=auto_result.transform.rotation_deg,
                    rotation_rad=auto_result.transform.rotation_rad,
                    tx=auto_result.transform.tx,
                    ty=auto_result.transform.ty,
                ),
                confidence=confidence,
                residual_error=auto_result.residual_error,
                coordinate_space="CROPPED_PREVIEW_PIXELS",
                points_a=[AnchorPoint(x=p.x, y=p.y) for p in auto_result.matched_points_a[:2]],
                points_b=[AnchorPoint(x=p.x, y=p.y) for p in auto_result.matched_points_b[:2]],
                reference_width=w_a,
                reference_height=h_a,
                target_width=w_b,
                target_height=h_b,
                computed_at=datetime.now(timezone.utc),
            )
            session.status = SessionStatus.ALIGNED
            storage_service.save_session(session)
            logger.info(f"Alignment saved to session {session_id}")
            
        except AutoAlignError as e:
            # Auto-alignment failed - return fallback with usable response
            logger.warning(f"Auto-alignment failed: {e.code} - {e.message}")
            
            compose_status = ComposeStatus.AUTO_FAILED_FALLBACK_MANUAL
            confidence = 0.0
            reason = f"{e.code}: {e.message}"
            warnings.append("Auto-alignment failed. Manual adjustment required.")
            
            if e.debug:
                # Include diagnostic info from failed attempt
                alignment_state = AlignmentState(
                    method="identity",
                    confidence=0.0,
                    transform=Transform2D(),  # Identity
                    phase_response=e.debug.phase_response,
                    overlap_ratio=e.debug.overlap_ratio,
                    refinement_method=e.debug.refinement_method_used,
                    warnings=[e.debug.rejection_reason or "Unknown reason"],
                )
            else:
                alignment_state = AlignmentState(
                    method="identity",
                    confidence=0.0,
                    transform=Transform2D(),
                    warnings=[str(e.message)],
                )
    
    elif request.manual_transform:
        # Manual transform provided
        alignment_transform = request.manual_transform
        confidence = 1.0  # Manual is always "confident"
        
        alignment_state = AlignmentState(
            method="manual",
            confidence=1.0,
            transform=alignment_transform,
        )
        
        # Save manual alignment to session for differences endpoint
        session.alignment = AlignmentInfo(
            transform_type="similarity",
            matrix_2x3=[
                [alignment_transform.scale_x * math.cos(math.radians(alignment_transform.rotation_deg)),
                 -alignment_transform.scale_x * math.sin(math.radians(alignment_transform.rotation_deg)),
                 alignment_transform.translate_x],
                [alignment_transform.scale_x * math.sin(math.radians(alignment_transform.rotation_deg)),
                 alignment_transform.scale_x * math.cos(math.radians(alignment_transform.rotation_deg)),
                 alignment_transform.translate_y],
            ],
            params=TransformParams(
                scale=alignment_transform.scale_x,
                rotation_deg=alignment_transform.rotation_deg,
                rotation_rad=math.radians(alignment_transform.rotation_deg),
                tx=alignment_transform.translate_x,
                ty=alignment_transform.translate_y,
            ),
            confidence=1.0,
            residual_error=0.0,
            coordinate_space="CROPPED_PREVIEW_PIXELS",
            points_a=[AnchorPoint(x=0, y=0), AnchorPoint(x=100, y=100)],  # Placeholder
            points_b=[AnchorPoint(x=0, y=0), AnchorPoint(x=100, y=100)],  # Placeholder
            reference_width=w_a,
            reference_height=h_a,
            target_width=w_b,
            target_height=h_b,
            computed_at=datetime.now(timezone.utc),
        )
        session.status = SessionStatus.ALIGNED
        storage_service.save_session(session)
        logger.info(f"Manual alignment saved to session {session_id}")
    
    # ============================================================
    # Extract layers from both images
    # ============================================================
    
    layers: list[LayerInfo] = []
    layer_states: list[LayerState] = []
    
    # Extract layer from image A (pair1) - single layer per image
    extraction_a = layer_extraction_service.extract_layers(image_a, LayerSource.PAIR1)
    
    for extracted in extraction_a.layers:
        layer_id = f"{session_id[:8]}_a_{uuid4().hex[:6]}"
        
        # Convert to base64 PNG
        png_base64 = layer_extraction_service.layer_to_png_base64(extracted.image)
        
        layer_info = composition_service.build_layer_info(
            extracted,
            LayerSource.PAIR1,
            layer_id,
            png_base64=png_base64,
        )
        layers.append(layer_info)
        
        # Build layer state (pair1 is the reference - no transform)
        layer_state = composition_service.build_layer_state(
            layer_info,
            transform=Transform2D(),  # No transform for reference layer
            visible=True,
        )
        layer_states.append(layer_state)
    
    # Extract layer from image B (pair2) - single layer per image
    extraction_b = layer_extraction_service.extract_layers(image_b, LayerSource.PAIR2)
    
    for extracted in extraction_b.layers:
        layer_id = f"{session_id[:8]}_b_{uuid4().hex[:6]}"
        
        # Convert to base64 PNG
        png_base64 = layer_extraction_service.layer_to_png_base64(extracted.image)
        
        layer_info = composition_service.build_layer_info(
            extracted,
            LayerSource.PAIR2,
            layer_id,
            png_base64=png_base64,
        )
        layers.append(layer_info)
        
        # Build layer state (pair2 gets the alignment transform)
        layer_state = composition_service.build_layer_state(
            layer_info,
            transform=alignment_transform,  # Apply alignment transform
            visible=True,
        )
        layer_states.append(layer_state)
    
    # ============================================================
    # Build composition state
    # ============================================================
    
    # Canvas size is the larger of the two images
    canvas_width = max(w_a, w_b)
    canvas_height = max(h_a, h_b)
    
    composition_state = composition_service.build_composition_state(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        layer_states=layer_states,
        alignment=alignment_state,
        dpi=session.preview_a.dpi,
    )
    
    # ============================================================
    # DEBUG: Generate internal overlay.png if requested
    # IMPORTANT: This is for internal testing ONLY - NEVER returned via API
    # ============================================================
    
    if request.debug:
        logger.info("Debug mode: generating internal overlay.png (NOT returned via API)")
        
        try:
            from app.services.align import SimilarityTransform
            
            # Reconstruct similarity transform
            transform = SimilarityTransform(
                scale=alignment_transform.scale_x,
                rotation_rad=math.radians(alignment_transform.rotation_deg),
                tx=alignment_transform.translate_x,
                ty=alignment_transform.translate_y,
            )
            
            config = OverlayConfig(
                color_a=(0, 0, 255),      # Red in BGR
                color_b=(255, 255, 0),    # Cyan in BGR
                alpha_a=0.7,
                alpha_b=0.7,
            )
            
            result = overlay_service.generate_overlay(
                image_a=image_a,
                image_b=image_b,
                transform=transform,
                config=config,
            )
            
            # Save to session directory (internal only)
            session_dir = storage_service.get_session_dir(session_id)
            debug_dir = session_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            debug_overlay_path = debug_dir / "overlay.png"
            cv2.imwrite(str(debug_overlay_path), result.overlay_image)
            
            logger.info(f"Debug overlay saved to {debug_overlay_path} (internal only)")
            
        except Exception as e:
            logger.warning(f"Debug overlay generation failed: {e}")
            warnings.append(f"Debug overlay generation failed: {e}")
    
    # ============================================================
    # Compute Bounds (contentBounds for editor, geometryBounds for export)
    # ============================================================
    
    # contentBounds: Union of all visible layer bounds + padding
    # Used for editor canvas - must NEVER clip any content
    content_x, content_y, content_w, content_h = bounds_computation_service.compute_content_bounds(
        layers, layer_states, canvas_width, canvas_height, padding=CANVAS_PADDING_PX
    )
    
    # geometryBounds: Geometry-only bounds for export cropping
    # Use image_a as reference for geometry bounds (it's the base layer)
    geo_x, geo_y, geo_w, geo_h = bounds_computation_service.compute_geometry_bounds_from_image(
        image_a, margin=EXPORT_MARGIN_PX
    )
    
    bounds_info = LayerBoundsInfo(
        content_bounds=BoundingBox(x=content_x, y=content_y, width=content_w, height=content_h),
        geometry_bounds=BoundingBox(x=geo_x, y=geo_y, width=geo_w, height=geo_h),
    )
    
    logger.info(
        f"Bounds computed: contentBounds=({content_x:.0f},{content_y:.0f},{content_w:.0f},{content_h:.0f}), "
        f"geometryBounds=({geo_x:.0f},{geo_y:.0f},{geo_w:.0f},{geo_h:.0f})"
    )
    
    # ============================================================
    # Build response
    # ============================================================
    
    processing_time = int((time.time() - start_time) * 1000)
    now = datetime.now(timezone.utc)
    
    logger.info(
        f"Compose complete for session {session_id}: "
        f"status={compose_status.value}, "
        f"layers={len(layers)}, "
        f"confidence={confidence:.1%}, "
        f"time={processing_time}ms"
    )
    
    return ComposeResponse(
        session_id=session_id,
        status=compose_status,
        confidence=confidence,
        layers=layers,
        state=composition_state,
        bounds=bounds_info,
        warnings=warnings,
        reason=reason,
        processing_time_ms=processing_time,
        generated_at=now,
    )


@router.post(
    "/{session_id}/compose/update-state",
    response_model=ComposeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="Update composition state",
    description="""
Update the composition state with manual adjustments.

This endpoint allows the frontend to send updated layer states
(transforms, visibility, opacity, etc.) back to the backend
for persistence or re-generation of layers if needed.

Note: This does NOT generate a flattened overlay. The frontend
is responsible for rendering and export.
""",
)
async def update_composition_state(
    session_id: str,
    state: CompositionState,
) -> ComposeResponse:
    """
    Update composition state with manual adjustments.
    
    The frontend can send updated state back for persistence.
    This does NOT generate any flattened images.
    """
    start_time = time.time()
    
    session = get_session_or_404(session_id)
    
    # For now, just acknowledge the update and return updated state
    # In a production system, you might persist this to the session
    
    processing_time = int((time.time() - start_time) * 1000)
    now = datetime.now(timezone.utc)
    
    logger.info(f"Composition state updated for session {session_id}")
    
    # Return the updated state (layers would need to be re-fetched or cached)
    return ComposeResponse(
        session_id=session_id,
        status=ComposeStatus.SUCCESS,
        confidence=state.alignment.confidence,
        layers=[],  # Frontend should use cached layers
        state=state,
        warnings=["Layers not included - use cached layers from initial compose call"],
        processing_time_ms=processing_time,
        generated_at=now,
    )
