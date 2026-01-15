"""
Static image serving endpoint.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.services.storage import storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


@router.get(
    "/{image_id}",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG image"},
        404: {"description": "Image not found"},
    },
)
async def get_image(image_id: str) -> FileResponse:
    """
    Serve a generated image by its ID.
    
    Image IDs follow the format: preview_{a|b}_{session_id_prefix}
    """
    logger.debug(f"Image request: {image_id}")
    
    image_path = storage_service.get_image_by_id(image_id)
    
    if image_path is None or not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "IMAGE_NOT_FOUND",
                "message": f"Image '{image_id}' does not exist or has expired",
            },
        )
    
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"{image_id}.png",
    )
