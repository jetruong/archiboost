"""
Upload endpoint for PDF and PNG files.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.config import settings
from app.models.responses import UploadResponse, FileUploadInfo, ErrorResponse
from app.models.session import SessionStatus, FileType
from app.services.storage import storage_service
from app.services.rasterize import rasterize_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


def determine_file_type(content_type: str) -> FileType:
    """Determine the file type from content type."""
    if content_type == "image/png":
        return FileType.PNG
    return FileType.PDF


def validate_png_image(content: bytes) -> tuple[bool, str]:
    """
    Validate that content is a valid PNG image.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Decode image from bytes
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return False, "File is not a valid PNG image"
        
        # Check that image has reasonable dimensions
        height, width = image.shape[:2]
        if width < 10 or height < 10:
            return False, f"Image too small: {width}x{height} pixels"
        
        # Optional: Could add max size check here too
        
        return True, ""
        
    except Exception as e:
        return False, f"Cannot validate PNG: {e}"


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or validation error"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_files(
    file_a: UploadFile = File(..., description="First file - Drawing A (PDF or PNG)"),
    file_b: UploadFile = File(..., description="Second file - Drawing B (PDF or PNG)"),
) -> UploadResponse:
    """
    Upload two files to start a comparison session.
    
    Accepts both PDF and PNG files. You can mix file types (e.g., PDF + PNG).
    Creates a new session and stores the uploaded files for processing.
    """
    logger.info(f"Upload request: file_a={file_a.filename} ({file_a.content_type}), file_b={file_b.filename} ({file_b.content_type})")
    
    # Validate content types
    for label, file in [("file_a", file_a), ("file_b", file_b)]:
        if file.content_type not in settings.allowed_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_FILE_TYPE",
                    "message": f"File '{label}' must be a PDF document or PNG image",
                    "details": {
                        "field": label,
                        "received_type": file.content_type,
                        "expected_types": settings.allowed_content_types,
                    },
                },
            )
    
    # Read file contents and check sizes
    content_a = await file_a.read()
    content_b = await file_b.read()
    
    for label, content, filename in [
        ("file_a", content_a, file_a.filename),
        ("file_b", content_b, file_b.filename),
    ]:
        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": f"File '{filename}' exceeds the {settings.max_file_size_mb}MB limit",
                    "details": {
                        "field": label,
                        "size_bytes": len(content),
                        "max_bytes": settings.max_file_size_bytes,
                    },
                },
            )
    
    # Determine file types
    file_type_a = determine_file_type(file_a.content_type)
    file_type_b = determine_file_type(file_b.content_type)
    
    # Create session
    session = storage_service.create_session()
    
    try:
        # Save files with appropriate file types
        file_info_a = await storage_service.save_uploaded_file(
            session.session_id, content_a, file_a.filename, "A", file_type=file_type_a
        )
        file_info_b = await storage_service.save_uploaded_file(
            session.session_id, content_b, file_b.filename, "B", file_type=file_type_b
        )
        
        # Validate files based on their types
        for label, file_info, content in [("file_a", file_info_a, content_a), ("file_b", file_info_b, content_b)]:
            if file_info.file_type == FileType.PDF:
                # Validate PDFs using PyMuPDF
                is_valid, error_msg = rasterize_service.validate_pdf(Path(file_info.storage_path))
                if not is_valid:
                    # Determine error type
                    if "pages" in error_msg.lower():
                        code = "MULTI_PAGE_PDF"
                    else:
                        code = "INVALID_FILE_TYPE"
                    
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": code,
                            "message": error_msg,
                            "details": {"field": label, "filename": file_info.original_filename},
                        },
                    )
            else:
                # Validate PNG images
                is_valid, error_msg = validate_png_image(content)
                if not is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "INVALID_FILE_TYPE",
                            "message": error_msg,
                            "details": {"field": label, "filename": file_info.original_filename},
                        },
                    )
        
        # Update session with file info
        session.file_a = file_info_a
        session.file_b = file_info_b
        session.status = SessionStatus.UPLOADED
        storage_service.save_session(session)
        
        logger.info(f"Session {session.session_id} created successfully")
        
        return UploadResponse(
            session_id=session.session_id,
            file_a=FileUploadInfo(
                id=file_info_a.id,
                filename=file_info_a.original_filename,
                size_bytes=file_info_a.size_bytes,
                uploaded_at=file_info_a.uploaded_at,
            ),
            file_b=FileUploadInfo(
                id=file_info_b.id,
                filename=file_info_b.original_filename,
                size_bytes=file_info_b.size_bytes,
                uploaded_at=file_info_b.uploaded_at,
            ),
            status=session.status,
            created_at=session.created_at,
            expires_at=session.expires_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "PROCESSING_ERROR",
                "message": "Failed to process uploaded files",
                "details": {"error": str(e)},
            },
        )
