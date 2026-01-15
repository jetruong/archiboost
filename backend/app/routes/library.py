"""
Library routes for persistent file storage.

These endpoints allow users to:
- Upload files to a persistent library
- List/search library files
- Use library files when creating sessions
"""

import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.config import settings
from app.models.library import (
    LibraryFile,
    LibraryFileType,
    LibraryUploadResponse,
    LibraryListResponse,
    LibraryFileResponse,
    CreateSessionFromLibraryRequest,
    CreateSessionFromLibraryResponse,
)
from app.models.responses import ErrorResponse, FileUploadInfo
from app.models.session import FileType, FileInfo
from app.services.library import library_service
from app.services.storage import storage_service
from app.services.rasterize import rasterize_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/library", tags=["library"])


# ============================================================
# Upload to Library
# ============================================================

@router.post(
    "/upload",
    response_model=LibraryUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_to_library(
    file: UploadFile = File(..., description="PDF or PNG file to upload"),
    display_name: Optional[str] = Query(default=None, description="Display name for the file"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags"),
    description: Optional[str] = Query(default=None, description="File description"),
) -> LibraryUploadResponse:
    """
    Upload a file to the persistent library.
    
    Files in the library persist indefinitely and can be used across
    multiple comparison sessions.
    """
    logger.info(f"Library upload: {file.filename} ({file.content_type})")
    
    # Validate content type
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_FILE_TYPE",
                "message": "File must be a PDF document or PNG image",
                "details": {
                    "received_type": file.content_type,
                    "expected_types": settings.allowed_content_types,
                },
            },
        )
    
    # Read content
    content = await file.read()
    
    # Check size
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds the {settings.max_file_size_mb}MB limit",
                "details": {
                    "size_bytes": len(content),
                    "max_bytes": settings.max_file_size_bytes,
                },
            },
        )
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    try:
        # Add to library
        library_file = await library_service.add_file(
            content=content,
            filename=file.filename,
            content_type=file.content_type,
            display_name=display_name,
            tags=tag_list,
            description=description,
        )
        
        return LibraryUploadResponse(
            file=library_file,
            message="File uploaded to library successfully",
        )
        
    except Exception as e:
        logger.error(f"Library upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "UPLOAD_FAILED",
                "message": f"Failed to upload file: {e}",
            },
        )


# ============================================================
# List Library Files
# ============================================================

@router.get(
    "/files",
    response_model=LibraryListResponse,
)
async def list_library_files(
    file_type: Optional[str] = Query(default=None, description="Filter by type: 'pdf' or 'png'"),
    tags: Optional[str] = Query(default=None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(default=None, description="Search in filename/name"),
) -> LibraryListResponse:
    """
    List files in the library with optional filtering.
    """
    # Parse file type
    ft = None
    if file_type:
        try:
            ft = LibraryFileType(file_type.lower())
        except ValueError:
            pass
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    files = library_service.list_files(
        file_type=ft,
        tags=tag_list,
        search=search,
    )
    
    return LibraryListResponse(
        files=files,
        total=len(files),
    )


# ============================================================
# Get Single File
# ============================================================

@router.get(
    "/files/{file_id}",
    response_model=LibraryFileResponse,
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def get_library_file(file_id: str) -> LibraryFileResponse:
    """
    Get details of a specific library file.
    """
    file = library_service.get_file(file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "FILE_NOT_FOUND",
                "message": f"Library file not found: {file_id}",
            },
        )
    
    return LibraryFileResponse(file=file)


# ============================================================
# Get File Content
# ============================================================

@router.get(
    "/files/{file_id}/download",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def download_library_file(file_id: str) -> FileResponse:
    """
    Download the original file from the library.
    """
    file = library_service.get_file(file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FILE_NOT_FOUND", "message": f"File not found: {file_id}"},
        )
    
    path = Path(file.storage_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FILE_MISSING", "message": "File data not found on disk"},
        )
    
    media_type = "image/png" if file.file_type == LibraryFileType.PNG else "application/pdf"
    
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=file.filename,
    )


# ============================================================
# Get File Preview
# ============================================================

@router.get(
    "/files/{file_id}/preview",
    responses={
        404: {"model": ErrorResponse, "description": "Preview not found"},
    },
)
async def get_library_file_preview(file_id: str) -> FileResponse:
    """
    Get the preview image for a library file.
    """
    preview_path = library_service.get_preview_path(file_id)
    if not preview_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "PREVIEW_NOT_FOUND", "message": "Preview not available"},
        )
    
    return FileResponse(
        path=preview_path,
        media_type="image/png",
    )


# ============================================================
# Delete File
# ============================================================

@router.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def delete_library_file(file_id: str):
    """
    Delete a file from the library.
    """
    success = library_service.delete_file(file_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FILE_NOT_FOUND", "message": f"File not found: {file_id}"},
        )


# ============================================================
# Update File Metadata
# ============================================================

@router.patch(
    "/files/{file_id}",
    response_model=LibraryFileResponse,
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def update_library_file(
    file_id: str,
    display_name: Optional[str] = Query(default=None),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags (replaces existing)"),
    description: Optional[str] = Query(default=None),
) -> LibraryFileResponse:
    """
    Update metadata for a library file.
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    file = library_service.update_file(
        file_id=file_id,
        display_name=display_name,
        tags=tag_list,
        description=description,
    )
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FILE_NOT_FOUND", "message": f"File not found: {file_id}"},
        )
    
    return LibraryFileResponse(file=file)


# ============================================================
# Create Session from Library Files
# ============================================================

@router.post(
    "/create-session",
    response_model=CreateSessionFromLibraryResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Library file not found"},
    },
)
async def create_session_from_library(
    file_a_id: Optional[str] = Query(default=None, description="Library file ID for Drawing A"),
    file_b_id: Optional[str] = Query(default=None, description="Library file ID for Drawing B"),
    file_a: Optional[UploadFile] = File(default=None, description="Upload for Drawing A (if not using library)"),
    file_b: Optional[UploadFile] = File(default=None, description="Upload for Drawing B (if not using library)"),
) -> CreateSessionFromLibraryResponse:
    """
    Create a new session using files from the library and/or fresh uploads.
    
    You can:
    - Use two library files: provide file_a_id and file_b_id
    - Use two uploads: provide file_a and file_b
    - Mix: provide one library ID and one upload
    
    At least one file must be provided for each drawing.
    """
    from datetime import datetime, timezone
    
    # Validate that we have both files
    has_a = file_a_id or file_a
    has_b = file_b_id or file_b
    
    if not has_a or not has_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_FILES",
                "message": "Both Drawing A and Drawing B must be provided (via library ID or upload)",
            },
        )
    
    # Create session
    session = storage_service.create_session()
    session_dir = storage_service.get_session_dir(session.session_id)
    
    file_a_source = "unknown"
    file_b_source = "unknown"
    
    try:
        # Handle Drawing A
        if file_a_id:
            # Use library file
            library_file = library_service.get_file(file_a_id)
            if not library_file:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "FILE_NOT_FOUND", "message": f"Library file not found: {file_a_id}"},
                )
            
            dest_path = library_service.copy_to_session(file_a_id, session_dir, "A")
            if not dest_path:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"code": "COPY_FAILED", "message": "Failed to copy library file A"},
                )
            
            file_type_a = FileType.PNG if library_file.file_type == LibraryFileType.PNG else FileType.PDF
            session.file_a = FileInfo(
                id=library_file.id,
                original_filename=library_file.filename,
                size_bytes=library_file.size_bytes,
                storage_path=str(dest_path),
                uploaded_at=datetime.now(timezone.utc),
                file_type=file_type_a,
            )
            file_a_source = "library"
        else:
            # Use upload
            if not file_a:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "MISSING_FILE_A", "message": "File A upload required"},
                )
            
            content = await file_a.read()
            file_type_a = FileType.PNG if file_a.content_type == "image/png" else FileType.PDF
            
            file_info_a = await storage_service.save_uploaded_file(
                session.session_id, content, file_a.filename, "A", file_type=file_type_a
            )
            session.file_a = file_info_a
            file_a_source = "upload"
        
        # Handle Drawing B
        if file_b_id:
            # Use library file
            library_file = library_service.get_file(file_b_id)
            if not library_file:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "FILE_NOT_FOUND", "message": f"Library file not found: {file_b_id}"},
                )
            
            dest_path = library_service.copy_to_session(file_b_id, session_dir, "B")
            if not dest_path:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"code": "COPY_FAILED", "message": "Failed to copy library file B"},
                )
            
            file_type_b = FileType.PNG if library_file.file_type == LibraryFileType.PNG else FileType.PDF
            session.file_b = FileInfo(
                id=library_file.id,
                original_filename=library_file.filename,
                size_bytes=library_file.size_bytes,
                storage_path=str(dest_path),
                uploaded_at=datetime.now(timezone.utc),
                file_type=file_type_b,
            )
            file_b_source = "library"
        else:
            # Use upload
            if not file_b:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "MISSING_FILE_B", "message": "File B upload required"},
                )
            
            content = await file_b.read()
            file_type_b = FileType.PNG if file_b.content_type == "image/png" else FileType.PDF
            
            file_info_b = await storage_service.save_uploaded_file(
                session.session_id, content, file_b.filename, "B", file_type=file_type_b
            )
            session.file_b = file_info_b
            file_b_source = "upload"
        
        # Save session
        storage_service.save_session(session)
        
        logger.info(
            f"Created session {session.session_id} from library: "
            f"A={file_a_source}, B={file_b_source}"
        )
        
        return CreateSessionFromLibraryResponse(
            session_id=session.session_id,
            file_a_source=file_a_source,
            file_b_source=file_b_source,
            message="Session created successfully",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session from library: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "SESSION_CREATION_FAILED", "message": str(e)},
        )
