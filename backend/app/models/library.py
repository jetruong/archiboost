"""
Library models for persistent file storage.

The library stores user-uploaded files that persist beyond session lifetime,
allowing users to reuse files across multiple comparisons.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class LibraryFileType(str, Enum):
    """Type of file in the library."""
    PDF = "pdf"
    PNG = "png"


class LibraryFile(BaseModel):
    """A file stored in the persistent library."""
    id: str = Field(description="Unique file identifier")
    filename: str = Field(description="Original filename")
    display_name: str = Field(description="User-friendly display name")
    file_type: LibraryFileType
    size_bytes: int
    storage_path: str
    
    # Preview info (generated after upload)
    preview_path: Optional[str] = None
    preview_width: Optional[int] = None
    preview_height: Optional[int] = None
    
    # Metadata
    uploaded_at: datetime
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    
    # Source info
    source: str = Field(default="upload", description="How file was added: 'upload', 'import'")


class LibraryIndex(BaseModel):
    """Index of all files in the library."""
    version: str = "1.0.0"
    files: List[LibraryFile] = Field(default_factory=list)
    updated_at: datetime


# ============================================================
# API Models
# ============================================================

class LibraryUploadResponse(BaseModel):
    """Response from uploading a file to the library."""
    file: LibraryFile
    message: str = "File uploaded successfully"


class LibraryListResponse(BaseModel):
    """Response listing library files."""
    files: List[LibraryFile]
    total: int


class LibraryFileResponse(BaseModel):
    """Response with single file details."""
    file: LibraryFile


class CreateSessionFromLibraryRequest(BaseModel):
    """Request to create a session from library files."""
    file_a_id: Optional[str] = Field(default=None, description="Library file ID for Drawing A")
    file_b_id: Optional[str] = Field(default=None, description="Library file ID for Drawing B")
    
    # For mixed mode: one from library, one uploaded fresh
    # If file_a_id is set, file_a will use that library file
    # If file_a_id is None, expect file_a to be uploaded


class CreateSessionFromLibraryResponse(BaseModel):
    """Response from creating session from library."""
    session_id: str
    file_a_source: str = Field(description="'library' or 'upload'")
    file_b_source: str = Field(description="'library' or 'upload'")
    message: str
