"""
Library service for managing persistent file storage.

Files uploaded to the library persist beyond session lifetime and can be
reused across multiple comparisons.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from app.config import settings
from app.models.library import (
    LibraryFile,
    LibraryFileType,
    LibraryIndex,
)
from app.services.rasterize import rasterize_service
from app.services.crop import crop_service

logger = logging.getLogger(__name__)


class LibraryService:
    """
    Service for managing the persistent file library.
    
    Library structure:
    storage_root/
      library/
        index.json          # Index of all files
        files/
          {file_id}/
            original.{pdf|png}  # Original file
            preview.png         # Generated preview
            metadata.json       # File metadata
    """
    
    def __init__(self, storage_root: Optional[Path] = None):
        self.storage_root = storage_root or settings.storage_root
        self.library_dir = self.storage_root / "library"
        self.files_dir = self.library_dir / "files"
        self.index_path = self.library_dir / "index.json"
        
        # Ensure directories exist
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self._index = self._load_index()
    
    def _load_index(self) -> LibraryIndex:
        """Load the library index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                return LibraryIndex.model_validate(data)
            except Exception as e:
                logger.error(f"Failed to load library index: {e}")
        
        # Create new index
        return LibraryIndex(
            files=[],
            updated_at=datetime.now(timezone.utc),
        )
    
    def _save_index(self) -> None:
        """Save the library index to disk."""
        self._index.updated_at = datetime.now(timezone.utc)
        with open(self.index_path, "w") as f:
            json.dump(self._index.model_dump(mode="json"), f, indent=2, default=str)
    
    def _get_file_dir(self, file_id: str) -> Path:
        """Get the directory for a specific file."""
        return self.files_dir / file_id
    
    def _determine_file_type(self, filename: str, content_type: Optional[str]) -> LibraryFileType:
        """Determine file type from filename and content type."""
        if content_type == "image/png":
            return LibraryFileType.PNG
        if filename.lower().endswith(".png"):
            return LibraryFileType.PNG
        return LibraryFileType.PDF
    
    def _generate_preview(
        self,
        file_path: Path,
        file_type: LibraryFileType,
        preview_path: Path,
        dpi: int = 150,
    ) -> tuple[int, int]:
        """
        Generate a preview image for a file.
        
        Returns (width, height) of the preview.
        """
        if file_type == LibraryFileType.PNG:
            # Load PNG directly
            bgr_image = cv2.imread(str(file_path))
            if bgr_image is None:
                raise ValueError(f"Failed to load PNG: {file_path}")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        else:
            # Rasterize PDF
            raster_result = rasterize_service.rasterize_page(file_path, page_number=0, dpi=dpi)
            rgb_image = raster_result.image
        
        # Crop whitespace
        crop_result = crop_service.crop_whitespace(rgb_image)
        
        # Save preview
        bgr_preview = cv2.cvtColor(crop_result.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(preview_path), bgr_preview)
        
        return crop_result.metadata.cropped_width, crop_result.metadata.cropped_height
    
    async def add_file(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        display_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> LibraryFile:
        """
        Add a file to the library.
        
        Args:
            content: File content as bytes
            filename: Original filename
            content_type: MIME type
            display_name: User-friendly name (defaults to filename)
            tags: Optional tags for categorization
            description: Optional description
        
        Returns:
            LibraryFile metadata
        """
        file_id = str(uuid.uuid4())
        file_type = self._determine_file_type(filename, content_type)
        
        # Create file directory
        file_dir = self._get_file_dir(file_id)
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine storage paths
        extension = "png" if file_type == LibraryFileType.PNG else "pdf"
        original_path = file_dir / f"original.{extension}"
        preview_path = file_dir / "preview.png"
        
        # Save original file
        original_path.write_bytes(content)
        logger.info(f"Saved library file: {original_path}")
        
        # Generate preview
        try:
            preview_width, preview_height = self._generate_preview(
                original_path, file_type, preview_path
            )
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            preview_width, preview_height = None, None
        
        # Create file record
        library_file = LibraryFile(
            id=file_id,
            filename=filename,
            display_name=display_name or filename,
            file_type=file_type,
            size_bytes=len(content),
            storage_path=str(original_path),
            preview_path=str(preview_path) if preview_path.exists() else None,
            preview_width=preview_width,
            preview_height=preview_height,
            uploaded_at=datetime.now(timezone.utc),
            tags=tags or [],
            description=description,
            source="upload",
        )
        
        # Add to index
        self._index.files.append(library_file)
        self._save_index()
        
        logger.info(f"Added file to library: {file_id} ({filename})")
        return library_file
    
    def get_file(self, file_id: str) -> Optional[LibraryFile]:
        """Get a file by ID."""
        for f in self._index.files:
            if f.id == file_id:
                return f
        return None
    
    def list_files(
        self,
        file_type: Optional[LibraryFileType] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> List[LibraryFile]:
        """
        List files in the library with optional filtering.
        
        Args:
            file_type: Filter by file type
            tags: Filter by tags (any match)
            search: Search in filename and display_name
        
        Returns:
            List of matching files
        """
        files = self._index.files
        
        if file_type:
            files = [f for f in files if f.file_type == file_type]
        
        if tags:
            files = [f for f in files if any(t in f.tags for t in tags)]
        
        if search:
            search_lower = search.lower()
            files = [
                f for f in files
                if search_lower in f.filename.lower()
                or search_lower in f.display_name.lower()
                or (f.description and search_lower in f.description.lower())
            ]
        
        # Sort by upload date (newest first)
        files.sort(key=lambda f: f.uploaded_at, reverse=True)
        
        return files
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from the library.
        
        Returns True if file was deleted, False if not found.
        """
        file = self.get_file(file_id)
        if not file:
            return False
        
        # Remove from index
        self._index.files = [f for f in self._index.files if f.id != file_id]
        self._save_index()
        
        # Delete file directory
        file_dir = self._get_file_dir(file_id)
        if file_dir.exists():
            shutil.rmtree(file_dir)
        
        logger.info(f"Deleted library file: {file_id}")
        return True
    
    def update_file(
        self,
        file_id: str,
        display_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Optional[LibraryFile]:
        """
        Update file metadata.
        
        Returns updated file or None if not found.
        """
        for i, f in enumerate(self._index.files):
            if f.id == file_id:
                if display_name is not None:
                    f.display_name = display_name
                if tags is not None:
                    f.tags = tags
                if description is not None:
                    f.description = description
                
                self._index.files[i] = f
                self._save_index()
                return f
        
        return None
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get the storage path for a file."""
        file = self.get_file(file_id)
        if file:
            return Path(file.storage_path)
        return None
    
    def get_preview_path(self, file_id: str) -> Optional[Path]:
        """Get the preview image path for a file."""
        file = self.get_file(file_id)
        if file and file.preview_path:
            path = Path(file.preview_path)
            if path.exists():
                return path
        return None
    
    def copy_to_session(
        self,
        file_id: str,
        session_dir: Path,
        which: str,
    ) -> Optional[Path]:
        """
        Copy a library file to a session's input directory.
        
        Args:
            file_id: Library file ID
            session_dir: Session directory path
            which: "A" or "B"
        
        Returns:
            Path to the copied file, or None if file not found
        """
        file = self.get_file(file_id)
        if not file:
            return None
        
        source_path = Path(file.storage_path)
        if not source_path.exists():
            logger.error(f"Library file not found on disk: {source_path}")
            return None
        
        # Determine destination path
        inputs_dir = session_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        
        extension = "png" if file.file_type == LibraryFileType.PNG else "pdf"
        dest_path = inputs_dir / f"file_{which.lower()}.{extension}"
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied library file {file_id} to {dest_path}")
        
        return dest_path


# Global service instance
library_service = LibraryService()
