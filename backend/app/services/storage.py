"""
Storage service for managing sessions and files on the local filesystem.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.session import Session, SessionStatus, FileInfo, FileType

logger = logging.getLogger(__name__)


class StorageService:
    """Manages session storage and file operations."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.sessions_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> Session:
        """Create a new session with unique ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=settings.session_ttl_hours)
        
        session = Session(
            session_id=session_id,
            status=SessionStatus.UPLOADED,
            created_at=now,
            expires_at=expires_at,
        )
        
        # Create session directories
        session_dir = self.get_session_dir(session_id)
        (session_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (session_dir / "previews").mkdir(parents=True, exist_ok=True)
        (session_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created session {session_id}")
        return session
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get the directory path for a session."""
        return self.base_dir / session_id
    
    def get_session_path(self, session_id: str) -> Path:
        """Get the path to the session.json file."""
        return self.get_session_dir(session_id) / "session.json"
    
    def save_session(self, session: Session) -> None:
        """Persist session to disk."""
        path = self.get_session_path(session.session_id)
        session.save(path)
        logger.debug(f"Saved session {session.session_id}")
    
    def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from disk. Returns None if not found."""
        path = self.get_session_path(session_id)
        if not path.exists():
            return None
        try:
            session = Session.load(path)
            # Check expiration
            if datetime.now(timezone.utc) > session.expires_at:
                logger.warning(f"Session {session_id} has expired")
                return None
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists and is not expired."""
        return self.load_session(session_id) is not None
    
    async def save_uploaded_file(
        self,
        session_id: str,
        file_content: bytes,
        original_filename: str,
        which: str,  # "A" or "B"
        file_type: FileType = FileType.PDF,
    ) -> FileInfo:
        """Save an uploaded file and return its metadata."""
        file_id = str(uuid.uuid4())
        
        # Determine storage path based on file type
        session_dir = self.get_session_dir(session_id)
        extension = "png" if file_type == FileType.PNG else "pdf"
        filename = f"file_{which.lower()}.{extension}"
        storage_path = session_dir / "inputs" / filename
        
        # Write file
        storage_path.write_bytes(file_content)
        
        file_info = FileInfo(
            id=file_id,
            original_filename=original_filename,
            size_bytes=len(file_content),
            storage_path=str(storage_path),
            uploaded_at=datetime.now(timezone.utc),
            file_type=file_type,
        )
        
        logger.info(f"Saved file {original_filename} ({len(file_content)} bytes) as {storage_path}")
        return file_info
    
    def get_input_path(self, session_id: str, which: str, file_type: FileType = FileType.PDF) -> Path:
        """Get the path to an input file (PDF or PNG)."""
        extension = "png" if file_type == FileType.PNG else "pdf"
        return self.get_session_dir(session_id) / "inputs" / f"file_{which.lower()}.{extension}"
    
    def get_preview_path(self, session_id: str, which: str) -> Path:
        """Get the path to a preview image."""
        return self.get_session_dir(session_id) / "previews" / f"preview_{which.lower()}.png"
    
    def get_image_by_id(self, image_id: str) -> Optional[Path]:
        """
        Find an image by its ID across all sessions.
        
        Supported formats:
        - preview_{a|b}_{session_id_prefix}
        - overlay_{session_id_prefix}
        - diff_{session_id_prefix}
        """
        parts = image_id.split("_")
        if len(parts) < 2:
            return None
        
        image_type = parts[0]  # 'preview', 'overlay', or 'diff'
        
        if image_type == "preview":
            # Format: preview_a_550e8400 or preview_b_550e8400
            if len(parts) < 3:
                return None
            which = parts[1]  # 'a' or 'b'
            session_prefix = parts[2]
            
            for session_dir in self.base_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith(session_prefix):
                    preview_path = session_dir / "previews" / f"preview_{which}.png"
                    if preview_path.exists():
                        return preview_path
        
        elif image_type in ("overlay", "diff"):
            # Format: overlay_550e8400 or diff_550e8400
            session_prefix = parts[1]
            
            for session_dir in self.base_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith(session_prefix):
                    image_path = session_dir / "overlays" / f"{image_type}_{session_prefix}.png"
                    if image_path.exists():
                        return image_path
        
        return None
    
    def get_overlay_path(self, session_id: str) -> Path:
        """Get the path to the overlay image."""
        return self.get_session_dir(session_id) / "overlays" / f"overlay_{session_id[:8]}.png"
    
    def generate_image_id(self, session_id: str, which: str) -> str:
        """Generate a unique image ID for preview."""
        prefix = session_id[:8]
        return f"preview_{which.lower()}_{prefix}"
    
    def generate_overlay_id(self, session_id: str) -> str:
        """Generate a unique image ID for overlay."""
        return f"overlay_{session_id[:8]}"


# Global service instance
storage_service = StorageService()
