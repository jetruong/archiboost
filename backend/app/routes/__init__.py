"""
API route modules.
"""

from app.routes.upload import router as upload_router
from app.routes.sessions import router as sessions_router
from app.routes.images import router as images_router
from app.routes.compose import router as compose_router
from app.routes.library import router as library_router
from app.routes.differences import router as differences_router

__all__ = [
    "upload_router",
    "sessions_router",
    "images_router",
    "compose_router",
    "library_router",
    "differences_router",
]
