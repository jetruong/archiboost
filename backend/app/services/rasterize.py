"""
PDF rasterization service using PyMuPDF (fitz).
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

from app.config import settings
from app.models.session import PageMetadata

logger = logging.getLogger(__name__)


@dataclass
class RasterResult:
    """Result of PDF rasterization."""
    image: np.ndarray  # RGB image as numpy array
    page_metadata: PageMetadata


class RasterizeService:
    """Service for rasterizing PDF pages to images."""
    
    def __init__(self, default_dpi: int = None):
        self.default_dpi = default_dpi or settings.default_dpi
    
    def rasterize_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        dpi: int = None,
    ) -> RasterResult:
        """
        Rasterize a single page from a PDF to an RGB image.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page index (0-based)
            dpi: Resolution for rasterization (default from settings)
        
        Returns:
            RasterResult containing the image and metadata
        
        Raises:
            ValueError: If PDF cannot be opened or page doesn't exist
        """
        dpi = dpi or self.default_dpi
        start_time = time.time()
        
        logger.info(f"Rasterizing {pdf_path} page {page_number} at {dpi} DPI")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"Cannot open PDF: {e}")
        
        try:
            if page_number >= len(doc):
                raise ValueError(f"Page {page_number} does not exist (PDF has {len(doc)} pages)")
            
            page = doc[page_number]
            
            # Get page dimensions in points
            rect = page.rect
            pdf_width_pt = rect.width
            pdf_height_pt = rect.height
            
            # Calculate zoom factor for desired DPI
            # PDF default is 72 DPI
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array (RGB)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = image.reshape(pixmap.height, pixmap.width, 3)
            
            render_time_ms = int((time.time() - start_time) * 1000)
            
            metadata = PageMetadata(
                page_number=page_number,
                pdf_width_pt=pdf_width_pt,
                pdf_height_pt=pdf_height_pt,
                dpi=dpi,
                raster_width_px=pixmap.width,
                raster_height_px=pixmap.height,
                render_time_ms=render_time_ms,
            )
            
            logger.info(
                f"Rasterized to {pixmap.width}x{pixmap.height} px in {render_time_ms}ms"
            )
            
            return RasterResult(image=image, page_metadata=metadata)
            
        finally:
            doc.close()
    
    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF."""
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            raise ValueError(f"Cannot open PDF: {e}")
    
    def validate_pdf(self, pdf_path: Path) -> tuple[bool, str]:
        """
        Validate that a file is a valid PDF.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            if page_count > 1:
                return False, f"PDF has {page_count} pages; only single-page PDFs are supported in v1"
            
            return True, ""
            
        except fitz.fitz.FileDataError:
            return False, "File is not a valid PDF"
        except Exception as e:
            return False, f"Cannot validate PDF: {e}"


# Global service instance
rasterize_service = RasterizeService()
