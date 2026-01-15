"""
Application configuration settings.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    
    # File Storage
    data_dir: Path = Path("./data")
    
    # Storage Root for library files
    storage_root: Path = Path("./storage")
    
    # Upload Limits
    max_file_size_mb: int = 25
    allowed_content_types: list[str] = ["application/pdf", "image/png"]
    
    # Rasterization Settings
    default_dpi: int = 250
    max_dpi: int = 600
    min_dpi: int = 72
    
    # Crop Settings
    crop_threshold: int = 250  # Grayscale threshold for whitespace detection
    crop_padding: int = 20     # Pixels of padding around content
    min_content_area: int = 100  # Minimum pixel area to be considered content
    
    # Session Settings
    session_ttl_hours: int = 1
    
    # ============================================================
    # AUTO ALIGNMENT V2 SETTINGS
    # ============================================================
    
    # --- Linework Preprocessing ---
    # Canny edge detection parameters
    canny_threshold1: int = 50      # Lower hysteresis threshold
    canny_threshold2: int = 150     # Upper hysteresis threshold
    canny_aperture: int = 3         # Sobel aperture size (3, 5, or 7)
    
    # Morphological operations for linework enhancement
    linework_dilate_kernel: int = 2  # Kernel size for dilating edges
    linework_dilate_iterations: int = 1
    
    # Text suppression (remove small connected components)
    text_suppress_enabled: bool = True
    text_suppress_min_area: int = 100        # Min area in pixels to keep
    text_suppress_min_width: int = 5         # Min width in pixels to keep
    text_suppress_min_height: int = 5        # Min height in pixels to keep
    text_suppress_max_aspect_ratio: float = 10.0  # Max aspect ratio (width/height)
    
    # Contrast normalization
    normalize_contrast: bool = True
    clahe_clip_limit: float = 2.0    # CLAHE clip limit
    clahe_tile_size: int = 8         # CLAHE tile grid size
    
    # --- Rotation Strategy ---
    # Coarse rotation candidates (evaluated first)
    rotation_candidates_deg: List[float] = [0.0, 90.0, 180.0, 270.0]
    
    # Fine rotation refinement (around best coarse candidate)
    fine_rotation_enabled: bool = True         # Enable fine rotation search
    fine_rotation_range_deg: float = 15.0      # Search ±15° around best coarse rotation
    fine_rotation_step_deg: float = 3.0        # Step size for fine search
    fine_rotation_threshold: float = 0.5       # Only refine if coarse confidence < this
    
    # --- Phase Correlation (Primary Translation Estimation) ---
    phase_correlation_window: str = "hann"  # Window function: "hann", "hamming", "none"
    phase_correlation_min_response: float = 0.15  # Minimum response to accept
    
    # --- Scale Bounds ---
    # Reject alignments with scale outside these bounds
    # Note: Wide bounds allow comparing drawings at different DPIs/resolutions
    scale_min: float = 0.1           # Minimum allowed scale (10x smaller)
    scale_max: float = 10.0          # Maximum allowed scale (10x larger)
    scale_warn_min: float = 0.5      # Warn if scale below this
    scale_warn_max: float = 2.0      # Warn if scale above this
    
    # --- Translation Bounds ---
    # Maximum translation as fraction of image dimension
    # Note: Wide bounds allow drawings positioned anywhere in the frame
    translation_max_fraction: float = 2.0  # Max translation = 200% of image size
    
    # --- Overlap Requirements ---
    # Minimum overlap between warped B and A bounding boxes
    min_overlap_ratio: float = 0.1   # At least 10% overlap required
    
    # --- Feature-Based Refinement (Secondary, Conditional) ---
    # Feature detector priority (try in order)
    feature_detectors: List[str] = ["AKAZE", "SIFT", "ORB"]
    
    # When to use feature refinement
    feature_refinement_threshold: float = 0.5  # Use if phase confidence < this
    
    # Feature matching parameters
    feature_max_features: int = 2000       # Max features to detect
    feature_match_ratio: float = 0.75      # Lowe's ratio test threshold
    feature_min_matches: int = 10          # Minimum matches required
    
    # RANSAC parameters
    ransac_reproj_threshold: float = 5.0   # Reprojection error threshold (pixels)
    ransac_max_iters: int = 2000           # Maximum RANSAC iterations
    ransac_confidence: float = 0.99        # RANSAC confidence level
    
    # --- Confidence Thresholds ---
    # Final confidence scoring weights
    confidence_weight_phase: float = 0.4      # Weight for phase correlation response
    confidence_weight_overlap: float = 0.3    # Weight for bbox overlap ratio
    confidence_weight_inlier: float = 0.3     # Weight for inlier ratio (if refinement ran)
    
    # Minimum confidence to accept alignment
    auto_align_min_confidence: float = 0.35   # Below this -> AUTO_ALIGNMENT_FAILED
    
    # Confidence threshold for "high confidence" (skip refinement)
    auto_align_high_confidence: float = 0.7   # Above this -> skip feature refinement
    
    # ============================================================
    # DIFFERENCE DETECTION SETTINGS
    # ============================================================
    
    # Position tolerance in pixels - allows small translation/alignment errors
    # Both linework masks are dilated by this amount before comparison
    # Set to 0 for pixel-perfect comparison (very strict)
    # Set to 5-10 for tolerance of small alignment errors (recommended)
    diff_position_tolerance: int = 5
    
    # Minimum region area in pixels to consider a difference
    # Regions smaller than this are filtered out as noise
    diff_min_region_area: int = 100
    
    # Morphological kernel size for cleanup operations
    diff_morph_kernel_size: int = 5
    
    # Number of morphological iterations for gap closing
    diff_morph_iterations: int = 2
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"
    
    class Config:
        env_prefix = "ARCHIBOOST_"
        env_file = ".env"
        extra = "ignore"  # Allow extra env vars like GOOGLE_API_KEY


# Global settings instance
settings = Settings()
