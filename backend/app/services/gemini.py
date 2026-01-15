"""
Gemini AI service for generating summaries of architectural drawing differences.

Uses Google's Gemini Flash model (2.0 or 2.5) - a Vision Language Model (VLM) - 
to analyze annotated overlay images and provide concise, region-based summaries 
of detected differences.
"""

import base64
import io
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Load .env file from backend directory
def _load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value

_load_env()

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. AI summaries will be unavailable.")


class GeminiSummaryService:
    """Service for generating AI summaries of drawing differences."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini service.
        
        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
                     Set in .env file: GOOGLE_API_KEY=your_key_here
        """
        # Read from environment variable (set in .env file)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        # Try 2.5 first, fallback to 2.0 if not available
        self.model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")  # Gemini 2.0 Flash (VLM - Vision Language Model)
        self._configured = False
        self.debug_save_enabled = os.environ.get("GEMINI_DEBUG_SAVE", "false").lower() in ("1", "true", "yes", "on")
        self.debug_save_dir = Path(
            os.environ.get(
                "GEMINI_DEBUG_DIR",
                Path(__file__).resolve().parent.parent.parent / "logs" / "gemini",
            )
        )
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._configured = True
                logger.info(f"Gemini service configured with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
        elif not GEMINI_AVAILABLE:
            logger.warning("Gemini unavailable: google-generativeai not installed. Install with: pip install google-generativeai")
        else:
            logger.warning(
                "Gemini unavailable: API key not set. "
                "Set GOOGLE_API_KEY in your .env file. "
                "Get your key from: https://makersuite.google.com/app/apikey"
            )
    
    @property
    def is_available(self) -> bool:
        """Check if Gemini service is available."""
        return GEMINI_AVAILABLE and self._configured
    
    @property
    def is_vlm(self) -> bool:
        """Check if the model is a Vision Language Model (VLM)."""
        # Gemini models support vision (image + text input)
        return True
    
    @property
    def model_display_name(self) -> str:
        """Get a human-readable model name."""
        if "2.5" in self.model_name:
            return "Gemini 2.5 Flash"
        elif "2.0" in self.model_name:
            return "Gemini 2.0 Flash"
        elif "1.5" in self.model_name:
            return "Gemini 1.5 Flash"
        else:
            return "Gemini Flash"
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 PNG string."""
        # Encode as PNG
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image as PNG")
        
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
    
    def _build_prompt(self, regions: List[dict]) -> str:
        """
        Build the prompt for Gemini to summarize differences.
        
        Args:
            regions: List of region dictionaries with id, bbox, area, centroid
        
        Returns:
            Formatted prompt string
        """
        region_descriptions = []
        for region in regions:
            region_descriptions.append(
                f"- Region {region['id']}: located at bbox ({region['bbox'][0]}, {region['bbox'][1]}, "
                f"{region['bbox'][2]}x{region['bbox'][3]}), area={region['area']} pixels"
            )
        
        regions_text = "\n".join(region_descriptions) if region_descriptions else "No significant regions detected."
        
        prompt = f"""You are analyzing an annotated overlay image comparing two architectural drawings. The image shows:
- RED linework = Drawing A (the original/reference drawing)
- CYAN linework = Drawing B (the revised/comparison drawing)  
- Where lines overlap perfectly, they appear darker (both drawings match)
- Yellow highlighted areas indicate detected differences between the drawings
- Red bounding boxes labeled R1, R2, etc. mark specific difference regions
- The region IDs correspond to the list below

Detected difference regions:
{regions_text}

Please provide a concise summary of the differences (2-4 sentences). For each significant region:
1. Reference the region label (e.g., "Region 1", "Region 2")
2. Describe what appears to have changed:
   - RED only = content removed in revision (present in A, missing in B)
   - CYAN only = content added in revision (missing in A, present in B)
   - Slight offset of both colors = content moved or adjusted
3. Note the general location in the drawing

Important guidelines:
- Be concise and technical
- State uncertainty when the change is unclear (e.g., "appears to be", "may indicate")
- Focus on architectural significance (walls, fixtures, dimensions, annotations)
- If differences seem minor (small areas or line weight variations), mention this
- Do not speculate beyond what's visible in the image

Begin your response directly with the summary, no introductory phrases."""
        
        return prompt

    def _maybe_save_debug_artifacts(
        self,
        prompt: str,
        annotated_overlay: np.ndarray,
        regions: List[dict],
    ) -> None:
        """Optionally persist prompt/image/regions for debugging."""
        if not self.debug_save_enabled:
            return
        
        try:
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            base = self.debug_save_dir / f"gemini_{ts}"
            
            # Prompt
            base.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
            # Regions JSON
            base.with_suffix(".regions.json").write_text(json.dumps(regions, indent=2), encoding="utf-8")
            # Overlay image (BGR)
            cv2.imwrite(str(base.with_suffix(".png")), annotated_overlay)
            
            logger.info(f"Saved Gemini debug artifacts to {self.debug_save_dir}")
        except Exception as e:
            logger.warning(f"Failed to save Gemini debug artifacts: {e}")
    
    async def summarize_differences(
        self,
        annotated_overlay: np.ndarray,
        regions: List[dict],
        timeout: float = 30.0,
    ) -> Optional[str]:
        """
        Generate an AI summary of the differences shown in the annotated overlay.
        
        Args:
            annotated_overlay: The annotated overlay image with labeled regions
            regions: List of region dictionaries from diff detection
            timeout: Request timeout in seconds
        
        Returns:
            AI-generated summary string, or None if unavailable/failed
        """
        if not self.is_available:
            logger.warning("Gemini service not available, returning fallback summary")
            return self._generate_fallback_summary(regions)
        
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(annotated_overlay)
            
            # Build prompt
            prompt = self._build_prompt(regions)
            self._maybe_save_debug_artifacts(prompt, annotated_overlay, regions)
            
            # Log what we're sending to the LLM (prompt + image metadata/base64 length)
            try:
                logger.info("Gemini prompt:\n%s", prompt)
                logger.info(
                    "Annotated overlay for Gemini: shape=%s, base64_len=%d, preview='%s...'",
                    annotated_overlay.shape,
                    len(image_b64),
                    image_b64[:64],
                )
            except Exception as log_err:
                logger.debug(f"Failed to log Gemini payload details: {log_err}")
            
            # Create the model
            model = genai.GenerativeModel(self.model_name)
            
            # Create the image part
            image_part = {
                "mime_type": "image/png",
                "data": image_b64
            }
            
            # Generate content
            response = model.generate_content(
                [prompt, image_part],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 1000,
                }
            )
            
            summary = response.text.strip()
            logger.info(f"Generated AI summary: {len(summary)} characters")
            
            return summary
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_summary(regions)
    
    def _generate_fallback_summary(self, regions: List[dict]) -> str:
        """
        Generate a fallback summary when AI is unavailable.
        
        Args:
            regions: List of region dictionaries
        
        Returns:
            Simple message indicating feature is unavailable
        """
        return "AI-powered difference analysis is not available at this time."


# Global service instance
gemini_service = GeminiSummaryService()
