"""
Unit tests for the crop service.
"""

import numpy as np
import pytest

from app.services.crop import CropService, BoundingBox


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""
    
    def test_to_dict(self):
        """Test BoundingBox to dict conversion."""
        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        result = bbox.to_dict()
        
        assert result == {"x": 10, "y": 20, "width": 100, "height": 200}
    
    def test_from_xyxy(self):
        """Test BoundingBox creation from corner coordinates."""
        bbox = BoundingBox.from_xyxy(10, 20, 110, 220)
        
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 200


class TestCropService:
    """Tests for CropService."""
    
    @pytest.fixture
    def crop_service(self):
        """Create a CropService instance with default settings."""
        return CropService(threshold=250, padding=10, min_content_area=10)
    
    def create_test_image(
        self,
        width: int = 200,
        height: int = 200,
        content_x: int = 50,
        content_y: int = 50,
        content_w: int = 100,
        content_h: int = 100,
        background: int = 255,
        content: int = 0,
    ) -> np.ndarray:
        """Create a test image with a black rectangle on white background."""
        image = np.full((height, width, 3), background, dtype=np.uint8)
        image[content_y:content_y + content_h, content_x:content_x + content_w] = content
        return image
    
    def test_detect_content_bbox_basic(self, crop_service):
        """Test basic content detection with centered black rectangle."""
        # Create 200x200 white image with 100x100 black square at (50, 50)
        image = self.create_test_image(
            width=200, height=200,
            content_x=50, content_y=50,
            content_w=100, content_h=100,
        )
        
        bbox = crop_service.detect_content_bbox(image)
        
        assert bbox.x == 50
        assert bbox.y == 50
        assert bbox.width == 100
        assert bbox.height == 100
    
    def test_detect_content_bbox_corner(self, crop_service):
        """Test content detection with content in corner."""
        image = self.create_test_image(
            width=200, height=200,
            content_x=0, content_y=0,
            content_w=50, content_h=50,
        )
        
        bbox = crop_service.detect_content_bbox(image)
        
        assert bbox.x == 0
        assert bbox.y == 0
        assert bbox.width == 50
        assert bbox.height == 50
    
    def test_detect_content_bbox_bottom_right(self, crop_service):
        """Test content detection with content in bottom-right."""
        image = self.create_test_image(
            width=200, height=200,
            content_x=150, content_y=150,
            content_w=50, content_h=50,
        )
        
        bbox = crop_service.detect_content_bbox(image)
        
        assert bbox.x == 150
        assert bbox.y == 150
        assert bbox.width == 50
        assert bbox.height == 50
    
    def test_detect_content_bbox_no_content(self, crop_service):
        """Test that detection fails on all-white image."""
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        
        with pytest.raises(ValueError, match="No significant content detected"):
            crop_service.detect_content_bbox(image)
    
    def test_detect_content_bbox_threshold_sensitivity(self):
        """Test that threshold affects detection."""
        # Create image with gray content (value 200)
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        image[50:150, 50:150] = 200  # Gray square
        
        # With threshold 250, gray (200) is detected as content
        service_sensitive = CropService(threshold=250, padding=10, min_content_area=10)
        bbox = service_sensitive.detect_content_bbox(image)
        assert bbox.width == 100
        
        # With threshold 180, gray (200) is NOT detected as content
        service_strict = CropService(threshold=180, padding=10, min_content_area=10)
        with pytest.raises(ValueError):
            service_strict.detect_content_bbox(image)
    
    def test_apply_padding_basic(self, crop_service):
        """Test padding application."""
        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        
        padded = crop_service.apply_padding(bbox, 200, 200, padding=10)
        
        assert padded.x == 40
        assert padded.y == 40
        assert padded.width == 120
        assert padded.height == 120
    
    def test_apply_padding_clamped_to_edges(self, crop_service):
        """Test that padding is clamped to image boundaries."""
        bbox = BoundingBox(x=5, y=5, width=100, height=100)
        
        padded = crop_service.apply_padding(bbox, 200, 200, padding=10)
        
        # x and y should be clamped to 0
        assert padded.x == 0
        assert padded.y == 0
        assert padded.width == 115  # 0 to 105 + 10 = 115
        assert padded.height == 115
    
    def test_apply_padding_clamped_to_far_edges(self, crop_service):
        """Test padding clamping at far edges."""
        bbox = BoundingBox(x=150, y=150, width=50, height=50)
        
        padded = crop_service.apply_padding(bbox, 200, 200, padding=20)
        
        # Should be clamped at x=200, y=200
        assert padded.x == 130
        assert padded.y == 130
        assert padded.x + padded.width == 200
        assert padded.y + padded.height == 200
    
    def test_crop_whitespace_full_pipeline(self, crop_service):
        """Test full cropping pipeline."""
        image = self.create_test_image(
            width=300, height=300,
            content_x=100, content_y=100,
            content_w=100, content_h=100,
        )
        
        result = crop_service.crop_whitespace(image, padding=10)
        
        # Check cropped dimensions
        assert result.metadata.cropped_width == 120  # 100 + 2*10 padding
        assert result.metadata.cropped_height == 120
        
        # Check crop image shape
        assert result.image.shape == (120, 120, 3)
        
        # Check metadata
        assert result.metadata.original_width == 300
        assert result.metadata.original_height == 300
        assert result.metadata.content_bbox["width"] == 100
        assert result.metadata.content_bbox["height"] == 100
        assert result.metadata.threshold_used == 250
        assert result.metadata.padding_applied == 10
    
    def test_crop_whitespace_ratio(self, crop_service):
        """Test whitespace ratio calculation."""
        # Create 200x200 image with 100x100 content = 25% content, 75% whitespace
        image = self.create_test_image(
            width=200, height=200,
            content_x=50, content_y=50,
            content_w=100, content_h=100,
        )
        
        result = crop_service.crop_whitespace(image, padding=0)
        
        # Content is 100x100 = 10000 pixels
        # Total is 200x200 = 40000 pixels
        # Whitespace ratio = 1 - (10000/40000) = 0.75
        assert result.metadata.whitespace_ratio == 0.75
    
    def test_crop_preserves_content(self):
        """Test that cropping preserves the actual content pixels."""
        # Use a crop service with no padding for this test
        service = CropService(threshold=250, padding=0, min_content_area=10)
        
        image = self.create_test_image(
            width=200, height=200,
            content_x=50, content_y=50,
            content_w=100, content_h=100,
        )
        
        result = service.crop_whitespace(image, padding=0)
        
        # The cropped image should be all black (the content)
        assert result.image.shape == (100, 100, 3)
        assert np.all(result.image == 0)


class TestCropServiceGrayscale:
    """Tests for grayscale image handling."""
    
    def test_detect_content_bbox_grayscale(self):
        """Test content detection on grayscale image."""
        service = CropService(threshold=250, padding=10, min_content_area=10)
        
        # Create grayscale image
        image = np.full((200, 200), 255, dtype=np.uint8)
        image[50:150, 50:150] = 0  # Black square
        
        bbox = service.detect_content_bbox(image)
        
        assert bbox.x == 50
        assert bbox.y == 50
        assert bbox.width == 100
        assert bbox.height == 100
