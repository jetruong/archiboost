"""
Unit tests for overlay pipeline - verifies mask extraction, warping, and tinting.
"""

import math
import pytest
import numpy as np
import cv2

from app.services.overlay import OverlayService, OverlayConfig
from app.services.align import SimilarityTransform


class TestLineMaskExtraction:
    """Tests for line mask extraction from images."""
    
    @pytest.fixture
    def service(self):
        return OverlayService()
    
    def test_white_background_no_lines(self, service):
        """Pure white image should produce empty mask."""
        # Create white image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        mask = service.extract_line_mask(image)
        
        assert mask.shape == (200, 200)
        assert np.all(mask == 0), "White background should produce no lines"
    
    def test_black_image_all_lines(self, service):
        """Pure black image should produce full mask."""
        # Create black image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        mask = service.extract_line_mask(image)
        
        assert np.all(mask == 255), "Black image should be all lines"
    
    def test_horizontal_line_detected(self, service):
        """Horizontal black line on white should be detected."""
        # Create white image with black horizontal line
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image[100:105, :, :] = 0  # Black horizontal line
        
        mask = service.extract_line_mask(image)
        
        # Check line area is detected
        line_pixels = np.sum(mask[100:105, :] == 255)
        assert line_pixels > 0, "Horizontal line should be detected"
        
        # Check non-line area is empty
        non_line_pixels = np.sum(mask[:100, :] == 255)
        assert non_line_pixels == 0, "Non-line area should be empty"
    
    def test_vertical_line_detected(self, service):
        """Vertical black line on white should be detected."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image[:, 100:105, :] = 0  # Black vertical line
        
        mask = service.extract_line_mask(image)
        
        # Check line area is detected
        line_pixels = np.sum(mask[:, 100:105] == 255)
        assert line_pixels > 0, "Vertical line should be detected"
    
    def test_gray_line_threshold(self, service):
        """Lines with different gray values should be handled by threshold."""
        # Dark gray (should be detected as line)
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image[50:55, :, :] = 100  # Dark gray line
        
        mask = service.extract_line_mask(image, threshold=240)
        dark_gray_pixels = np.sum(mask[50:55, :] == 255)
        assert dark_gray_pixels > 0, "Dark gray line should be detected"
        
        # Light gray (should NOT be detected with default threshold)
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image[50:55, :, :] = 250  # Light gray line
        
        mask = service.extract_line_mask(image, threshold=240)
        light_gray_pixels = np.sum(mask[50:55, :] == 255)
        assert light_gray_pixels == 0, "Light gray should not be detected as line"
    
    def test_grayscale_input(self, service):
        """Grayscale image input should work."""
        # Grayscale image (2D)
        image = np.ones((200, 200), dtype=np.uint8) * 255
        image[100:105, :] = 0
        
        mask = service.extract_line_mask(image)
        
        assert mask.shape == (200, 200)
        assert np.sum(mask[100:105, :] == 255) > 0


class TestImageWarping:
    """Tests for image warping with similarity transforms."""
    
    @pytest.fixture
    def service(self):
        return OverlayService()
    
    def test_identity_warp(self, service):
        """Identity transform should not change image."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        image[50:100, 50:100, :] = 255  # White square
        
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        warped, validity_mask = service.warp_image(image, transform, (200, 200))
        
        # Core region should be unchanged
        np.testing.assert_array_equal(warped[50:100, 50:100, :], image[50:100, 50:100, :])
        # Validity mask should be all 255 for identity transform
        assert np.all(validity_mask == 255), "Identity warp should have full validity"
    
    def test_translation_warp(self, service):
        """Translation should shift image content."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        image[50:70, 50:70, :] = 255  # White square at (50,50)
        
        # Translate by (30, 20)
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=30.0, ty=20.0)
        
        warped, validity_mask = service.warp_image(image, transform, (200, 200))
        
        # The white square should now be around (80, 70)
        assert np.mean(warped[70:90, 80:100, :]) > 200, "Translated region should be white"
        # Areas where original image was shifted from should have 0 validity
        assert np.sum(validity_mask[:20, :]) == 0, "Top edge should be invalid after translation"
    
    def test_scale_warp(self, service):
        """Scaling should resize content."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Draw a centered black square
        image[80:120, 80:120, :] = 0  # 40x40 black square centered at (100,100)
        
        # Scale by 0.5 (content shrinks)
        transform = SimilarityTransform(scale=0.5, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        warped, validity_mask = service.warp_image(image, transform, (200, 200))
        
        # After scaling by 0.5, the black square should be at (40,40) with size 20x20
        # The original (80,120) becomes (40,60) after 0.5x scaling
        black_region = warped[40:60, 40:60, :]
        assert np.mean(black_region) < 50, "Scaled black region should still be black"
        # Validity mask should only be valid in the scaled region
        assert np.sum(validity_mask[100:, :]) == 0, "Bottom half should be invalid after 0.5x scale"
    
    def test_rotation_warp(self, service):
        """Rotation should rotate content."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Draw horizontal line through center
        image[95:105, 50:150, :] = 0  # Horizontal black line
        
        # Rotate 90 degrees
        transform = SimilarityTransform(scale=1.0, rotation_rad=math.pi/2, tx=0.0, ty=0.0)
        
        warped, validity_mask = service.warp_image(image, transform, (200, 200))
        
        # After 90° rotation, horizontal line becomes vertical
        # The rotation is around origin, so result moves
        # Just verify the image has been modified
        assert not np.array_equal(warped, image), "Rotation should change image"
        # Validity mask should exist
        assert validity_mask is not None, "Validity mask should be returned"


class TestTintedLayerCreation:
    """Tests for creating tinted BGRA layers from masks."""
    
    @pytest.fixture
    def service(self):
        return OverlayService()
    
    def test_empty_mask_produces_transparent_layer(self, service):
        """Empty mask should produce fully transparent layer."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        layer = service.create_tinted_layer(mask, color=(0, 0, 255), alpha=0.7)
        
        assert layer.shape == (100, 100, 4)
        assert np.all(layer[:, :, 3] == 0), "Alpha should be 0 for empty mask"
    
    def test_full_mask_produces_colored_layer(self, service):
        """Full mask should produce fully colored layer."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        layer = service.create_tinted_layer(mask, color=(0, 0, 255), alpha=0.7)
        
        # Check color (BGR)
        assert np.all(layer[:, :, 0] == 0), "Blue channel"
        assert np.all(layer[:, :, 1] == 0), "Green channel"  
        assert np.all(layer[:, :, 2] == 255), "Red channel"
        # Check alpha (0.7 * 255 = 178.5 -> 178)
        assert np.all(layer[:, :, 3] == 178), "Alpha should be 0.7 * 255"
    
    def test_partial_mask(self, service):
        """Partial mask should produce mixed layer."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # Center square is mask
        
        layer = service.create_tinted_layer(mask, color=(255, 255, 0), alpha=0.5)
        
        # Masked region should have color and alpha
        assert np.all(layer[25:75, 25:75, 3] > 0), "Masked region should have alpha"
        # Unmasked region should be transparent
        assert np.all(layer[0:25, :, 3] == 0), "Unmasked region should be transparent"


class TestOverlayComposite:
    """Tests for compositing tinted layers."""
    
    @pytest.fixture
    def service(self):
        return OverlayService()
    
    def test_empty_layers_produce_white_background(self, service):
        """Empty layers should show white background."""
        layer_a = np.zeros((100, 100, 4), dtype=np.uint8)
        layer_b = np.zeros((100, 100, 4), dtype=np.uint8)
        
        result = service.composite_layers(layer_a, layer_b)
        
        assert result.shape == (100, 100, 3)
        # Should be white
        assert np.all(result == 255), "Empty layers should show white background"
    
    def test_single_layer_shows_tint(self, service):
        """Single layer should show its tint color."""
        layer_a = np.zeros((100, 100, 4), dtype=np.uint8)
        layer_a[40:60, 40:60, :] = [0, 0, 255, 178]  # Red square with alpha
        layer_b = np.zeros((100, 100, 4), dtype=np.uint8)
        
        result = service.composite_layers(layer_a, layer_b)
        
        # Red region should be visible
        red_region = result[40:60, 40:60, :]
        assert np.mean(red_region[:, :, 2]) > 200, "Red channel should be high"
    
    def test_overlapping_layers_darkened(self, service):
        """Overlapping regions should be darkened."""
        # Both layers have content in same region
        layer_a = np.zeros((100, 100, 4), dtype=np.uint8)
        layer_a[40:60, 40:60, :] = [0, 0, 255, 178]  # Red
        
        layer_b = np.zeros((100, 100, 4), dtype=np.uint8)
        layer_b[40:60, 40:60, :] = [255, 255, 0, 178]  # Cyan
        
        result = service.composite_layers(layer_a, layer_b)
        
        # Overlapping region should be darker than pure cyan
        overlap_region = result[40:60, 40:60, :]
        # When both layers overlap, the result should be darkened
        assert np.mean(overlap_region) < 200, "Overlapping region should be darker"


class TestFullOverlayPipeline:
    """End-to-end tests for the full overlay generation pipeline."""
    
    @pytest.fixture
    def service(self):
        return OverlayService()
    
    def create_simple_drawing(self, width, height, line_coords):
        """Create a simple white image with black lines."""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        for y1, y2, x1, x2 in line_coords:
            image[y1:y2, x1:x2, :] = 0
        return image
    
    def test_identical_images_produce_single_color_overlay(self, service):
        """Identical images with identity transform should show consistent coloring."""
        # Create identical images with a cross pattern
        image_a = self.create_simple_drawing(200, 200, [
            (95, 105, 50, 150),  # Horizontal line
            (50, 150, 95, 105),  # Vertical line
        ])
        image_b = image_a.copy()
        
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        result = service.generate_overlay(image_a, image_b, transform)
        
        assert result.overlay_image.shape[:2] == (200, 200)
        assert result.width == 200
        assert result.height == 200
        
        # The overlap regions should be dark (both colors present)
        # Check center area where cross is
        center = result.overlay_image[95:105, 95:105, :]
        assert np.mean(center) < 150, "Overlapping lines should be dark"
    
    def test_translated_overlay_shows_both_colors(self, service):
        """Translated image should show both red and cyan regions."""
        # Image A: horizontal line
        image_a = self.create_simple_drawing(200, 200, [(95, 105, 50, 150)])
        
        # Image B: same line (will be translated)
        image_b = self.create_simple_drawing(200, 200, [(95, 105, 50, 150)])
        
        # Translate B by 20 pixels horizontally
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=20.0, ty=0.0)
        
        result = service.generate_overlay(image_a, image_b, transform)
        
        # Should have non-white pixels (lines are present)
        non_white_pixels = np.sum(result.overlay_image < 250)
        assert non_white_pixels > 0, "Should have visible lines"
    
    def test_diff_mask_generated(self, service):
        """Diff mask should be generated and show differences."""
        # Different images
        image_a = self.create_simple_drawing(200, 200, [(95, 105, 50, 150)])  # Horizontal
        image_b = self.create_simple_drawing(200, 200, [(50, 150, 95, 105)])  # Vertical
        
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        result = service.generate_overlay(image_a, image_b, transform)
        
        assert result.diff_mask is not None
        assert result.diff_mask.shape == (200, 200)
        # XOR of different patterns should have non-zero pixels
        assert np.sum(result.diff_mask > 0) > 0, "Diff mask should show differences"
    
    def test_overlay_preserves_line_information(self, service):
        """Overlay should preserve line positions from both images."""
        # Image A: horizontal line at y=50
        image_a = self.create_simple_drawing(200, 200, [(48, 52, 50, 150)])
        
        # Image B: horizontal line at y=100
        image_b = self.create_simple_drawing(200, 200, [(98, 102, 50, 150)])
        
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        result = service.generate_overlay(image_a, image_b, transform)
        
        # Line A region (y=50) should have non-white pixels
        line_a_region = result.overlay_image[48:52, 75:125, :]
        assert np.mean(line_a_region) < 250, "Line A should be visible"
        
        # Line B region (y=100) should have non-white pixels
        line_b_region = result.overlay_image[98:102, 75:125, :]
        assert np.mean(line_b_region) < 250, "Line B should be visible"
    
    def test_processing_time_recorded(self, service):
        """Processing time should be recorded in result."""
        image_a = self.create_simple_drawing(200, 200, [(95, 105, 50, 150)])
        image_b = image_a.copy()
        
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        result = service.generate_overlay(image_a, image_b, transform)
        
        assert result.processing_time_ms >= 0
    
    def test_custom_config(self, service):
        """Custom configuration should be applied."""
        image_a = self.create_simple_drawing(200, 200, [(95, 105, 50, 150)])
        image_b = image_a.copy()
        
        transform = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        custom_config = OverlayConfig(
            color_a=(0, 255, 0),  # Green
            color_b=(255, 0, 255),  # Magenta
            alpha_a=0.5,
            alpha_b=0.5,
            line_threshold=200,
        )
        
        result = service.generate_overlay(image_a, image_b, transform, config=custom_config)
        
        assert result.config.color_a == (0, 255, 0)
        assert result.config.alpha_a == 0.5


class TestOverlayWithRealTransforms:
    """Tests combining alignment transforms with overlay generation."""
    
    @pytest.fixture
    def overlay_service(self):
        return OverlayService()
    
    def create_simple_drawing(self, width, height, line_coords):
        """Create a simple white image with black lines."""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        for y1, y2, x1, x2 in line_coords:
            image[y1:y2, x1:x2, :] = 0
        return image
    
    def test_scaled_overlay(self, overlay_service):
        """Overlay with scaled image B."""
        # Image A: 10px thick line
        image_a = self.create_simple_drawing(400, 400, [(195, 205, 100, 300)])
        
        # Image B: 5px thick line (will be scaled 2x)
        image_b = self.create_simple_drawing(400, 400, [(197, 202, 100, 200)])
        
        # Scale B by 2x
        transform = SimilarityTransform(scale=2.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        result = overlay_service.generate_overlay(image_a, image_b, transform)
        
        # Should produce valid overlay
        assert result.overlay_image.shape == (400, 400, 3)
    
    def test_rotated_overlay(self, overlay_service):
        """Overlay with rotated image B."""
        # Image A: horizontal line
        image_a = self.create_simple_drawing(400, 400, [(195, 205, 50, 350)])
        
        # Image B: horizontal line (will be rotated 45 degrees)
        image_b = self.create_simple_drawing(400, 400, [(195, 205, 50, 350)])
        
        # Rotate B by 45 degrees
        transform = SimilarityTransform(
            scale=1.0, 
            rotation_rad=math.pi/4, 
            tx=200.0,  # Translate to keep in view
            ty=-100.0
        )
        
        result = overlay_service.generate_overlay(image_a, image_b, transform)
        
        # Should produce valid overlay
        assert result.overlay_image.shape == (400, 400, 3)
        # Lines should not perfectly overlap (rotation applied)
        # Check that diff mask shows differences
        assert result.diff_mask is not None
