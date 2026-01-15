"""
Unit tests for alignment math - verifies transform maps points correctly.

Note: Default rotation constraint is NONE (no rotation) for architectural drawings.
Tests that need rotation must explicitly use rotation_constraint=RotationConstraint.FREE.
"""

import math
import pytest
import numpy as np

from app.services.align import (
    AlignService,
    SimilarityTransform,
    Point2D,
    AlignmentError,
    RotationConstraint,
)


class TestPoint2D:
    """Tests for Point2D class."""
    
    def test_to_array(self):
        """Point converts to numpy array correctly."""
        p = Point2D(x=100.5, y=200.5)
        arr = p.to_array()
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, [100.5, 200.5])
    
    def test_from_dict(self):
        """Point can be created from dict."""
        p = Point2D.from_dict({"x": 50, "y": 75})
        assert p.x == 50.0
        assert p.y == 75.0


class TestSimilarityTransform:
    """Tests for SimilarityTransform class."""
    
    def test_identity_transform(self):
        """Identity transform: scale=1, rotation=0, translation=0."""
        t = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        p = Point2D(x=100, y=200)
        result = t.transform_point(p)
        
        assert abs(result.x - p.x) < 1e-10
        assert abs(result.y - p.y) < 1e-10
    
    def test_translation_only(self):
        """Transform with only translation."""
        t = SimilarityTransform(scale=1.0, rotation_rad=0.0, tx=50.0, ty=-30.0)
        
        p = Point2D(x=100, y=200)
        result = t.transform_point(p)
        
        assert abs(result.x - 150) < 1e-10
        assert abs(result.y - 170) < 1e-10
    
    def test_scale_only(self):
        """Transform with only scaling."""
        t = SimilarityTransform(scale=2.0, rotation_rad=0.0, tx=0.0, ty=0.0)
        
        p = Point2D(x=100, y=50)
        result = t.transform_point(p)
        
        assert abs(result.x - 200) < 1e-10
        assert abs(result.y - 100) < 1e-10
    
    def test_rotation_90_degrees(self):
        """Transform with 90-degree rotation."""
        t = SimilarityTransform(scale=1.0, rotation_rad=math.pi/2, tx=0.0, ty=0.0)
        
        # Point at (100, 0) should rotate to (0, 100)
        p = Point2D(x=100, y=0)
        result = t.transform_point(p)
        
        assert abs(result.x - 0) < 1e-10
        assert abs(result.y - 100) < 1e-10
    
    def test_rotation_180_degrees(self):
        """Transform with 180-degree rotation."""
        t = SimilarityTransform(scale=1.0, rotation_rad=math.pi, tx=0.0, ty=0.0)
        
        # Point at (100, 50) should rotate to (-100, -50)
        p = Point2D(x=100, y=50)
        result = t.transform_point(p)
        
        assert abs(result.x - (-100)) < 1e-10
        assert abs(result.y - (-50)) < 1e-10
    
    def test_combined_transform(self):
        """Transform with scale, rotation, and translation combined."""
        # Scale by 2, rotate 90 degrees, translate by (10, 20)
        t = SimilarityTransform(scale=2.0, rotation_rad=math.pi/2, tx=10.0, ty=20.0)
        
        # Point at (100, 0):
        # - Scale: (200, 0)
        # - Rotate 90°: (0, 200)
        # - Translate: (10, 220)
        p = Point2D(x=100, y=0)
        result = t.transform_point(p)
        
        assert abs(result.x - 10) < 1e-10
        assert abs(result.y - 220) < 1e-10
    
    def test_matrix_2x3_format(self):
        """Matrix should be in correct format for cv2.warpAffine."""
        t = SimilarityTransform(scale=1.5, rotation_rad=math.pi/4, tx=100.0, ty=50.0)
        
        m = t.matrix_2x3
        assert m.shape == (2, 3)
        
        # Check matrix structure: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        s = 1.5
        cos_t = math.cos(math.pi/4)
        sin_t = math.sin(math.pi/4)
        
        assert abs(m[0, 0] - s * cos_t) < 1e-10
        assert abs(m[0, 1] - (-s * sin_t)) < 1e-10
        assert abs(m[0, 2] - 100.0) < 1e-10
        assert abs(m[1, 0] - s * sin_t) < 1e-10
        assert abs(m[1, 1] - s * cos_t) < 1e-10
        assert abs(m[1, 2] - 50.0) < 1e-10
    
    def test_rotation_deg_property(self):
        """Rotation in degrees computed correctly."""
        t = SimilarityTransform(scale=1.0, rotation_rad=math.pi/4, tx=0.0, ty=0.0)
        assert abs(t.rotation_deg - 45.0) < 1e-10
    
    def test_params_dict(self):
        """Parameters can be exported as dict."""
        t = SimilarityTransform(scale=1.5, rotation_rad=0.5, tx=100.0, ty=50.0)
        params = t.to_params_dict()
        
        assert "scale" in params
        assert "rotation_deg" in params
        assert "rotation_rad" in params
        assert "tx" in params
        assert "ty" in params


class TestAlignService:
    """Tests for AlignService alignment computation."""
    
    @pytest.fixture
    def service(self):
        return AlignService()
    
    def test_identity_alignment(self, service):
        """Same points should yield identity-like transform."""
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=100)]
        points_b = [Point2D(x=100, y=100), Point2D(x=300, y=100)]
        
        result = service.compute_similarity_transform(points_a, points_b)
        
        assert abs(result.transform.scale - 1.0) < 1e-6
        assert abs(result.transform.rotation_deg) < 0.01
        assert result.confidence == 1.0
        assert result.residual_error < 1e-6
    
    def test_translation_alignment(self, service):
        """Points offset by translation should compute correct translation."""
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=100)]
        points_b = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
        
        result = service.compute_similarity_transform(points_a, points_b)
        
        assert abs(result.transform.scale - 1.0) < 1e-6
        assert abs(result.transform.tx - 50.0) < 1e-6
        assert abs(result.transform.ty - 50.0) < 1e-6
    
    def test_scale_alignment(self, service):
        """Points at different scales should compute correct scale factor."""
        # A has distance 200, B has distance 100 -> scale = 2.0
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=100)]
        points_b = [Point2D(x=100, y=100), Point2D(x=200, y=100)]
        
        result = service.compute_similarity_transform(points_a, points_b)
        
        assert abs(result.transform.scale - 2.0) < 1e-6
        assert abs(result.transform.rotation_deg) < 0.01
    
    def test_rotation_alignment(self, service):
        """Rotated points should compute correct rotation angle (with FREE constraint)."""
        # A: horizontal vector
        # B: vertical vector (90 degrees rotated)
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=100)]  # horizontal
        points_b = [Point2D(x=100, y=100), Point2D(x=100, y=300)]  # vertical (rotated -90°)
        
        # Must use FREE rotation to allow arbitrary rotation
        result = service.compute_similarity_transform(
            points_a, points_b, rotation_constraint=RotationConstraint.FREE
        )
        
        assert abs(result.transform.scale - 1.0) < 1e-6
        # A is 0°, B is 90°, so rotation = 0° - 90° = -90°
        assert abs(result.transform.rotation_deg - (-90.0)) < 0.01
    
    def test_rotation_snap_90(self, service):
        """SNAP_90 constraint should snap to nearest 90°."""
        # Small rotation that should snap to 0°
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=110)]  # ~3° from horizontal
        points_b = [Point2D(x=100, y=100), Point2D(x=300, y=100)]  # horizontal
        
        result = service.compute_similarity_transform(
            points_a, points_b, rotation_constraint=RotationConstraint.SNAP_90
        )
        
        # Should snap to 0°
        assert abs(result.transform.rotation_deg) < 0.01
    
    def test_rotation_none(self, service):
        """NONE constraint should force rotation to 0° (default)."""
        # Points that would normally produce rotation
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=200)]  # diagonal
        points_b = [Point2D(x=100, y=100), Point2D(x=300, y=100)]  # horizontal
        
        result = service.compute_similarity_transform(
            points_a, points_b, rotation_constraint=RotationConstraint.NONE
        )
        
        # Should be 0°
        assert abs(result.transform.rotation_deg) < 0.01
    
    def test_combined_alignment(self, service):
        """Combined scale, rotation, and translation (with FREE constraint)."""
        # Create points where B needs to be:
        # - Scaled by 1.5
        # - Rotated by 45 degrees
        # - Translated
        
        # For testing, we'll verify the transform maps p1_b -> p1_a and p2_b -> p2_a
        points_a = [Point2D(x=100, y=200), Point2D(x=300, y=400)]
        points_b = [Point2D(x=50, y=100), Point2D(x=150, y=200)]
        
        # Use FREE rotation to allow the full transform
        result = service.compute_similarity_transform(
            points_a, points_b, rotation_constraint=RotationConstraint.FREE
        )
        
        # Verify mapping
        mapped_p1 = result.transform.transform_point(points_b[0])
        mapped_p2 = result.transform.transform_point(points_b[1])
        
        assert abs(mapped_p1.x - points_a[0].x) < 1e-4
        assert abs(mapped_p1.y - points_a[0].y) < 1e-4
        assert abs(mapped_p2.x - points_a[1].x) < 1e-4
        assert abs(mapped_p2.y - points_a[1].y) < 1e-4
    
    def test_residual_error_is_zero_for_exact_fit(self, service):
        """Residual error should be essentially zero for valid two-point alignment."""
        points_a = [Point2D(x=100, y=100), Point2D(x=400, y=300)]
        points_b = [Point2D(x=50, y=50), Point2D(x=200, y=150)]
        
        result = service.compute_similarity_transform(points_a, points_b)
        
        # Two-point similarity transform should have zero residual
        assert result.residual_error < 1e-6
    
    def test_degenerate_points_raises_error(self, service):
        """Points too close together should raise AlignmentError."""
        # Points only 5 pixels apart (below minimum)
        points_a = [Point2D(x=100, y=100), Point2D(x=103, y=104)]
        points_b = [Point2D(x=100, y=100), Point2D(x=300, y=300)]
        
        with pytest.raises(AlignmentError) as excinfo:
            service.compute_similarity_transform(points_a, points_b)
        
        assert excinfo.value.code == "DEGENERATE_POINTS"
        assert "image a" in excinfo.value.message.lower()
    
    def test_degenerate_points_in_b_raises_error(self, service):
        """Points too close in B should also raise error."""
        points_a = [Point2D(x=100, y=100), Point2D(x=300, y=300)]
        points_b = [Point2D(x=100, y=100), Point2D(x=105, y=105)]
        
        with pytest.raises(AlignmentError) as excinfo:
            service.compute_similarity_transform(points_a, points_b)
        
        assert excinfo.value.code == "DEGENERATE_POINTS"
        assert "image b" in excinfo.value.message.lower()
    
    def test_wrong_point_count_raises_error(self, service):
        """Wrong number of points should raise error."""
        points_a = [Point2D(x=100, y=100)]  # Only 1 point
        points_b = [Point2D(x=100, y=100), Point2D(x=300, y=300)]
        
        with pytest.raises(AlignmentError) as excinfo:
            service.compute_similarity_transform(points_a, points_b)
        
        assert excinfo.value.code == "INVALID_POINT_COUNT"
    
    def test_point_bounds_validation(self, service):
        """Points outside image bounds should be detected."""
        points = [Point2D(x=500, y=100), Point2D(x=600, y=200)]
        
        with pytest.raises(AlignmentError) as excinfo:
            service.validate_points_in_bounds(points, width=400, height=400, image_name="A")
        
        assert excinfo.value.code == "POINT_OUT_OF_BOUNDS"
    
    def test_negative_point_detected(self, service):
        """Negative coordinates should be detected as out of bounds."""
        points = [Point2D(x=-10, y=100), Point2D(x=200, y=200)]
        
        with pytest.raises(AlignmentError) as excinfo:
            service.validate_points_in_bounds(points, width=400, height=400, image_name="B")
        
        assert excinfo.value.code == "POINT_OUT_OF_BOUNDS"
    
    def test_confidence_reduced_for_extreme_scale(self, service):
        """Confidence should be reduced for extreme scale factors."""
        # Scale factor > 2.0
        points_a = [Point2D(x=100, y=100), Point2D(x=600, y=100)]  # distance = 500
        points_b = [Point2D(x=100, y=100), Point2D(x=150, y=100)]  # distance = 50, scale = 10
        
        result = service.compute_similarity_transform(points_a, points_b)
        
        assert result.confidence < 1.0
    
    def test_confidence_reduced_for_large_rotation(self, service):
        """Confidence should be reduced for large rotation angles (with FREE constraint)."""
        # Rotation > 15 degrees - must use FREE to actually get rotation
        import math
        # Create 30-degree rotation
        dist = 200
        angle_b = math.radians(0)
        angle_a = math.radians(30)
        
        points_a = [
            Point2D(x=100, y=100),
            Point2D(x=100 + dist * math.cos(angle_a), y=100 + dist * math.sin(angle_a))
        ]
        points_b = [
            Point2D(x=100, y=100),
            Point2D(x=100 + dist * math.cos(angle_b), y=100 + dist * math.sin(angle_b))
        ]
        
        result = service.compute_similarity_transform(
            points_a, points_b, rotation_constraint=RotationConstraint.FREE
        )
        
        assert result.confidence < 1.0


class TestTransformMapsPointsCorrectly:
    """Specific tests to verify the core requirement: transform maps points correctly."""
    
    @pytest.fixture
    def service(self):
        return AlignService()
    
    def test_arbitrary_points_map_correctly_with_free_rotation(self, service):
        """Arbitrary point pairs should map p1B->p1A and p2B->p2A within tolerance (FREE rotation)."""
        # Arbitrary realistic architectural drawing coordinates
        points_a = [Point2D(x=234.5, y=567.8), Point2D(x=456.7, y=789.0)]
        points_b = [Point2D(x=100.1, y=200.2), Point2D(x=300.3, y=400.4)]
        
        # Use FREE rotation to get exact mapping
        result = service.compute_similarity_transform(
            points_a, points_b, rotation_constraint=RotationConstraint.FREE
        )
        
        # Map B points and verify they land on A points
        mapped_p1 = result.transform.transform_point(points_b[0])
        mapped_p2 = result.transform.transform_point(points_b[1])
        
        tolerance = 0.001  # Sub-pixel tolerance
        
        assert abs(mapped_p1.x - points_a[0].x) < tolerance, f"p1 x mismatch: {mapped_p1.x} vs {points_a[0].x}"
        assert abs(mapped_p1.y - points_a[0].y) < tolerance, f"p1 y mismatch: {mapped_p1.y} vs {points_a[0].y}"
        assert abs(mapped_p2.x - points_a[1].x) < tolerance, f"p2 x mismatch: {mapped_p2.x} vs {points_a[1].x}"
        assert abs(mapped_p2.y - points_a[1].y) < tolerance, f"p2 y mismatch: {mapped_p2.y} vs {points_a[1].y}"
    
    def test_axis_aligned_points_map_correctly_default(self, service):
        """Axis-aligned points should map correctly with default (NONE) rotation."""
        # Same angle vectors - no rotation needed
        points_a = [Point2D(x=100, y=100), Point2D(x=400, y=400)]  # 45° diagonal
        points_b = [Point2D(x=50, y=50), Point2D(x=200, y=200)]    # Same 45° diagonal
        
        result = service.compute_similarity_transform(points_a, points_b)
        
        # Rotation should be 0
        assert abs(result.transform.rotation_deg) < 0.01
        
        # Scale should be 2 (300 / 150)
        assert abs(result.transform.scale - 2.0) < 0.01
        
        # Centroids should match after transform
        mapped_centroid = result.transform.transform_point(Point2D(x=125, y=125))
        expected_centroid = Point2D(x=250, y=250)
        assert abs(mapped_centroid.x - expected_centroid.x) < 1
        assert abs(mapped_centroid.y - expected_centroid.y) < 1
    
    def test_many_random_pairs_free_rotation(self, service):
        """Test with many random point pairs with FREE rotation to ensure robustness."""
        np.random.seed(42)  # Deterministic
        
        for i in range(20):
            # Generate random points (ensuring minimum distance)
            p1_a = Point2D(x=np.random.uniform(50, 400), y=np.random.uniform(50, 400))
            offset_a = np.random.uniform(100, 300)
            angle_a = np.random.uniform(0, 2*math.pi)
            p2_a = Point2D(x=p1_a.x + offset_a*math.cos(angle_a), 
                          y=p1_a.y + offset_a*math.sin(angle_a))
            
            p1_b = Point2D(x=np.random.uniform(50, 400), y=np.random.uniform(50, 400))
            offset_b = np.random.uniform(100, 300)
            angle_b = np.random.uniform(0, 2*math.pi)
            p2_b = Point2D(x=p1_b.x + offset_b*math.cos(angle_b),
                          y=p1_b.y + offset_b*math.sin(angle_b))
            
            # Use FREE rotation to test full transform capability
            result = service.compute_similarity_transform(
                [p1_a, p2_a], [p1_b, p2_b], 
                rotation_constraint=RotationConstraint.FREE
            )
            
            mapped_p1 = result.transform.transform_point(p1_b)
            mapped_p2 = result.transform.transform_point(p2_b)
            
            tolerance = 0.01
            assert abs(mapped_p1.x - p1_a.x) < tolerance, f"Iteration {i}: p1 x mismatch"
            assert abs(mapped_p1.y - p1_a.y) < tolerance, f"Iteration {i}: p1 y mismatch"
            assert abs(mapped_p2.x - p2_a.x) < tolerance, f"Iteration {i}: p2 x mismatch"
            assert abs(mapped_p2.y - p2_a.y) < tolerance, f"Iteration {i}: p2 y mismatch"
