"""
Core alignment types and math utilities.

Provides fundamental types for similarity transforms used throughout the alignment pipeline.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RotationConstraint(str, Enum):
    """Rotation constraint modes for alignment."""
    NONE = "NONE"              # No rotation allowed (translation + scale only)
    SNAP_90 = "SNAP_90"        # Snap to nearest 90° (0°, 90°, 180°, 270°)
    FREE = "FREE"              # Allow any rotation


@dataclass
class Point2D:
    """A 2D point."""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @classmethod
    def from_dict(cls, d: dict) -> "Point2D":
        return cls(x=float(d["x"]), y=float(d["y"]))


@dataclass
class SimilarityTransform:
    """
    A similarity transform: uniform scale + rotation + translation.
    
    Maps point p_B in image B to point p_A in image A:
        p_A = s * R(θ) * p_B + t
    
    The 2x3 affine matrix is:
        [[s*cos(θ), -s*sin(θ), tx],
         [s*sin(θ),  s*cos(θ), ty]]
    """
    scale: float
    rotation_rad: float
    tx: float
    ty: float
    
    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.rotation_rad)
    
    @property
    def matrix_2x3(self) -> np.ndarray:
        """Get the 2x3 affine transformation matrix."""
        cos_t = math.cos(self.rotation_rad)
        sin_t = math.sin(self.rotation_rad)
        s = self.scale
        
        return np.array([
            [s * cos_t, -s * sin_t, self.tx],
            [s * sin_t,  s * cos_t, self.ty]
        ], dtype=np.float64)
    
    @property
    def matrix_2x3_list(self) -> list:
        """Get the 2x3 matrix as a nested list for JSON serialization."""
        m = self.matrix_2x3
        return [[float(m[0, 0]), float(m[0, 1]), float(m[0, 2])],
                [float(m[1, 0]), float(m[1, 1]), float(m[1, 2])]]
    
    def transform_point(self, p: Point2D) -> Point2D:
        """Apply this transform to a point."""
        vec = p.to_array()
        m = self.matrix_2x3
        result = m[:, :2] @ vec + m[:, 2]
        return Point2D(x=result[0], y=result[1])
    
    def to_params_dict(self) -> dict:
        """Get transform parameters as a dictionary."""
        return {
            "scale": round(self.scale, 6),
            "rotation_deg": round(self.rotation_deg, 4),
            "rotation_rad": round(self.rotation_rad, 6),
            "tx": round(self.tx, 4),
            "ty": round(self.ty, 4),
        }


class AlignmentError(Exception):
    """Error during alignment computation."""
    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
