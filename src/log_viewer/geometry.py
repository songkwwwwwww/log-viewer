"""Shared geometry data structures used across map and scene models."""

from dataclasses import dataclass


@dataclass
class Point3D:
    """Represents a point in 3D Cartesian coordinates."""

    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    """Represents a 3D rotation in quaternion form (w, x, y, z)."""

    w: float
    x: float
    y: float
    z: float
