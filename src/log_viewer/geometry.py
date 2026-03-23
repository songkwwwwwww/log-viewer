"""Shared geometry data structures used across map and scene models.

These structures provide basic 3D primitives for representing positions and 
orientations in a Cartesian coordinate system.
"""

from dataclasses import dataclass


@dataclass
class Point3D:
    """Represents a point in 3D Cartesian coordinates.
    
    Attributes:
        x: X-coordinate (typically East/Forward in meters).
        y: Y-coordinate (typically North/Left in meters).
        z: Z-coordinate (typically Up in meters).
    """

    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    """Represents a 3D rotation in quaternion form (w, x, y, z).
    
    Used for representing orientations of objects without gimbal lock.
    
    Attributes:
        w: Scalar component.
        x: Vector x component.
        y: Vector y component.
        z: Vector z component.
    """

    w: float
    x: float
    y: float
    z: float
