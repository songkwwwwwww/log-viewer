"""Data structures for logged dynamic objects and scene frames."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .geometry import Point3D, Quaternion


@dataclass
class PoseAtTime:
    """A position and orientation snapshot at a specific point in time."""

    timestamp: float
    position: Point3D
    orientation: Quaternion


@dataclass
class ObjectState:
    """Represents the real-time state of an entity (vehicle, pedestrian)."""

    id: str
    type: str
    position: Point3D
    velocity: Point3D
    acceleration: Point3D
    orientation: Quaternion
    size: Tuple[float, float, float]
    future_trajectory: Optional[List[PoseAtTime]] = None


@dataclass
class SceneFrame:
    """A collection of all object states at a specific timestamp."""

    timestamp: float
    objects: Dict[str, ObjectState]
