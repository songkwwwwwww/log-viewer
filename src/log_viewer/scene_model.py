"""Data structures for logged dynamic objects and scene frames.

This module defines the state of dynamic entities (vehicles, pedestrians, etc.)
over time, as captured in simulation logs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .geometry import Point3D, Quaternion


@dataclass
class PoseAtTime:
    """A position and orientation snapshot at a specific point in time.

    Used primarily for representing future trajectories or historical paths.

    Attributes:
        timestamp: Time in seconds.
        position: 3D coordinates in meters.
        orientation: Rotation as a quaternion.
    """

    timestamp: float
    position: Point3D
    orientation: Quaternion


@dataclass
class ObjectState:
    """Represents the real-time state of an entity (vehicle, pedestrian).

    Attributes:
        id: Unique identifier for the object.
        type: Classification of the object (e.g., "vehicle", "pedestrian").
        position: 3D coordinates of the object's center in meters.
        velocity: Velocity vector in meters/second.
        acceleration: Acceleration vector in meters/second^2.
        orientation: Rotation as a quaternion.
        size: Dimensions of the object (length, width, height) in meters.
        future_trajectory: Optional list of predicted future poses.
    """

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
    """A collection of all object states at a specific timestamp.

    Represents a single "tick" or "snapshot" of the world state.

    Attributes:
        timestamp: Simulation time in seconds.
        objects: Dictionary mapping object IDs to their state in this frame.
    """

    timestamp: float
    objects: Dict[str, ObjectState]
