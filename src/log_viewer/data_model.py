"""Module defining the core data structures for Log Viewer."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


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


@dataclass
class LaneBoundary:
    """Defines the geometry and visual style of a lane edge."""

    id: str  # ex: "road_1_lane_-1_left"
    style: str  # solid, dashed, etc.
    points: List[Point3D]


@dataclass
class Lane:
    """Represents a single drivable or walkable lane segment."""

    id: str
    road_id: str
    type: str  # e.g., driving, sidewalk
    left_boundary: LaneBoundary
    right_boundary: LaneBoundary
    center_line: List[Point3D]
    is_left: bool = False  # True = travels opposite to reference line direction


@dataclass
class RoadLink:
    """Represents a directional connection between two roads in OpenDRIVE."""

    road_id: str
    successor_road_id: Optional[str]  # None if connects to a junction
    successor_type: str  # "road" or "junction"
    end_point: Point3D  # last point of this road's centerline
    successor_start_point: Optional[Point3D]  # first point of successor road, if known


@dataclass
class XodrMapData:
    """Contains all lanes and connectivity extracted from an OpenDRIVE map."""

    lanes: List[Lane]
    road_links: List[RoadLink] = None  # road-to-road successor connections

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.road_links is None:
            self.road_links = []


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
