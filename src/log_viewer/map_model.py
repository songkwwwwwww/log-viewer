"""Data structures for parsed OpenDRIVE map content."""

from dataclasses import dataclass
from typing import List, Optional

from .geometry import Point3D


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
