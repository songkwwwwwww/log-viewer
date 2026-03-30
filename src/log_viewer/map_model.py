"""Data structures for parsed OpenDRIVE map content.

This module defines the structural representation of an OpenDRIVE road network,
focusing on the geometric boundaries, lane types, and connectivity between roads.
"""

from dataclasses import dataclass
from typing import List, Optional

from .geometry import Point3D


@dataclass
class LaneBoundary:
    """Defines the geometry and visual style of a lane edge.

    Attributes:
        id: Unique identifier for the boundary (e.g., "road_1_lane_-1_left").
        style: Visual style of the boundary line (e.g., "solid", "dashed").
        points: A sequence of 3D points defining the boundary's shape in meters.
    """

    id: str
    style: str
    points: List[Point3D]


@dataclass
class Lane:
    """Represents a single drivable or walkable lane segment.

    A lane is defined by its boundaries and its functional type. It is part of
    a road and follows the road's reference line.

    Attributes:
        id: Identifier for the lane (usually the integer ID from OpenDRIVE).
        road_id: ID of the road this lane belongs to.
        type: Functional type (e.g., "driving", "sidewalk", "biking", "shoulder").
        left_boundary: The boundary to the left relative to the road direction.
        right_boundary: The boundary to the right relative to the road direction.
        center_line: Calculated center line of the lane.
        is_left: True if the lane is on the left side of the road reference line
            (usually implies travel direction opposite to the reference line).
    """

    id: str
    road_id: str
    type: str
    left_boundary: LaneBoundary
    right_boundary: LaneBoundary
    center_line: List[Point3D]
    is_left: bool = False


@dataclass
class RoadMark:
    """A single road marking segment (solid line, dashed segment, etc.).

    Mirrors the RoadMark struct in libOpenDRIVE/include/RoadMark.h.

    Attributes:
        left_pts: 3D points along the left (inner) edge of the mark.
        right_pts: 3D points along the right (outer) edge of the mark.
        mark_type: Marking type string (e.g. "solid", "broken").
        color: Color hint from OpenDRIVE (e.g. "white", "yellow", "standard").
    """

    left_pts: List[Point3D]
    right_pts: List[Point3D]
    mark_type: str
    color: str


@dataclass
class RoadLink:
    """Represents a directional connection between two roads in OpenDRIVE.

    Used for visualizing the connectivity and flow between different road segments.

    Attributes:
        road_id: ID of the source road.
        successor_road_id: ID of the connected road, if it's a direct road-to-road link.
        successor_type: Type of connection ("road" or "junction").
        end_point: The last point of the source road's reference line.
        successor_start_point: The first point of the successor road's reference line.
    """

    road_id: str
    successor_road_id: Optional[str]
    successor_type: str
    end_point: Point3D
    successor_start_point: Optional[Point3D]


@dataclass
class XodrMapData:
    """Contains all lanes and connectivity extracted from an OpenDRIVE map.

    This is the top-level container for all static map information used by the viewer.

    Attributes:
        lanes: List of all parsed Lane objects.
        road_links: List of connectivity links between roads.
        road_marks: List of all parsed RoadMark objects.
    """

    lanes: List[Lane]
    road_links: List[RoadLink] = None
    road_marks: List[RoadMark] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.road_links is None:
            self.road_links = []
        if self.road_marks is None:
            self.road_marks = []
