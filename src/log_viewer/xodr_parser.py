import xml.etree.ElementTree as ET
import math
from dataclasses import dataclass
from typing import List, Optional

from .geometry import Point3D
from .map_model import Lane, LaneBoundary, RoadLink, XodrMapData


@dataclass
class ElevationEntry:
    """Represents a single elevation record in a cubic polynomial profile."""

    s: float
    a: float
    b: float
    c: float
    d: float


class ElevationProfile:
    """Manages multiple elevation entries and evaluates Z at any s position."""

    def __init__(self, entries: List[ElevationEntry]):
        """Initialize with a list of elevation entries, sorted by s."""
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_z(self, s: float) -> float:
        """Calculate the elevation Z at distance s along the road."""
        if not self.entries:
            return 0.0

        # Find the entry starting at or just before s
        active_entry = self.entries[0]
        for entry in self.entries:
            if s >= entry.s:
                active_entry = entry
            else:
                break

        ds = s - active_entry.s
        return (
            active_entry.a
            + active_entry.b * ds
            + active_entry.c * ds**2
            + active_entry.d * ds**3
        )


class XodrParser:
    """A minimal OpenDRIVE (XODR) parser for visualization purposes."""

    def __init__(self, file_path: str):
        """Initialize parser from file path."""
        self.file_path = file_path
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()

    def parse(self) -> XodrMapData:
        """Parse the OpenDRIVE XML into XodrMapData."""
        lanes_to_render = []
        road_links = []

        # First pass: collect reference line endpoints per road
        road_ref_endpoints: dict = {}  # road_id -> (start_point, end_point)
        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")
            ref_pts = self._parse_plan_view(road)
            if ref_pts:
                road_ref_endpoints[road_id] = (ref_pts[0], ref_pts[-1])

        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")

            # 1. Parse Elevation Profile
            elevation_profile = self._parse_elevation_profile(road)

            # 2. Parse Reference Line (planView)
            ref_line_pts = self._parse_plan_view(
                road, elevation_profile=elevation_profile
            )
            if not ref_line_pts:
                continue

            # 2. Parse successor link
            link_tag = road.find("link")
            if link_tag is not None:
                succ_tag = link_tag.find("successor")
                if succ_tag is not None:
                    succ_type = succ_tag.get("elementType", "road")
                    succ_id = succ_tag.get("elementId")
                    end_pt = ref_line_pts[-1]
                    succ_start = None
                    if succ_type == "road" and succ_id in road_ref_endpoints:
                        succ_start = road_ref_endpoints[succ_id][0]
                    road_links.append(
                        RoadLink(
                            road_id=road_id,
                            successor_road_id=succ_id if succ_type == "road" else None,
                            successor_type=succ_type,
                            end_point=end_pt,
                            successor_start_point=succ_start,
                        )
                    )

            # 3. Extract lane information
            lanes = road.find("lanes")
            if lanes is None:
                continue

            lane_section = lanes.find("laneSection")
            if lane_section is None:
                continue

            # Process left lanes: IDs are positive, sorted ascending (innermost first)
            left = lane_section.find("left")
            if left is not None:
                left_lanes = sorted(
                    left.findall("lane"),
                    key=lambda t: int(t.get("id", 0)),
                )  # ascending: [1, 2, 3, ...]
                inner_offset = 0.0
                for lane_tag in left_lanes:
                    width = self._get_lane_width(lane_tag)
                    outer_offset = inner_offset + width
                    parsed_lane = self._process_lane(
                        lane_tag,
                        road_id,
                        ref_line_pts,
                        inner_offset=inner_offset,
                        outer_offset=outer_offset,
                        is_left=True,
                    )
                    if parsed_lane:
                        lanes_to_render.append(parsed_lane)
                    inner_offset = outer_offset

            # Process right lanes: IDs are negative,
            # sorted ascending by abs (innermost first)
            right = lane_section.find("right")
            if right is not None:
                right_lanes = sorted(
                    right.findall("lane"),
                    key=lambda t: abs(int(t.get("id", 0))),
                )  # ascending by magnitude: [-1, -2, -3, ...]
                inner_offset = 0.0
                for lane_tag in right_lanes:
                    width = self._get_lane_width(lane_tag)
                    outer_offset = inner_offset + width
                    parsed_lane = self._process_lane(
                        lane_tag,
                        road_id,
                        ref_line_pts,
                        inner_offset=inner_offset,
                        outer_offset=outer_offset,
                        is_left=False,
                    )
                    if parsed_lane:
                        lanes_to_render.append(parsed_lane)
                    inner_offset = outer_offset

        return XodrMapData(lanes=lanes_to_render, road_links=road_links)

    def _get_lane_width(self, lane_tag: ET.Element) -> float:
        """Return the constant width (a-coefficient) of a lane."""
        width_tag = lane_tag.find("width")
        if width_tag is not None:
            return float(width_tag.get("a", 3.0))
        return 3.0

    def _parse_elevation_profile(self, road: ET.Element) -> ElevationProfile:
        """Parse the elevationProfile for a road."""
        entries = []
        profile_tag = road.find("elevationProfile")
        if profile_tag is not None:
            for elev in profile_tag.findall("elevation"):
                entries.append(
                    ElevationEntry(
                        s=float(elev.get("s", 0)),
                        a=float(elev.get("a", 0)),
                        b=float(elev.get("b", 0)),
                        c=float(elev.get("c", 0)),
                        d=float(elev.get("d", 0)),
                    )
                )
        return ElevationProfile(entries)

    def _parse_plan_view(
        self,
        road: ET.Element,
        step_size: float = 2.0,
        elevation_profile: Optional[ElevationProfile] = None,
    ) -> List[Point3D]:
        """Parse the reference line geometries into a list of points."""
        plan_view = road.find("planView")
        if plan_view is None:
            return []

        pts = []
        for geom in plan_view.findall("geometry"):
            s_start = float(geom.get("s", 0))
            x0 = float(geom.get("x", 0))
            y0 = float(geom.get("y", 0))
            hdg = float(geom.get("hdg", 0))
            length = float(geom.get("length", 0))

            # Basic parsing for lines and arcs
            if geom.find("line") is not None:
                num_steps = max(2, int(length / step_size))
                for i in range(num_steps + 1):
                    t = i / num_steps
                    ds = t * length
                    s_curr = s_start + ds
                    x = x0 + ds * math.cos(hdg)
                    y = y0 + ds * math.sin(hdg)
                    z = (
                        elevation_profile.get_z(s_curr)
                        if elevation_profile
                        else 0.0
                    )
                    pts.append(Point3D(x, y, z))

            elif geom.find("arc") is not None:
                arc = geom.find("arc")
                curvature = float(arc.get("curvature", 0))
                num_steps = max(2, int(length / step_size))
                for i in range(num_steps + 1):
                    t = i / num_steps
                    ds = t * length
                    s_curr = s_start + ds
                    if curvature == 0:
                        x = x0 + ds * math.cos(hdg)
                        y = y0 + ds * math.sin(hdg)
                    else:
                        radius = 1.0 / curvature
                        cx = x0 - radius * math.sin(hdg)
                        cy = y0 + radius * math.cos(hdg)
                        theta = hdg - math.pi / 2 + ds * curvature
                        x = cx + radius * math.cos(theta)
                        y = cy + radius * math.sin(theta)
                    z = (
                        elevation_profile.get_z(s_curr)
                        if elevation_profile
                        else 0.0
                    )
                    pts.append(Point3D(x, y, z))

        return pts

    def _process_lane(
        self,
        lane_tag: ET.Element,
        road_id: str,
        ref_line: List[Point3D],
        inner_offset: float,
        outer_offset: float,
        is_left: bool,
    ) -> Optional[Lane]:
        """Create a Lane using pre-computed inner/outer cumulative offsets."""
        lane_id = lane_tag.get("id", "0")
        lane_type = lane_tag.get("type", "driving")

        # Only render driving/sidewalk lanes for clarity
        if lane_type not in ["driving", "sidewalk", "biking", "shoulder"]:
            return None

        center_offset = (inner_offset + outer_offset) / 2.0

        # For right lanes, offsets go in the opposite direction (negative side)
        sign = 1.0 if is_left else -1.0

        center_pts = []
        inner_pts = []
        outer_pts = []

        for i in range(len(ref_line)):
            p1 = ref_line[max(0, i - 1)]
            p2 = ref_line[min(len(ref_line) - 1, i + 1)]

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            norm = math.hypot(dx, dy)
            if norm == 0:
                nx, ny = 0.0, 1.0
            else:
                nx, ny = -dy / norm, dx / norm  # Left-pointing normal

            px, py, pz = ref_line[i].x, ref_line[i].y, ref_line[i].z

            center_pts.append(
                Point3D(
                    px + nx * sign * center_offset,
                    py + ny * sign * center_offset,
                    pz,
                )
            )
            inner_pts.append(
                Point3D(
                    px + nx * sign * inner_offset,
                    py + ny * sign * inner_offset,
                    pz,
                )
            )
            outer_pts.append(
                Point3D(
                    px + nx * sign * outer_offset,
                    py + ny * sign * outer_offset,
                    pz,
                )
            )

        # Left lanes: "left boundary" is outer, "right boundary" is inner
        # Right lanes: "left boundary" is inner, "right boundary" is outer
        if is_left:
            lb_left = LaneBoundary(
                id=f"{road_id}_{lane_id}_left", style="solid", points=outer_pts
            )
            lb_right = LaneBoundary(
                id=f"{road_id}_{lane_id}_right", style="solid", points=inner_pts
            )
        else:
            lb_left = LaneBoundary(
                id=f"{road_id}_{lane_id}_left", style="solid", points=inner_pts
            )
            lb_right = LaneBoundary(
                id=f"{road_id}_{lane_id}_right", style="solid", points=outer_pts
            )

        return Lane(
            id=f"{road_id}_{lane_id}",
            road_id=road_id,
            type=lane_type,
            left_boundary=lb_left,
            right_boundary=lb_right,
            center_line=center_pts,
            is_left=is_left,
        )


def parse_xodr(file_path: str) -> XodrMapData:
    """Convenience function to parse an XODR file."""
    parser = XodrParser(file_path)
    return parser.parse()
