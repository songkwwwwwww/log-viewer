import xml.etree.ElementTree as ET
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass
class LaneOffsetEntry:
    """Represents a single laneOffset record (cubic polynomial)."""

    s: float
    a: float
    b: float
    c: float
    d: float


class LaneOffsetProfile:
    """Manages multiple laneOffset entries and evaluates lateral offset at any s."""

    def __init__(self, entries: List[LaneOffsetEntry]):
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_offset(self, s: float) -> float:
        if not self.entries:
            return 0.0

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


@dataclass
class WidthEntry:
    """Represents a single width record (cubic polynomial) within a lane section."""

    s_offset: float  # s-offset relative to the lane section start
    a: float
    b: float
    c: float
    d: float


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
        road_ref_endpoints: dict = {}
        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")
            pts_with_s = self._parse_plan_view(road)
            if pts_with_s:
                road_ref_endpoints[road_id] = (pts_with_s[0][0], pts_with_s[-1][0])

        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")

            # 1. Parse Elevation Profile
            elevation_profile = self._parse_elevation_profile(road)

            # 2. Parse Reference Line (planView) — returns (Point3D, s) pairs
            ref_line_with_s = self._parse_plan_view(
                road, elevation_profile=elevation_profile
            )
            if not ref_line_with_s:
                continue

            # 3. Parse successor link
            link_tag = road.find("link")
            if link_tag is not None:
                succ_tag = link_tag.find("successor")
                if succ_tag is not None:
                    succ_type = succ_tag.get("elementType", "road")
                    succ_id = succ_tag.get("elementId")
                    end_pt = ref_line_with_s[-1][0]
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

            # 4. Extract lane information
            lanes = road.find("lanes")
            if lanes is None:
                continue

            # Parse laneOffset profile (Fix 2)
            lane_offset_profile = self._parse_lane_offset_profile(lanes)

            # Process all lane sections (Fix 1: multiple lane sections per road)
            lane_sections = lanes.findall("laneSection")
            if not lane_sections:
                continue

            for idx, lane_section in enumerate(lane_sections):
                s0 = float(lane_section.get("s", 0.0))
                s1 = (
                    float(lane_sections[idx + 1].get("s", math.inf))
                    if idx + 1 < len(lane_sections)
                    else math.inf
                )

                # Slice reference line points for this section
                section_pts_with_s = [
                    (pt, s) for pt, s in ref_line_with_s if s0 <= s <= s1
                ]
                if not section_pts_with_s:
                    continue

                ref_pts = [pt for pt, _ in section_pts_with_s]
                s_vals = [s for _, s in section_pts_with_s]

                # Compute lane offset at each s point (Fix 2)
                lane_offsets = [lane_offset_profile.get_offset(s) for s in s_vals]

                # Process left lanes: IDs are positive, sorted ascending (innermost first)
                left = lane_section.find("left")
                if left is not None:
                    left_lanes = sorted(
                        left.findall("lane"),
                        key=lambda t: int(t.get("id", 0)),
                    )
                    # Left lanes build in the +normal direction; start from lane_offset
                    inner_offsets = list(lane_offsets)
                    for lane_tag in left_lanes:
                        widths = self._get_lane_widths_at_s(lane_tag, s_vals, s0)
                        outer_offsets = [
                            inner_offsets[i] + widths[i] for i in range(len(ref_pts))
                        ]
                        parsed_lane = self._process_lane(
                            lane_tag,
                            road_id,
                            ref_pts,
                            inner_offsets,
                            outer_offsets,
                            is_left=True,
                        )
                        if parsed_lane:
                            lanes_to_render.append(parsed_lane)
                        inner_offsets = outer_offsets

                # Process right lanes: IDs are negative, sorted ascending by magnitude
                right = lane_section.find("right")
                if right is not None:
                    right_lanes = sorted(
                        right.findall("lane"),
                        key=lambda t: abs(int(t.get("id", 0))),
                    )
                    # Right lanes build in the -normal direction; negate lane_offset
                    # so that sign * inner_offsets = +lane_offset physically
                    inner_offsets = [-lo for lo in lane_offsets]
                    for lane_tag in right_lanes:
                        widths = self._get_lane_widths_at_s(lane_tag, s_vals, s0)
                        outer_offsets = [
                            inner_offsets[i] - widths[i] for i in range(len(ref_pts))
                        ]
                        parsed_lane = self._process_lane(
                            lane_tag,
                            road_id,
                            ref_pts,
                            inner_offsets,
                            outer_offsets,
                            is_left=False,
                        )
                        if parsed_lane:
                            lanes_to_render.append(parsed_lane)
                        inner_offsets = outer_offsets

        return XodrMapData(lanes=lanes_to_render, road_links=road_links)

    def _parse_lane_offset_profile(self, lanes: ET.Element) -> LaneOffsetProfile:
        """Parse all <laneOffset> entries from the <lanes> element."""
        entries = []
        for lo in lanes.findall("laneOffset"):
            entries.append(
                LaneOffsetEntry(
                    s=float(lo.get("s", 0.0)),
                    a=float(lo.get("a", 0.0)),
                    b=float(lo.get("b", 0.0)),
                    c=float(lo.get("c", 0.0)),
                    d=float(lo.get("d", 0.0)),
                )
            )
        return LaneOffsetProfile(entries)

    def _get_lane_widths_at_s(
        self, lane_tag: ET.Element, s_vals: List[float], section_s0: float
    ) -> List[float]:
        """Return lane width at each s position, evaluating the cubic polynomial. (Fix 4)"""
        width_tags = lane_tag.findall("width")
        if not width_tags:
            return [3.0] * len(s_vals)

        entries = sorted(
            [
                WidthEntry(
                    s_offset=float(wt.get("sOffset", 0.0)),
                    a=float(wt.get("a", 3.0)),
                    b=float(wt.get("b", 0.0)),
                    c=float(wt.get("c", 0.0)),
                    d=float(wt.get("d", 0.0)),
                )
                for wt in width_tags
            ],
            key=lambda e: e.s_offset,
        )

        widths = []
        for s in s_vals:
            ds_from_section = s - section_s0
            active = entries[0]
            for entry in entries:
                if ds_from_section >= entry.s_offset:
                    active = entry
                else:
                    break
            ds = ds_from_section - active.s_offset
            w = active.a + active.b * ds + active.c * ds**2 + active.d * ds**3
            widths.append(max(0.0, w))
        return widths

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
    ) -> List[Tuple[Point3D, float]]:
        """Parse the reference line geometries into a list of (Point3D, s) pairs."""
        plan_view = road.find("planView")
        if plan_view is None:
            return []

        pts: List[Tuple[Point3D, float]] = []
        for geom in plan_view.findall("geometry"):
            s_start = float(geom.get("s", 0))
            x0 = float(geom.get("x", 0))
            y0 = float(geom.get("y", 0))
            hdg = float(geom.get("hdg", 0))
            length = float(geom.get("length", 0))

            def _z(s: float) -> float:
                return elevation_profile.get_z(s) if elevation_profile else 0.0

            # Skip duplicate start point for all segments after the first
            skip_first = len(pts) > 0

            if geom.find("line") is not None:
                num_steps = max(2, int(length / step_size))
                for i in range(num_steps + 1):
                    if i == 0 and skip_first:
                        continue
                    ds = (i / num_steps) * length
                    s_curr = s_start + ds
                    pts.append(
                        (
                            Point3D(
                                x0 + ds * math.cos(hdg),
                                y0 + ds * math.sin(hdg),
                                _z(s_curr),
                            ),
                            s_curr,
                        )
                    )

            elif geom.find("arc") is not None:
                arc = geom.find("arc")
                curvature = float(arc.get("curvature", 0))
                num_steps = max(2, int(length / step_size))
                for i in range(num_steps + 1):
                    if i == 0 and skip_first:
                        continue
                    ds = (i / num_steps) * length
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
                    pts.append((Point3D(x, y, _z(s_curr)), s_curr))

            elif geom.find("spiral") is not None:
                # Fix 3: Clothoid / Euler spiral via numerical integration
                spiral = geom.find("spiral")
                curv_start = float(spiral.get("curvStart", 0))
                curv_end = float(spiral.get("curvEnd", 0))
                num_steps = max(2, int(length / step_size))
                dcurv = (curv_end - curv_start) / length if length > 0 else 0.0
                x, y = x0, y0
                ds_step = length / num_steps
                for i in range(num_steps + 1):
                    if i == 0:
                        if not skip_first:
                            pts.append((Point3D(x, y, _z(s_start)), s_start))
                        continue
                    ds_prev = (i - 1) * ds_step
                    # Use midpoint curvature for heading integration (midpoint rule)
                    ds_mid = ds_prev + ds_step / 2.0
                    h_mid = hdg + curv_start * ds_mid + 0.5 * dcurv * ds_mid**2
                    x += ds_step * math.cos(h_mid)
                    y += ds_step * math.sin(h_mid)
                    s_curr = s_start + i * ds_step
                    pts.append((Point3D(x, y, _z(s_curr)), s_curr))

            elif geom.find("paramPoly3") is not None:
                # Fix 3: Parametric cubic polynomial
                pp3 = geom.find("paramPoly3")
                aU = float(pp3.get("aU", 0))
                bU = float(pp3.get("bU", 0))
                cU = float(pp3.get("cU", 0))
                dU = float(pp3.get("dU", 0))
                aV = float(pp3.get("aV", 0))
                bV = float(pp3.get("bV", 0))
                cV = float(pp3.get("cV", 0))
                dV = float(pp3.get("dV", 0))
                p_range = pp3.get("pRange", "normalized")
                num_steps = max(2, int(length / step_size))
                cos_h, sin_h = math.cos(hdg), math.sin(hdg)
                for i in range(num_steps + 1):
                    if i == 0 and skip_first:
                        continue
                    t = i / num_steps
                    p = t * length if p_range == "arcLength" else t
                    u = aU + bU * p + cU * p**2 + dU * p**3
                    v = aV + bV * p + cV * p**2 + dV * p**3
                    x = x0 + u * cos_h - v * sin_h
                    y = y0 + u * sin_h + v * cos_h
                    s_curr = s_start + t * length
                    pts.append((Point3D(x, y, _z(s_curr)), s_curr))

        return pts

    def _process_lane(
        self,
        lane_tag: ET.Element,
        road_id: str,
        ref_line: List[Point3D],
        inner_offsets: List[float],
        outer_offsets: List[float],
        is_left: bool,
    ) -> Optional[Lane]:
        """Create a Lane using per-point inner/outer cumulative offsets."""
        lane_id = lane_tag.get("id", "0")
        lane_type = lane_tag.get("type", "driving")

        # Only render drivable and pedestrian lane types
        if lane_type not in ["driving", "sidewalk", "biking", "shoulder"]:
            return None

        # For right lanes the offsets are already negative (built outward in -normal dir)
        # We use abs and the sign to reconstruct actual lateral displacement
        center_pts = []
        inner_pts = []
        outer_pts = []

        n = len(ref_line)
        for i in range(n):
            p1 = ref_line[max(0, i - 1)]
            p2 = ref_line[min(n - 1, i + 1)]

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            norm = math.hypot(dx, dy)
            if norm == 0:
                nx, ny = 0.0, 1.0
            else:
                nx, ny = -dy / norm, dx / norm  # Left-pointing normal

            px, py, pz = ref_line[i].x, ref_line[i].y, ref_line[i].z
            io = inner_offsets[i]
            oo = outer_offsets[i]
            co = (io + oo) / 2.0

            center_pts.append(Point3D(px + nx * co, py + ny * co, pz))
            inner_pts.append(Point3D(px + nx * io, py + ny * io, pz))
            outer_pts.append(Point3D(px + nx * oo, py + ny * oo, pz))

        # Left lanes: outer edge is the left boundary, inner edge is the right boundary
        # Right lanes: inner edge (closer to center) is the left boundary, outer is right
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
