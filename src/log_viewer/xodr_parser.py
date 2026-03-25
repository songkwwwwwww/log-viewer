"""Parses OpenDRIVE (.xodr) files into a 3D road network model.

This module implements a subset of the OpenDRIVE specification, focusing on 
extracting road geometries, lane structures, and connectivity for 3D visualization.

Key OpenDRIVE concepts handled:
- Reference Line (planView): The "spine" of the road, defined by geometric primitives 
  (lines, arcs, spirals, polynomials).
- s-coordinate: Longitudinal distance along the reference line.
- t-coordinate: Lateral offset from the reference line.
- Lane Sections: Segments of the road where the lane configuration (number of lanes, 
  widths) is constant.
- Profiles: Elevation (z), Superelevation (banking/roll), and Lane Offset (lateral shift 
  of the entire lane group).
"""

import xml.etree.ElementTree as ET
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .geometry import Point3D
from .map_model import Lane, LaneBoundary, RoadLink, XodrMapData

# Linearization tolerance (metres) used for adaptive sampling — mirrors the eps
# value passed to libOpenDRIVE's approximate_linear() calls.
_SAMPLING_EPS = 0.1


def _normalize_3d(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalizes a 3D vector to unit length."""
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n == 0:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross_product(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Computes the cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _compute_3d_tangent(
    ref_line: List[Point3D], i: int
) -> Tuple[float, float, float]:
    """Computes the 3D tangent vector (e_s) at a point on the reference line.
    
    Uses central difference between neighboring points.
    """
    n = len(ref_line)
    p1 = ref_line[max(0, i - 1)]
    p2 = ref_line[min(n - 1, i + 1)]
    return _normalize_3d((p2.x - p1.x, p2.y - p1.y, p2.z - p1.z))


def _compute_e_t(
    e_s: Tuple[float, float, float], theta: float
) -> Tuple[float, float, float]:
    """Computes the lateral unit vector (e_t) accounting for road superelevation.

    The superelevation angle theta represents the "roll" or "banking" of the road.
    When theta=0, e_t is simply the 2D left-pointing normal to the reference line.
    
    Calculation mirrors Road::get_xyz() in libOpenDRIVE/src/Road.cpp.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return _normalize_3d((
        cos_t * -e_s[1] + sin_t * -e_s[2] * e_s[0],
        cos_t *  e_s[0] + sin_t * -e_s[2] * e_s[1],
        sin_t * (e_s[0] ** 2 + e_s[1] ** 2),
    ))


@dataclass
class ElevationEntry:
    """Represents a single elevation record in a cubic polynomial profile."""

    s: float
    a: float
    b: float
    c: float
    d: float


def _cubic_poly_sample_s_values(
    s_entry: float,
    s_entry_end: float,
    c: float,
    d: float,
    s_start: float,
    s_end: float,
    eps: float,
) -> List[float]:
    """Compute s-values needing explicit samples within a cubic polynomial segment.

    Mirrors libOpenDRIVE CubicPoly::approximate_linear logic:
    - Linear (c==0, d==0): only endpoints needed.
    - Quadratic (d==0, c!=0): uniform steps of 2*sqrt(|eps/c|).
    - Cubic: subdivide segment into ~10 uniform steps as a practical approximation.

    Returns s-values clipped to [s_start, s_end].
    """
    seg_s_start = max(s_entry, s_start)
    seg_s_end = min(s_entry_end, s_end)
    if seg_s_end <= seg_s_start:
        return []

    result = [seg_s_start, seg_s_end]

    if d == 0.0 and c == 0.0:
        pass  # linear — endpoints suffice
    elif d == 0.0 and c != 0.0:
        # quadratic: step derived from curvature tolerance
        step = 2.0 * math.sqrt(abs(eps / c)) if abs(c) > 1e-12 else (seg_s_end - seg_s_start)
        s = seg_s_start + step
        while s < seg_s_end:
            result.append(s)
            s += step
    else:
        # cubic: ~10 uniform subdivisions as a practical approximation
        n = max(2, int((seg_s_end - seg_s_start) / (10.0 * eps)))
        step = (seg_s_end - seg_s_start) / n
        s = seg_s_start + step
        while s < seg_s_end:
            result.append(s)
            s += step

    return result


class ElevationProfile:
    """Manages multiple elevation entries and evaluates Z at any s position.

    Elevation is defined as a sequence of cubic polynomials relative to the
    longitudinal position s.
    """

    def __init__(self, entries: List[ElevationEntry]):
        """Initialize with a list of elevation entries, sorted by s."""
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_z(self, s: float) -> float:
        """Calculate the elevation Z at distance s along the road."""
        if not self.entries:
            return 0.0

        # Find the "active" polynomial entry for the given s
        active_entry = self.entries[0]
        for entry in self.entries:
            if s >= entry.s:
                active_entry = entry
            else:
                break

        ds = s - active_entry.s
        # z(ds) = a + b*ds + c*ds^2 + d*ds^3
        return (
            active_entry.a
            + active_entry.b * ds
            + active_entry.c * ds**2
            + active_entry.d * ds**3
        )

    def get_sample_s_values(
        self, s_start: float, s_end: float, eps: float
    ) -> List[float]:
        """Return s-values where this profile needs explicit sampling.

        Mirrors libOpenDRIVE CubicProfile::approximate_linear.
        """
        s_set: set = set()
        s_set.add(s_start)
        s_set.add(s_end)
        entries = self.entries
        for i, entry in enumerate(entries):
            next_s = entries[i + 1].s if i + 1 < len(entries) else math.inf
            s_set.update(
                _cubic_poly_sample_s_values(entry.s, next_s, entry.c, entry.d, s_start, s_end, eps)
            )
        return sorted(v for v in s_set if s_start <= v <= s_end)


@dataclass
class LaneOffsetEntry:
    """Represents a single laneOffset record (cubic polynomial)."""

    s: float
    a: float
    b: float
    c: float
    d: float


class LaneOffsetProfile:
    """Manages lateral offset of the entire lane group from the reference line."""

    def __init__(self, entries: List[LaneOffsetEntry]):
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_offset(self, s: float) -> float:
        """Evaluates the lateral offset at position s."""
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

    def get_sample_s_values(
        self, s_start: float, s_end: float, eps: float
    ) -> List[float]:
        """Return s-values where this profile needs explicit sampling."""
        s_set: set = set()
        s_set.add(s_start)
        s_set.add(s_end)
        entries = self.entries
        for i, entry in enumerate(entries):
            next_s = entries[i + 1].s if i + 1 < len(entries) else math.inf
            s_set.update(
                _cubic_poly_sample_s_values(entry.s, next_s, entry.c, entry.d, s_start, s_end, eps)
            )
        return sorted(v for v in s_set if s_start <= v <= s_end)


@dataclass
class WidthEntry:
    """Represents a single width record (cubic polynomial) within a lane section."""

    s_offset: float  # s-offset relative to the lane section start
    a: float
    b: float
    c: float
    d: float


@dataclass
class SuperelevationEntry:
    """Represents a single superelevation record (road banking angle, radians)."""

    s: float
    a: float
    b: float
    c: float
    d: float


class SuperelevationProfile:
    """Evaluates road superelevation (roll angle) at any s position."""

    def __init__(self, entries: List[SuperelevationEntry]):
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_value(self, s: float) -> float:
        """Returns the superelevation angle in radians at position s."""
        if not self.entries:
            return 0.0
        active = self.entries[0]
        for entry in self.entries:
            if s >= entry.s:
                active = entry
            else:
                break
        ds = s - active.s
        return active.a + active.b * ds + active.c * ds**2 + active.d * ds**3

    def get_sample_s_values(
        self, s_start: float, s_end: float, eps: float
    ) -> List[float]:
        """Return s-values where this profile needs explicit sampling."""
        s_set: set = set()
        s_set.add(s_start)
        s_set.add(s_end)
        entries = self.entries
        for i, entry in enumerate(entries):
            next_s = entries[i + 1].s if i + 1 < len(entries) else math.inf
            s_set.update(
                _cubic_poly_sample_s_values(entry.s, next_s, entry.c, entry.d, s_start, s_end, eps)
            )
        return sorted(v for v in s_set if s_start <= v <= s_end)


@dataclass
class CrossfallEntry:
    """Represents a lateral slope angle record."""
    s: float
    a: float
    b: float
    c: float
    d: float
    side: str  # "left", "right", or "both"


class CrossfallProfile:
    """Evaluates crossfall (lateral slope angle) at any s position.
    
    Crossfall is used to drain water from the road surface and is applied
    independently to the left or right sides.
    """

    def __init__(self, entries: List[CrossfallEntry]):
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_crossfall(self, s: float, is_left_lane: bool) -> float:
        """Returns the crossfall angle in radians for a specific side."""
        if not self.entries:
            return 0.0
        active = self.entries[0]
        for entry in self.entries:
            if s >= entry.s:
                active = entry
            else:
                break
        
        # Check if this profile applies to the requested side
        if active.side == "left" and not is_left_lane:
            return 0.0
        if active.side == "right" and is_left_lane:
            return 0.0
            
        ds = s - active.s
        return active.a + active.b * ds + active.c * ds**2 + active.d * ds**3


class XodrParser:
    """A minimal OpenDRIVE (XODR) parser for visualization purposes.
    
    This parser extracts road geometries and lane structures, performing the 
    necessary coordinate transformations to generate 3D meshes.
    """

    def __init__(self, file_path: str):
        """Initialize parser from file path."""
        self.file_path = file_path
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()

    def parse(self) -> XodrMapData:
        """Parse the OpenDRIVE XML into XodrMapData."""
        lanes_to_render = []
        road_links = []

        # First pass: collect reference line endpoints per road to resolve connectivity
        road_ref_endpoints: dict = {}
        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")
            pts_with_s = self._parse_plan_view(road)
            if pts_with_s:
                road_ref_endpoints[road_id] = (pts_with_s[0][0], pts_with_s[-1][0])

        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")

            # 1. Parse Profiles (Elevation, Superelevation, Crossfall)
            elevation_profile = self._parse_elevation_profile(road)
            superelevation_profile, crossfall_profile = self._parse_lateral_profile(road)

            # 2. Parse Reference Line (planView) — returns (Point3D, s) pairs sampled along segments
            ref_line_with_s = self._parse_plan_view(
                road, elevation_profile=elevation_profile
            )
            if not ref_line_with_s:
                continue

            # 3. Parse road connectivity (successor links)
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

            # Parse laneOffset profile (applies lateral shift to all lanes)
            lane_offset_profile = self._parse_lane_offset_profile(lanes)

            # Collect all s-values that need explicit sampling:
            # lane section boundaries + lane_offset/superelevation change-points
            # + per-lane width entry boundaries.
            # Mirrors libOpenDRIVE Road::get_lane_mesh s-value union strategy.
            road_s_start = ref_line_with_s[0][1]
            road_s_end = ref_line_with_s[-1][1]
            extra_s: set = set()

            # Lane section boundaries
            for ls in lanes.findall("laneSection"):
                extra_s.add(float(ls.get("s", 0.0)))

            # Lane offset profile change-points
            extra_s.update(
                lane_offset_profile.get_sample_s_values(road_s_start, road_s_end, _SAMPLING_EPS)
            )

            # Superelevation change-points
            extra_s.update(
                superelevation_profile.get_sample_s_values(road_s_start, road_s_end, _SAMPLING_EPS)
            )

            # Width entry boundary s-values for every lane in every lane section
            for ls in lanes.findall("laneSection"):
                ls_s0 = float(ls.get("s", 0.0))
                for side in (ls.find("left"), ls.find("right")):
                    if side is None:
                        continue
                    for lane_tag in side.findall("lane"):
                        for wt in lane_tag.findall("width"):
                            extra_s.add(ls_s0 + float(wt.get("sOffset", 0.0)))

            ref_line_with_s = _insert_boundary_points(
                ref_line_with_s, sorted(extra_s)
            )

            # Process all lane sections within the road
            lane_sections = lanes.findall("laneSection")
            if not lane_sections:
                continue

            for idx, lane_section in enumerate(lane_sections):
                s0 = float(lane_section.get("s", 0.0))
                is_last_section = idx + 1 >= len(lane_sections)
                s1 = (
                    float(lane_sections[idx + 1].get("s", math.inf))
                    if not is_last_section
                    else math.inf
                )

                # Slice reference line points for this section.
                # Include [s0, s1] so the section has a proper end vertex at the
                # boundary. The boundary point at s == s1 is evaluated using this
                # section's width polynomial (section_s0-relative), which is
                # correct because width polynomials are defined up to the next
                # section boundary. The last section uses s0 <= s.
                if is_last_section:
                    section_pts_with_s = [
                        (pt, s) for pt, s in ref_line_with_s if s >= s0
                    ]
                else:
                    section_pts_with_s = [
                        (pt, s) for pt, s in ref_line_with_s if s0 <= s <= s1
                    ]
                if not section_pts_with_s:
                    continue

                ref_pts = [pt for pt, _ in section_pts_with_s]
                s_vals = [s for _, s in section_pts_with_s]

                # Compute lane offset at each sampled s point
                lane_offsets = [lane_offset_profile.get_offset(s) for s in s_vals]

                # Process left lanes (OpenDRIVE IDs are positive, build outward from center)
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
                            s_vals,
                            is_left=True,
                            superelevation_profile=superelevation_profile,
                            crossfall_profile=crossfall_profile,
                        )
                        if parsed_lane:
                            lanes_to_render.append(parsed_lane)
                        inner_offsets = outer_offsets

                # Process right lanes (OpenDRIVE IDs are negative, build outward)
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
                            s_vals,
                            is_left=False,
                            superelevation_profile=superelevation_profile,
                            crossfall_profile=crossfall_profile,
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
        """Return lane width at each s position, evaluating the cubic polynomial width profile."""
        width_tags = lane_tag.findall("width")
        if not width_tags:
            # Default to 3m if no width records are present
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

    def _parse_lateral_profile(
        self, road: ET.Element
    ) -> Tuple[SuperelevationProfile, CrossfallProfile]:
        """Parse <lateralProfile> for superelevation and crossfall."""
        superelev_entries = []
        crossfall_entries = []
        lateral_tag = road.find("lateralProfile")
        if lateral_tag is not None:
            for se in lateral_tag.findall("superelevation"):
                superelev_entries.append(
                    SuperelevationEntry(
                        s=float(se.get("s", 0)),
                        a=float(se.get("a", 0)),
                        b=float(se.get("b", 0)),
                        c=float(se.get("c", 0)),
                        d=float(se.get("d", 0)),
                    )
                )
            for cf in lateral_tag.findall("crossFall"):
                crossfall_entries.append(
                    CrossfallEntry(
                        s=float(cf.get("s", 0)),
                        a=float(cf.get("a", 0)),
                        b=float(cf.get("b", 0)),
                        c=float(cf.get("c", 0)),
                        d=float(cf.get("d", 0)),
                        side=cf.get("side", "both"),
                    )
                )
        return SuperelevationProfile(superelev_entries), CrossfallProfile(crossfall_entries)

    def _parse_plan_view(
        self,
        road: ET.Element,
        eps: float = 0.1,
        elevation_profile: Optional[ElevationProfile] = None,
    ) -> List[Tuple[Point3D, float]]:
        """Parse the reference line geometries into a list of (Point3D, s) pairs.

        Uses adaptive sampling per geometry type, mirroring libOpenDRIVE:
        - Line: endpoints only (already linear).
        - Arc: ~1-degree intervals based on curvature (0.01/|curvature|).
        - Spiral: uniform steps of 10*eps.
        - ParamPoly3: uniform steps of eps.
        Elevation profile change-points are merged into every segment's sample set.
        """
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
            s_end = s_start + length

            def _z(s: float) -> float:
                return elevation_profile.get_z(s) if elevation_profile else 0.0

            # Skip duplicate start point if this isn't the first segment of the road
            skip_first = len(pts) > 0

            # Build the sorted set of s-values to sample for this geometry segment.
            # Always include both endpoints; then merge elevation change-points.
            s_set: set = {s_start, s_end}
            if elevation_profile:
                s_set.update(elevation_profile.get_sample_s_values(s_start, s_end, eps))

            # ---------- Line Geometry ----------
            if geom.find("line") is not None:
                # Lines are already linear — endpoints only (libOpenDRIVE Line::approximate_linear)
                pass  # s_set already has s_start and s_end

            # ---------- Arc Geometry (Constant Curvature) ----------
            elif geom.find("arc") is not None:
                arc_elem = geom.find("arc")
                curvature = float(arc_elem.get("curvature", 0))
                if abs(curvature) > 1e-12:
                    # ~1-degree steps: s_step = 0.01 / |curvature|
                    # Mirrors libOpenDRIVE Arc::approximate_linear (TODO stub)
                    s_step = 0.01 / abs(curvature)
                    s = s_start
                    while s < s_end:
                        s_set.add(s)
                        s += s_step
                # else: zero curvature arc is a line — endpoints suffice

            # ---------- Spiral Geometry (Clothoid / Euler Spiral) ----------
            elif geom.find("spiral") is not None:
                # Uniform steps of 10*eps.
                # Mirrors libOpenDRIVE Spiral::approximate_linear (TODO stub)
                s_step = 10.0 * eps
                s = s_start
                while s < s_end:
                    s_set.add(s)
                    s += s_step

            # ---------- Parametric Cubic Polynomial Geometry ----------
            elif geom.find("paramPoly3") is not None:
                # Uniform steps of eps for fine-grained sampling
                s_step = eps
                s = s_start
                while s < s_end:
                    s_set.add(s)
                    s += s_step

            # Evaluate geometry at each sample s-value in sorted order.
            # For arc and paramPoly3 we need the element attributes; re-fetch here.
            arc_elem = geom.find("arc")
            pp3_elem = geom.find("paramPoly3")
            spiral_elem = geom.find("spiral")

            if arc_elem is not None:
                curvature = float(arc_elem.get("curvature", 0))
            if spiral_elem is not None:
                curv_start_sp = float(spiral_elem.get("curvStart", 0))
                curv_end_sp = float(spiral_elem.get("curvEnd", 0))
                dcurv = (curv_end_sp - curv_start_sp) / length if length > 0 else 0.0
            if pp3_elem is not None:
                aU = float(pp3_elem.get("aU", 0))
                bU = float(pp3_elem.get("bU", 0))
                cU = float(pp3_elem.get("cU", 0))
                dU = float(pp3_elem.get("dU", 0))
                aV = float(pp3_elem.get("aV", 0))
                bV = float(pp3_elem.get("bV", 0))
                cV = float(pp3_elem.get("cV", 0))
                dV = float(pp3_elem.get("dV", 0))
                p_range = pp3_elem.get("pRange", "normalized")
                cos_h = math.cos(hdg)
                sin_h = math.sin(hdg)

            # For spiral: integrate heading incrementally — must iterate in order
            if spiral_elem is not None:
                sorted_s = sorted(s_set)
                x, y = x0, y0
                prev_s = s_start
                for s_curr in sorted_s:
                    if s_curr == s_start:
                        if not skip_first:
                            pts.append((Point3D(x, y, _z(s_start)), s_start))
                        continue
                    ds_step = s_curr - prev_s
                    ds_mid = (prev_s - s_start) + ds_step / 2.0
                    h_mid = hdg + curv_start_sp * ds_mid + 0.5 * dcurv * ds_mid**2
                    x += ds_step * math.cos(h_mid)
                    y += ds_step * math.sin(h_mid)
                    pts.append((Point3D(x, y, _z(s_curr)), s_curr))
                    prev_s = s_curr
            else:
                for s_curr in sorted(s_set):
                    if s_curr == s_start and skip_first:
                        continue
                    ds = s_curr - s_start

                    if geom.find("line") is not None:
                        x = x0 + ds * math.cos(hdg)
                        y = y0 + ds * math.sin(hdg)
                    elif arc_elem is not None:
                        if abs(curvature) < 1e-12:
                            x = x0 + ds * math.cos(hdg)
                            y = y0 + ds * math.sin(hdg)
                        else:
                            radius = 1.0 / curvature
                            cx = x0 - radius * math.sin(hdg)
                            cy = y0 + radius * math.cos(hdg)
                            theta = hdg - math.pi / 2 + ds * curvature
                            x = cx + radius * math.cos(theta)
                            y = cy + radius * math.sin(theta)
                    elif pp3_elem is not None:
                        t = ds / length if length > 0 else 0.0
                        p = ds if p_range == "arcLength" else t
                        u = aU + bU * p + cU * p**2 + dU * p**3
                        v = aV + bV * p + cV * p**2 + dV * p**3
                        x = x0 + u * cos_h - v * sin_h
                        y = y0 + u * sin_h + v * cos_h
                    else:
                        continue

                    pts.append((Point3D(x, y, _z(s_curr)), s_curr))

        return pts

    def _process_lane(
        self,
        lane_tag: ET.Element,
        road_id: str,
        ref_line: List[Point3D],
        inner_offsets: List[float],
        outer_offsets: List[float],
        s_vals: List[float],
        is_left: bool,
        superelevation_profile: "SuperelevationProfile",
        crossfall_profile: "CrossfallProfile",
    ) -> Optional[Lane]:
        """Constructs a 3D Lane object from road geometry and lateral profiles.

        This function calculates the exact 3D position of lane boundaries and 
        centerlines by applying lateral offsets (t), superelevation (roll), 
        and crossfall to the road's reference line.

        Coordinate System:
          - e_s: Unit tangent vector (longitudinal direction).
          - e_t: Unit lateral vector (horizontal direction, with roll).
          - e_h: Unit vertical vector (upwards normal).
        """
        lane_id = lane_tag.get("id", "0")
        lane_type = lane_tag.get("type", "driving")

        # Only render drivable and pedestrian-relevant lane types
        if lane_type not in ["driving", "sidewalk", "biking", "shoulder"]:
            return None

        center_pts = []
        inner_pts = []
        outer_pts = []

        n = len(ref_line)
        for i in range(n):
            s = s_vals[i]
            px, py, pz = ref_line[i].x, ref_line[i].y, ref_line[i].z

            # 1. Compute local orthonormal frame (e_s, e_t, e_h)
            e_s = _compute_3d_tangent(ref_line, i)
            theta = superelevation_profile.get_value(s)
            e_t = _compute_e_t(e_s, theta)
            e_h = _normalize_3d(_cross_product(e_s, e_t))

            # 2. Determine lateral offsets for inner, outer, and center paths
            io = inner_offsets[i]
            oo = outer_offsets[i]
            co = (io + oo) / 2.0

            # 3. Apply Crossfall: h_t = -tan(crossfall) * |t|
            # This creates a slight lateral slope for drainage.
            crossfall = crossfall_profile.get_crossfall(s, is_left)
            tan_cf = math.tan(crossfall)

            def _make_pt(t_offset: float) -> Point3D:
                """Projects a point from the reference line using local coordinates."""
                h_t = -tan_cf * abs(t_offset)
                return Point3D(
                    px + e_t[0] * t_offset + e_h[0] * h_t,
                    py + e_t[1] * t_offset + e_h[1] * h_t,
                    pz + e_t[2] * t_offset + e_h[2] * h_t,
                )

            center_pts.append(_make_pt(co))
            inner_pts.append(_make_pt(io))
            outer_pts.append(_make_pt(oo))

        # Determine Left/Right boundaries based on lane side (OpenDRIVE orientation)
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


def _insert_boundary_points(
    ref_line_with_s: List[Tuple[Point3D, float]],
    boundary_s_values: List[float],
) -> List[Tuple[Point3D, float]]:
    """Ensures that lane section boundaries are explicitly represented in the ref line.

    This prevents geometric "seams" or gaps between adjacent lane sections by 
    interpolating and inserting points at the exact s-coordinates where sections start/end.
    """
    result = list(ref_line_with_s)
    insert_offset = 0
    for s_target in sorted(boundary_s_values):
        j = insert_offset
        while j < len(result) and result[j][1] < s_target:
            j += 1
        if j >= len(result):
            break
        if abs(result[j][1] - s_target) < 1e-9:
            insert_offset = j + 1
            continue  # point is already present
        if j == 0:
            continue  # target is before road start; skip
            
        # Interpolate between result[j-1] and result[j]
        pt_a, s_a = result[j - 1]
        pt_b, s_b = result[j]
        t = (s_target - s_a) / (s_b - s_a)
        pt_new = Point3D(
            pt_a.x + t * (pt_b.x - pt_a.x),
            pt_a.y + t * (pt_b.y - pt_a.y),
            pt_a.z + t * (pt_b.z - pt_a.z),
        )
        result.insert(j, (pt_new, s_target))
        insert_offset = j + 1
    return result


def parse_xodr(file_path: str) -> XodrMapData:
    """Convenience function to parse an OpenDRIVE file into a map model."""
    parser = XodrParser(file_path)
    return parser.parse()
