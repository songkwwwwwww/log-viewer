"""Regression tests for OpenDRIVE lane sampling fidelity."""

from pathlib import Path

from log_viewer.xodr_parser import parse_xodr


def test_line_geometry_lane_uses_width_profile_samples():
    """Ensure cubic lane width adds interior samples on a straight road.

    Town07 road 42 / lane 3 is a useful regression case:
    - the reference line is a single <line>
    - lane 3 has a cubic width profile
    - there is no non-zero laneOffset or superelevation to add samples

    The old implementation produced only the two line endpoints.
    """
    xodr_path = Path(__file__).resolve().parents[1] / "assets" / "Town07.xodr"
    map_data = parse_xodr(str(xodr_path))

    lane = next(lane for lane in map_data.lanes if lane.id == "42_3")
    assert lane.type == "shoulder"
    assert len(lane.left_boundary.points) > 2
    assert len(lane.left_boundary.points) == len(lane.right_boundary.points)
    assert len(lane.center_line) == len(lane.left_boundary.points)
