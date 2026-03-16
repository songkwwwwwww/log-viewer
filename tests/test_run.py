import math
import os
import time
from typing import List

from log_viewer.data_model import (
    ObjectState,
    Point3D,
    Quaternion,
    SceneFrame,
    XodrMapData,
)
from log_viewer.viewer import LogViewer
from log_viewer.xodr_parser import parse_xodr


def heading_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle (radians) to a Quaternion (w, x, y, z)."""
    return Quaternion(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))


def generate_simulation_frames(
    map_data: XodrMapData, duration: float = 5.0, fps: int = 10
) -> List[SceneFrame]:
    """Generate frames with objects moving along lanes."""
    frames = []
    num_frames = int(duration * fps)
    dt = 1.0 / fps

    # Filter driving lanes
    driving_lanes = [lane for lane in map_data.lanes if lane.type == "driving"]
    if not driving_lanes:
        return []

    # Select a few lanes for simulation
    selected_lanes = driving_lanes[:3]
    vehicles = []

    for idx, lane in enumerate(selected_lanes):
        vehicles.append(
            {
                "id": f"v{idx}",
                "lane": lane,
                "speed": 10.0 + idx * 2.0,  # Each vehicle has a different speed
                "size": (4.5, 2.0, 1.5),
            }
        )

    for i in range(num_frames):
        timestamp = i * dt
        objects = {}

        for v in vehicles:
            lane = v["lane"]
            lane_pts = lane.center_line
            # Total length of the lane
            total_dist = 0.0
            segment_lengths = []
            for j in range(len(lane_pts) - 1):
                d = math.hypot(
                    lane_pts[j + 1].x - lane_pts[j].x, lane_pts[j + 1].y - lane_pts[j].y
                )
                segment_lengths.append(d)
                total_dist += d

            # Current distance traveled
            # Flip direction if it's a left lane (OpenDRIVE left lanes are backward)
            dist_traveled = (v["speed"] * timestamp) % total_dist
            if lane.is_left:
                dist_traveled = total_dist - dist_traveled

            # Find position and tangent on the path
            curr_dist = 0.0
            pos = lane_pts[0]
            yaw = 0.0

            if lane.is_left:
                # Find position from end for left lanes
                reverse_dist = total_dist - dist_traveled
                temp_dist = 0.0
                for j in range(len(segment_lengths)):
                    if temp_dist + segment_lengths[j] >= reverse_dist:
                        ratio = (reverse_dist - temp_dist) / segment_lengths[j]
                        p1 = lane_pts[j]
                        p2 = lane_pts[j + 1]
                        pos = Point3D(
                            p1.x + ratio * (p2.x - p1.x),
                            p1.y + ratio * (p2.y - p1.y),
                            0.0,
                        )
                        # Yaw is toward previous point for left lanes
                        yaw = math.atan2(p1.y - p2.y, p1.x - p2.x)
                        break
                    temp_dist += segment_lengths[j]
            else:
                for j in range(len(segment_lengths)):
                    if curr_dist + segment_lengths[j] >= dist_traveled:
                        ratio = (dist_traveled - curr_dist) / segment_lengths[j]
                        p1 = lane_pts[j]
                        p2 = lane_pts[j + 1]
                        pos = Point3D(
                            p1.x + ratio * (p2.x - p1.x),
                            p1.y + ratio * (p2.y - p1.y),
                            0.0,
                        )
                        yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
                        break
                    curr_dist += segment_lengths[j]

            objects[v["id"]] = ObjectState(
                id=v["id"],
                type="vehicle",
                position=pos,
                velocity=Point3D(
                    v["speed"] * math.cos(yaw), v["speed"] * math.sin(yaw), 0
                ),
                acceleration=Point3D(0, 0, 0),
                orientation=heading_to_quaternion(yaw),
                size=v["size"],
            )

        frames.append(SceneFrame(timestamp=timestamp, objects=objects))

    return frames


def test_viewer():
    """Run an enhanced visual test with dynamic object simulation."""
    # 1. Provide Map Data
    xodr_path = os.path.join(os.path.dirname(__file__), "..", "assets", "Town01.xodr")
    if os.path.exists(xodr_path):
        map_data = parse_xodr(xodr_path)
    else:
        print("Warning: Town01.xodr not found. Map will be empty.")
        map_data = XodrMapData(lanes=[])

    # 2. Init viewer
    viewer = LogViewer()
    viewer.init_xodr(map_data)

    # 3. Simulate frames
    print("Generating simulation frames...")
    frames = generate_simulation_frames(map_data, duration=10.0, fps=10)

    # 4. Render
    print(f"Rendering {len(frames)} frames...")
    for frame in frames:
        viewer.render_state(frame)
        time.sleep(0.05)


if __name__ == "__main__":
    test_viewer()
