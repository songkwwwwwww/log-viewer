"""Integration test and demonstration script for the Log Viewer.

This script simulates dynamic objects (vehicles) moving along lanes in an 
OpenDRIVE map and visualizes their movement in the Rerun-based viewer.
"""

import math
import os
import time
from typing import List, Dict

from log_viewer.geometry import Point3D, Quaternion
from log_viewer.map_model import XodrMapData
from log_viewer.scene_model import ObjectState, SceneFrame
from log_viewer.viewer import LogViewer
from log_viewer.xodr_parser import parse_xodr


def heading_to_quaternion(yaw: float) -> Quaternion:
    """Convert a 2D yaw angle (radians) to a 3D Quaternion.

    Args:
        yaw: Rotation angle around the Z-axis in radians.

    Returns:
        A Quaternion object representing the rotation.
    """
    return Quaternion(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))


def generate_simulation_frames(
    map_data: XodrMapData, duration: float = 5.0, fps: int = 10
) -> List[SceneFrame]:
    """Generates synthetic simulation frames with objects moving along lanes.

    This function picks a few driving lanes from the map and creates vehicles
    that travel along the lane's centerline at a constant speed.

    Args:
        map_data: Parsed OpenDRIVE map data.
        duration: Total simulation time in seconds.
        fps: Frames per second (sampling rate).

    Returns:
        A list of SceneFrame objects representing the simulated timeline.
    """
    frames = []
    num_frames = int(duration * fps)
    dt = 1.0 / fps

    # Filter for drivable lanes
    driving_lanes = [lane for lane in map_data.lanes if lane.type == "driving"]
    if not driving_lanes:
        return []

    # Select up to 3 lanes for the simulation
    selected_lanes = driving_lanes[:3]
    vehicles = []

    for idx, lane in enumerate(selected_lanes):
        vehicles.append(
            {
                "id": f"v{idx}",
                "lane": lane,
                "speed": 10.0 + idx * 2.0,  # Each vehicle has a distinct speed
                "size": (4.5, 2.0, 1.5),
            }
        )

    for i in range(num_frames):
        timestamp = i * dt
        objects: Dict[str, ObjectState] = {}

        for v in vehicles:
            lane = v["lane"]
            lane_pts = lane.center_line
            
            # Calculate total lane length and segment lengths for interpolation
            total_dist = 0.0
            segment_lengths = []
            for j in range(len(lane_pts) - 1):
                d = math.hypot(
                    lane_pts[j + 1].x - lane_pts[j].x, 
                    lane_pts[j + 1].y - lane_pts[j].y
                )
                segment_lengths.append(d)
                total_dist += d

            # Determine the longitudinal distance traveled
            # OpenDRIVE left lanes travel in reverse direction relative to the reference line
            dist_traveled = (v["speed"] * timestamp) % total_dist
            if lane.is_left:
                dist_traveled = total_dist - dist_traveled

            # Interpolate position and heading along the lane centerline
            curr_dist = 0.0
            pos = lane_pts[0]
            yaw = 0.0

            if lane.is_left:
                # Find position by tracing back from the end for left lanes
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
                sub_type="car",
                is_static=False,
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
    """Run an integration test with dynamic object simulation in Rerun.
    
    This function:
    1. Loads a sample OpenDRIVE map (Town01.xodr).
    2. Initializes the LogViewer.
    3. Generates synthetic simulation frames.
    4. Streams the frames to the Rerun viewer.
    """
    # 1. Load Map Data
    xodr_path = os.path.join(os.path.dirname(__file__), "..", "assets", "Town01.xodr")
    if os.path.exists(xodr_path):
        map_data = parse_xodr(xodr_path)
    else:
        print(f"Warning: {xodr_path} not found. Map will be empty.")
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
        time.sleep(0.01)  # Throttle delivery to the viewer


if __name__ == "__main__":
    test_viewer()
