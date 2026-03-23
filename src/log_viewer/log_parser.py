"""Parses JSON format simulation logs into SceneFrames.

This module handles the ingestion of simulation logs, which can be in either
a single JSON array format or a JSONL (JSON Lines) format where each line
is a separate JSON object representing a single frame.

Expected JSON Object Structure per frame:
{
    "timestamp": 123.456,
    "objects": [
        {
            "id": "car_1",
            "type": "vehicle",
            "position": {"x": 10.0, "y": 20.0, "z": 0.5},
            "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
            "acceleration": {"x": 0.1, "y": 0.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "size": [4.5, 2.0, 1.5],
            "future_trajectory": [
                {
                    "timestamp": 124.0,
                    "position": {"x": 15.0, "y": 20.0, "z": 0.5},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                }
            ]
        }
    ]
}
"""

import json
from pathlib import Path
from typing import List

from .geometry import Point3D, Quaternion
from .scene_model import ObjectState, PoseAtTime, SceneFrame


class LogParser:
    """Parses JSON or JSONL format simulation logs into SceneFrames."""

    def __init__(self, file_path: str):
        """Initialize the LogParser with a file path.

        Args:
            file_path: Path to the .json or .jsonl log file.
        """
        self.file_path = file_path

    def parse(self) -> List[SceneFrame]:
        """Parse the simulation logs into a list of SceneFrames.

        Supports both .json (single array) and .jsonl (one frame per line) formats.

        Returns:
            A list of SceneFrame objects sorted by timestamp (if provided in the log).
        """
        frames = []

        if self.file_path.endswith(".jsonl"):
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    frame_data = json.loads(line)
                    frames.append(self._parse_frame(frame_data))
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # List of frames
                    for frame_data in data:
                        frames.append(self._parse_frame(frame_data))
                else:
                    # Single frame object
                    frames.append(self._parse_frame(data))

        return frames

    def _parse_frame(self, data: dict) -> SceneFrame:
        """Helper to convert a dictionary representing a frame into a SceneFrame object."""
        timestamp = data.get("timestamp", 0.0)
        objects_dict = {}

        for obj_data in data.get("objects", []):
            obj_id = obj_data.get("id", "unknown")

            # Position (meters)
            pos_dict = obj_data.get("position", {"x": 0, "y": 0, "z": 0})
            position = Point3D(
                pos_dict.get("x", 0), pos_dict.get("y", 0), pos_dict.get("z", 0)
            )

            # Velocity (meters/second)
            vel_dict = obj_data.get("velocity", {"x": 0, "y": 0, "z": 0})
            velocity = Point3D(
                vel_dict.get("x", 0), vel_dict.get("y", 0), vel_dict.get("z", 0)
            )

            # Acceleration (meters/second^2)
            acc_dict = obj_data.get("acceleration", {"x": 0, "y": 0, "z": 0})
            acceleration = Point3D(
                acc_dict.get("x", 0), acc_dict.get("y", 0), acc_dict.get("z", 0)
            )

            # Orientation (quaternion w,x,y,z)
            ori_dict = obj_data.get("orientation", {"w": 1, "x": 0, "y": 0, "z": 0})
            orientation = Quaternion(
                ori_dict.get("w", 1),
                ori_dict.get("x", 0),
                ori_dict.get("y", 0),
                ori_dict.get("z", 0),
            )

            # Size (length, width, height in meters)
            size_list = obj_data.get("size", [4.5, 2.0, 1.5])  # default car size
            size = (size_list[0], size_list[1], size_list[2])

            # Future Trajectory (list of poses)
            future_trajectory = None
            if "future_trajectory" in obj_data:
                future_trajectory = []
                for pt_data in obj_data["future_trajectory"]:
                    pt_ts = pt_data.get("timestamp", timestamp)
                    pt_pos = pt_data.get("position", {"x": 0, "y": 0, "z": 0})
                    pt_ori = pt_data.get(
                        "orientation", {"w": 1, "x": 0, "y": 0, "z": 0}
                    )
                    future_trajectory.append(
                        PoseAtTime(
                            timestamp=pt_ts,
                            position=Point3D(
                                pt_pos.get("x", 0),
                                pt_pos.get("y", 0),
                                pt_pos.get("z", 0),
                            ),
                            orientation=Quaternion(
                                pt_ori.get("w", 1),
                                pt_ori.get("x", 0),
                                pt_ori.get("y", 0),
                                pt_ori.get("z", 0),
                            ),
                        )
                    )

            obj_state = ObjectState(
                id=obj_id,
                type=obj_data.get("type", "unknown"),
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                orientation=orientation,
                size=size,
                future_trajectory=future_trajectory,
            )
            objects_dict[obj_id] = obj_state

        return SceneFrame(timestamp=timestamp, objects=objects_dict)


def parse_log(file_path: str) -> List[SceneFrame]:
    """Convenience function to parse a JSON/JSONL log file and return SceneFrames.

    Args:
        file_path: Path to the log file.

    Returns:
        List of SceneFrame objects.
    """
    return LogParser(file_path).parse()
