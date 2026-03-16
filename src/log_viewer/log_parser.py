"""Parses JSON format simulation logs into SceneFrames."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .geometry import Point3D, Quaternion
from .scene_model import ObjectState, PoseAtTime, SceneFrame


class LogParser:
    """Parses JSON or JSONL format simulation logs into SceneFrames."""

    def __init__(
        self,
        file_path: str,
        transform_matrix: Optional[np.ndarray] = None,
    ):
        """Initialize the LogParser with a file path."""
        self.file_path = file_path
        self.transform_matrix = transform_matrix

    def parse(self) -> List[SceneFrame]:
        """Parse the JSON or JSONL format simulation logs into SceneFrames."""
        frames = []

        # Determine if it's JSON array or JSONL based on file extension
        # In a real app we might peek at the first character
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
                    for frame_data in data:
                        frames.append(self._parse_frame(frame_data))
                else:
                    # Maybe it's a single frame dict
                    frames.append(self._parse_frame(data))

        return frames

    def _parse_frame(self, data: dict) -> SceneFrame:
        timestamp = data.get("timestamp", 0.0)
        objects_dict = {}

        for obj_data in data.get("objects", []):
            obj_id = obj_data.get("id", "unknown")

            # Position
            pos_dict = obj_data.get("position", {"x": 0, "y": 0, "z": 0})
            position = self._transform_point(
                pos_dict.get("x", 0), pos_dict.get("y", 0), pos_dict.get("z", 0)
            )

            # Velocity
            vel_dict = obj_data.get("velocity", {"x": 0, "y": 0, "z": 0})
            velocity = Point3D(
                vel_dict.get("x", 0), vel_dict.get("y", 0), vel_dict.get("z", 0)
            )

            # Acceleration
            acc_dict = obj_data.get("acceleration", {"x": 0, "y": 0, "z": 0})
            acceleration = Point3D(
                acc_dict.get("x", 0), acc_dict.get("y", 0), acc_dict.get("z", 0)
            )

            # Orientation
            ori_dict = obj_data.get("orientation", {"w": 1, "x": 0, "y": 0, "z": 0})
            orientation = Quaternion(
                ori_dict.get("w", 1),
                ori_dict.get("x", 0),
                ori_dict.get("y", 0),
                ori_dict.get("z", 0),
            )

            # Size
            size_list = obj_data.get("size", [4.5, 2.0, 1.5])  # default car size
            size = (size_list[0], size_list[1], size_list[2])

            # Future Trajectory
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
                            position=self._transform_point(
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

    def _transform_point(self, x: float, y: float, z: float) -> Point3D:
        """Apply the optional homogeneous transform to a point."""
        if self.transform_matrix is None:
            return Point3D(x, y, z)

        homogeneous_point = np.array([x, y, z, 1.0], dtype=float)
        transformed_point = self.transform_matrix @ homogeneous_point
        return Point3D(
            float(transformed_point[0]),
            float(transformed_point[1]),
            float(transformed_point[2]),
        )


def load_transform_matrix(meta_file_path: str) -> np.ndarray:
    """Load a sim-to-xodr_enu 4x4 transform matrix from a metadata JSON file."""
    meta_path = Path(meta_file_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    has_sim_to_xodr_enu = "sim_to_xodr_enu" in metadata
    has_xodr_enu_to_sim = "xodr_enu_to_sim" in metadata

    if has_sim_to_xodr_enu and has_xodr_enu_to_sim:
        raise ValueError(
            "Metadata file must contain only one of "
            "'sim_to_xodr_enu' or 'xodr_enu_to_sim'."
        )

    if has_sim_to_xodr_enu:
        matrix = np.array(metadata["sim_to_xodr_enu"], dtype=float)
    elif has_xodr_enu_to_sim:
        inverse_matrix = np.array(metadata["xodr_enu_to_sim"], dtype=float)
        if inverse_matrix.shape != (4, 4):
            raise ValueError(
                "xodr_enu_to_sim must be a 4x4 array in the metadata file."
            )
        matrix = np.linalg.inv(inverse_matrix)
    else:
        raise ValueError(
            "Metadata file must contain either 'sim_to_xodr_enu' "
            "or 'xodr_enu_to_sim'."
        )

    if matrix.shape != (4, 4):
        raise ValueError(
            "sim_to_xodr_enu must be a 4x4 array in the metadata file."
        )

    return matrix


def parse_log(
    file_path: str,
    transform_matrix: Optional[np.ndarray] = None,
    meta_file_path: Optional[str] = None,
) -> List[SceneFrame]:
    """Parse a file using LogParser and return a list of SceneFrames."""
    if transform_matrix is not None and meta_file_path is not None:
        raise ValueError("Provide either transform_matrix or meta_file_path, not both.")

    if meta_file_path is not None:
        transform_matrix = load_transform_matrix(meta_file_path)

    parser = LogParser(file_path, transform_matrix=transform_matrix)
    return parser.parse()
