"""CLI entry point for replaying a map and object log in the viewer.

This script loads an OpenDRIVE map and a corresponding object log (JSON/JSONL)
 and visualizes the simulation playback using the Rerun viewer.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .log_parser import parse_log
from .viewer import LogViewer
from .xodr_parser import parse_xodr


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for log replay.
    
    Returns:
        An ArgumentParser instance with map and log path arguments.
    """
    parser = argparse.ArgumentParser(
        description="Replay an OpenDRIVE map and JSON/JSONL object log in Rerun.",
    )
    parser.add_argument(
        "--map",
        required=True,
        dest="map_path",
        help="Path to the OpenDRIVE .xodr map file.",
    )
    parser.add_argument(
        "--log",
        required=True,
        dest="log_path",
        help="Path to the JSON or JSONL object log file.",
    )
    parser.add_argument(
        "--application-id",
        default="log_viewer",
        help="Rerun application id used for the viewer session.",
    )
    parser.add_argument(
        "--sampling-eps",
        type=float,
        default=0.1,
        dest="sampling_eps",
        help=(
            "Map linearization tolerance in metres (default: 0.1). "
            "Higher values (0.5–1.0) load faster with lower visual fidelity."
        ),
    )
    return parser


def main() -> None:
    """Main execution loop: parse inputs, load data, and stream to Rerun.
    
    1. Parses CLI arguments.
    2. Loads and parses the OpenDRIVE map (.xodr).
    3. Loads and parses the object simulation log (.json/.jsonl).
    4. Initializes the LogViewer.
    5. Iterates through simulation frames and renders each to the viewer.
    """
    args = build_parser().parse_args()

    map_path = Path(args.map_path)
    log_path = Path(args.log_path)

    if not map_path.is_file():
        raise FileNotFoundError(f"Map file not found: {map_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    print(f"Loading map from: {map_path}")
    map_data = parse_xodr(str(map_path), eps=args.sampling_eps)

    print(f"Loading log from: {log_path}")
    frames = parse_log(str(log_path))

    viewer = LogViewer(application_id=args.application_id)
    viewer.init_xodr(map_data)

    # Pre-compute scalar timeseries per object and send as columnar batches.
    # This replaces per-frame rr.log() calls for 7 scalar channels,
    # reducing IPC overhead from ~7*N*F calls to 7*N calls.
    print("Pre-computing scalar timeseries...")
    obj_ts: dict = defaultdict(list)       # obj_id -> [timestamp, ...]
    obj_speed: dict = defaultdict(list)
    obj_accel: dict = defaultdict(list)
    obj_pos_x: dict = defaultdict(list)
    obj_pos_y: dict = defaultdict(list)
    obj_pos_z: dict = defaultdict(list)
    obj_vel_x: dict = defaultdict(list)
    obj_vel_y: dict = defaultdict(list)

    for frame in frames:
        for obj_id, obj in frame.objects.items():
            obj_ts[obj_id].append(frame.timestamp)
            speed = math.sqrt(
                obj.velocity.x ** 2 + obj.velocity.y ** 2 + obj.velocity.z ** 2
            )
            accel = math.sqrt(
                obj.acceleration.x ** 2
                + obj.acceleration.y ** 2
                + obj.acceleration.z ** 2
            )
            obj_speed[obj_id].append(speed)
            obj_accel[obj_id].append(accel)
            obj_pos_x[obj_id].append(obj.position.x)
            obj_pos_y[obj_id].append(obj.position.y)
            obj_pos_z[obj_id].append(obj.position.z)
            obj_vel_x[obj_id].append(obj.velocity.x)
            obj_vel_y[obj_id].append(obj.velocity.y)

    viewer.send_scalar_columns(
        obj_ts, obj_speed, obj_accel,
        obj_pos_x, obj_pos_y, obj_pos_z,
        obj_vel_x, obj_vel_y,
    )

    print(f"Sending {len(frames)} frames to Rerun...")
    for frame in frames:
        viewer.render_state(frame)

    print("Replay data loaded. Use the Rerun timeline to inspect frames.")


if __name__ == "__main__":
    main()
