"""CLI entry point for replaying a map and object log in the viewer.

This script loads an OpenDRIVE map and a corresponding object log (JSON/JSONL)
 and visualizes the simulation playback using the Rerun viewer.
"""

import argparse
from pathlib import Path

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
    map_data = parse_xodr(str(map_path))

    print(f"Loading log from: {log_path}")
    frames = parse_log(str(log_path))

    viewer = LogViewer(application_id=args.application_id)
    viewer.init_xodr(map_data)

    print(f"Sending {len(frames)} frames to Rerun...")
    for frame in frames:
        viewer.render_state(frame)

    print("Replay data loaded. Use the Rerun timeline to inspect frames.")


if __name__ == "__main__":
    main()
