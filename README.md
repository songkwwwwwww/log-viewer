# Log Viewer

**Log Viewer** is a lightweight standalone viewer designed to debug and visualize data from autonomous driving simulators and recording systems in 3D.

It renders timestamp-based JSON object logs (vehicles, pedestrians, etc.) on top of [OpenDRIVE (XODR)](https://www.asam.net/standards/detail/opendrive/) road geometry. Built with the **[Rerun (rerun-sdk)](https://rerun.io/)** visualization engine, it provides smooth 3D rendering and intuitive timeline controls (Play, Pause, Scrub).

---

## 🚀 Getting Started

### Prerequisites
This project uses [`uv`](https://github.com/astral-sh/uv), a modern Python package manager, for dependency and environment management.
- Python >= 3.9
- [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup and Installation
Sync dependencies in the project root:
```bash
uv sync
```
*(`uv` will automatically create a `.venv` and install required dependencies like `rerun-sdk` and `numpy`.)*

---

## 🎬 Running the Example

You can launch the viewer immediately using the built-in dummy data and map assets:

```bash
uv run tests/test_run.py
```

Running this command will launch the **Rerun Viewer**, where you can see:
1. Lane visualization based on `assets/Town01.xodr`.
2. Vehicle movement (blue boxes) and trajectory lines from `dummy_log.json` (or generated simulation).
3. **Timeline Scrubber** at the bottom for playback control and time seeking.

### Replaying Your Own Map and Log

If you already have an `.xodr` map and a log file matching the documented JSON/JSONL format, run:

```bash
uv run log-viewer-replay --map path/to/map.xodr --log path/to/log.jsonl
```

This command parses the map, loads all frames from the log, and sends them to the Rerun viewer for timeline-based inspection.

---

## 🗂 Project Structure

```text
log-viewer/
├── assets/                  # XODR map assets
├── docs/                    # Design documentation (design-doc.md)
├── pyproject.toml           # Python packaging and dependencies
├── tests/
│   ├── dummy_log.json       # Sample object logs in JSON format
│   └── test_run.py          # Test script for initializing and running the viewer
└── src/
    └── log_viewer/          # Core source code
        ├── geometry.py      # Shared geometry primitives
        ├── map_model.py     # OpenDRIVE map data models
        ├── scene_model.py   # Dynamic object/logging data models
        ├── data_model.py    # Backward-compatible model re-exports
        ├── replay_log.py    # CLI entry point for map + log replay
        ├── viewer.py        # Rendering interface with Rerun SDK
        ├── xodr_parser.py   # OpenDRIVE geometry parsing logic
        └── log_parser.py    # Time-series log parsing logic
```

---

## 📝 Data Formats

### 1. Map Data (.xodr)
Loads OpenDRIVE (`.xodr`) maps. It currently supports `line` and `arc` segments in `<planView>` and renders lane boundaries based on width coefficients.

### 2. Simulation Log (.json / .jsonl)
Supports single JSON arrays or line-delimited JSON (JSONL) matching the `SceneFrame` structure.
```json
[
  {
    "timestamp": 0.0,
    "objects": [
      {
        "id": "car_001",
        "type": "vehicle",
        "position": {"x": 10.5, "y": -2.0, "z": 0.0},
        "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
        "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "size": [4.5, 2.0, 1.5],
        "future_trajectory": [
          {
            "timestamp": 0.5,
            "position": {"x": 13.0, "y": -2.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
          }
        ]
      }
    ]
  }
]
```
*(For detailed data hierarchy, refer to `docs/design-doc.md`.)*
