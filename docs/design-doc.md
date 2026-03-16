# Design Document: Log Viewer

## 1. Overview
This document defines the architecture and implementation design for the **XODR (OpenDRIVE) based road map and log visualization tool (Log Viewer)**.

The primary goal of this tool is to provide an intuitive 3D viewer for analyzing and debugging autonomous driving simulations and real-world driving logs.
- XODR (OpenDRIVE) based road map parsing and 3D visualization.
- Visualization of objects (vehicles, pedestrians, etc.) on top of the XODR map based on timestamped log data.
- Identification and display of 3D Bounding Boxes and unique object IDs.
- Visualization of historical and future moving paths (trajectories).

**The initial version focuses on a lightweight standalone viewer for debugging and analysis rather than a complex multi-process simulation environment.**

---

## 2. Goals & Non-goals

### 2.1 Primary Goals
- Lightweight rendering of XODR map assets (lanes, boundaries).
- Time-series log playback control (Play, Pause, Step Forward/Backward, Manual Scrubbing).
- Support for two main operation modes:
  1. **Python API Mode (Real-time Injector)**
     - Provide XODR map on initialization.
     - Construct a `SceneFrame` object for each frame in a main loop and call the `render_state(...)` API directly.
  2. **Log Replay Mode (File Replay)**
     - Read specified log files (JSON/JSONL) to assemble full frame data and replay via the UI.

### 2.2 Secondary Goals
- Visualization of past/future trajectory lines per object.
- Camera controls (Zoom, Pan, Orbit, Top View / 3D Perspective View toggle).
- Detailed state inspection (speed, acceleration, etc.) when clicking an object.

### 2.3 Non-goals (Initial Version)
- Vehicle dynamics simulation or control logic.
- Raw sensor data visualization (LiDAR point clouds, camera images).
- HD Map lane topology editing or generation logic.
- Complex mesh rendering for photorealism (trees, buildings, etc.).

---

## 3. Implementation Strategy

### 3.1 Implementation Language: Python
**Conclusion: Python is recommended.**

- **Rationale**:
  - Naturally satisfies the requirement for a "Python API."
  - Familiar to data scientists and autonomous driving engineers for custom scripting and logic modification.
  - Leverages the powerful ecosystem for data preprocessing (NumPy, Pandas, SciPy).
  - While C++ is superior for extreme rendering optimization, a "simple debugging viewer" benefits more from Python's rapid development iteration speed.

### 3.2 Visualization Library
Required features: 3D viewer and timeline controller in Python.

1. **Rerun (rerun-sdk) [Recommended]**
   - **Description**: A modern open-source logging and visualization tool (Rust backend, Python API) built for time-series 3D spatial debugging.
   - **Pros**: Handles primitives like 3D Bounding Boxes, lines (trajectories), and points very efficiently. Crucially, it has a **built-in timeline widget and replay UI**, eliminating the need to build a GUI from scratch.
2. **Open3D / VisPy**
   - **Pros**: Good for custom rendering control.
   - **Cons**: High development overhead to build UI layers like playback/pause and timeline sliders (e.g., using PyQt/PySide6).

**Conclusion**: Use **Rerun** to deliver high-quality 3D object visualization and log replay control with minimal effort.

---

## 4. High-Level Design & System Architecture

The system consists of a data processing layer (Parser), a control layer (Core API), and a visualization layer (Engine).

```text
  [ Log File(JSON/JSONL) ]       [ HD Map (.xodr) ]
           |                            |
           v                            v
 +--------------------+      +--------------------+
 |     LogParser      |      |     XodrParser     |
 +--------------------+      +--------------------+
           | ('SceneFrame')             | ('Lanes', 'Boundaries')
           v                            v
 +======================================================+
 |                   LogViewer Core                     |
 |  - init_xodr(map_data, transform_matrix=None)        |
 |  - render_state(frame)                               |
 |  - load_log(file_path)                               |
 +======================================================+
                           |  (Rerun Python API)
                           v
 +======================================================+
 |        Visualization Engine (Rerun Viewer)           |
 |  - 3D Renderer (Map & Object BBox)                   |
 |  - Timeline Controller (Play/Pause/Scrub)            |
 +======================================================+
```

1. **XodrParser**: Parses OpenDRIVE format and extracts only geometry for visualization (Lane, Boundary coordinates).
2. **LogParser**: Reads historical logs and organizes them by timestamp into `SceneFrame` structures.
3. **LogViewer Core**: The main public API; pushes real-time data via `render_state()` or uploads full datasets to the visualization engine via `load_log()`.
4. **Visualization Engine**: The Rerun Viewer process handles the actual rendering and playback controls.

---

## 5. Detailed Data Model

Uses a Cartesian coordinate system (X, Y, Z, Quaternions). XODR visualization begins by transforming lane and boundary geometry data.

For implementation, the data model is split by responsibility:
- `geometry.py`: shared primitives such as `Point3D` and `Quaternion`
- `map_model.py`: OpenDRIVE-derived map structures
- `scene_model.py`: logged dynamic object and frame structures

### 5.1 Geometry Foundations

```python
@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float
```

### 5.2 Map Data

Constructs line and surface arrays optimized for visualization rather than the heavy OpenDRIVE structure.

```python
@dataclass
class LaneBoundary:
    id: str                 # ex: "road_1_lane_-1_left"
    style: str              # e.g., solid, dashed
    points: List[Point3D]   # sampled points for drawing curves/polygons

@dataclass
class Lane:
    id: str
    road_id: str
    type: str               # determines surface color/texture (driving, sidewalk, etc.)
    left_boundary: LaneBoundary
    right_boundary: LaneBoundary
    center_line: List[Point3D] # used for trajectory generation or path visualization
    is_left: bool = False   # lane direction relative to reference line

@dataclass
class RoadLink:
    road_id: str
    successor_road_id: Optional[str]
    successor_type: str
    end_point: Point3D
    successor_start_point: Optional[Point3D]

@dataclass
class XodrMapData:
    lanes: List[Lane]
    road_links: List[RoadLink]
```

### 5.3 Scene Dynamic Data

Defines the real-time state of objects per timestamp.

```python
@dataclass
class PoseAtTime:
    timestamp: float
    position: Point3D
    orientation: Quaternion

@dataclass
class ObjectState:
    id: str                                  # Unique ID (ex: "car_001")
    type: str                                # vehicle, pedestrian, etc.
    position: Point3D                        # center point or rear-axle (x, y, z)
    velocity: Point3D                        # (vx, vy, vz)
    acceleration: Point3D                    # (ax, ay, az)
    orientation: Quaternion                  # rotation in 3D (w, x, y, z)
    size: Tuple[float, float, float]         # Bounding Box dimensions (length, width, height)
    future_trajectory: List[PoseAtTime]      # (Optional) predicted/planned trajectory

@dataclass
class SceneFrame:
    timestamp: float
    objects: Dict[str, ObjectState]          # id -> ObjectState mapping
```

### 5.4 JSON Log Format

Standard structure for frame-by-frame logging. Supports JSON arrays or JSON Lines (JSONL).

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

- **JSON Lines (.jsonl)**: For very large logs, `.jsonl` is more efficient than a single massive array.

---

## 6. Risks & Future Work

1. **Coordinate System Alignment**
   - XODR map coordinates may differ from external log data (e.g., ENU, NED, or local frames).
   - *Mitigation*: The `init_xodr` API accepts an optional 4x4 `transform_matrix` to align data.
2. **Geometry Vertex Overhead**
   - Rendering vast XODR maps as dense `Point3D` sets can be resource-intensive.
   - *Mitigation*: Implement dynamic sampling based on distance (LOD) or cache static meshes on initial load.
