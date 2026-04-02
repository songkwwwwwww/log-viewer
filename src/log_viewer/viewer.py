"""Visualization engine utilizing rerun-sdk for 3D road and object playback.

This module provides the LogViewer class, which translates parsed map and
simulation data into Rerun entities (meshes, boxes, line strips, etc.)
for interactive visualization.
"""

import math

import rerun as rr
import rerun.blueprint as rrb
import numpy as np

from .map_model import XodrMapData
from .scene_model import SceneFrame

# Lane type color palette [R, G, B, A]
# Used to differentiate between driving lanes, sidewalks, etc.
_LANE_COLORS = {
    "driving": [80, 80, 80, 255],
    "sidewalk": [120, 120, 120, 255],
    "biking": [150, 60, 60, 255],
    "shoulder": [100, 95, 90, 255],
}

# Road mark color palette [R, G, B, A] — keyed by OpenDRIVE color attribute.
_ROAD_MARK_COLORS = {
    "white": [255, 255, 255, 255],
    "yellow": [255, 220, 0, 255],
    "standard": [255, 255, 255, 255],
    "blue": [0, 100, 255, 255],
    "green": [0, 200, 80, 255],
    "red": [220, 50, 50, 255],
    "orange": [255, 140, 0, 255],
}

# Object type colors [R, G, B] — fallback when sub_type is not recognized.
_OBJECT_COLORS = {
    "vehicle": [0, 100, 255],
    "pedestrian": [255, 100, 0],
}

# Sub-type color overrides [R, G, B].
# When an object has a recognized sub_type, this palette takes priority over
# the type-based color above.
_SUB_TYPE_COLORS = {
    "yellow": [255, 200, 0],
    "blue": [0, 100, 255],
    "red": [220, 50, 50],
    "unknown": [160, 160, 160],
}


class LogViewer:
    """Core Log Viewer class using rerun-sdk for 3D visualization.

    This class handles the initialization of the Rerun viewer, the layout
    of the visualization workspace (blueprint), and the logging of
    static map data and dynamic simulation frames.
    """

    def __init__(self, application_id: str = "log_viewer"):
        """Initialize the LogViewer engine.

        Sets up the Rerun blueprint which defines the layout of the viewer:
        - A main 3D Spatial View for the map and objects.
        - Multiple Time Series Views for telemetry data (speed, acceleration, position).

        Args:
            application_id: Unique string identifier for the Rerun session.
        """
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                # Left: 3D spatial view (map + objects)
                rrb.Spatial3DView(
                    name="3D View",
                    origin="/",
                    contents=[
                        "map/**",
                        "objects/*/box",
                    ],
                ),
                rrb.Vertical(
                    # Right-top: speed & acceleration telemetry
                    rrb.TimeSeriesView(
                        name="Speed & Acceleration",
                        origin="/",
                        contents=[
                            "objects/+/scalars/speed",
                            "objects/+/scalars/acceleration",
                        ],
                    ),
                    # Right-mid: 2D position telemetry
                    rrb.TimeSeriesView(
                        name="Position (X/Y)",
                        origin="/",
                        contents=[
                            "objects/+/scalars/position_x",
                            "objects/+/scalars/position_y",
                        ],
                    ),
                    # Right-bottom: velocity components
                    rrb.TimeSeriesView(
                        name="Velocity (Vx/Vy)",
                        origin="/",
                        contents=[
                            "objects/+/scalars/velocity_x",
                            "objects/+/scalars/velocity_y",
                        ],
                    ),
                    row_shares=[1, 1, 1],
                ),
                column_shares=[2, 1],
            ),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="expanded"),
            rrb.TimePanel(state="expanded"),
        )
        # Initialize Rerun and spawn the viewer window
        rr.init(application_id, spawn=True, default_blueprint=blueprint)
        self._logged_static_objects: set = set()

    @staticmethod
    def _build_strip_mesh(left_pts, right_pts):
        """Build a triangle-strip mesh from left/right boundary point lists.

        Returns (vertices, indices) as numpy arrays, or (None, None) if the
        inputs are too short or mismatched.
        """
        n = len(left_pts)
        if n != len(right_pts) or n < 2:
            return None, None

        vertices = np.empty((2 * n, 3), dtype=np.float32)
        for i in range(n):
            lp, rp = left_pts[i], right_pts[i]
            vertices[2 * i] = (lp.x, lp.y, lp.z)
            vertices[2 * i + 1] = (rp.x, rp.y, rp.z)

        idx = np.arange(n - 1, dtype=np.uint32)
        l1 = 2 * idx
        r1 = l1 + 1
        l2 = l1 + 2
        r2 = l1 + 3
        tri_a = np.stack([l1, r1, l2], axis=1)
        tri_b = np.stack([r1, r2, l2], axis=1)
        indices = np.empty((2 * (n - 1), 3), dtype=np.uint32)
        indices[0::2] = tri_a
        indices[1::2] = tri_b

        return vertices, indices

    @staticmethod
    def _pts_to_array(points):
        """Convert a list of Point3D to a (N, 3) float32 numpy array."""
        return np.array(
            [[p.x, p.y, p.z] for p in points], dtype=np.float32
        )

    def init_xodr(self, map_data: XodrMapData):
        """Initialize and log the OpenDRIVE map in the viewer.

        All map geometry is batched into a small number of global Rerun entities
        to minimise the entity tree size and rr.log() call count:
          - Lane surfaces: one Mesh3D per lane type (4 entities)
          - Lane boundaries: single LineStrips3D (1 entity)
          - Center lines: single LineStrips3D (1 entity)
          - Direction arrows: single Arrows3D (1 entity)
          - Road marks: single Mesh3D (1 entity, unchanged)

        Args:
            map_data: Parsed XODR map data containing lanes and boundaries.
        """
        # ---------- Accumulators for batched geometry ----------
        # Lane surfaces grouped by type: type -> (verts_list, indices_list, offset)
        surf_verts: dict = {t: [] for t in _LANE_COLORS}
        surf_indices: dict = {t: [] for t in _LANE_COLORS}
        surf_vert_offset: dict = {t: 0 for t in _LANE_COLORS}

        all_boundary_strips = []   # all boundary LineStrips3D strips
        all_center_strips = []     # all center line strips
        all_arrow_origins = []     # direction arrow origins
        all_arrow_vectors = []     # direction arrow vectors

        arrow_scale = 2.0

        for lane in map_data.lanes:
            left_pts = lane.left_boundary.points
            right_pts = lane.right_boundary.points

            # ---------- 1. Lane surface: accumulate per lane type ----------
            vertices, indices = self._build_strip_mesh(left_pts, right_pts)
            if vertices is not None and lane.type in surf_verts:
                offset = surf_vert_offset[lane.type]
                surf_verts[lane.type].append(vertices)
                surf_indices[lane.type].append(indices + offset)
                surf_vert_offset[lane.type] += len(vertices)

            # ---------- 2. Lane boundaries: accumulate all strips ----------
            for boundary in (lane.left_boundary, lane.right_boundary):
                if boundary.points:
                    all_boundary_strips.append(self._pts_to_array(boundary.points))

            # ---------- 3. Center lines + direction arrows ----------
            cps = lane.center_line
            if len(cps) >= 2:
                all_center_strips.append(self._pts_to_array(cps))

                dir_sign = -1.0 if lane.is_left else 1.0
                last = len(cps) - 2
                mid = min(len(cps) // 2, last)
                for idx in sorted(set([0, mid, last])):
                    p0 = cps[idx]
                    p1 = cps[idx + 1]
                    dx = p1.x - p0.x
                    dy = p1.y - p0.y
                    norm = math.sqrt(dx * dx + dy * dy) or 1.0
                    all_arrow_origins.append([p0.x, p0.y, p0.z + 0.3])
                    all_arrow_vectors.append([
                        dx / norm * arrow_scale * dir_sign,
                        dy / norm * arrow_scale * dir_sign,
                        0.0,
                    ])

        # ---------- Flush batched lane surfaces (one rr.log per lane type) ----------
        for lane_type, verts_list in surf_verts.items():
            if not verts_list:
                continue
            combined_verts = np.concatenate(verts_list)
            combined_indices = np.concatenate(surf_indices[lane_type])
            color = _LANE_COLORS[lane_type]
            rr.log(
                f"map/surfaces/{lane_type}",
                rr.Mesh3D(
                    vertex_positions=combined_verts,
                    triangle_indices=combined_indices,
                    albedo_factor=color,
                ),
                static=True,
            )

        # ---------- Flush boundaries (single rr.log) ----------
        if all_boundary_strips:
            rr.log(
                "map/boundaries",
                rr.LineStrips3D(all_boundary_strips, colors=[200, 200, 200]),
                static=True,
            )

        # ---------- Flush center lines (single rr.log) ----------
        if all_center_strips:
            rr.log(
                "map/centers",
                rr.LineStrips3D(all_center_strips, colors=[255, 255, 0]),
                static=True,
            )

        # ---------- Flush direction arrows (single rr.log) ----------
        if all_arrow_origins:
            rr.log(
                "map/directions",
                rr.Arrows3D(
                    origins=all_arrow_origins,
                    vectors=all_arrow_vectors,
                    colors=[255, 200, 0],
                    radii=0.2,
                ),
                static=True,
            )

        # ---------- Road marks: merge into a single batched mesh ----------
        all_mark_verts = []
        all_mark_indices = []
        all_mark_colors = []
        vert_offset = 0
        for mark in map_data.road_marks:
            left_pts = mark.left_pts
            right_pts = mark.right_pts
            n = len(left_pts)
            if n != len(right_pts) or n < 2:
                continue
            verts = np.empty((2 * n, 3), dtype=np.float32)
            for i in range(n):
                lp, rp = left_pts[i], right_pts[i]
                verts[2 * i] = (lp.x, lp.y, lp.z)
                verts[2 * i + 1] = (rp.x, rp.y, rp.z)

            idx = np.arange(n - 1, dtype=np.uint32)
            l1 = 2 * idx + vert_offset
            r1 = l1 + 1
            l2 = l1 + 2
            r2 = l1 + 3
            tri_a = np.stack([l1, r1, l2], axis=1)
            tri_b = np.stack([r1, r2, l2], axis=1)
            mark_indices = np.empty((2 * (n - 1), 3), dtype=np.uint32)
            mark_indices[0::2] = tri_a
            mark_indices[1::2] = tri_b

            color = _ROAD_MARK_COLORS.get(mark.color, [255, 255, 255, 255])
            per_vert_colors = np.tile(
                np.array(color, dtype=np.uint8), (2 * n, 1)
            )

            all_mark_verts.append(verts)
            all_mark_indices.append(mark_indices)
            all_mark_colors.append(per_vert_colors)
            vert_offset += 2 * n

        if all_mark_verts:
            rr.log(
                "map/road_marks",
                rr.Mesh3D(
                    vertex_positions=np.concatenate(all_mark_verts),
                    triangle_indices=np.concatenate(all_mark_indices),
                    vertex_colors=np.concatenate(all_mark_colors),
                ),
                static=True,
            )

    def send_scalar_columns(
        self,
        obj_ts: dict,
        obj_speed: dict,
        obj_accel: dict,
        obj_pos_x: dict,
        obj_pos_y: dict,
        obj_pos_z: dict,
        obj_vel_x: dict,
        obj_vel_y: dict,
    ):
        """Send all scalar timeseries as columnar batches via rr.send_columns().

        This replaces per-frame rr.log() calls for scalars, reducing IPC
        overhead from ~7*N*F calls to 7*N calls (N=objects, F=frames).
        """
        scalar_channels = [
            ("speed", obj_speed),
            ("acceleration", obj_accel),
            ("position_x", obj_pos_x),
            ("position_y", obj_pos_y),
            ("position_z", obj_pos_z),
            ("velocity_x", obj_vel_x),
            ("velocity_y", obj_vel_y),
        ]
        for obj_id, ts_list in obj_ts.items():
            ts_arr = np.array(ts_list, dtype=np.float64)
            time_col = rr.TimeColumn("sim_time", timestamp=ts_arr)
            for channel_name, channel_data in scalar_channels:
                values = np.array(channel_data[obj_id], dtype=np.float64)
                rr.send_columns(
                    f"objects/{obj_id}/scalars/{channel_name}",
                    indexes=[time_col],
                    columns=rr.Scalars.columns(scalars=values),
                )

    def render_state(self, frame: SceneFrame):
        """Render a single frame of simulation state.

        Logs 3D bounding boxes and future trajectories for all objects.
        Scalar timeseries are sent separately via send_scalar_columns().

        Args:
            frame: The scene frame containing all object states at a given timestamp.
        """
        rr.set_time("sim_time", timestamp=frame.timestamp)

        for obj_id, obj in frame.objects.items():
            obj_path = f"objects/{obj_id}"

            # Skip re-logging static objects that have already been logged
            if obj.is_static and obj_id in self._logged_static_objects:
                continue

            speed = math.sqrt(obj.velocity.x**2 + obj.velocity.y**2 + obj.velocity.z**2)
            accel = math.sqrt(
                obj.acceleration.x**2 + obj.acceleration.y**2 + obj.acceleration.z**2
            )

            color = _SUB_TYPE_COLORS.get(
                obj.sub_type, _OBJECT_COLORS.get(obj.type, [200, 200, 200])
            )
            rr.log(
                f"{obj_path}/box",
                rr.Boxes3D(
                    half_sizes=[[obj.size[0] / 2, obj.size[1] / 2, obj.size[2] / 2]],
                    centers=[[obj.position.x, obj.position.y, obj.position.z]],
                    quaternions=[
                        [
                            obj.orientation.x,
                            obj.orientation.y,
                            obj.orientation.z,
                            obj.orientation.w,
                        ]
                    ],
                    colors=[color],
                    labels=[f"{obj.type}_{obj.id}"],
                ),
                rr.AnyValues(
                    object_id=obj.id,
                    object_type=obj.type,
                    object_sub_type=obj.sub_type,
                    is_static=obj.is_static,
                    position_x=obj.position.x,
                    position_y=obj.position.y,
                    position_z=obj.position.z,
                    speed_mps=round(speed, 3),
                    acceleration_mps2=round(accel, 3),
                    velocity_x=obj.velocity.x,
                    velocity_y=obj.velocity.y,
                ),
                static=obj.is_static,
            )

            if obj.is_static:
                self._logged_static_objects.add(obj_id)

            # ---- Future trajectory ----
            if obj.future_trajectory:
                traj_pts = [
                    [p.position.x, p.position.y, p.position.z]
                    for p in obj.future_trajectory
                ]
                if traj_pts:
                    traj_pts.insert(0, [obj.position.x, obj.position.y, obj.position.z])
                    rr.log(
                        f"{obj_path}/trajectory",
                        rr.LineStrips3D([traj_pts], colors=[255, 255, 0]),
                    )

    def load_log(self, file_path: str):
        """Parse a JSON/CSV and replay it (Placeholder)."""
        print(f"Loading log from: {file_path}")
        # Parse log, build SceneFrames and call render_state in a loop
