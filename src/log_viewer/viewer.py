"""Visualization engine utilizing rerun-sdk for 3D road and object playback.

This module provides the LogViewer class, which translates parsed map and
simulation data into Rerun entities (meshes, boxes, line strips, etc.)
for interactive visualization.
"""

import math

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from typing import Optional

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

# Object type colors [R, G, B]
_OBJECT_COLORS = {
    "vehicle": [0, 100, 255],
    "pedestrian": [255, 100, 0],
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

    def init_xodr(self, map_data: XodrMapData):
        """Initialize and log the OpenDRIVE map in the viewer.

        Iterates through all parsed lanes and logs them as 3D meshes (surfaces),
        line strips (boundaries and centerlines), and arrows (direction).

        Args:
            map_data: Parsed XODR map data containing lanes and boundaries.
        """

        for lane in map_data.lanes:
            # Each lane is logged under a unique entity path hierarchy
            entity_path = f"map/roads/{lane.road_id}/lanes/{lane.id}"

            # ---------- 1. Lane surface (Mesh3D) ----------
            # We construct a triangle strip mesh between the left and right boundaries
            left_pts = lane.left_boundary.points
            right_pts = lane.right_boundary.points

            if len(left_pts) == len(right_pts) and len(left_pts) >= 2:
                vertices = []
                indices = []
                for i in range(len(left_pts)):
                    # Add vertices for both boundaries at each longitudinal step
                    vertices.append([left_pts[i].x, left_pts[i].y, left_pts[i].z])
                    vertices.append([right_pts[i].x, right_pts[i].y, right_pts[i].z])

                    if i < len(left_pts) - 1:
                        # Define two triangles per segment to form a quad
                        idx_l1 = 2 * i
                        idx_r1 = 2 * i + 1
                        idx_l2 = 2 * i + 2
                        idx_r2 = 2 * i + 3
                        indices.append([idx_l1, idx_r1, idx_l2])
                        indices.append([idx_r1, idx_r2, idx_l2])

                color = _LANE_COLORS.get(lane.type, [100, 100, 100, 255])
                rr.log(
                    f"{entity_path}/surface",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=indices,
                        albedo_factor=color,
                    ),
                    # Attach lane metadata — visible in Rerun's Selection Panel
                    rr.AnyValues(
                        lane_id=lane.id,
                        road_id=lane.road_id,
                        lane_type=lane.type,
                    ),
                    static=True,  # Map data is static across all time
                )

            # ---------- 2. Lane boundaries (LineStrips3D) ----------
            # Log the left and right boundary lines as distinct entities
            for boundary, name in [
                (lane.left_boundary, "left_boundary"),
                (lane.right_boundary, "right_boundary"),
            ]:
                pts = [[p.x, p.y, p.z] for p in boundary.points]
                if pts:
                    rr.log(
                        f"{entity_path}/{name}",
                        rr.LineStrips3D([pts], colors=[200, 200, 200]),
                        static=True,
                    )

            # ---------- 3. Center line + direction arrows ----------
            # Log the centerline and add directional arrows for orientation
            cps = lane.center_line
            if len(cps) >= 2:
                center_pts = [[p.x, p.y, p.z] for p in cps]
                rr.log(
                    f"{entity_path}/center",
                    rr.LineStrips3D([center_pts], colors=[255, 255, 0]),
                    static=True,
                )

                # Left lanes travel opposite to the reference line direction
                # (OpenDRIVE spec). Flip the computed tangent for those lanes.
                dir_sign = -1.0 if lane.is_left else 1.0

                # Place direction arrows at start, midpoint, and end of the center line
                arrow_indices = [0, len(cps) // 2, len(cps) - 2]
                origins = []
                vectors = []
                arrow_scale = 2.0  # length in world units

                for idx in arrow_indices:
                    p0 = cps[idx]
                    p1 = cps[idx + 1]
                    dx = p1.x - p0.x
                    dy = p1.y - p0.y
                    norm = math.sqrt(dx * dx + dy * dy) or 1.0
                    origins.append([p0.x, p0.y, p0.z + 0.3])
                    vectors.append(
                        [
                            dx / norm * arrow_scale * dir_sign,
                            dy / norm * arrow_scale * dir_sign,
                            0.0,
                        ]
                    )

                rr.log(
                    f"{entity_path}/direction",
                    rr.Arrows3D(
                        origins=origins,
                        vectors=vectors,
                        colors=[255, 200, 0],  # Amber
                        radii=0.2,
                    ),
                    static=True,
                )

    def render_state(self, frame: SceneFrame):
        """Render a single frame of simulation state.

        Logs:
          - 3D bounding boxes for all objects.
          - Future trajectory line per object.
          - Per-object scalar timeseries (speed, acceleration, position, velocity).

        Args:
            frame: The scene frame containing all object states at a given timestamp.
        """
        # Set the current simulation time in Rerun
        rr.set_time_seconds("sim_time", frame.timestamp)

        for obj_id, obj in frame.objects.items():
            obj_path = f"objects/{obj_id}"

            # ---- Derived values for telemetry ----
            speed = math.sqrt(obj.velocity.x**2 + obj.velocity.y**2 + obj.velocity.z**2)
            accel = math.sqrt(
                obj.acceleration.x**2 + obj.acceleration.y**2 + obj.acceleration.z**2
            )

            # ---- Per-object 3D bounding box + Selection metadata ----
            # Bounding boxes represent the physical extent of the object
            color = _OBJECT_COLORS.get(obj.type, [200, 200, 200])
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
                # Visible in Selection -> Data panel when box is clicked
                rr.AnyValues(
                    object_id=obj.id,
                    object_type=obj.type,
                    position_x=obj.position.x,
                    position_y=obj.position.y,
                    position_z=obj.position.z,
                    speed_mps=round(speed, 3),
                    acceleration_mps2=round(accel, 3),
                    velocity_x=obj.velocity.x,
                    velocity_y=obj.velocity.y,
                ),
            )

            # ---- Scalar timeseries per object ----
            # These are plotted in the TimeSeriesViews defined in the blueprint
            rr.log(f"{obj_path}/scalars/speed", rr.Scalar(speed))
            rr.log(f"{obj_path}/scalars/acceleration", rr.Scalar(accel))
            rr.log(f"{obj_path}/scalars/position_x", rr.Scalar(obj.position.x))
            rr.log(f"{obj_path}/scalars/position_y", rr.Scalar(obj.position.y))
            rr.log(f"{obj_path}/scalars/position_z", rr.Scalar(obj.position.z))
            rr.log(f"{obj_path}/scalars/velocity_x", rr.Scalar(obj.velocity.x))
            rr.log(f"{obj_path}/scalars/velocity_y", rr.Scalar(obj.velocity.y))

            # ---- Future trajectory ----
            # Draws a line representing the predicted future path of the object
            if obj.future_trajectory:
                traj_pts = [
                    [p.position.x, p.position.y, p.position.z]
                    for p in obj.future_trajectory
                ]
                if traj_pts:
                    # Prepend current position to connect the line to the object
                    traj_pts.insert(0, [obj.position.x, obj.position.y, obj.position.z])
                    rr.log(
                        f"{obj_path}/trajectory",
                        rr.LineStrips3D([traj_pts], colors=[255, 255, 0]),
                    )

    def load_log(self, file_path: str):
        """Parse a JSON/CSV and replay it (Placeholder)."""
        print(f"Loading log from: {file_path}")
        # Parse log, build SceneFrames and call render_state in a loop
