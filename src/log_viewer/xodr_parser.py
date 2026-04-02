"""Parses OpenDRIVE (.xodr) files into a 3D road network model.

This module implements a subset of the OpenDRIVE specification, focusing on 
extracting road geometries, lane structures, and connectivity for 3D visualization.

Key OpenDRIVE concepts handled:
- Reference Line (planView): The "spine" of the road, defined by geometric primitives 
  (lines, arcs, spirals, polynomials).
- s-coordinate: Longitudinal distance along the reference line.
- t-coordinate: Lateral offset from the reference line.
- Lane Sections: Segments of the road where the lane configuration (number of lanes, 
  widths) is constant.
- Profiles: Elevation (z), Superelevation (banking/roll), and Lane Offset (lateral shift 
  of the entire lane group).
"""

import xml.etree.ElementTree as ET
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.special import fresnel as _scipy_fresnel

from .geometry import Point3D
from .map_model import Lane, LaneBoundary, RoadLink, RoadMark, XodrMapData

# Linearization tolerance (metres) used for adaptive sampling — mirrors the eps
# value passed to libOpenDRIVE's approximate_linear() calls.
_SAMPLING_EPS = 0.1


def _odr_spiral(
    s: float, c_dot: float
) -> Tuple[float, float, float]:
    """Compute clothoid (Euler spiral) point at arc length s from the spiral origin.

    Mirrors odrSpiral() in third_party/libOpenDRIVE/src/Geometries/Spiral/odrSpiral.cpp.
    Uses scipy Fresnel integrals instead of the CEPHES polynomial approximation,
    which gives the same result without accumulated numerical integration error.

    Args:
        s: Arc length along the spiral from the standard spiral origin (curv=0).
        c_dot: Curvature rate [1/m²] = (curv_end - curv_start) / length.

    Returns:
        (x, y, t): Position in the spiral's local frame and tangent angle [rad].
    """
    a = math.sqrt(math.pi / abs(c_dot))
    # scipy.special.fresnel returns (S, C) for input s/a
    S, C = _scipy_fresnel(s / a)
    x = C * a
    y = S * a
    if c_dot < 0.0:
        y = -y
    t = s * s * c_dot * 0.5
    return x, y, t


class _ParamPoly3:
    """Evaluates a ParamPoly3 geometry at any arc-length s.

    Mirrors ParamPoly3 in libOpenDRIVE/src/Geometries/ParamPoly3.cpp.

    When pRange="arcLength" the XML coefficients are defined with p=s (metres),
    so they must be rescaled to the p∈[0,1] domain before Bézier conversion.
    After conversion, an arc-length LUT maps each s-value to the correct Bézier
    parameter t, matching libOpenDRIVE's CubicBezier2D::get_t() behaviour.
    """

    # Number of LUT samples used to build the arc-length table.
    _LUT_STEPS = 200

    def __init__(
        self,
        aU: float, bU: float, cU: float, dU: float,
        aV: float, bV: float, cV: float, dV: float,
        length: float,
        p_range: str,
    ) -> None:
        """Build the arc-length LUT for this geometry segment.

        Args:
            aU..dV: Polynomial coefficients from the XML.
            length: Geometry segment length [m].
            p_range: "normalized" (p∈[0,1]) or "arcLength" (p=s in metres).
        """
        # When pRange="arcLength" the coefficients encode p in metres.
        # Rescale to p∈[0,1] so the polynomial and Bézier logic is uniform.
        # Mirrors the constructor rescaling in libOpenDRIVE ParamPoly3.cpp.
        if p_range == "arcLength" and length > 0:
            bU *= length;  cU *= length ** 2;  dU *= length ** 3
            bV *= length;  cV *= length ** 2;  dV *= length ** 3

        self._aU = aU; self._bU = bU; self._cU = cU; self._dU = dU
        self._aV = aV; self._bV = bV; self._cV = cV; self._dV = dV
        self._length = length

        # Build arc-length → t LUT by sampling the curve at uniform t steps.
        # Mirrors CubicBezier2D constructor in libOpenDRIVE/include/CubicBezier.hpp.
        steps = self._LUT_STEPS
        self._arclen_t: List[Tuple[float, float]] = [(0.0, 0.0)]
        arclen = 0.0
        prev_u, prev_v = self._eval_uv(0.0)
        for i in range(1, steps + 1):
            t = i / steps
            u, v = self._eval_uv(t)
            arclen += math.hypot(u - prev_u, v - prev_v)
            self._arclen_t.append((arclen, t))
            prev_u, prev_v = u, v

        self._total_arclen = arclen
        # Mirror libOpenDRIVE ParamPoly3.cpp:41: cubic_bezier.arclen_t[length]=1.0.
        # This sentinel pins ds=length → t=1 so that get_t(ds) accepts the full
        # OpenDRIVE s-coordinate range [0, length] directly.
        if length > 0:
            self._arclen_t.append((length, 1.0))

    def _eval_uv(self, p: float) -> Tuple[float, float]:
        """Evaluate the polynomial at parameter p ∈ [0, 1]."""
        u = self._aU + self._bU * p + self._cU * p ** 2 + self._dU * p ** 3
        v = self._aV + self._bV * p + self._cV * p ** 2 + self._dV * p ** 3
        return u, v

    def _arclen_to_t(self, s: float) -> float:
        """Map arc length s (from segment start) to Bézier parameter t.

        Mirrors CubicBezier2D::get_t() — linear interpolation within the LUT.
        """
        if not self._arclen_t:
            return 0.0
        max_arclen = self._arclen_t[-1][0]
        if max_arclen <= 0.0:
            return 0.0
        # Clamp to valid range (mirrors: arclen_adj = min(arclen, valid_length)).
        s = max(0.0, min(s, max_arclen))
        # Binary search in the LUT.
        lo, hi = 0, len(self._arclen_t) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._arclen_t[mid][0] <= s:
                lo = mid
            else:
                hi = mid
        al0, t0 = self._arclen_t[lo]
        al1, t1 = self._arclen_t[hi]
        if al1 == al0:
            return t0
        return t0 + (s - al0) / (al1 - al0) * (t1 - t0)

    def _arclen_s_to_t(self, ds: float) -> float:
        """Map arc-length offset from segment start to Bézier parameter t.

        Mirrors ParamPoly3::get_xy(s): cubic_bezier.get_t(s - s0).
        The LUT is keyed by OpenDRIVE arc-length (with arclen_t[length]=1.0),
        so ds can be passed directly without rescaling.
        """
        return self._arclen_to_t(ds)

    def get_xy(self, ds: float, x0: float, y0: float, hdg: float) -> Tuple[float, float]:
        """Compute world (x, y) for arc-length offset ds from segment start.

        Mirrors ParamPoly3::get_xy() in libOpenDRIVE.
        """
        t = self._arclen_s_to_t(ds)
        u, v = self._eval_uv(t)
        cos_h = math.cos(hdg)
        sin_h = math.sin(hdg)
        x = x0 + u * cos_h - v * sin_h
        y = y0 + u * sin_h + v * cos_h
        return x, y

    def get_tangent(self, ds: float, hdg: float) -> Tuple[float, float]:
        """Compute the analytic unit tangent (dx, dy) at arc-length offset ds.

        Mirrors ParamPoly3::derivative() in libOpenDRIVE, which evaluates the
        Bézier derivative and rotates it by hdg.
        """
        t = self._arclen_s_to_t(ds)
        # Derivative of the cubic polynomial w.r.t. p
        du = self._bU + 2.0 * self._cU * t + 3.0 * self._dU * t ** 2
        dv = self._bV + 2.0 * self._cV * t + 3.0 * self._dV * t ** 2
        cos_h = math.cos(hdg)
        sin_h = math.sin(hdg)
        dx = cos_h * du - sin_h * dv
        dy = sin_h * du + cos_h * dv
        n = math.hypot(dx, dy)
        if n > 0:
            return dx / n, dy / n
        return math.cos(hdg), math.sin(hdg)


def _normalize_3d(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalizes a 3D vector to unit length."""
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n == 0:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)



def _cubic_bezier_1d_get_control_points(
    a: float, b: float, c: float, d: float
) -> Tuple[float, float, float, float]:
    """Return Bézier control points for a cubic polynomial a + b*p + c*p^2 + d*p^3."""
    p0 = a
    p1 = b / 3.0 + a
    p2 = c / 3.0 + 2.0 * p1 - p0
    p3 = d + 3.0 * p2 - 3.0 * p1 + p0
    return p0, p1, p2, p3


def _cubic_bezier_1d_get_coefficients(
    ctrl_pts: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Return polynomial coefficients for a cubic Bézier curve."""
    p_a, p_b, p_c, p_d = ctrl_pts
    return (
        p_a,
        3.0 * p_b - 3.0 * p_a,
        3.0 * p_c - 6.0 * p_b + 3.0 * p_a,
        p_d - 3.0 * p_c + 3.0 * p_b - p_a,
    )


def _approximate_linear_quad_bezier_1d(
    ctrl_pts: Tuple[float, float, float], eps: float
) -> List[float]:
    """Approximate a quadratic Bézier curve by linear segments."""
    param_c = ctrl_pts[0] - 2.0 * ctrl_pts[1] + ctrl_pts[2]
    if abs(param_c) <= 1e-12:
        step_size = 1.0
    else:
        step_size = min(math.sqrt((4.0 * eps) / abs(param_c)), 1.0)

    p_vals = []
    p = 0.0
    while p < 1.0:
        p_vals.append(p)
        p += step_size
    if not p_vals or p_vals[-1] != 1.0:
        p_vals.append(1.0)
    return p_vals


def _cubic_bezier_1d_subcurve(
    ctrl_pts: Tuple[float, float, float, float], t_start: float, t_end: float
) -> Tuple[float, float, float, float]:
    """Extract the control points of a cubic Bézier sub-curve."""

    def _f_cubic_t123(t1: float, t2: float, t3: float) -> float:
        return (
            (1.0 - t3)
            * (
                (1.0 - t2)
                * ((1.0 - t1) * ctrl_pts[0] + t1 * ctrl_pts[1])
                + t2 * ((1.0 - t1) * ctrl_pts[1] + t1 * ctrl_pts[2])
            )
            + t3
            * (
                (1.0 - t2)
                * ((1.0 - t1) * ctrl_pts[1] + t1 * ctrl_pts[2])
                + t2 * ((1.0 - t1) * ctrl_pts[2] + t1 * ctrl_pts[3])
            )
        )

    return (
        _f_cubic_t123(t_start, t_start, t_start),
        _f_cubic_t123(t_start, t_start, t_end),
        _f_cubic_t123(t_start, t_end, t_end),
        _f_cubic_t123(t_end, t_end, t_end),
    )


def _cubic_bezier_1d_approximate_linear(
    ctrl_pts: Tuple[float, float, float, float], eps: float
) -> List[float]:
    """Approximate a cubic Bézier curve by linear segments.

    Mirrors CubicBezier<double, 1>::approximate_linear() in libOpenDRIVE.
    """
    _, _, _, coeff_3 = _cubic_bezier_1d_get_coefficients(ctrl_pts)
    if abs(coeff_3) <= 1e-12:
        seg_size = 1.0
    else:
        seg_size = pow((0.5 * eps) / ((1.0 / 54.0) * abs(coeff_3)), 1.0 / 3.0)

    seg_intervals = []
    t = 0.0
    while t < 1.0:
        seg_intervals.append((t, min(t + seg_size, 1.0)))
        t += seg_size
    if not seg_intervals:
        seg_intervals.append((0.0, 1.0))
    elif (1.0 - seg_intervals[-1][1]) < 1e-6:
        seg_intervals[-1] = (seg_intervals[-1][0], 1.0)
    else:
        seg_intervals.append((seg_intervals[-1][1], 1.0))

    t_vals = [0.0]
    for t0, t1 in seg_intervals:
        sub_curve = _cubic_bezier_1d_subcurve(ctrl_pts, t0, t1)
        p_b_quad_0 = 0.25 * sub_curve[0] + 0.75 * sub_curve[1]
        p_b_quad_1 = 0.25 * sub_curve[3] + 0.75 * sub_curve[2]
        p_m_quad = 0.5 * (p_b_quad_0 + p_b_quad_1)

        for p_sub in _approximate_linear_quad_bezier_1d(
            (sub_curve[0], p_b_quad_0, p_m_quad), 0.5 * eps
        ):
            t_vals.append(t0 + p_sub * (t1 - t0) * 0.5)
        t_vals.pop()

        for p_sub in _approximate_linear_quad_bezier_1d(
            (p_m_quad, p_b_quad_1, sub_curve[3]), 0.5 * eps
        ):
            t_vals.append(t0 + (t1 - t0) * 0.5 + p_sub * (t1 - t0) * 0.5)
        t_vals.pop()

    t_vals.append(1.0)
    return sorted(set(t_vals))


@dataclass
class _CubicPoly:
    """Absolute-s cubic polynomial representation used by libOpenDRIVE."""

    a: float
    b: float
    c: float
    d: float

    @classmethod
    def from_relative(
        cls, s_origin: float, a: float, b: float, c: float, d: float
    ) -> "_CubicPoly":
        """Convert a ds-based polynomial into absolute-s form."""
        return cls(
            a=a - b * s_origin + c * s_origin * s_origin - d * s_origin**3,
            b=b - 2.0 * c * s_origin + 3.0 * d * s_origin * s_origin,
            c=c - 3.0 * d * s_origin,
            d=d,
        )

    def evaluate(self, s: float) -> float:
        return self.a + self.b * s + self.c * s * s + self.d * s * s * s

    def derivative(self, s: float) -> float:
        return self.b + 2.0 * self.c * s + 3.0 * self.d * s * s

    def negate(self) -> "_CubicPoly":
        return _CubicPoly(-self.a, -self.b, -self.c, -self.d)

    def max_value(self, s_start: float, s_end: float) -> float:
        if self.d != 0.0:
            disc = self.c * self.c - 3.0 * self.b * self.d
            if disc < 0.0:
                return max(self.evaluate(s_start), self.evaluate(s_end))
            s_extr = (math.sqrt(disc) - self.c) / (3.0 * self.d)
            max_val_1 = self.evaluate(min(max(s_extr, s_start), s_end))
            max_val_2 = self.evaluate(min(max(-s_extr, s_start), s_end))
            return max(max_val_1, max_val_2)
        if self.c != 0.0:
            s_extr = (-self.b) / (2.0 * self.c)
            return self.evaluate(min(max(s_extr, s_start), s_end))
        return max(self.evaluate(s_start), self.evaluate(s_end))

    def get_sample_s_values(
        self, eps: float, s_start: float, s_end: float
    ) -> List[float]:
        """Return the s-values needed to linearly approximate this polynomial."""
        if s_start == s_end:
            return []
        if self.d == 0.0 and self.c == 0.0:
            return [s_start, s_end]
        if self.d == 0.0 and self.c != 0.0:
            step = 2.0 * math.sqrt(abs(eps / self.c))
            s_vals = []
            s = s_start
            while s < s_end:
                s_vals.append(s)
                s += step
        else:
            s_0 = s_start
            s_1 = s_end
            d_p = (
                -self.d * s_0**3
                + self.d * s_1**3
                - 3.0 * self.d * s_0 * s_1 * s_1
                + 3.0 * self.d * s_0 * s_0 * s_1
            )
            c_p = (
                3.0 * self.d * s_0**3
                + 3.0 * self.d * s_0 * s_1 * s_1
                - 6.0 * self.d * s_0 * s_0 * s_1
                + self.c * s_0 * s_0
                + self.c * s_1 * s_1
                - 2.0 * self.c * s_0 * s_1
            )
            b_p = (
                -3.0 * self.d * s_0**3
                + 3.0 * self.d * s_0 * s_0 * s_1
                - 2.0 * self.c * s_0 * s_0
                + 2.0 * self.c * s_0 * s_1
                - self.b * s_0
                + self.b * s_1
            )
            a_p = self.d * s_0**3 + self.c * s_0 * s_0 + self.b * s_0 + self.a
            ctrl_pts = _cubic_bezier_1d_get_control_points(a_p, b_p, c_p, d_p)
            p_vals = _cubic_bezier_1d_approximate_linear(ctrl_pts, eps)
            s_vals = [s_start]
            for p in p_vals:
                s_vals.append(p * (s_end - s_start) + s_start)

        if (s_end - s_vals[-1]) < 1e-9 and len(s_vals) != 1:
            s_vals[-1] = s_end
        else:
            s_vals.append(s_end)

        return sorted(set(s_vals))


class _CubicProfile:
    """Piecewise absolute-s cubic profile mirroring libOpenDRIVE CubicProfile."""

    def __init__(self, segments: Optional[Dict[float, _CubicPoly]] = None):
        self.segments = dict(sorted((segments or {}).items()))
        self._cached_keys: tuple = tuple(self.segments)
        # Numpy arrays for vectorised evaluate_array() — built once at construction.
        if self.segments:
            self._keys_np = np.array(self._cached_keys, dtype=np.float64)
            self._coeffs_np = np.array(
                [[p.a, p.b, p.c, p.d] for p in self.segments.values()],
                dtype=np.float64,
            )
        else:
            self._keys_np = np.empty(0, dtype=np.float64)
            self._coeffs_np = np.empty((0, 4), dtype=np.float64)

    def _keys(self) -> tuple:
        return self._cached_keys

    def get_poly(self, s: float, extend_start: bool = True) -> Optional[_CubicPoly]:
        if not self.segments:
            return None

        keys = self._cached_keys
        if not extend_start and s < keys[0]:
            return None

        idx = bisect_right(keys, s) - 1
        if idx < 0:
            idx = 0
        return self.segments[keys[idx]]

    def evaluate(
        self, s: float, default_val: float = 0.0, extend_start: bool = True
    ) -> float:
        poly = self.get_poly(s, extend_start=extend_start)
        if poly is None:
            return default_val
        return poly.evaluate(s)

    def evaluate_array(self, s_arr: np.ndarray) -> np.ndarray:
        """Evaluate the profile at an array of s-values using vectorised numpy ops.

        Significantly faster than calling evaluate() in a Python loop when
        len(s_arr) is large (typical lane has 50–300 sample points).
        """
        if not self.segments:
            return np.zeros(len(s_arr), dtype=np.float64)
        idx = np.searchsorted(self._keys_np, s_arr, side="right") - 1
        idx = np.clip(idx, 0, len(self._cached_keys) - 1)
        a = self._coeffs_np[idx, 0]
        b = self._coeffs_np[idx, 1]
        c = self._coeffs_np[idx, 2]
        d = self._coeffs_np[idx, 3]
        return a + b * s_arr + c * s_arr ** 2 + d * s_arr ** 3

    def derivative(
        self, s: float, default_val: float = 0.0, extend_start: bool = True
    ) -> float:
        poly = self.get_poly(s, extend_start=extend_start)
        if poly is None:
            return default_val
        return poly.derivative(s)

    def negate(self) -> "_CubicProfile":
        return _CubicProfile(
            {s0: poly.negate() for s0, poly in self.segments.items()}
        )

    def add(self, other: "_CubicProfile") -> "_CubicProfile":
        if not other.segments:
            return _CubicProfile(dict(self.segments))
        if not self.segments:
            return _CubicProfile(dict(other.segments))

        s0_vals = sorted(set(self.segments).union(other.segments))
        segments: Dict[float, _CubicPoly] = {}
        for s0 in s0_vals:
            this_poly = self.get_poly(s0, extend_start=False)
            other_poly = other.get_poly(s0, extend_start=False)
            if this_poly is None:
                segments[s0] = other_poly
            elif other_poly is None:
                segments[s0] = this_poly
            else:
                segments[s0] = _CubicPoly(
                    this_poly.a + other_poly.a,
                    this_poly.b + other_poly.b,
                    this_poly.c + other_poly.c,
                    this_poly.d + other_poly.d,
                )
        return _CubicProfile(segments)

    def max_value(self, s_start: float, s_end: float) -> float:
        if s_start == s_end or not self.segments:
            return 0.0

        keys = self._cached_keys
        end_idx = bisect_left(keys, s_end)
        start_idx = bisect_right(keys, s_start) - 1
        if start_idx < 0:
            start_idx = 0

        max_vals = []
        for idx in range(start_idx, end_idx):
            s0 = keys[idx]
            poly = self.segments[s0]
            s_start_poly = max(s0, s_start)
            next_s0 = keys[idx + 1] if idx + 1 < end_idx else None
            s_end_poly = s_end if next_s0 is None else min(next_s0, s_end)
            max_vals.append(poly.max_value(s_start_poly, s_end_poly))
        return max(max_vals, default=0.0)

    def get_sample_s_values(
        self, eps: float, s_start: float, s_end: float
    ) -> List[float]:
        if s_start == s_end or not self.segments:
            return []

        keys = self._cached_keys
        end_idx = bisect_left(keys, s_end)
        start_idx = bisect_right(keys, s_start) - 1
        if start_idx < 0:
            start_idx = 0

        s_vals: set = set()
        for idx in range(start_idx, end_idx):
            s0 = keys[idx]
            poly = self.segments[s0]
            s_start_poly = max(s0, s_start)
            next_s0 = keys[idx + 1] if idx + 1 < end_idx else None
            s_end_poly = s_end if next_s0 is None else min(next_s0, s_end)
            s_vals.update(poly.get_sample_s_values(eps, s_start_poly, s_end_poly))
        return sorted(s_vals)


def _zero_cubic_profile() -> _CubicProfile:
    """Return a constant-zero profile defined over the whole road."""
    return _CubicProfile({0.0: _CubicPoly.from_relative(0.0, 0.0, 0.0, 0.0, 0.0)})


def _thin_s_values(s_values: List[float], eps: float) -> List[float]:
    """Remove redundant sample points closer than eps, mirroring libOpenDRIVE.

    Mirrors the thinning loop in Road::get_lane_mesh() (Road.cpp): iterates
    forward and removes s_iter+1 when it is within eps of s_iter, but never
    removes the last point (std::next(s_iter, 2) != end() guard).
    """
    result = list(s_values)
    idx = 0
    while idx + 2 < len(result):
        if (result[idx + 1] - result[idx]) <= eps:
            result.pop(idx + 1)
        else:
            idx += 1
    return result



def _compute_e_t(
    e_s: Tuple[float, float, float], theta: float
) -> Tuple[float, float, float]:
    """Computes the lateral unit vector (e_t) accounting for road superelevation.

    The superelevation angle theta represents the "roll" or "banking" of the road.
    When theta=0, e_t is simply the 2D left-pointing normal to the reference line.
    
    Calculation mirrors Road::get_xyz() in libOpenDRIVE/src/Road.cpp.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return _normalize_3d((
        cos_t * -e_s[1] + sin_t * -e_s[2] * e_s[0],
        cos_t *  e_s[0] + sin_t * -e_s[2] * e_s[1],
        sin_t * (e_s[0] ** 2 + e_s[1] ** 2),
    ))


@dataclass
class ElevationEntry:
    """Represents a single elevation record in a cubic polynomial profile."""

    s: float
    a: float
    b: float
    c: float
    d: float

class ElevationProfile:
    """Manages multiple elevation entries and evaluates Z at any s position.

    Elevation is defined as a sequence of cubic polynomials relative to the
    longitudinal position s.
    """

    def __init__(self, entries: List[ElevationEntry]):
        """Initialize with a list of elevation entries, sorted by s."""
        segments = {0.0: _CubicPoly.from_relative(0.0, 0.0, 0.0, 0.0, 0.0)}
        for entry in sorted(entries, key=lambda e: e.s):
            segments[entry.s] = _CubicPoly.from_relative(
                entry.s, entry.a, entry.b, entry.c, entry.d
            )
        self._profile = _CubicProfile(segments)

    def get_z(self, s: float) -> float:
        """Calculate the elevation Z at distance s along the road."""
        return self._profile.evaluate(s)

    def get_dz(self, s: float) -> float:
        """Return dz/ds at position s."""
        return self._profile.derivative(s)

    def get_sample_s_values(
        self, s_start: float, s_end: float, eps: float
    ) -> List[float]:
        """Return s-values where this profile needs explicit sampling.

        Mirrors libOpenDRIVE CubicProfile::approximate_linear.
        """
        return self._profile.get_sample_s_values(eps, s_start, s_end)


@dataclass
class LaneOffsetEntry:
    """Represents a single laneOffset record (cubic polynomial)."""

    s: float
    a: float
    b: float
    c: float
    d: float


class LaneOffsetProfile:
    """Manages lateral offset of the entire lane group from the reference line."""

    def __init__(self, entries: List[LaneOffsetEntry]):
        segments = {0.0: _CubicPoly.from_relative(0.0, 0.0, 0.0, 0.0, 0.0)}
        for entry in sorted(entries, key=lambda e: e.s):
            segments[entry.s] = _CubicPoly.from_relative(
                entry.s, entry.a, entry.b, entry.c, entry.d
            )
        self._profile = _CubicProfile(segments)

    def get_offset(self, s: float) -> float:
        """Evaluates the lateral offset at position s."""
        return self._profile.evaluate(s)

    def get_sample_s_values(
        self, s_start: float, s_end: float, eps: float
    ) -> List[float]:
        """Return s-values where this profile needs explicit sampling."""
        return self._profile.get_sample_s_values(eps, s_start, s_end)


@dataclass
class LaneHeightEntry:
    """Represents a lane height offset record."""

    s: float
    inner: float
    outer: float


class LaneHeightProfile:
    """Evaluates lane height offset at any s position."""

    def __init__(self, entries: List[LaneHeightEntry]):
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_height_offset(self, s: float, p_t: float) -> float:
        """Returns the interpolated height offset."""
        if not self.entries:
            return 0.0

        idx = 0
        for i, entry in enumerate(self.entries):
            if s >= entry.s:
                idx = i
            else:
                break

        active = self.entries[idx]
        inner_height = active.inner
        outer_height = active.outer

        h_t = p_t * (outer_height - inner_height) + inner_height

        if idx + 1 < len(self.entries):
            next_entry = self.entries[idx + 1]
            ds = next_entry.s - active.s
            if ds > 1e-9:
                dh_inner = (next_entry.inner - inner_height) / ds * (s - active.s)
                dh_outer = (next_entry.outer - outer_height) / ds * (s - active.s)
                h_t += p_t * (dh_outer - dh_inner) + dh_inner

        return h_t

    def get_sample_s_values(self, s_start: float, s_end: float) -> List[float]:
        """Return explicit s-values from lane height records."""
        return [e.s for e in self.entries if s_start <= e.s <= s_end]


@dataclass
class WidthEntry:
    """Represents a single width record (cubic polynomial) within a lane section."""

    s_offset: float  # s-offset relative to the lane section start
    a: float
    b: float
    c: float
    d: float


@dataclass
class SuperelevationEntry:
    """Represents a single superelevation record (road banking angle, radians)."""

    s: float
    a: float
    b: float
    c: float
    d: float


class SuperelevationProfile:
    """Evaluates road superelevation (roll angle) at any s position."""

    def __init__(self, entries: List[SuperelevationEntry]):
        segments = {0.0: _CubicPoly.from_relative(0.0, 0.0, 0.0, 0.0, 0.0)}
        for entry in sorted(entries, key=lambda e: e.s):
            segments[entry.s] = _CubicPoly.from_relative(
                entry.s, entry.a, entry.b, entry.c, entry.d
            )
        self._profile = _CubicProfile(segments)

    def get_value(self, s: float) -> float:
        """Returns the superelevation angle in radians at position s."""
        return self._profile.evaluate(s)

    def get_sample_s_values(
        self, s_start: float, s_end: float, eps: float
    ) -> List[float]:
        """Return s-values where this profile needs explicit sampling."""
        return self._profile.get_sample_s_values(eps, s_start, s_end)


@dataclass
class CrossfallEntry:
    """Represents a lateral slope angle record."""
    s: float
    a: float
    b: float
    c: float
    d: float
    side: str  # "left", "right", or "both"


class CrossfallProfile:
    """Evaluates crossfall (lateral slope angle) at any s position.
    
    Crossfall is used to drain water from the road surface and is applied
    independently to the left or right sides.
    """

    def __init__(self, entries: List[CrossfallEntry]):
        self.entries = sorted(entries, key=lambda e: e.s)

    def get_crossfall(self, s: float, is_left_lane: bool) -> float:
        """Returns the crossfall angle in radians for a specific side."""
        if not self.entries:
            return 0.0
        active = self.entries[0]
        for entry in self.entries:
            if s >= entry.s:
                active = entry
            else:
                break
        
        # Check if this profile applies to the requested side
        if active.side == "left" and not is_left_lane:
            return 0.0
        if active.side == "right" and is_left_lane:
            return 0.0
            
        ds = s - active.s
        return active.a + active.b * ds + active.c * ds**2 + active.d * ds**3


@dataclass
class _GeomSegment:
    """Cached descriptor for a single planView geometry element."""

    kind: str  # "line", "arc", "spiral", "paramPoly3"
    s_start: float
    length: float
    hdg: float
    x0: float
    y0: float
    # Arc
    curvature: float = 0.0
    # Spiral (precomputed)
    c_dot: float = 0.0
    s0_spiral: float = 0.0
    x0_spiral: float = 0.0
    y0_spiral: float = 0.0
    a0_spiral: float = 0.0
    hdg_offset: float = 0.0
    # ParamPoly3
    pp3: Optional[_ParamPoly3] = None

    @property
    def s_end(self) -> float:
        return self.s_start + self.length


class _RoadGeometryCache:
    """Caches parsed planView geometry for a road to avoid redundant XML parsing.

    The original ``_parse_plan_view`` re-parses the XML ``<planView>`` element
    and reconstructs ``_ParamPoly3`` objects (with 200-step LUT) on every call.
    For a road with N lanes this happens N+3 times.  This class parses once and
    stores lightweight ``_GeomSegment`` descriptors that can be evaluated at
    arbitrary s-values cheaply.
    """

    def __init__(self, road: ET.Element) -> None:
        self.segments: List[_GeomSegment] = []
        plan_view = road.find("planView")
        if plan_view is None:
            return
        for geom in plan_view.findall("geometry"):
            s_start = float(geom.get("s", 0))
            x0 = float(geom.get("x", 0))
            y0 = float(geom.get("y", 0))
            hdg = float(geom.get("hdg", 0))
            length = float(geom.get("length", 0))

            if geom.find("line") is not None:
                self.segments.append(_GeomSegment(
                    kind="line", s_start=s_start, length=length,
                    hdg=hdg, x0=x0, y0=y0,
                ))
            elif geom.find("arc") is not None:
                curvature = float(geom.find("arc").get("curvature", 0))
                self.segments.append(_GeomSegment(
                    kind="arc", s_start=s_start, length=length,
                    hdg=hdg, x0=x0, y0=y0, curvature=curvature,
                ))
            elif geom.find("spiral") is not None:
                sp = geom.find("spiral")
                curv_start = float(sp.get("curvStart", 0))
                curv_end = float(sp.get("curvEnd", 0))
                c_dot = (curv_end - curv_start) / length if length > 0 else 0.0
                s0_spiral = curv_start / c_dot if abs(c_dot) > 1e-12 else 0.0
                if abs(c_dot) > 1e-12:
                    x0_sp, y0_sp, a0_sp = _odr_spiral(s0_spiral, c_dot)
                else:
                    x0_sp, y0_sp, a0_sp = 0.0, 0.0, 0.0
                self.segments.append(_GeomSegment(
                    kind="spiral", s_start=s_start, length=length,
                    hdg=hdg, x0=x0, y0=y0,
                    c_dot=c_dot, s0_spiral=s0_spiral,
                    x0_spiral=x0_sp, y0_spiral=y0_sp, a0_spiral=a0_sp,
                    hdg_offset=hdg - a0_sp,
                ))
            elif geom.find("paramPoly3") is not None:
                pp3_elem = geom.find("paramPoly3")
                pp3 = _ParamPoly3(
                    aU=float(pp3_elem.get("aU", 0)),
                    bU=float(pp3_elem.get("bU", 0)),
                    cU=float(pp3_elem.get("cU", 0)),
                    dU=float(pp3_elem.get("dU", 0)),
                    aV=float(pp3_elem.get("aV", 0)),
                    bV=float(pp3_elem.get("bV", 0)),
                    cV=float(pp3_elem.get("cV", 0)),
                    dV=float(pp3_elem.get("dV", 0)),
                    length=length,
                    p_range=pp3_elem.get("pRange", "normalized"),
                )
                self.segments.append(_GeomSegment(
                    kind="paramPoly3", s_start=s_start, length=length,
                    hdg=hdg, x0=x0, y0=y0, pp3=pp3,
                ))

    def get_default_s_set(
        self,
        eps: float,
        elevation_profile: Optional["ElevationProfile"] = None,
    ) -> List[Tuple["Point3D", float, Tuple[float, float]]]:
        """Evaluate using default adaptive sampling (no explicit s-values)."""
        return self._evaluate_impl(eps, elevation_profile, sample_s_values=None)

    def evaluate(
        self,
        sample_s_values: List[float],
        elevation_profile: Optional["ElevationProfile"] = None,
    ) -> List[Tuple["Point3D", float, Tuple[float, float]]]:
        """Evaluate at explicit s-values."""
        return self._evaluate_impl(0.1, elevation_profile, sample_s_values)

    def _evaluate_impl(
        self,
        eps: float,
        elevation_profile: Optional["ElevationProfile"],
        sample_s_values: Optional[List[float]],
    ) -> List[Tuple["Point3D", float, Tuple[float, float]]]:
        """Core evaluation logic, mirrors the original _parse_plan_view.

        Uses vectorised numpy operations for line/arc/spiral segments; falls back
        to a scalar loop only for paramPoly3 (which has a LUT-based evaluation).
        """
        pts: List[Tuple[Point3D, float, Tuple[float, float]]] = []

        for seg in self.segments:
            skip_first = len(pts) > 0

            # ---- Build sorted s-value array for this segment ----
            if sample_s_values is not None:
                s_filtered = [s for s in sample_s_values
                               if seg.s_start <= s <= seg.s_end]
                if not s_filtered:
                    continue
                s_arr = np.array(sorted(set(s_filtered)), dtype=np.float64)
            else:
                s_set: set = {seg.s_start, seg.s_end}
                if elevation_profile:
                    s_set.update(
                        elevation_profile.get_sample_s_values(seg.s_start, seg.s_end, eps)
                    )
                if seg.kind == "arc" and abs(seg.curvature) > 1e-12:
                    s_step = 0.01 / abs(seg.curvature)
                    s_set.update(np.arange(seg.s_start, seg.s_end, s_step).tolist())
                elif seg.kind == "spiral":
                    s_set.update(np.arange(seg.s_start, seg.s_end, 10.0 * eps).tolist())
                elif seg.kind == "paramPoly3":
                    s_set.update(np.arange(seg.s_start, seg.s_end, eps).tolist())
                s_arr = np.array(sorted(s_set), dtype=np.float64)

            # Drop the duplicate start point when joining segments
            if skip_first and len(s_arr) > 0 and abs(s_arr[0] - seg.s_start) < 1e-9:
                s_arr = s_arr[1:]
            if len(s_arr) == 0:
                continue

            ds_arr = s_arr - seg.s_start

            # ---- Vectorised geometry evaluation per segment kind ----
            if seg.kind == "line":
                cos_h = math.cos(seg.hdg)
                sin_h = math.sin(seg.hdg)
                x_arr = seg.x0 + ds_arr * cos_h
                y_arr = seg.y0 + ds_arr * sin_h
                dx_arr = np.full(len(s_arr), cos_h)
                dy_arr = np.full(len(s_arr), sin_h)

            elif seg.kind == "arc":
                if abs(seg.curvature) < 1e-12:
                    cos_h = math.cos(seg.hdg)
                    sin_h = math.sin(seg.hdg)
                    x_arr = seg.x0 + ds_arr * cos_h
                    y_arr = seg.y0 + ds_arr * sin_h
                    dx_arr = np.full(len(s_arr), cos_h)
                    dy_arr = np.full(len(s_arr), sin_h)
                else:
                    radius = 1.0 / seg.curvature
                    cx = seg.x0 - radius * math.sin(seg.hdg)
                    cy = seg.y0 + radius * math.cos(seg.hdg)
                    theta = seg.hdg - math.pi / 2 + ds_arr * seg.curvature
                    x_arr = cx + radius * np.cos(theta)
                    y_arr = cy + radius * np.sin(theta)
                    ang = math.pi / 2 - seg.curvature * ds_arr - seg.hdg
                    dx_arr = np.sin(ang)
                    dy_arr = np.cos(ang)

            elif seg.kind == "spiral":
                if abs(seg.c_dot) <= 1e-12:
                    cos_h = math.cos(seg.hdg)
                    sin_h = math.sin(seg.hdg)
                    x_arr = seg.x0 + ds_arr * cos_h
                    y_arr = seg.y0 + ds_arr * sin_h
                    dx_arr = np.full(len(s_arr), cos_h)
                    dy_arr = np.full(len(s_arr), sin_h)
                else:
                    a = math.sqrt(math.pi / abs(seg.c_dot))
                    spiral_s = ds_arr + seg.s0_spiral
                    S, C = _scipy_fresnel(spiral_s / a)
                    xs_sp = C * a
                    ys_sp = S * a if seg.c_dot >= 0 else -S * a
                    tang_s = spiral_s ** 2 * seg.c_dot * 0.5
                    cos_off = math.cos(seg.hdg_offset)
                    sin_off = math.sin(seg.hdg_offset)
                    x_arr = (cos_off * (xs_sp - seg.x0_spiral)
                             - sin_off * (ys_sp - seg.y0_spiral) + seg.x0)
                    y_arr = (sin_off * (xs_sp - seg.x0_spiral)
                             + cos_off * (ys_sp - seg.y0_spiral) + seg.y0)
                    tang_angle = tang_s + seg.hdg_offset
                    dx_arr = np.cos(tang_angle)
                    dy_arr = np.sin(tang_angle)

            elif seg.kind == "paramPoly3":
                # paramPoly3 uses a LUT-based arc-length inversion; keep scalar loop.
                x_list, y_list, dx_list, dy_list = [], [], [], []
                for ds in ds_arr:
                    xv, yv = seg.pp3.get_xy(float(ds), seg.x0, seg.y0, seg.hdg)
                    dxv, dyv = seg.pp3.get_tangent(float(ds), seg.hdg)
                    x_list.append(xv); y_list.append(yv)
                    dx_list.append(dxv); dy_list.append(dyv)
                x_arr = np.array(x_list)
                y_arr = np.array(y_list)
                dx_arr = np.array(dx_list)
                dy_arr = np.array(dy_list)

            else:
                continue

            # ---- Elevation z-values (batched) ----
            if elevation_profile is not None:
                z_arr = elevation_profile._profile.evaluate_array(s_arr)
            else:
                z_arr = np.zeros(len(s_arr))

            # ---- Append results ----
            for i in range(len(s_arr)):
                pts.append((
                    Point3D(float(x_arr[i]), float(y_arr[i]), float(z_arr[i])),
                    float(s_arr[i]),
                    (float(dx_arr[i]), float(dy_arr[i])),
                ))

        return pts


class XodrParser:
    """A minimal OpenDRIVE (XODR) parser for visualization purposes.

    This parser extracts road geometries and lane structures, performing the 
    necessary coordinate transformations to generate 3D meshes.
    """

    def __init__(self, file_path: str, eps: float = _SAMPLING_EPS):
        """Initialize parser from file path.

        Args:
            file_path: Path to the .xodr file.
            eps: Linearization tolerance in metres for adaptive sampling.
                 Lower values produce denser (more accurate) meshes; higher
                 values speed up parsing at the cost of visual fidelity.
        """
        self.file_path = file_path
        self._eps = eps
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()

    def parse(self) -> XodrMapData:
        """Parse the OpenDRIVE XML into XodrMapData."""
        lanes_to_render = []
        road_links = []
        road_marks_to_render = []

        # First pass: build geometry caches only (no geometry evaluation here).
        # Reference line endpoints are collected in the second pass where elevation
        # profiles are available, avoiding a redundant double evaluation per road.
        road_geom_caches: Dict[str, _RoadGeometryCache] = {}
        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")
            road_geom_caches[road_id] = _RoadGeometryCache(road)

        road_ref_endpoints: dict = {}
        for road in self.root.findall("road"):
            road_id = road.get("id", "unknown")
            geom_cache = road_geom_caches[road_id]

            # 1. Parse Profiles (Elevation, Superelevation, Crossfall)
            elevation_profile = self._parse_elevation_profile(road)
            superelevation_profile, crossfall_profile = self._parse_lateral_profile(road)

            # 2. Evaluate Reference Line from cached geometry
            ref_line_with_s = geom_cache.get_default_s_set(
                self._eps, elevation_profile=elevation_profile
            )
            if not ref_line_with_s:
                continue

            # Collect endpoints for road link resolution (previously done in pass 1)
            road_ref_endpoints[road_id] = (ref_line_with_s[0][0], ref_line_with_s[-1][0])

            # 3. Parse road connectivity (successor links)
            link_tag = road.find("link")
            if link_tag is not None:
                succ_tag = link_tag.find("successor")
                if succ_tag is not None:
                    succ_type = succ_tag.get("elementType", "road")
                    succ_id = succ_tag.get("elementId")
                    end_pt = ref_line_with_s[-1][0]
                    succ_start = None
                    if succ_type == "road" and succ_id in road_ref_endpoints:
                        succ_start = road_ref_endpoints[succ_id][0]
                    road_links.append(
                        RoadLink(
                            road_id=road_id,
                            successor_road_id=succ_id if succ_type == "road" else None,
                            successor_type=succ_type,
                            end_point=end_pt,
                            successor_start_point=succ_start,
                        )
                    )

            # 4. Extract lane information
            lanes = road.find("lanes")
            if lanes is None:
                continue

            # Parse laneOffset profile (applies lateral shift to all lanes)
            lane_offset_profile = self._parse_lane_offset_profile(lanes)
            lane_offset_cubic = lane_offset_profile._profile

            # Base reference-line s-values, shared by all lanes on this road.
            # Each lane later extends this set with inner/outer border and
            # superelevation-specific samples, mirroring Road::get_lane_mesh().
            road_s_start = ref_line_with_s[0][1]
            road_s_end = ref_line_with_s[-1][1]
            road_sample_s_values = [s for _, s, _ in ref_line_with_s]

            # Process all lane sections within the road
            lane_sections = lanes.findall("laneSection")
            if not lane_sections:
                continue

            for idx, lane_section in enumerate(lane_sections):
                s0 = float(lane_section.get("s", 0.0))
                s1 = (
                    float(lane_sections[idx + 1].get("s", road_s_end))
                    if idx + 1 < len(lane_sections)
                    else road_s_end
                )
                zero_profile = _zero_cubic_profile()

                # Process center lane (id=0) road marks — outer border is the
                # reference line itself offset by lane_offset_cubic (width=0).
                center = lane_section.find("center")
                if center is not None:
                    center_section_pts = geom_cache.evaluate(
                        [s for s in road_sample_s_values if s0 <= s <= s1],
                        elevation_profile=elevation_profile,
                    )
                    if center_section_pts:
                        center_s_vals = [s for _, s, _ in center_section_pts]
                        for lane_tag in center.findall("lane"):
                            road_marks_to_render.extend(
                                self._parse_roadmarks_for_lane(
                                    lane_tag,
                                    s0,
                                    s1,
                                    lane_offset_cubic,
                                    center_section_pts,
                                    center_s_vals,
                                    elevation_profile,
                                    superelevation_profile,
                                )
                            )

                # Process left lanes (OpenDRIVE IDs are positive, build outward from center)
                left = lane_section.find("left")
                if left is not None:
                    left_lanes = sorted(
                        left.findall("lane"),
                        key=lambda t: int(t.get("id", 0)),
                    )
                    prev_outer_no_offset: Optional[_CubicProfile] = None
                    for lane_tag in left_lanes:
                        lane_width = self._build_lane_width_profile(lane_tag, s0)
                        lane_height = self._build_lane_height_profile(lane_tag, s0)
                        inner_no_offset = (
                            zero_profile if prev_outer_no_offset is None else prev_outer_no_offset
                        )
                        outer_no_offset = (
                            lane_width
                            if prev_outer_no_offset is None
                            else prev_outer_no_offset.add(lane_width)
                        )
                        inner_border = inner_no_offset.add(lane_offset_cubic)
                        outer_border = outer_no_offset.add(lane_offset_cubic)
                        s_vals = self._collect_lane_sample_s_values(
                            road_sample_s_values,
                            s0,
                            s1,
                            inner_border,
                            outer_border,
                            superelevation_profile,
                            lane_height,
                        )
                        section_pts_with_s = geom_cache.evaluate(
                            s_vals, elevation_profile=elevation_profile,
                        )
                        if not section_pts_with_s:
                            prev_outer_no_offset = outer_no_offset
                            continue
                        ref_pts = [pt for pt, _, _ in section_pts_with_s]
                        tangents = [dxy for _, _, dxy in section_pts_with_s]
                        s_vals = [s for _, s, _ in section_pts_with_s]
                        s_arr = np.array(s_vals, dtype=np.float64)
                        inner_offsets = inner_border.evaluate_array(s_arr)
                        outer_offsets = outer_border.evaluate_array(s_arr)
                        parsed_lane = self._process_lane(
                            lane_tag,
                            road_id,
                            ref_pts,
                            inner_offsets,
                            outer_offsets,
                            s_arr,
                            tangents,
                            is_left=True,
                            elevation_profile=elevation_profile,
                            superelevation_profile=superelevation_profile,
                            crossfall_profile=crossfall_profile,
                            lane_height_profile=lane_height,
                        )
                        if parsed_lane:
                            lanes_to_render.append(parsed_lane)
                        road_marks_to_render.extend(
                            self._parse_roadmarks_for_lane(
                                lane_tag,
                                s0,
                                s1,
                                outer_border,
                                section_pts_with_s,
                                s_vals,
                                elevation_profile,
                                superelevation_profile,
                            )
                        )
                        prev_outer_no_offset = outer_no_offset

                # Process right lanes (OpenDRIVE IDs are negative, build outward)
                right = lane_section.find("right")
                if right is not None:
                    right_lanes = sorted(
                        right.findall("lane"),
                        key=lambda t: abs(int(t.get("id", 0))),
                    )
                    prev_outer_no_offset = None
                    for lane_tag in right_lanes:
                        lane_width = self._build_lane_width_profile(lane_tag, s0).negate()
                        lane_height = self._build_lane_height_profile(lane_tag, s0)
                        inner_no_offset = (
                            zero_profile if prev_outer_no_offset is None else prev_outer_no_offset
                        )
                        outer_no_offset = (
                            lane_width
                            if prev_outer_no_offset is None
                            else prev_outer_no_offset.add(lane_width)
                        )
                        inner_border = inner_no_offset.add(lane_offset_cubic)
                        outer_border = outer_no_offset.add(lane_offset_cubic)
                        s_vals = self._collect_lane_sample_s_values(
                            road_sample_s_values,
                            s0,
                            s1,
                            inner_border,
                            outer_border,
                            superelevation_profile,
                            lane_height,
                        )
                        section_pts_with_s = geom_cache.evaluate(
                            s_vals, elevation_profile=elevation_profile,
                        )
                        if not section_pts_with_s:
                            prev_outer_no_offset = outer_no_offset
                            continue
                        ref_pts = [pt for pt, _, _ in section_pts_with_s]
                        tangents = [dxy for _, _, dxy in section_pts_with_s]
                        s_vals = [s for _, s, _ in section_pts_with_s]
                        s_arr = np.array(s_vals, dtype=np.float64)
                        inner_offsets = inner_border.evaluate_array(s_arr)
                        outer_offsets = outer_border.evaluate_array(s_arr)
                        parsed_lane = self._process_lane(
                            lane_tag,
                            road_id,
                            ref_pts,
                            inner_offsets,
                            outer_offsets,
                            s_arr,
                            tangents,
                            is_left=False,
                            elevation_profile=elevation_profile,
                            superelevation_profile=superelevation_profile,
                            crossfall_profile=crossfall_profile,
                            lane_height_profile=lane_height,
                        )
                        if parsed_lane:
                            lanes_to_render.append(parsed_lane)
                        road_marks_to_render.extend(
                            self._parse_roadmarks_for_lane(
                                lane_tag,
                                s0,
                                s1,
                                outer_border,
                                section_pts_with_s,
                                s_vals,
                                elevation_profile,
                                superelevation_profile,
                            )
                        )
                        prev_outer_no_offset = outer_no_offset

        return XodrMapData(
            lanes=lanes_to_render,
            road_links=road_links,
            road_marks=road_marks_to_render,
        )

    def _parse_roadmarks_for_lane(
        self,
        lane_tag: ET.Element,
        s0: float,
        s1: float,
        outer_border: "_CubicProfile",
        section_pts_with_s: List[Tuple["Point3D", float, Tuple[float, float]]],
        s_vals: List[float],
        elevation_profile: Optional["ElevationProfile"],
        superelevation_profile: "SuperelevationProfile",
    ) -> List[RoadMark]:
        """Parse road mark geometry for a single lane.

        Mirrors Road::get_roadmark_mesh() in libOpenDRIVE/src/Road.cpp.
        For each <roadMark> element on the lane:
          - Solid/continuous marks: one segment spanning the full roadmark range.
          - Dashed marks (with <type><line> children): repeated dash segments.

        Each segment's 3D quad is computed by placing two parallel edges at:
          t_edge_a = outer_border(s) + width/2 + t_offset
          t_edge_b = t_edge_a - width

        Args:
            lane_tag: The <lane> XML element.
            s0: Lane section start s-coordinate.
            s1: Lane section end s-coordinate.
            outer_border: Lateral offset profile for the lane outer edge.
            section_pts_with_s: Reference line samples (Point3D, s, tangent).
            s_vals: Sorted s-values used for sampling this lane.
            elevation_profile: Road elevation profile.
            superelevation_profile: Road superelevation profile.

        Returns:
            List of RoadMark objects with 3D left_pts and right_pts.
        """
        _STANDARD_WIDTH = 0.12
        _BOLD_WIDTH = 0.25

        ref_pts = [pt for pt, _, _ in section_pts_with_s]
        tangents = [dxy for _, _, dxy in section_pts_with_s]

        def _surface_pt(s: float, t: float) -> Point3D:
            """Compute 3D surface point at (s, t) — mirrors Road::get_surface_pt()."""
            idx = bisect_right(s_vals, s) - 1
            idx = max(0, min(idx, len(ref_pts) - 1))
            px, py, pz = ref_pts[idx].x, ref_pts[idx].y, ref_pts[idx].z
            dx2d, dy2d = tangents[idx]
            dz = elevation_profile.get_dz(s) if elevation_profile else 0.0
            e_s = _normalize_3d((dx2d, dy2d, dz))
            theta = superelevation_profile.get_value(s)
            e_t = _compute_e_t(e_s, theta)
            return Point3D(
                px + e_t[0] * t,
                py + e_t[1] * t,
                pz + e_t[2] * t,
            )

        result: List[RoadMark] = []
        roadmark_tags = lane_tag.findall("roadMark")

        for rm_idx, rm_tag in enumerate(roadmark_tags):
            rm_s_offset = float(rm_tag.get("sOffset", 0.0))
            rm_s_offset = max(0.0, rm_s_offset)
            rm_s0 = s0 + rm_s_offset

            # End of this roadmark group = start of next, or lane section end
            if rm_idx + 1 < len(roadmark_tags):
                next_offset = float(
                    roadmark_tags[rm_idx + 1].get("sOffset", 0.0)
                )
                rm_s1 = s0 + max(0.0, next_offset)
            else:
                rm_s1 = s1

            if rm_s0 >= rm_s1:
                continue

            rm_type = rm_tag.get("type", "none")
            if rm_type == "none":
                continue

            rm_color = rm_tag.get("color", "standard")
            rm_weight = rm_tag.get("weight", "standard")
            rm_width_attr = float(rm_tag.get("width", -1))

            base_width = _BOLD_WIDTH if rm_weight == "bold" else _STANDARD_WIDTH

            type_node = rm_tag.find("type")
            if type_node is not None:
                # Dashed/patterned marks: iterate <line> children
                line_width_type = float(type_node.get("width", -1))
                for line_tag in type_node.findall("line"):
                    lw0 = float(line_tag.get("width", -1))
                    width = (
                        lw0 if lw0 > 0
                        else (line_width_type if line_width_type > 0 else base_width)
                    )
                    length = float(line_tag.get("length", 0.0))
                    space = float(line_tag.get("space", 0.0))
                    t_offset = float(line_tag.get("tOffset", 0.0))
                    s_offset_line = max(0.0, float(line_tag.get("sOffset", 0.0)))

                    if (length + space) == 0:
                        continue

                    s_start_dash = rm_s0 + s_offset_line
                    while s_start_dash < rm_s1:
                        s_end_dash = min(rm_s1, s_start_dash + length)
                        if s_end_dash > s_start_dash:
                            mark = self._compute_roadmark_geometry(
                                s_start_dash,
                                s_end_dash,
                                t_offset,
                                width,
                                outer_border,
                                s_vals,
                                _surface_pt,
                                rm_type,
                                rm_color,
                            )
                            if mark is not None:
                                result.append(mark)
                        s_start_dash += length + space
            else:
                # Solid / continuous mark
                width = (
                    rm_width_attr if rm_width_attr > 0 else base_width
                )
                mark = self._compute_roadmark_geometry(
                    rm_s0,
                    rm_s1,
                    0.0,
                    width,
                    outer_border,
                    s_vals,
                    _surface_pt,
                    rm_type,
                    rm_color,
                )
                if mark is not None:
                    result.append(mark)

        return result

    def _compute_roadmark_geometry(
        self,
        s_start: float,
        s_end: float,
        t_offset: float,
        width: float,
        outer_border: "_CubicProfile",
        s_vals: List[float],
        surface_pt_fn,
        mark_type: str,
        color: str,
    ) -> Optional[RoadMark]:
        """Compute 3D quad-strip geometry for a single road mark segment.

        Mirrors Road::get_roadmark_mesh() in libOpenDRIVE/src/Road.cpp:353-379.
        Samples at s-values within [s_start, s_end] and computes two parallel
        edges at t = outer_border(s) ± width/2 + t_offset.

        Returns None if fewer than 2 sample points exist.
        """
        seg_s_vals = [s for s in s_vals if s_start <= s <= s_end]
        if s_start not in seg_s_vals:
            seg_s_vals = [s_start] + seg_s_vals
        if s_end not in seg_s_vals:
            seg_s_vals = seg_s_vals + [s_end]
        seg_s_vals = sorted(set(seg_s_vals))

        if len(seg_s_vals) < 2:
            return None

        left_pts: List[Point3D] = []
        right_pts: List[Point3D] = []
        for s in seg_s_vals:
            outer = outer_border.evaluate(s)
            t_a = outer + width * 0.5 + t_offset
            t_b = t_a - width
            left_pts.append(surface_pt_fn(s, t_a))
            right_pts.append(surface_pt_fn(s, t_b))

        return RoadMark(
            left_pts=left_pts,
            right_pts=right_pts,
            mark_type=mark_type,
            color=color,
        )

    def _parse_lane_offset_profile(self, lanes: ET.Element) -> LaneOffsetProfile:
        """Parse all <laneOffset> entries from the <lanes> element."""
        entries = []
        for lo in lanes.findall("laneOffset"):
            entries.append(
                LaneOffsetEntry(
                    s=float(lo.get("s", 0.0)),
                    a=float(lo.get("a", 0.0)),
                    b=float(lo.get("b", 0.0)),
                    c=float(lo.get("c", 0.0)),
                    d=float(lo.get("d", 0.0)),
                )
            )
        return LaneOffsetProfile(entries)

    def _build_lane_height_profile(
        self, lane_tag: ET.Element, section_s0: float
    ) -> LaneHeightProfile:
        """Parse lane height records into a LaneHeightProfile."""
        entries = []
        for ht in lane_tag.findall("height"):
            s_offset = float(ht.get("sOffset", 0.0))
            inner = float(ht.get("inner", 0.0))
            outer = float(ht.get("outer", 0.0))
            entries.append(
                LaneHeightEntry(
                    s=section_s0 + s_offset, inner=inner, outer=outer
                )
            )
        return LaneHeightProfile(entries)

    def _build_lane_width_profile(
        self, lane_tag: ET.Element, section_s0: float
    ) -> _CubicProfile:
        """Parse lane width records into a cubic profile on absolute s."""
        width_tags = lane_tag.findall("width")
        if not width_tags:
            return _CubicProfile(
                {section_s0: _CubicPoly.from_relative(section_s0, 3.0, 0.0, 0.0, 0.0)}
            )

        entries = sorted(
            (
                WidthEntry(
                    s_offset=float(wt.get("sOffset", 0.0)),
                    a=float(wt.get("a", 3.0)),
                    b=float(wt.get("b", 0.0)),
                    c=float(wt.get("c", 0.0)),
                    d=float(wt.get("d", 0.0)),
                )
                for wt in width_tags
            ),
            key=lambda entry: entry.s_offset,
        )
        segments: Dict[float, _CubicPoly] = {}
        for entry in entries:
            s_abs = section_s0 + entry.s_offset
            segments[s_abs] = _CubicPoly.from_relative(
                s_abs, entry.a, entry.b, entry.c, entry.d
            )
        return _CubicProfile(segments)

    def _collect_lane_sample_s_values(
        self,
        road_sample_s_values: List[float],
        s_start: float,
        s_end: float,
        inner_border: _CubicProfile,
        outer_border: _CubicProfile,
        superelevation_profile: SuperelevationProfile,
        lane_height_profile: "LaneHeightProfile",
    ) -> List[float]:
        """Collect lane-specific sample s-values, mirroring Road::get_lane_mesh()."""
        s_vals = {
            s for s in road_sample_s_values if s_start <= s <= s_end
        }
        s_vals.update(inner_border.get_sample_s_values(self._eps, s_start, s_end))
        s_vals.update(outer_border.get_sample_s_values(self._eps, s_start, s_end))
        s_vals.update(lane_height_profile.get_sample_s_values(s_start, s_end))

        t_max = outer_border.max_value(s_start, s_end)
        superelev_eps = (
            math.atan(self._eps / abs(t_max))
            if abs(t_max) > 1e-12
            else self._eps
        )
        s_vals.update(
            superelevation_profile.get_sample_s_values(
                s_start, s_end, superelev_eps
            )
        )
        s_vals.add(s_start)
        s_vals.add(s_end)
        return _thin_s_values(sorted(s_vals), self._eps)

    def _parse_elevation_profile(self, road: ET.Element) -> ElevationProfile:
        """Parse the elevationProfile for a road."""
        entries = []
        profile_tag = road.find("elevationProfile")
        if profile_tag is not None:
            for elev in profile_tag.findall("elevation"):
                entries.append(
                    ElevationEntry(
                        s=float(elev.get("s", 0)),
                        a=float(elev.get("a", 0)),
                        b=float(elev.get("b", 0)),
                        c=float(elev.get("c", 0)),
                        d=float(elev.get("d", 0)),
                    )
                )
        return ElevationProfile(entries)

    def _parse_lateral_profile(
        self, road: ET.Element
    ) -> Tuple[SuperelevationProfile, CrossfallProfile]:
        """Parse <lateralProfile> for superelevation and crossfall."""
        superelev_entries = []
        crossfall_entries = []
        lateral_tag = road.find("lateralProfile")
        if lateral_tag is not None:
            for se in lateral_tag.findall("superelevation"):
                superelev_entries.append(
                    SuperelevationEntry(
                        s=float(se.get("s", 0)),
                        a=float(se.get("a", 0)),
                        b=float(se.get("b", 0)),
                        c=float(se.get("c", 0)),
                        d=float(se.get("d", 0)),
                    )
                )
            for crossfall_tag_name in ("crossfall", "crossFall"):
                for cf in lateral_tag.findall(crossfall_tag_name):
                    crossfall_entries.append(
                        CrossfallEntry(
                            s=float(cf.get("s", 0)),
                            a=float(cf.get("a", 0)),
                            b=float(cf.get("b", 0)),
                            c=float(cf.get("c", 0)),
                            d=float(cf.get("d", 0)),
                            side=cf.get("side", "both"),
                        )
                    )
        return SuperelevationProfile(superelev_entries), CrossfallProfile(crossfall_entries)

    def _process_lane(
        self,
        lane_tag: ET.Element,
        road_id: str,
        ref_line: List[Point3D],
        inner_offsets: np.ndarray,
        outer_offsets: np.ndarray,
        s_vals: np.ndarray,
        tangents: List[Tuple[float, float]],
        is_left: bool,
        elevation_profile: Optional[ElevationProfile],
        superelevation_profile: "SuperelevationProfile",
        crossfall_profile: "CrossfallProfile",
        lane_height_profile: "LaneHeightProfile",
    ) -> Optional[Lane]:
        """Constructs a 3D Lane object from road geometry and lateral profiles.

        Uses vectorised numpy operations to compute lane boundary points in bulk
        instead of a per-point Python loop.

        Coordinate System:
          - e_s: Unit tangent vector (longitudinal direction).
          - e_t: Unit lateral vector (horizontal direction, with roll).
          - e_h: Unit vertical vector (upwards normal).
        """
        lane_id = lane_tag.get("id", "0")
        lane_type = lane_tag.get("type", "driving")

        # Only render drivable and pedestrian-relevant lane types
        if lane_type not in ["driving", "sidewalk", "biking", "shoulder"]:
            return None

        # level=true means this lane should remain horizontal even when the road
        # has superelevation (banking). Mirrors the lane.level branch in
        # Road::get_surface_pt() in libOpenDRIVE/src/Road.cpp (lines 178-183).
        lane_level = lane_tag.get("level", "false").lower() == "true"

        n = len(ref_line)

        # --- Build (N, 3) reference-line position array ---
        ref_pts = np.empty((n, 3), dtype=np.float64)
        for i, pt in enumerate(ref_line):
            ref_pts[i] = (pt.x, pt.y, pt.z)

        # --- Build (N, 2) tangent array and dz array ---
        tang_arr = np.array(tangents, dtype=np.float64)  # (N, 2)
        if elevation_profile is not None:
            dz_arr = elevation_profile._profile.evaluate_array(s_vals)
        else:
            dz_arr = np.zeros(n, dtype=np.float64)

        # --- e_s: (N, 3) unit tangent ---
        e_s = np.stack([tang_arr[:, 0], tang_arr[:, 1], dz_arr], axis=1)
        norms = np.linalg.norm(e_s, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        e_s = e_s / norms

        # --- superelevation theta: (N,) ---
        theta_arr = superelevation_profile._profile.evaluate_array(s_vals)

        # --- e_t: (N, 3) vectorised _compute_e_t ---
        cos_t = np.cos(theta_arr)
        sin_t = np.sin(theta_arr)
        e_t = np.stack([
            cos_t * (-e_s[:, 1]) + sin_t * (-e_s[:, 2]) * e_s[:, 0],
            cos_t * e_s[:, 0]    + sin_t * (-e_s[:, 2]) * e_s[:, 1],
            sin_t * (e_s[:, 0] ** 2 + e_s[:, 1] ** 2),
        ], axis=1)
        norms_t = np.linalg.norm(e_t, axis=1, keepdims=True)
        norms_t = np.where(norms_t == 0, 1.0, norms_t)
        e_t = e_t / norms_t

        # --- e_h: (N, 3) cross product e_s × e_t ---
        e_h = np.cross(e_s, e_t)

        # --- crossfall h_t: (N,) per offset ---
        # Evaluate crossfall per point via scalar loop (CrossfallProfile has no
        # _CubicProfile backing, and is almost always zero in practice).
        cf_arr = np.array(
            [crossfall_profile.get_crossfall(float(s), is_left) for s in s_vals],
            dtype=np.float64,
        )
        tan_cf = np.tan(cf_arr)

        io = inner_offsets  # (N,)
        oo = outer_offsets  # (N,)
        co = (io + oo) * 0.5

        def _h_t(t_offset: np.ndarray) -> np.ndarray:
            if lane_level:
                h_inner = -tan_cf * np.abs(io)
                h = h_inner + np.tan(theta_arr) * (t_offset - io)
            else:
                h = -tan_cf * np.abs(t_offset)

            if lane_height_profile.entries:
                # lane_height is rare; fall back to scalar loop
                denom = np.where(oo != io, oo - io, 1.0)
                p_t = np.where(oo != io, (t_offset - io) / denom, 0.0)
                for k in range(n):
                    h[k] += lane_height_profile.get_height_offset(
                        float(s_vals[k]), float(p_t[k])
                    )
            return h

        def _project(t_offset: np.ndarray) -> List[Point3D]:
            h = _h_t(t_offset)
            xyz = ref_pts + e_t * t_offset[:, None] + e_h * h[:, None]
            return [Point3D(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
                    for i in range(n)]

        center_pts = _project(co)
        inner_pts = _project(io)
        outer_pts = _project(oo)

        # Determine Left/Right boundaries based on lane side (OpenDRIVE orientation)
        if is_left:
            lb_left = LaneBoundary(
                id=f"{road_id}_{lane_id}_left", style="solid", points=outer_pts
            )
            lb_right = LaneBoundary(
                id=f"{road_id}_{lane_id}_right", style="solid", points=inner_pts
            )
        else:
            lb_left = LaneBoundary(
                id=f"{road_id}_{lane_id}_left", style="solid", points=inner_pts
            )
            lb_right = LaneBoundary(
                id=f"{road_id}_{lane_id}_right", style="solid", points=outer_pts
            )

        return Lane(
            id=f"{road_id}_{lane_id}",
            road_id=road_id,
            type=lane_type,
            left_boundary=lb_left,
            right_boundary=lb_right,
            center_line=center_pts,
            is_left=is_left,
        )
def parse_xodr(file_path: str, eps: float = _SAMPLING_EPS) -> XodrMapData:
    """Convenience function to parse an OpenDRIVE file into a map model.

    Args:
        file_path: Path to the .xodr file.
        eps: Linearization tolerance in metres (default 0.1).
             Use higher values (0.5–1.0) for faster loading with lower fidelity.
    """
    parser = XodrParser(file_path, eps=eps)
    return parser.parse()
