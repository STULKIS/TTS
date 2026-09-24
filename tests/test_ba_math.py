"""Pure-Python unit tests for Better Anotate geometry helpers (no bpy needed)."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATH_PATH = ROOT / "better_anotate" / "math_utils.py"


def load_math_utils():
    spec = importlib.util.spec_from_file_location("ba_math_utils", MATH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mu = load_math_utils()


class ExpandStrokeTests(unittest.TestCase):
    def test_freehand_returns_input(self):
        pts = [(0, 0, 0), (1, 0, 0), (2, 1, 0)]
        self.assertEqual(mu.expand_stroke("FREEHAND", pts), [pts])

    def test_freehand_single_point_makes_a_dot(self):
        self.assertEqual(mu.expand_stroke("FREEHAND", [(1, 2, 3)]), [[(1, 2, 3)]])

    def test_empty_input_returns_nothing(self):
        for tool in ("FREEHAND", "LINE", "ARROW", "RECT", "CIRCLE", "HIGHLIGHT"):
            self.assertEqual(mu.expand_stroke(tool, []), [])

    def test_line_uses_endpoints(self):
        result = mu.expand_stroke("LINE", [(0, 0, 0), (5, 5, 1)])
        self.assertEqual(result, [[(0, 0, 0), (5, 5, 1)]])

    def test_line_keeps_only_endpoints_of_dense_input(self):
        pts = [(i, 0, 0) for i in range(10)]
        result = mu.expand_stroke("LINE", pts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [pts[0], pts[-1]])

    def test_arrow_has_shaft_and_two_heads(self):
        polys = mu.expand_stroke("ARROW", [(0, 0, 0), (10, 0, 0)], (0, 0, 1))
        self.assertEqual(len(polys), 3)
        shaft, left, right = polys
        self.assertEqual(shaft[0], (0, 0, 0))
        self.assertEqual(shaft[-1], (10, 0, 0))
        # head lines both start at the tip
        self.assertEqual(left[0], (10, 0, 0))
        self.assertEqual(right[0], (10, 0, 0))
        # head corners are symmetric about the shaft axis (y = 0)
        self.assertAlmostEqual(left[1][1], -right[1][1])
        self.assertAlmostEqual(abs(left[1][1]), 1.5, places=6)
        # head sits behind the tip
        self.assertLess(left[1][0], 10)

    def test_arrow_degenerate_is_dropped(self):
        self.assertEqual(mu.expand_stroke("ARROW", [(1, 1, 1), (1, 1, 1)]), [])

    def test_rect_is_closed_and_axis_aligned(self):
        polys = mu.expand_stroke("RECT", [(0, 0, 0), (4, 2, 0)], (0, 0, 1))
        self.assertEqual(len(polys), 1)
        poly = polys[0]
        self.assertEqual(poly[0], poly[-1])
        self.assertEqual(len(poly), 5)
        xs = {round(p[0], 6) for p in poly}
        ys = {round(p[1], 6) for p in poly}
        self.assertEqual(xs, {0.0, 4.0})
        self.assertEqual(ys, {0.0, 2.0})

    def test_rect_with_zero_width_is_dropped(self):
        self.assertEqual(mu.expand_stroke("RECT", [(0, 0, 0), (0, 5, 0)]), [])

    def test_circle_is_closed_with_enough_segments(self):
        polys = mu.expand_stroke("CIRCLE", [(0, 0, 0), (2, 0, 0)], (0, 0, 1))
        self.assertEqual(len(polys), 1)
        poly = polys[0]
        self.assertEqual(poly[0], poly[-1])
        self.assertGreaterEqual(len(poly), 33)
        # every point sits on the circle of radius 1 centered at (1, 0, 0)
        for x, y, z in poly:
            self.assertAlmostEqual(math.hypot(x - 1.0, y - 0.0), 1.0, places=6)
            self.assertAlmostEqual(z, 0.0, places=6)

    def test_circle_respects_plane_normal(self):
        # circle in the XZ plane (normal along Y)
        polys = mu.expand_stroke("CIRCLE", [(0, 0, 0), (2, 0, 0)], (0, 1, 0))
        poly = polys[0]
        for x, y, z in poly:
            self.assertAlmostEqual(math.hypot(x - 1.0, z), 1.0, places=6)
            self.assertAlmostEqual(y, 0.0, places=6)

    def test_unknown_tool_falls_back_to_line(self):
        result = mu.expand_stroke("MYSTERY", [(0, 0, 0), (1, 1, 1)])
        self.assertEqual(result, [[(0, 0, 0), (1, 1, 1)]])


class ProjectClipTests(unittest.TestCase):
    # simple orthographic-ish mapping: ndc = (2x/w - 1, 2y/h - 1) with w=1
    ROWS = (
        (2.0 / 100.0, 0.0, 0.0, -1.0),
        (0.0, 2.0 / 80.0, 0.0, -1.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    def test_center_maps_to_center(self):
        # x=50 -> ndc 0 ; y=40 -> ndc 0
        xy = mu.project_clip(self.ROWS, (50.0, 40.0, 0.0), 100, 80)
        self.assertIsNotNone(xy)
        self.assertAlmostEqual(xy[0], 50.0)
        self.assertAlmostEqual(xy[1], 40.0)

    def test_corner_maps_to_origin(self):
        xy = mu.project_clip(self.ROWS, (0.0, 0.0, 0.0), 100, 80)
        self.assertIsNotNone(xy)
        self.assertAlmostEqual(xy[0], 0.0)
        self.assertAlmostEqual(xy[1], 0.0)

    def test_behind_camera_is_rejected(self):
        rows = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),  # w = -z
        )
        self.assertIsNone(mu.project_clip(rows, (0.0, 0.0, 5.0, ), 10, 10))


class HitTestTests(unittest.TestCase):
    def test_distance_midpoint(self):
        self.assertAlmostEqual(
            mu.point_segment_distance(0, 1, -1, 0, 1, 0), 1.0
        )

    def test_distance_clamps_to_segment(self):
        # beyond endpoint B
        d = mu.point_segment_distance(10, 0, 0, 0, 1, 0)
        self.assertAlmostEqual(d, 9.0)

    def test_degenerate_segment(self):
        d = mu.point_segment_distance(3, 4, 0, 0, 0, 0)
        self.assertAlmostEqual(d, 5.0)

    def test_polyline_hit_within_radius(self):
        poly = [(0, 0), (10, 0)]
        self.assertTrue(mu.polyline_hit(poly, 5.0, 1.0, 2.0))
        self.assertFalse(mu.polyline_hit(poly, 5.0, 5.0, 2.0))

    def test_polyline_hit_endpoint(self):
        poly = [(0, 0), (10, 0)]
        self.assertTrue(mu.polyline_hit(poly, -1.0, 0.0, 1.5))

    def test_single_point_hit(self):
        self.assertTrue(mu.polyline_hit([(5, 5)], 6.0, 5.0, 1.5))
        self.assertFalse(mu.polyline_hit([(5, 5)], 9.0, 5.0, 1.5))

    def test_circle_polyline_closed(self):
        poly = mu.circle_polyline_2d(10, 10, 5, segments=16)
        self.assertEqual(poly[0], poly[-1])
        self.assertEqual(len(poly), 17)


if __name__ == "__main__":
    unittest.main()
