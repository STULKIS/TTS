"""Pure geometry helpers for Better Anotate.

This module must stay importable without ``bpy``/``mathutils`` so it can be
unit-tested with plain Python.  All points are ``(x, y, z)`` tuples in either
world space (3D viewport strokes) or 2D view space (other editors, z = 0).
"""

from __future__ import annotations

import math

__all__ = (
    "expand_stroke",
    "project_clip",
    "point_segment_distance",
    "polyline_hit",
    "circle_polyline_2d",
)

# ---------------------------------------------------------------------------
# small vector helpers (tuple based, no mathutils)
# ---------------------------------------------------------------------------


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a, fallback=(1.0, 0.0, 0.0)):
    ln = _length(a)
    if ln < 1e-12:
        return fallback
    return (a[0] / ln, a[1] / ln, a[2] / ln)


# ---------------------------------------------------------------------------
# stroke expansion: anchors -> polylines
# ---------------------------------------------------------------------------

#: Number of segments used to approximate circles / arcs.
CIRCLE_SEGMENTS = 48


def expand_stroke(tool, points, normal=(0.0, 0.0, 1.0)):
    """Expand stored anchor points into one or more polylines.

    ``tool`` is one of ``FREEHAND``, ``HIGHLIGHT``, ``LINE``, ``ARROW``,
    ``RECT``, ``CIRCLE``.  The result is a list of polylines, each a list of
    3D points.  Degenerate input returns an empty list (no stroke drawn).
    """
    pts = [tuple(p) for p in points]
    if not pts:
        return []

    if tool in ("FREEHAND", "HIGHLIGHT"):
        return [pts] if len(pts) >= 1 else []

    if tool == "LINE":
        if len(pts) < 2:
            return [pts]
        return [[pts[0], pts[-1]]]

    if len(pts) < 2:
        return []
    p0, p1 = pts[0], pts[-1]
    n = _normalize(tuple(normal), fallback=(0.0, 0.0, 1.0))
    d = _sub(p1, p0)

    if tool == "ARROW":
        return _expand_arrow(p0, p1, n)
    if tool == "RECT":
        return _expand_rect(p0, p1, n)
    if tool == "CIRCLE":
        return _expand_circle(p0, p1, n)
    # Unknown tool: fall back to a straight line so data is never lost.
    return [[p0, p1]]


def _expand_arrow(p0, p1, n):
    d = _sub(p1, p0)
    ln = _length(d)
    if ln < 1e-9:
        return []
    direction = _scale(d, 1.0 / ln)
    head_len = min(ln * 0.3, ln - 1e-6)
    head_half = head_len * 0.5
    side = _normalize(_cross(n, direction), fallback=(0.0, 1.0, 0.0))
    base = _sub(p1, _scale(direction, head_len))
    left = _add(base, _scale(side, head_half))
    right = _sub(base, _scale(side, head_half))
    return [[p0, p1], [p1, left], [p1, right]]


def _expand_rect(p0, p1, n):
    """Corner-to-corner rectangle on the stroke plane.

    The side directions come from a fixed orthonormal basis of the plane
    (derived from the normal), *not* from the diagonal ``p0 -> p1`` — using
    the diagonal would make the second projection identically zero.
    """
    d = _sub(p1, p0)
    # build a stable in-plane basis
    ref = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = _normalize(_sub(ref, _scale(n, _dot(ref, n))))
    e2 = _normalize(_cross(n, e1), fallback=(0.0, 1.0, 0.0))
    a = _dot(d, e1)
    b = _dot(d, e2)
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return []
    c00 = p0
    c10 = _add(p0, _scale(e1, a))
    c11 = _add(_add(p0, _scale(e1, a)), _scale(e2, b))
    c01 = _add(p0, _scale(e2, b))
    return [[c00, c10, c11, c01, c00]]


def _expand_circle(p0, p1, n):
    d = _sub(p1, p0)
    ln = _length(d)
    if ln < 1e-9:
        return []
    center = _scale(_add(p0, p1), 0.5)
    r = ln * 0.5
    e1 = _normalize(d)
    e2 = _normalize(_cross(n, e1), fallback=(0.0, 1.0, 0.0))
    poly = []
    for i in range(CIRCLE_SEGMENTS):
        t = (2.0 * math.pi * i) / CIRCLE_SEGMENTS
        c, s = math.cos(t), math.sin(t)
        pt = _add(center, _add(_scale(e1, r * c), _scale(e2, r * s)))
        poly.append(pt)
    poly.append(poly[0])  # close
    return [poly]


def circle_polyline_2d(cx, cy, radius, segments=40):
    """2D circle polyline (used for the eraser cursor)."""
    if radius <= 0.0:
        return []
    poly = []
    for i in range(segments):
        t = (2.0 * math.pi * i) / segments
        poly.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
    poly.append(poly[0])
    return poly


# ---------------------------------------------------------------------------
# projection / hit-testing
# ---------------------------------------------------------------------------


def project_clip(rows, co, width, height):
    """Project a world point through a 4x4 clip matrix to region pixels.

    ``rows`` is an iterable of 4 rows of 4 floats (e.g. iteration over a
    ``mathutils.Matrix``).  Returns ``(x, y)`` in region coordinates
    (origin bottom-left) or ``None`` when the point is behind the camera.
    """
    x, y, z = co
    vals = (x, y, z, 1.0)
    cx = cy = cw = 0.0
    for i, row in enumerate(rows):
        acc = row[0] * vals[0] + row[1] * vals[1] + row[2] * vals[2] + row[3] * vals[3]
        if i == 0:
            cx = acc
        elif i == 1:
            cy = acc
        elif i == 3:
            cw = acc
    if cw <= 1e-6:
        return None
    ndc_x = cx / cw
    ndc_y = cy / cw
    return ((ndc_x * 0.5 + 0.5) * width, (ndc_y * 0.5 + 0.5) * height)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Shortest distance from point P to segment AB (2D)."""
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def polyline_hit(poly_2d, px, py, radius):
    """True when (px, py) lies within ``radius`` of any part of the polyline."""
    if not poly_2d:
        return False
    if len(poly_2d) == 1:
        x, y = poly_2d[0]
        return math.hypot(px - x, py - y) <= radius
    r_sq = radius * radius
    for i in range(len(poly_2d) - 1):
        ax, ay = poly_2d[i]
        bx, by = poly_2d[i + 1]
        # cheap bbox reject before full distance computation
        if (
            min(ax, bx) - radius > px
            or max(ax, bx) + radius < px
            or min(ay, by) - radius > py
            or max(ay, by) + radius < py
        ):
            continue
        if point_segment_distance(px, py, ax, ay, bx, by) <= radius:
            return True
    # single-point hit test (closed shapes still pass through edges above)
    if len(poly_2d) == 1:
        x, y = poly_2d[0]
        if (px - x) ** 2 + (py - y) ** 2 <= r_sq:
            return True
    return False
