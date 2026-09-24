"""Screen-space overlay rendering for Better Anotate.

Strokes are drawn with the ``gpu`` module through space draw-handlers:

* ``SpaceView3D``  — POST_VIEW draws world-space strokes with native 3D
  projection (depth test disabled so annotations stay readable), POST_PIXEL
  draws the eraser cursor.
* every other supported editor — POST_PIXEL draws strokes stored in view
  space, re-projected through the region's ``view2d`` on each redraw so they
  pan/zoom with the content.

All GPU calls live inside draw callbacks, so importing/registration is safe
in background mode (no GL context).
"""

from __future__ import annotations

import bpy

from . import math_utils as mu
from . import state
from .props import get_settings

# space type -> (Space class name, space type idname)
SUPPORTED_SPACES = (
    ("SpaceView3D", "VIEW_3D"),
    ("SpaceImageEditor", "IMAGE_EDITOR"),
    ("SpaceNodeEditor", "NODE_EDITOR"),
    ("SpaceSequenceEditor", "SEQUENCE_EDITOR"),
    ("SpaceClipEditor", "CLIP_EDITOR"),
)

SUPPORTED_SPACE_IDS = tuple(item[1] for item in SUPPORTED_SPACES)

_shader = None


# ---------------------------------------------------------------------------
# view2d helpers (runtime-only RNA, signatures handled defensively)
# ---------------------------------------------------------------------------


def make_view2d_fns(region):
    """Return (to_view, to_region) callables for a region with a view2d."""
    v2d = getattr(region, "view2d", None)
    if v2d is None:
        # Fallback: raw pixel coordinates (strokes stay screen-locked).
        return (lambda x, y: (x, y)), (lambda x, y: (x, y))

    def to_view(x, y):
        try:
            return tuple(v2d.region_to_view(x, y))
        except TypeError:
            try:
                return tuple(v2d.region_to_view((x, y)))
            except TypeError:
                return (x, y)
        except Exception:
            return (x, y)

    def to_region(x, y):
        for args in ((x, y, False), (x, y), ((x, y), False)):
            try:
                return tuple(v2d.view_to_region(*args))
            except TypeError:
                continue
            except Exception:
                return (x, y)
        return (x, y)

    return to_view, to_region


# ---------------------------------------------------------------------------
# shared drawing
# ---------------------------------------------------------------------------


def _get_shader():
    global _shader
    if _shader is not None:
        return _shader
    import gpu

    for name in ("UNIFORM_COLOR", "3D_UNIFORM_COLOR"):
        try:
            _shader = gpu.shader.from_builtin(name)
            return _shader
        except Exception:
            continue
    return None


def _stroke_color(settings, stroke, layer_opacity):
    col = tuple(stroke.color)
    if len(col) == 3:
        col = (col[0], col[1], col[2], 1.0)
    alpha = col[3] * layer_opacity
    if stroke.tool == "HIGHLIGHT":
        alpha *= 0.45
    return (col[0], col[1], col[2], alpha)


def _stroke_width(stroke):
    width = float(stroke.thickness)
    if stroke.tool == "HIGHLIGHT":
        width *= 3.0
    return max(width, 1.0)


def _visible_layers(settings):
    return {layer.uid: (not layer.hide) for layer in settings.layers}


def _iter_visible_strokes(settings, editor):
    visible = _visible_layers(settings)
    for stroke in settings.strokes:
        if stroke.editor != editor:
            continue
        if stroke.uid in state.pending_erase:
            continue
        if not visible.get(stroke.layer_uid, True):
            continue
        yield stroke


def _layer_opacity(settings, layer_uid):
    for layer in settings.layers:
        if layer.uid == layer_uid:
            return layer.opacity
    return 1.0


def _draw_polylines(shader, polylines, color, width, project):
    """Draw polylines via LINE_STRIP / POINTS; ``project`` maps co->(x, y)."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    rgba = color
    if rgba[3] < 1.0:
        gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(width)
    gpu.state.point_size_set(max(width, 2.0))
    try:
        for poly in polylines:
            projected = []
            ok = True
            for co in poly:
                xy = project(co)
                if xy is None:
                    ok = False
                    break
                projected.append(xy)
            if not ok or not projected:
                continue
            if len(projected) == 1:
                batch = batch_for_shader(shader, "POINTS", {"pos": projected})
            else:
                batch = batch_for_shader(shader, "LINE_STRIP", {"pos": projected})
            shader.bind()
            shader.uniform_float("color", rgba)
            batch.draw(shader)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        if rgba[3] < 1.0:
            gpu.state.blend_set("NONE")


def _stroke_polylines(stroke):
    pts = [(p.x, p.y, p.z) for p in stroke.points]
    return mu.expand_stroke(stroke.tool, pts, tuple(stroke.normal))


def _preview_polylines(preview):
    return mu.expand_stroke(
        preview["tool"], preview["points"], preview.get("normal", (0, 0, 1))
    )


def _draw_error_once(exc):
    global _draw_error_reported
    if not _draw_error_reported:
        _draw_error_reported = True
        print(f"Better Anotate overlay error: {exc!r}")


# ---------------------------------------------------------------------------
# 3D viewport
# ---------------------------------------------------------------------------


def draw_view3d(context):
    settings = get_settings(context)
    if settings is None or not settings.show_overlay:
        return
    region = context.region
    rv3d = getattr(region, "data", None)
    if rv3d is None or not hasattr(rv3d, "perspective_matrix"):
        return
    shader = _get_shader()
    if shader is None:
        return

    import gpu

    mat = rv3d.perspective_matrix
    rows = [tuple(row) for row in mat]
    width, height = region.width, region.height

    def project(co):
        return mu.project_clip(rows, co, width, height)

    prev_depth = gpu.state.depth_test_get()
    prev_depth_mask = gpu.state.depth_mask_get()
    prev_blend = gpu.state.blend_get()
    prev_line = gpu.state.line_width_get()
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    try:
        for stroke in _iter_visible_strokes(settings, "VIEW_3D"):
            color = _stroke_color(
                settings, stroke, _layer_opacity(settings, stroke.layer_uid)
            )
            _draw_polylines(
                shader, _stroke_polylines(stroke), color, _stroke_width(stroke), project
            )

        preview = state.preview
        if preview is not None and preview.get("editor") == "VIEW_3D":
            _draw_polylines(
                shader,
                _preview_polylines(preview),
                preview["color"],
                preview["thickness"],
                project,
            )
    finally:
        gpu.state.depth_test_set(prev_depth)
        gpu.state.depth_mask_set(prev_depth_mask)
        gpu.state.blend_set(prev_blend)
        gpu.state.line_width_set(prev_line)


def draw_view3d_eraser(context):
    """POST_PIXEL: eraser cursor for the 3D viewport."""
    circle = state.eraser_circle
    if circle is None:
        return
    settings = get_settings(context)
    if settings is None:
        return
    _draw_eraser_cursor(context, circle)


def _draw_eraser_cursor(context, circle):
    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = _get_shader()
    if shader is None:
        return
    cx, cy, radius = circle
    poly = mu.circle_polyline_2d(cx, cy, radius, segments=40)
    if not poly:
        return
    prev_line = gpu.state.line_width_get()
    prev_blend = gpu.state.blend_get()
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(1.5)
    try:
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": poly})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
        batch.draw(shader)
        # inner crosshair
        cross = [(cx - 4, cy), (cx + 4, cy), (cx, cy), (cx, cy - 4), (cx, cy + 4)]
        batch = batch_for_shader(shader, "LINES", {"pos": cross})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
        batch.draw(shader)
    finally:
        gpu.state.line_width_set(prev_line)
        gpu.state.blend_set(prev_blend)


# ---------------------------------------------------------------------------
# 2D editors (image, node, sequencer, clip)
# ---------------------------------------------------------------------------


def make_draw_2d(space_type_id):
    def draw_2d(context):
        settings = get_settings(context)
        if settings is None or not settings.show_overlay:
            return
        region = context.region
        if region is None or region.type != "WINDOW":
            return
        shader = _get_shader()
        if shader is None:
            return
        _to_region_view, to_region = make_view2d_fns(region)

        def project(co):
            return to_region(co[0], co[1])

        try:
            for stroke in _iter_visible_strokes(settings, space_type_id):
                color = _stroke_color(
                    settings, stroke, _layer_opacity(settings, stroke.layer_uid)
                )
                _draw_polylines(
                    shader,
                    _stroke_polylines(stroke),
                    color,
                    _stroke_width(stroke),
                    project,
                )

            preview = state.preview
            if preview is not None and preview.get("editor") == space_type_id:
                _draw_polylines(
                    shader,
                    _preview_polylines(preview),
                    preview["color"],
                    preview["thickness"],
                    project,
                )

            if state.eraser_circle is not None and space_type_id != "VIEW_3D":
                _draw_eraser_cursor(context, state.eraser_circle)
        except Exception as exc:  # never let drawing crash the editor
            _draw_error_once(exc)

    return draw_2d


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def _draw_view3d_safe():
    try:
        draw_view3d(bpy.context)
    except Exception as exc:  # pragma: no cover - defensive
        _draw_error_once(exc)


def _draw_view3d_eraser_safe():
    try:
        draw_view3d_eraser(bpy.context)
    except Exception as exc:  # pragma: no cover - defensive
        _draw_error_once(exc)


def register_handlers():
    if state.handlers:
        return
    for class_name, space_id in SUPPORTED_SPACES:
        space_cls = getattr(bpy.types, class_name, None)
        if space_cls is None:
            continue
        try:
            if space_id == "VIEW_3D":
                handle = space_cls.draw_handler_add(
                    _draw_view3d_safe, (), "WINDOW", "POST_VIEW"
                )
                state.handlers.append((space_cls, handle, "WINDOW", "POST_VIEW"))
                handle = space_cls.draw_handler_add(
                    _draw_view3d_eraser_safe, (), "WINDOW", "POST_PIXEL"
                )
                state.handlers.append((space_cls, handle, "WINDOW", "POST_PIXEL"))
            else:
                callback = make_draw_2d(space_id)
                handle = space_cls.draw_handler_add(
                    callback, (), "WINDOW", "POST_PIXEL"
                )
                state.handlers.append((space_cls, handle, "WINDOW", "POST_PIXEL"))
        except Exception as exc:  # pragma: no cover - defensive
            _draw_error_once(exc)


def unregister_handlers():
    for space_cls, handle, region_type, draw_type in state.handlers:
        try:
            space_cls.draw_handler_remove(handle, region_type)
        except Exception:  # already gone
            pass
    state.handlers = []


def tag_redraw_all(context, space_type_id=None):
    """Request a redraw of every (or matching) editor region."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if space_type_id is not None and area.type != space_type_id:
                continue
            area.tag_redraw()
