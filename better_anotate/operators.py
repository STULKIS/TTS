"""Operators for Better Anotate: drawing, erasing, layers and conversion."""

from __future__ import annotations

import bpy
from bpy_extras import view3d_utils

from . import gp_convert
from . import math_utils as mu
from . import state
from .overlay import SUPPORTED_SPACE_IDS, make_view2d_fns, tag_redraw_all
from .props import (
    active_layer,
    append_stroke,
    alloc_uid,
    ensure_layer,
    get_settings,
    layer_by_uid,
    stroke_points,
)

MIN_POINT_DISTANCE_PX = 2.0
MAX_STROKE_POINTS = 5000
SHAPE_TOOLS = {"LINE", "ARROW", "RECT", "CIRCLE"}


def _supported_context(context):
    """True when the current space/region can host annotations."""
    space = getattr(context, "space_data", None)
    region = getattr(context, "region", None)
    if space is None or region is None:
        return False
    if getattr(space, "type", None) not in SUPPORTED_SPACE_IDS:
        return False
    if getattr(region, "type", None) != "WINDOW":
        return False
    return True


# ---------------------------------------------------------------------------
# coordinate conversion
# ---------------------------------------------------------------------------


class _Cursor3D:
    """Mouse -> 3D conversion for the viewport, honoring the Draw On mode."""

    def __init__(self, context, draw_on):
        self.region = context.region
        self.rv3d = context.region.data
        self.scene = context.scene
        self.draw_on = draw_on
        self.depsgraph = context.evaluated_depsgraph_get()
        cursor = self.scene.cursor
        self.cursor_loc = (cursor.location.x, cursor.location.y, cursor.location.z)
        self.plane_normal = self._view_direction()
        self.plane_point = self.cursor_loc
        self._first_done = False

    def _view_direction(self):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, (0.0, 0.0))
        target = view3d_utils.region_2d_to_location_3d(
            self.region, self.rv3d, (0.0, 0.0), self.cursor_loc
        )
        d = (
            target[0] - origin[0],
            target[1] - origin[1],
            target[2] - origin[2],
        )
        ln = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
        if ln < 1e-9:
            return (0.0, 0.0, 1.0)
        return (d[0] / ln, d[1] / ln, d[2] / ln)

    def __call__(self, mx, my):
        if self.draw_on == "SURFACE":
            origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, (mx, my))
            direction = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, (mx, my))
            hit, location, _normal, _index, _ob, _mat = self.scene.ray_cast(
                self.depsgraph, origin, direction
            )
            if hit:
                if not self._first_done:
                    self.plane_normal = tuple(_normal)
                    self.plane_point = tuple(location)
                self._first_done = True
                return tuple(location)
        # view plane through the 3D cursor (or the first surface hit)
        co = view3d_utils.region_2d_to_location_3d(
            self.region, self.rv3d, (mx, my), self.plane_point
        )
        if not self._first_done:
            self.plane_normal = self._view_direction()
            self.plane_point = tuple(co)
            self._first_done = True
        return tuple(co)


# ---------------------------------------------------------------------------
# draw (modal)
# ---------------------------------------------------------------------------


class BETTER_ANOTATE_OT_draw(bpy.types.Operator):
    """Draw an annotation stroke (Ctrl+Left drag, or the Annotate tool)"""

    bl_idname = "better_anotate.draw"
    bl_label = "Draw Annotation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not _supported_context(context):
            return False
        return get_settings(context) is not None

    def invoke(self, context, event):
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}
        layer = ensure_layer(settings)
        if layer.hide or layer.lock:
            self.report(
                {"ERROR"},
                f"Layer '{layer.name}' is {'hidden' if layer.hide else 'locked'}",
            )
            return {"CANCELLED"}

        space = context.space_data
        region = context.region
        self._space_type = space.type
        self._layer_uid = layer.uid
        self._color = tuple(settings.color)
        self._thickness = float(settings.thickness)
        self._tool = settings.tool
        mx, my = event.mouse_region_x, event.mouse_region_y

        if self._space_type == "VIEW_3D":
            if region.data is None or not hasattr(region.data, "perspective_matrix"):
                return {"CANCELLED"}
            self._to_world = _Cursor3D(context, settings.draw_on)
            self._normal = None
            first = self._to_world(mx, my)
            self._points = [first]
            self._normal = self._to_world.plane_normal
        else:
            to_view, _to_region = make_view2d_fns(region)
            vx, vy = to_view(mx, my)
            self._to_view = to_view
            self._points = [(vx, vy, 0.0)]
            self._normal = (0.0, 0.0, 1.0)

        self._last_px = (mx, my)
        state.preview = {
            "editor": self._space_type,
            "tool": self._tool,
            "points": list(self._points),
            "normal": self._normal,
            "color": self._color,
            "thickness": self._thickness,
            "layer_uid": self._layer_uid,
        }
        context.window_manager.modal_handler_add(self)
        tag_redraw_all(context, self._space_type)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        settings = get_settings(context)
        if settings is None:
            state.preview = None
            return {"CANCELLED"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            state.preview = None
            tag_redraw_all(context, self._space_type)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            mx, my = event.mouse_region_x, event.mouse_region_y
            if self._space_type == "VIEW_3D":
                co = self._to_world(mx, my)
            else:
                vx, vy = self._to_view(mx, my)
                co = (vx, vy, 0.0)

            if self._tool in SHAPE_TOOLS:
                if len(self._points) > 1:
                    self._points = [self._points[0], co]
                else:
                    self._points.append(co)
            else:
                dx = mx - self._last_px[0]
                dy = my - self._last_px[1]
                if (dx * dx + dy * dy) ** 0.5 >= MIN_POINT_DISTANCE_PX:
                    if len(self._points) < MAX_STROKE_POINTS:
                        self._points.append(co)
                    self._last_px = (mx, my)
            if state.preview is not None:
                state.preview["points"] = list(self._points)
            tag_redraw_all(context, self._space_type)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            points = list(self._points)
            state.preview = None
            if self._tool in SHAPE_TOOLS:
                if len(points) < 2 or points[0] == points[-1]:
                    tag_redraw_all(context, self._space_type)
                    return {"CANCELLED"}
                points = [points[0], points[-1]]  # store anchors only
            if not points:
                tag_redraw_all(context, self._space_type)
                return {"CANCELLED"}
            append_stroke(
                settings,
                layer_uid=self._layer_uid,
                editor=self._space_type,
                tool=self._tool,
                color=self._color,
                thickness=self._thickness,
                normal=self._normal,
                points=points,
            )
            tag_redraw_all(context, self._space_type)
            return {"FINISHED"}

        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# eraser (modal)
# ---------------------------------------------------------------------------


def _stroke_polylines_px(settings, stroke, project):
    pts = stroke_points(stroke)
    polylines = mu.expand_stroke(stroke.tool, pts, tuple(stroke.normal))
    out = []
    for poly in polylines:
        projected = []
        ok = True
        for co in poly:
            xy = project(co)
            if xy is None:
                ok = False
                break
            projected.append(xy)
        if ok and projected:
            out.append(projected)
    return out


def _projector(context, space_type):
    """Return a world/view point -> region pixel projector for this editor."""
    region = context.region
    if space_type == "VIEW_3D":
        rv3d = region.data
        rows = [tuple(row) for row in rv3d.perspective_matrix]
        width, height = region.width, region.height

        def project(co):
            return mu.project_clip(rows, co, width, height)

        return project
    _to_view, to_region = make_view2d_fns(region)

    def project(co):
        return to_region(co[0], co[1])

    return project


class BETTER_ANOTATE_OT_erase(bpy.types.Operator):
    """Erase annotations (Ctrl+Shift+Left drag)"""

    bl_idname = "better_anotate.erase"
    bl_label = "Erase Annotations"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not _supported_context(context):
            return False
        return get_settings(context) is not None

    def invoke(self, context, event):
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}
        self._space_type = context.space_data.type
        self._radius = max(10.0, float(settings.thickness) * 2.0)
        self._mx = event.mouse_region_x
        self._my = event.mouse_region_y
        self._removed = []  # snapshots for cancel-restore
        state.pending_erase = set()
        state.eraser_circle = (self._mx, self._my, self._radius)
        context.window_manager.modal_handler_add(self)
        tag_redraw_all(context, self._space_type)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        settings = get_settings(context)
        if settings is None:
            self._finish(context, commit=False)
            return {"CANCELLED"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._finish(context, commit=False)
            return {"CANCELLED"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            delta = 4.0 if event.type == "WHEELUPMOUSE" else -4.0
            self._radius = max(4.0, min(96.0, self._radius + delta))
            state.eraser_circle = (self._mx, self._my, self._radius)
            tag_redraw_all(context, self._space_type)
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE":
            self._mx = event.mouse_region_x
            self._my = event.mouse_region_y
            state.eraser_circle = (self._mx, self._my, self._radius)
            self._sweep(context, settings)
            tag_redraw_all(context, self._space_type)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._finish(context, commit=True)
            return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def _sweep(self, context, settings):
        project = _projector(context, self._space_type)
        visible = {layer.uid: (not layer.hide and not layer.lock) for layer in settings.layers}
        mx, my, radius = state.eraser_circle or (self._mx, self._my, self._radius)

        # index-based iteration so removals don't skip entries
        i = 0
        while i < len(settings.strokes):
            stroke = settings.strokes[i]
            if stroke.uid in state.pending_erase:
                i += 1
                continue
            if stroke.editor != self._space_type:
                i += 1
                continue
            if not visible.get(stroke.layer_uid, True):
                i += 1
                continue
            hit = False
            for poly in _stroke_polylines_px(settings, stroke, project):
                if mu.polyline_hit(poly, mx, my, radius):
                    hit = True
                    break
            if hit:
                self._removed.append(_snapshot_stroke(settings, stroke))
                state.pending_erase.add(stroke.uid)
                settings.strokes.remove(i)
                # removal shifts later entries; do not advance i
                continue
            i += 1

    def _finish(self, context, commit):
        space_type = self._space_type
        if not commit and self._removed:
            settings = get_settings(context)
            if settings is not None:
                for data in self._removed:
                    _restore_stroke(settings, data)
        self._removed = []
        state.pending_erase = set()
        state.eraser_circle = None
        tag_redraw_all(context, space_type)


def _snapshot_stroke(settings, stroke):
    layer = layer_by_uid(settings, stroke.layer_uid)
    return {
        "uid": stroke.uid,
        "layer_uid": stroke.layer_uid,
        "layer_index": settings.layers.find(layer.name) if layer else -1,
        "editor": stroke.editor,
        "tool": stroke.tool,
        "color": tuple(stroke.color),
        "thickness": stroke.thickness,
        "normal": tuple(stroke.normal),
        "points": stroke_points(stroke),
        "insert_index": _stroke_insert_index(settings, stroke),
    }


def _stroke_insert_index(settings, stroke):
    for i, s in enumerate(settings.strokes):
        if s.uid == stroke.uid:
            return i
    return len(settings.strokes)


def _restore_stroke(settings, data):
    new_stroke = append_stroke(
        settings,
        layer_uid=data["layer_uid"],
        editor=data["editor"],
        tool=data["tool"],
        color=data["color"],
        thickness=data["thickness"],
        normal=data["normal"],
        points=data["points"],
    )
    # keep the original uid so pending-erase bookkeeping stays coherent
    new_stroke.uid = data["uid"]
    index = data.get("insert_index", -1)
    if 0 <= index < len(settings.strokes) - 1:
        settings.strokes.move(len(settings.strokes) - 1, index)


# ---------------------------------------------------------------------------
# layer management
# ---------------------------------------------------------------------------


class BETTER_ANOTATE_OT_layer_add(bpy.types.Operator):
    """Add a new annotation layer"""

    bl_idname = "better_anotate.layer_add"
    bl_label = "Add Layer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return get_settings(context) is not None

    def execute(self, context):
        settings = get_settings(context)
        layer = settings.layers.add()
        layer.uid = alloc_uid(settings)
        layer.name = f"Layer {len(settings.layers)}"
        settings.active_layer = len(settings.layers) - 1
        return {"FINISHED"}


class BETTER_ANOTATE_OT_layer_remove(bpy.types.Operator):
    """Remove the layer and all of its strokes"""

    bl_idname = "better_anotate.layer_remove"
    bl_label = "Remove Layer"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.layers) > 0

    def execute(self, context):
        settings = get_settings(context)
        index = self.index if self.index >= 0 else settings.active_layer
        if not (0 <= index < len(settings.layers)):
            return {"CANCELLED"}
        layer = settings.layers[index]
        layer_uid = layer.uid
        settings.layers.remove(index)
        i = 0
        while i < len(settings.strokes):
            if settings.strokes[i].layer_uid == layer_uid:
                settings.strokes.remove(i)
            else:
                i += 1
        settings.active_layer = min(index, len(settings.layers) - 1)
        return {"FINISHED"}


class BETTER_ANOTATE_OT_layer_move(bpy.types.Operator):
    """Move the layer up or down the stack"""

    bl_idname = "better_anotate.layer_move"
    bl_label = "Move Layer"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)
    direction: bpy.props.IntProperty(default=1)  # +1 up (visually), -1 down

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.layers) > 1

    def execute(self, context):
        settings = get_settings(context)
        index = self.index if self.index >= 0 else settings.active_layer
        if not (0 <= index < len(settings.layers)):
            return {"CANCELLED"}
        # UI draws the stack top-down but collection index 0 is "first";
        # "up" in the UI = higher index (drawn above).
        target = index + (1 if self.direction > 0 else -1)
        if not (0 <= target < len(settings.layers)):
            return {"CANCELLED"}
        settings.layers.move(index, target)
        settings.active_layer = target
        return {"FINISHED"}


class BETTER_ANOTATE_OT_layer_activate(bpy.types.Operator):
    """Make this the active layer"""

    bl_idname = "better_anotate.layer_activate"
    bl_label = "Activate Layer"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.layers) > 0

    def execute(self, context):
        settings = get_settings(context)
        if 0 <= self.index < len(settings.layers):
            settings.active_layer = self.index
            return {"FINISHED"}
        return {"CANCELLED"}


class BETTER_ANOTATE_OT_layer_visibility(bpy.types.Operator):
    """Show or hide the layer (hides all of its strokes in the overlay)"""

    bl_idname = "better_anotate.layer_visibility"
    bl_label = "Toggle Layer Visibility"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.layers) > 0

    def execute(self, context):
        settings = get_settings(context)
        index = self.index if self.index >= 0 else settings.active_layer
        if not (0 <= index < len(settings.layers)):
            return {"CANCELLED"}
        layer = settings.layers[index]
        layer.hide = not layer.hide
        tag_redraw_all(context)
        return {"FINISHED"}


class BETTER_ANOTATE_OT_layer_lock(bpy.types.Operator):
    """Prevent drawing or erasing on this layer"""

    bl_idname = "better_anotate.layer_lock"
    bl_label = "Toggle Layer Lock"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.layers) > 0

    def execute(self, context):
        settings = get_settings(context)
        index = self.index if self.index >= 0 else settings.active_layer
        if not (0 <= index < len(settings.layers)):
            return {"CANCELLED"}
        layer = settings.layers[index]
        layer.lock = not layer.lock
        return {"FINISHED"}


class BETTER_ANOTATE_OT_clear_layer(bpy.types.Operator):
    """Delete every stroke on the active layer"""

    bl_idname = "better_anotate.clear_layer"
    bl_label = "Clear Layer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.strokes) > 0

    def execute(self, context):
        settings = get_settings(context)
        layer = active_layer(settings)
        if layer is None:
            self.report({"WARNING"}, "No active layer")
            return {"CANCELLED"}
        if layer.lock:
            self.report({"ERROR"}, f"Layer '{layer.name}' is locked")
            return {"CANCELLED"}
        i = 0
        while i < len(settings.strokes):
            if settings.strokes[i].layer_uid == layer.uid:
                settings.strokes.remove(i)
            else:
                i += 1
        tag_redraw_all(context)
        return {"FINISHED"}


class BETTER_ANOTATE_OT_delete_last(bpy.types.Operator):
    """Delete the most recently added stroke"""

    bl_idname = "better_anotate.delete_last"
    bl_label = "Delete Last Stroke"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.strokes) > 0

    def execute(self, context):
        settings = get_settings(context)
        if not len(settings.strokes):
            return {"CANCELLED"}
        settings.strokes.remove(len(settings.strokes) - 1)
        tag_redraw_all(context)
        return {"FINISHED"}


class BETTER_ANOTATE_OT_clear_all(bpy.types.Operator):
    """Delete every annotation in this scene"""

    bl_idname = "better_anotate.clear_all"
    bl_label = "Clear All Annotations"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and len(settings.strokes) > 0

    def execute(self, context):
        settings = get_settings(context)
        while len(settings.strokes):
            settings.strokes.remove(0)
        tag_redraw_all(context)
        return {"FINISHED"}


class BETTER_ANOTATE_OT_apply_style(bpy.types.Operator):
    """Apply the current color and thickness to all strokes on the active layer"""

    bl_idname = "better_anotate.apply_style"
    bl_label = "Apply Style to Layer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        if settings is None:
            return False
        layer = active_layer(settings)
        if layer is None:
            return False
        return any(s.layer_uid == layer.uid for s in settings.strokes)

    def execute(self, context):
        settings = get_settings(context)
        layer = active_layer(settings)
        if layer is None:
            return {"CANCELLED"}
        if layer.lock:
            self.report({"ERROR"}, f"Layer '{layer.name}' is locked")
            return {"CANCELLED"}
        color = tuple(settings.color)
        thickness = float(settings.thickness)
        count = 0
        for stroke in settings.strokes:
            if stroke.layer_uid == layer.uid:
                stroke.color = color
                stroke.thickness = thickness
                count += 1
        self.report({"INFO"}, f"Restyled {count} stroke(s)")
        tag_redraw_all(context)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# convert to grease pencil
# ---------------------------------------------------------------------------


class BETTER_ANOTATE_OT_convert_to_gp(bpy.types.Operator):
    """Convert 3D annotations into a real Grease Pencil object"""

    bl_idname = "better_anotate.convert_to_gp"
    bl_label = "Convert to Grease Pencil"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = get_settings(context)
        return settings is not None and any(
            s.editor == "VIEW_3D" for s in settings.strokes
        )

    def execute(self, context):
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}
        try:
            stats = gp_convert.convert(context, settings)
        except Exception as exc:
            self.report({"ERROR"}, f"Conversion failed: {exc}")
            return {"CANCELLED"}
        skipped = stats["skipped_2d"]
        msg = (
            f"Converted {stats['converted']} stroke(s), "
            f"{stats['points']} point(s) into '{gp_convert.GP_OBJECT_NAME}'"
        )
        if skipped:
            msg += f" ({skipped} 2D stroke(s) skipped)"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


CLASSES = (
    BETTER_ANOTATE_OT_draw,
    BETTER_ANOTATE_OT_erase,
    BETTER_ANOTATE_OT_layer_add,
    BETTER_ANOTATE_OT_layer_remove,
    BETTER_ANOTATE_OT_layer_move,
    BETTER_ANOTATE_OT_layer_activate,
    BETTER_ANOTATE_OT_layer_visibility,
    BETTER_ANOTATE_OT_layer_lock,
    BETTER_ANOTATE_OT_clear_layer,
    BETTER_ANOTATE_OT_clear_all,
    BETTER_ANOTATE_OT_delete_last,
    BETTER_ANOTATE_OT_apply_style,
    BETTER_ANOTATE_OT_convert_to_gp,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    state.clear_transient()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
