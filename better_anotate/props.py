"""Data model for Better Anotate.

Annotations live on the scene (``scene.better_anotate``) so they are saved
with the .blend file.  Layers behave like a paint-layer stack (think
UCUPaint): named, ordered, hideable/lockable, and every stroke stores its own
color, thickness and parent layer.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)

DEFAULT_COLOR = (1.0, 0.2, 0.08, 1.0)

TOOL_ITEMS = (
    ("FREEHAND", "Draw", "Freehand strokes"),
    ("LINE", "Line", "Straight line between two clicks"),
    ("ARROW", "Arrow", "Arrow from start to end point"),
    ("RECT", "Rectangle", "Rectangle from corner to corner"),
    ("CIRCLE", "Circle", "Circle from center to rim"),
    ("HIGHLIGHT", "Highlighter", "Thick translucent marker strokes"),
)

DRAW_ON_ITEMS = (
    ("VIEW", "View Plane", "Draw on a plane facing the view (through the 3D cursor)"),
    ("SURFACE", "Surface", "Draw on geometry under the cursor, falling back to the view plane"),
)


class BA_Point(bpy.types.PropertyGroup):
    """One stored point — world space for 3D strokes, view space for 2D."""

    x: FloatProperty()
    y: FloatProperty()
    z: FloatProperty(default=0.0)


class BA_Stroke(bpy.types.PropertyGroup):
    uid: IntProperty(default=0)
    layer_uid: IntProperty(default=0)
    editor: StringProperty(default="VIEW_3D")  # space type idname
    tool: StringProperty(default="FREEHAND")
    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    thickness: FloatProperty(name="Thickness", default=4.0, min=0.5, soft_max=64.0)
    normal: FloatVectorProperty(
        name="Plane Normal",
        size=3,
        default=(0.0, 0.0, 1.0),
        options={"HIDDEN"},
    )
    points: CollectionProperty(type=BA_Point)


class BA_Layer(bpy.types.PropertyGroup):
    uid: IntProperty(default=0)
    name: StringProperty(name="Name", default="Layer")
    hide: BoolProperty(name="Hidden", default=False)
    lock: BoolProperty(name="Locked", default=False)
    opacity: FloatProperty(
        name="Opacity",
        subtype="FACTOR",
        default=1.0,
        min=0.0,
        max=1.0,
    )


class BA_Settings(bpy.types.PropertyGroup):
    layers: CollectionProperty(type=BA_Layer)
    strokes: CollectionProperty(type=BA_Stroke)
    active_layer: IntProperty(default=-1)
    next_uid: IntProperty(default=1, min=1)

    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=DEFAULT_COLOR,
    )
    thickness: FloatProperty(
        name="Thickness",
        default=4.0,
        min=0.5,
        soft_max=64.0,
        subtype="PIXEL",
    )
    tool: EnumProperty(name="Tool", items=TOOL_ITEMS, default="FREEHAND")
    draw_on: EnumProperty(name="Draw On", items=DRAW_ON_ITEMS, default="VIEW")
    show_overlay: BoolProperty(
        name="Show Annotations",
        description="Draw annotations as a viewport overlay",
        default=True,
    )


# ---------------------------------------------------------------------------
# helpers used by operators / overlay
# ---------------------------------------------------------------------------


def get_settings(context) -> "BA_Settings | None":
    scene = getattr(context, "scene", None)
    if scene is None:
        return None
    return getattr(scene, "better_anotate", None)


def alloc_uid(settings: BA_Settings) -> int:
    uid = settings.next_uid
    settings.next_uid = uid + 1
    return uid


def ensure_layer(settings: BA_Settings) -> BA_Layer:
    """Return the active layer, creating a default one when needed."""
    idx = settings.active_layer
    if 0 <= idx < len(settings.layers):
        return settings.layers[idx]
    is_first = len(settings.layers) == 0
    layer = settings.layers.add()
    layer.uid = alloc_uid(settings)
    layer.name = "Notes" if is_first else f"Layer {len(settings.layers)}"
    settings.active_layer = len(settings.layers) - 1
    return layer


def active_layer(settings: BA_Settings) -> "BA_Layer | None":
    idx = settings.active_layer
    if 0 <= idx < len(settings.layers):
        return settings.layers[idx]
    return None


def layer_by_uid(settings: BA_Settings, uid: int) -> "BA_Layer | None":
    for layer in settings.layers:
        if layer.uid == uid:
            return layer
    return None


def stroke_count(settings: BA_Settings) -> int:
    return len(settings.strokes)


def append_stroke(
    settings: BA_Settings,
    *,
    layer_uid: int,
    editor: str,
    tool: str,
    color,
    thickness: float,
    normal,
    points,
) -> BA_Stroke:
    """Create a stroke entry from plain Python data and return it."""
    stroke = settings.strokes.add()
    stroke.uid = alloc_uid(settings)
    stroke.layer_uid = layer_uid
    stroke.editor = editor
    stroke.tool = tool
    stroke.color = tuple(color)
    stroke.thickness = float(thickness)
    stroke.normal = tuple(normal)
    for co in points:
        pt = stroke.points.add()
        pt.x, pt.y, pt.z = float(co[0]), float(co[1]), float(co[2])
    return stroke


def stroke_points(stroke: BA_Stroke):
    return [(p.x, p.y, p.z) for p in stroke.points]


CLASSES = (BA_Point, BA_Stroke, BA_Layer, BA_Settings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.better_anotate = bpy.props.PointerProperty(type=BA_Settings)


def unregister():
    if hasattr(bpy.types.Scene, "better_anotate"):
        del bpy.types.Scene.better_anotate
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
