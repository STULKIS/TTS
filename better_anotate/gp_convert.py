"""Convert Better Anotate strokes into a real Grease Pencil object.

Works on Blender 4.3+ (GPv3).  Between 4.3 and 4.5 the datablock collection
is ``bpy.data.grease_pencils_v3``; in 5.0 the legacy ``grease_pencils`` name
was reused for GPv3 (and GPv2 was removed).  We feature-detect instead of
version-sniffing.
"""

from __future__ import annotations

import math

import bpy

from . import math_utils as mu
from .props import stroke_points

GP_OBJECT_NAME = "Better Annotate"
GP_MATERIAL_NAME = "Better Annotate.Stroke"


def _gp_collection():
    coll = getattr(bpy.data, "grease_pencils_v3", None)
    if coll is not None:
        return coll
    coll = getattr(bpy.data, "grease_pencils", None)
    if coll is not None:
        return coll
    raise RuntimeError("No Grease Pencil datablock collection available")


def _window_view3d(context):
    """Return (region, rv3d) of the first VIEW_3D window area, if any."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return None, None
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW":
                    data = getattr(region, "data", None)
                    if data is not None and hasattr(data, "perspective_matrix"):
                        return region, data
    return None, None


def _stroke_pixel_world_size(context, world_center):
    """Approximate world-space size of one screen pixel at a stroke."""
    region, rv3d = _window_view3d(context)
    if region is None or rv3d is None:
        return 0.005
    loc = rv3d.view_location
    dx = world_center[0] - loc[0]
    dy = world_center[1] - loc[1]
    dz = world_center[2] - loc[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if rv3d.is_perspective and dist > 1e-6:
        view_plane = 2.0 * dist * math.tan(rv3d.angle * 0.5)
    else:
        view_plane = abs(rv3d.view_distance) or 1.0
    if region.height <= 0:
        return 0.005
    return max(view_plane / region.height, 1e-9)


def _ensure_material(gp):
    mat = bpy.data.materials.get(GP_MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(GP_MATERIAL_NAME)
    if len(gp.materials) == 0:
        gp.materials.append(mat)
    return 0  # material index used by strokes


def convert(context, settings):
    """Convert all 3D-view annotations to a Grease Pencil object.

    Returns a dict with counts: {converted, points, layers, skipped_2d}.
    """
    strokes_3d = [s for s in settings.strokes if s.editor == "VIEW_3D"]
    skipped_2d = len(settings.strokes) - len(strokes_3d)

    scene = context.scene
    frame_number = scene.frame_current

    gp = _gp_collection().new(GP_OBJECT_NAME)
    material_index = _ensure_material(gp)

    # one GP layer per annotation layer (uid -> gp layer)
    gp_layers = {}
    stats = {"converted": 0, "points": 0, "layers": 0, "skipped_2d": skipped_2d}

    for stroke in strokes_3d:
        layer = None
        for candidate in settings.layers:
            if candidate.uid == stroke.layer_uid:
                layer = candidate
                break
        layer_name = layer.name if layer is not None else "Notes"

        gp_layer = gp_layers.get(stroke.layer_uid)
        if gp_layer is None:
            gp_layer = gp.layers.new(layer_name)
            gp_layers[stroke.layer_uid] = gp_layer
            if layer is not None:
                if hasattr(gp_layer, "channel_color") and len(stroke.color) >= 3:
                    gp_layer.channel_color = stroke.color[:3]
                if hasattr(gp_layer, "opacity"):
                    gp_layer.opacity = layer.opacity
                if hasattr(gp_layer, "hide"):
                    gp_layer.hide = layer.hide
                if hasattr(gp_layer, "lock"):
                    gp_layer.lock = layer.lock
            gp_layer.frames.new(frame_number)
            stats["layers"] += 1

        frame = gp_layer.frames[-1]
        if frame.frame_number != frame_number:
            frame = gp_layer.frames.new(frame_number)
        if not hasattr(frame, "drawing"):
            raise RuntimeError(
                "this Blender's Grease Pencil Python API is too old — "
                "conversion needs Blender 4.5+ or 5.x"
            )
        drawing = frame.drawing

        points = stroke_points(stroke)
        polylines = mu.expand_stroke(stroke.tool, points, tuple(stroke.normal))
        if not polylines:
            continue

        # world-space radius from screen thickness at the stroke center
        center = points[0]
        px_world = _stroke_pixel_world_size(context, center)
        base_radius = max(stroke.thickness * 0.5 * px_world, 1e-6)
        # highlighter strokes are drawn thicker on the overlay
        if stroke.tool == "HIGHLIGHT":
            base_radius *= 3.0
        alpha = stroke.color[3] if len(stroke.color) >= 3 else 1.0

        for poly in polylines:
            if len(poly) == 0:
                continue
            drawing.add_strokes([len(poly)])
            gp_stroke = drawing.strokes[-1]
            gp_stroke.material_index = material_index
            for i, co in enumerate(poly):
                pt = gp_stroke.points[i]
                pt.position = co
                pt.radius = base_radius
                pt.opacity = alpha
                if hasattr(pt, "vertex_color"):
                    pt.vertex_color = stroke.color
            stats["converted"] += 1
            stats["points"] += len(poly)

    ob = bpy.data.objects.new(GP_OBJECT_NAME, gp)
    context.collection.objects.link(ob)
    if hasattr(ob, "show_in_front"):
        ob.show_in_front = True
    return stats
