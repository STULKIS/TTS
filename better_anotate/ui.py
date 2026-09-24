"""N-panel UI and toolbar tools for Better Anotate."""

from __future__ import annotations

import bpy

from .props import get_settings

# space type -> (panel class space type, space type idname) — keep in sync
# with overlay.SUPPORTED_SPACES.
PANEL_SPACES = (
    "VIEW_3D",
    "IMAGE_EDITOR",
    "NODE_EDITOR",
    "SEQUENCE_EDITOR",
    "CLIP_EDITOR",
)


class BA_PT_base:
    """Shared sidebar panel, registered once per supported editor."""

    bl_label = "Better Anotate"
    bl_idname = "BETTER_ANOTATE_PT_base"
    bl_category = "Annotate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = get_settings(context)
        if settings is None:
            layout.label(text="Scene settings missing", icon="ERROR")
            return

        # ---- tool / color / thickness --------------------------------
        col = layout.column()
        row = col.row(align=True)
        row.prop(settings, "tool", text="")
        row.operator("better_anotate.delete_last", text="", icon="X")

        col.prop(settings, "color")
        col.prop(settings, "thickness")

        space = getattr(context, "space_data", None)
        if space is not None and getattr(space, "type", None) == "VIEW_3D":
            col.prop(settings, "draw_on", text="Draw On")

        # ---- layer stack ---------------------------------------------
        box = col.box()
        header = box.row()
        header.label(text="Layers")
        spacer = header.row()
        spacer.alignment = "RIGHT"
        spacer.operator("better_anotate.layer_add", text="", icon="ADD")

        active_index = settings.active_layer
        for i, layer in enumerate(settings.layers):
            row = box.row(align=True)
            act = row.operator(
                "better_anotate.layer_activate",
                text="",
                icon="RADIOBUT_ON" if i == active_index else "RADIOBUT_OFF",
                emboss=False,
            )
            act.index = i
            row.prop(layer, "name", text="", emboss=(i == active_index))
            vis = row.operator(
                "better_anotate.layer_visibility",
                text="",
                icon="RESTRICT_VIEW_ON" if layer.hide else "RESTRICT_VIEW_OFF",
            )
            vis.index = i
            lock = row.operator(
                "better_anotate.layer_lock",
                text="",
                icon="LOCKED" if layer.lock else "UNLOCKED",
            )
            lock.index = i

        if not settings.layers:
            box.label(text="No layers yet", icon="INFO")

        if 0 <= active_index < len(settings.layers):
            footer = box.row(align=True)
            rem = footer.operator(
                "better_anotate.layer_remove", text="", icon="REMOVE"
            )
            rem.index = active_index
            up = footer.operator("better_anotate.layer_move", text="", icon="TRIA_UP")
            up.index = active_index
            up.direction = 1
            down = footer.operator(
                "better_anotate.layer_move", text="", icon="TRIA_DOWN"
            )
            down.index = active_index
            down.direction = -1
            footer.operator("better_anotate.clear_layer", text="", icon="TRASH")

        col.operator("better_anotate.apply_style", text="Apply Color/Size to Layer")

        # ---- actions -------------------------------------------------
        col.prop(settings, "show_overlay", text="Show Annotations")
        col.operator(
            "better_anotate.convert_to_gp",
            text="Convert 3D Strokes to Grease Pencil",
            icon="GREASEPENCIL",
        )
        col.separator()
        hint = col.column(align=True)
        hint.scale_y = 0.7
        hint.label(text="Draw: Ctrl + Left drag")
        hint.label(text="Erase: Ctrl + Shift + Left drag")
        if len(settings.strokes):
            hint.label(text=f"{len(settings.strokes)} stroke(s)")


def _make_panel(space_type):
    name = "BETTER_ANOTATE_PT_" + space_type.lower()
    return type(
        name,
        (BA_PT_base, bpy.types.Panel),
        {
            "bl_idname": name,
            "bl_space_type": space_type,
            "bl_region_type": "UI",
            "bl_category": "Annotate",
            "bl_label": "Better Anotate",
        },
    )


# ---------------------------------------------------------------------------
# toolbar tools
# ---------------------------------------------------------------------------


class BETTER_ANOTATE_TT_draw(bpy.types.WorkSpaceTool):
    """Draw annotations directly in the viewport"""

    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context_mode = None  # adjusted at registration if required
    bl_idname = "better_anotate.tool_draw"
    bl_label = "Annotate+"
    bl_description = "Draw colored, layered annotations (Better Anotate)"
    bl_icon = "GREASEPENCIL"
    bl_cursor = "CROSSHAIR"
    bl_keymap = (
        ("better_anotate.draw", {"type": "LEFTMOUSE", "value": "PRESS"}, None),
    )


class BETTER_ANOTATE_TT_draw_image(bpy.types.WorkSpaceTool):
    """Draw annotations over images"""

    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "TOOLS"
    bl_context_mode = None
    bl_idname = "better_anotate.tool_draw_image"
    bl_label = "Annotate+"
    bl_description = "Draw colored, layered annotations (Better Anotate)"
    bl_icon = "GREASEPENCIL"
    bl_cursor = "CROSSHAIR"
    bl_keymap = (
        ("better_anotate.draw", {"type": "LEFTMOUSE", "value": "PRESS"}, None),
    )


def _register_tool(tool_cls):
    """Register a toolbar tool, trying plausible context-mode keys."""
    tried = []
    for mode in (None, "objectmode"):
        tool_cls.bl_context_mode = mode
        try:
            bpy.utils.register_tool(tool_cls)
            return True
        except Exception as exc:
            tried.append((mode, str(exc)))
    print(
        f"Better Anotate: could not register tool {tool_cls.bl_idname} "
        f"(tried: {tried}) — the Ctrl+drag keymap still works."
    )
    return False


_tools_registered = []


def _register_tools():
    global _tools_registered
    for tool_cls in (BETTER_ANOTATE_TT_draw, BETTER_ANOTATE_TT_draw_image):
        if _register_tool(tool_cls):
            _tools_registered.append(tool_cls)


def _unregister_tools():
    global _tools_registered
    for tool_cls in reversed(_tools_registered):
        try:
            bpy.utils.unregister_tool(tool_cls)
        except Exception:
            pass
    _tools_registered = []


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

CLASSES: list = []


def register():
    global CLASSES
    CLASSES = [_make_panel(space) for space in PANEL_SPACES]
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _register_tools()


def unregister():
    _unregister_tools()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
