"""Better Anotate — layered, multi-color annotations for every editor.

Install as an extension (Blender 4.2+) via *Edit ▸ Preferences ▸ Add-ons ▸
Install from Disk*, picking the release zip, or drop this folder into your
scripts/addons directory.
"""

from __future__ import annotations

bl_info = {
    "name": "Better Anotate",
    "author": "STULKIS",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "N-Panel ▸ Annotate (3D View, Image, Node, Sequencer, Clip)",
    "description": (
        "Layered annotations with per-stroke colors, thickness and shapes — "
        "drawn as a non-destructive overlay in every editor"
    ),
    "doc_url": "",
    "tracker_url": "",
    "category": "3D View",
}

_MODULES = ("props", "operators", "ui", "keymap", "overlay")


def register():
    try:
        import bpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Better Anotate must be registered inside Blender") from exc

    from . import keymap, operators, overlay, props, ui

    props.register()
    operators.register()
    ui.register()
    keymap.register()
    overlay.register_handlers()


def unregister():
    try:
        from . import keymap, operators, overlay, props, ui
    except ImportError:  # pragma: no cover
        return

    overlay.unregister_handlers()
    keymap.unregister()
    ui.unregister()
    operators.unregister()
    props.unregister()


if __name__ == "__main__":  # pragma: no cover - script-load convenience
    register()
