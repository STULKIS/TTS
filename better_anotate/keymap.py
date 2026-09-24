"""Addon keymap for Better Anotate."""

from __future__ import annotations

import bpy

from . import state


def register():
    if state.keymap is not None:
        return
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc is None:
        # background mode / no user config — keymap simply stays unbound
        return
    km = kc.keymaps.new("Better Anotate", space_type="EMPTY", region_type="WINDOW")
    items = [
        km.keymap_items.new(
            "better_anotate.draw", "LEFTMOUSE", "PRESS", ctrl=True
        ),
        km.keymap_items.new(
            "better_anotate.erase", "LEFTMOUSE", "PRESS", ctrl=True, shift=True
        ),
    ]
    state.keymap = (km, items)


def unregister():
    if state.keymap is None:
        return
    km, items = state.keymap
    for item in items:
        try:
            km.keymap_items.remove(item)
        except Exception:
            pass
    try:
        kc = bpy.context.window_manager.keyconfigs.addon
        if kc is not None and km.name in kc.keymaps:
            kc.keymaps.remove(km)
    except Exception:
        pass
    state.keymap = None
