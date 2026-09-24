"""Mutable runtime state shared between operators and the overlay.

Kept separate from PropertyGroups on purpose: preview/eraser state is
transient (never saved to the .blend) and must not create undo steps.
"""

from __future__ import annotations

# In-progress stroke preview, dict or None:
#   editor, tool, points, normal, color, thickness, layer_uid
preview: dict | None = None

# UIDs of strokes currently being erased (drawn skipped until confirm).
pending_erase: set[int] = set()

# Eraser cursor in region pixels: (x, y, radius) or None.
eraser_circle: tuple | None = None

# Registered draw handlers as (space_class, handle, region_type, draw_type).
handlers: list = []

# Addon keymap as (keymap, [items]).
keymap: tuple | None = None

# Print a draw-time error only once to avoid spamming the console.
_draw_error_reported = False


def clear_transient() -> None:
    global preview, pending_erase, eraser_circle
    preview = None
    pending_erase = set()
    eraser_circle = None


def reset_all() -> None:
    global handlers, keymap
    clear_transient()
    handlers = []
    keymap = None
