# Better Anotate

**Layered, multi-color annotations with shapes — in every Blender editor where
you'd normally annotate.**

Blender's built-in *Annotate* tool keeps strokes on one little layer system
with a single color per layer. Better Anotate gives you a proper layer stack
(name / reorder / hide / lock / opacity — think UCUPaint), **a unique color
and thickness on every stroke**, shape tools, an eraser, and it draws
everything as a **non-destructive overlay**: nothing is added to your scene,
nothing shows up in renders, and it works in Object, Edit, Sculpt, Texture
Paint… any mode.

## Features

| | |
|---|---|
| **Layers** | Add, remove, rename, reorder, hide, lock, per-layer opacity. Strokes belong to a layer. |
| **Per-stroke color & thickness** | Pick a color/size before each stroke; change them any time. *Apply Color/Size to Layer* restyles a whole layer. |
| **Shape tools** | Freehand, Line, Arrow, Rectangle, Circle, Highlighter (thick translucent marker). |
| **Eraser** | Ctrl+Shift+drag removes strokes it touches (wheel resizes the brush). |
| **Draw On (3D)** | *View Plane* (facing the camera) or *Surface* (raycast onto geometry). |
| **Everywhere** | 3D Viewport, Image Editor, Node Editor, Sequencer, Movie Clip Editor. |
| **Overlay** | Never pollutes the scene, never renders, stays readable in any mode. |
| **Convert to Grease Pencil** | One click turns your 3D annotations into a real GP object you can render/edit. |

## Install

1. Build the zip: `python3 better_anotate/package.py` → `dist/better_anotate-0.1.0.zip`
2. Blender ▸ *Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk…* and pick the zip.
3. Enable **Better Anotate**.

(Blender 4.2+ uses the extension zip; `dist/better_anotate-legacy-0.1.0.zip`
is provided for classic *Install from Disk* as a legacy add-on.)

## Usage

| Action | How |
|---|---|
| Draw | **Ctrl + Left drag** in any supported editor, or the **Annotate+** toolbar tool (3D View / Image Editor) |
| Erase | **Ctrl + Shift + Left drag** (mouse wheel changes radius, Esc/RMB cancels) |
| Settings | N-Panel ▸ **Annotate** ▸ *Better Anotate* |
| Undo | Every stroke, erase session and layer edit is a normal undo step (Ctrl+Z) |

Annotations are stored per scene in the .blend file.

## Notes & limitations

* Annotations are a viewport overlay — they are **not** part of the render.
  Use *Convert 3D Strokes to Grease Pencil* when you need them in a render.
  Conversion only handles 3D-viewport strokes (2D-editor strokes stay
  overlay-only).
* If the built-in Annotate tool is active, its own LMB binding also draws —
  switch tools when you use ours from the toolbar.
* Supported: Blender **4.5 LTS and 5.x** (tested on 4.5.14 and 5.0.1).
  On older 4.x builds everything except *Convert to Grease Pencil* works —
  those versions lack the modern GP stroke Python API.

## Development

```bash
# geometry helpers (no Blender needed)
python3 -m unittest tests.test_ba_math -v

# full suite against a Blender Python (pip bpy)
pip install bpy==5.0.1          # Python 3.11
python -m unittest discover -s tests -v
```

Layout: `props.py` (data model) · `overlay.py` (gpu draw handlers) ·
`operators.py` (modal draw/erase + layer ops) · `math_utils.py` (pure
geometry) · `gp_convert.py` (Grease Pencil conversion) · `ui.py` (N-panel +
tools) · `keymap.py`.
