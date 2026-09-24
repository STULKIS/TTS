"""Smoke tests for the Better Anotate add-on (require the ``bpy`` module).

These tests are skipped automatically when Blender's Python module is not
available (plain ``python -m unittest discover``).  Run them inside Blender's
Python, e.g.::

    pip install bpy==5.0.1   # Python 3.11
    python -m unittest tests.test_better_anotate_bpy -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import bpy  # type: ignore

    HAS_BPY = True
except ImportError:
    HAS_BPY = False


@unittest.skipUnless(HAS_BPY, "bpy (Blender Python) is not available")
class BetterAnotateBpyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import better_anotate

        cls.addon = better_anotate
        # start from a clean registration every class run
        try:
            better_anotate.unregister()
        except Exception:
            pass
        better_anotate.register()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.addon.unregister()
        except Exception:
            pass

    # -- registration ------------------------------------------------------

    def test_register_creates_handlers_and_keymap(self):
        from better_anotate import state

        # 2 handlers for VIEW_3D + 1 per other supported editor
        self.assertEqual(len(state.handlers), 6)
        self.assertIsNotNone(state.keymap)
        km, items = state.keymap
        self.assertEqual([i.idname for i in items], ["better_anotate.draw", "better_anotate.erase"])

    def test_unregister_cleans_up_and_reregisters(self):
        from better_anotate import state

        self.addon.unregister()
        self.assertEqual(len(state.handlers), 0)
        self.assertIsNone(state.keymap)
        self.assertFalse(hasattr(bpy.types.Scene, "better_anotate"))
        self.addon.register()
        self.assertEqual(len(state.handlers), 6)
        self.assertTrue(hasattr(bpy.types.Scene, "better_anotate"))

    def test_all_icons_used_by_panels_exist(self):
        import re

        src = (ROOT / "better_anotate" / "ui.py").read_text(encoding="utf-8")
        params = bpy.types.UILayout.bl_rna.functions["operator"].parameters
        icon_items = {i.identifier for i in params["icon"].enum_items}
        non_icons = {
            "VIEW_3D", "IMAGE_EDITOR", "NODE_EDITOR", "SEQUENCE_EDITOR",
            "CLIP_EDITOR", "LEFTMOUSE", "PRESS", "BETTER_ANOTATE_PT_",
            "CROSSHAIR", "WINDOW", "TOOLS", "RIGHT", "EMPTY",
        }
        tokens = set(re.findall(r'"([A-Z][A-Z_0-9]{2,})"', src))
        missing = sorted(t for t in tokens if t not in icon_items and t not in non_icons)
        self.assertEqual(missing, [], f"unknown icons in ui.py: {missing}")

    def test_panels_exist_for_every_supported_space(self):
        names = {
            cls.__name__
            for cls in bpy.types.Panel.__subclasses__()
            if cls.__name__.startswith("BETTER_ANOTATE_PT_")
        }
        self.assertEqual(
            names,
            {
                "BETTER_ANOTATE_PT_view_3d",
                "BETTER_ANOTATE_PT_image_editor",
                "BETTER_ANOTATE_PT_node_editor",
                "BETTER_ANOTATE_PT_sequence_editor",
                "BETTER_ANOTATE_PT_clip_editor",
            },
        )

    # -- data model --------------------------------------------------------

    @staticmethod
    def _settings():
        return bpy.context.scene.better_anotate

    def test_ensure_layer_creates_default_once(self):
        from better_anotate.props import ensure_layer

        settings = self._settings()
        settings.layers.clear()
        settings.strokes.clear()
        settings.active_layer = -1
        first = ensure_layer(settings)
        second = ensure_layer(settings)
        self.assertEqual(len(settings.layers), 1)
        self.assertEqual(first.uid, second.uid)
        self.assertEqual(first.name, "Notes")

    def test_append_stroke_allocates_unique_uids(self):
        from better_anotate.props import append_stroke, ensure_layer

        settings = self._settings()
        settings.strokes.clear()
        layer = ensure_layer(settings)
        uids = set()
        for i in range(5):
            stroke = append_stroke(
                settings,
                layer_uid=layer.uid,
                editor="VIEW_3D",
                tool="FREEHAND",
                color=(1, 0, 0, 1),
                thickness=3.0,
                normal=(0, 0, 1),
                points=[(0, 0, 0), (i, 1, 0)],
            )
            uids.add(stroke.uid)
        self.assertEqual(len(uids), 5)
        self.assertEqual(len(settings.strokes), 5)

    def test_layer_ops_add_move_visibility_lock_remove(self):
        settings = self._settings()
        settings.layers.clear()
        settings.strokes.clear()
        self.assertEqual(bpy.ops.better_anotate.layer_add(), {"FINISHED"})
        self.assertEqual(bpy.ops.better_anotate.layer_add(), {"FINISHED"})
        self.assertEqual(len(settings.layers), 2)
        names_before = [l.name for l in settings.layers]

        # move first layer up (towards the end of the collection)
        self.assertEqual(
            bpy.ops.better_anotate.layer_move(index=0, direction=1), {"FINISHED"}
        )
        names_after = [l.name for l in settings.layers]
        self.assertEqual(names_after, [names_before[1], names_before[0]])
        self.assertEqual(settings.active_layer, 1)

        self.assertEqual(
            bpy.ops.better_anotate.layer_visibility(index=1), {"FINISHED"}
        )
        self.assertTrue(settings.layers[1].hide)
        self.assertEqual(bpy.ops.better_anotate.layer_lock(index=1), {"FINISHED"})
        self.assertTrue(settings.layers[1].lock)

        # removing the layer also removes its strokes
        layer_uid = settings.layers[1].uid
        from better_anotate.props import append_stroke

        append_stroke(
            settings,
            layer_uid=layer_uid,
            editor="VIEW_3D",
            tool="LINE",
            color=(0, 1, 0, 1),
            thickness=2.0,
            normal=(0, 0, 1),
            points=[(0, 0, 0), (1, 1, 0)],
        )
        self.assertEqual(
            bpy.ops.better_anotate.layer_remove(index=1), {"FINISHED"}
        )
        self.assertEqual(len(settings.layers), 1)
        self.assertFalse(any(s.layer_uid == layer_uid for s in settings.strokes))

    def test_clear_layer_and_clear_all(self):
        from better_anotate.props import append_stroke, ensure_layer

        settings = self._settings()
        settings.layers.clear()
        settings.strokes.clear()
        layer = ensure_layer(settings)
        for i in range(3):
            append_stroke(
                settings,
                layer_uid=layer.uid,
                editor="VIEW_3D",
                tool="FREEHAND",
                color=(1, 1, 1, 1),
                thickness=2.0,
                normal=(0, 0, 1),
                points=[(0, 0, 0), (i, 0, 1)],
            )
        self.assertEqual(bpy.ops.better_anotate.clear_layer(), {"FINISHED"})
        self.assertEqual(len(settings.strokes), 0)
        self.assertEqual(bpy.ops.better_anotate.delete_last.poll(), False)

    def test_apply_style_recolors_active_layer(self):
        from better_anotate.props import append_stroke, ensure_layer

        settings = self._settings()
        settings.layers.clear()
        settings.strokes.clear()
        layer = ensure_layer(settings)
        for i in range(2):
            append_stroke(
                settings,
                layer_uid=layer.uid,
                editor="VIEW_3D",
                tool="FREEHAND",
                color=(0, 0, 0, 1),
                thickness=9.0,
                normal=(0, 0, 1),
                points=[(0, 0, 0), (1, 0, 0)],
            )
        settings.color = (0.2, 0.4, 0.6, 1.0)
        settings.thickness = 7.5
        self.assertEqual(bpy.ops.better_anotate.apply_style(), {"FINISHED"})
        for stroke in settings.strokes:
            self.assertAlmostEqual(stroke.color[0], 0.2, places=5)
            self.assertAlmostEqual(stroke.thickness, 7.5, places=5)

    def test_delete_last_removes_most_recent(self):
        from better_anotate.props import append_stroke, ensure_layer

        settings = self._settings()
        settings.strokes.clear()
        layer = ensure_layer(settings)
        first = append_stroke(
            settings, layer_uid=layer.uid, editor="VIEW_3D", tool="FREEHAND",
            color=(1, 1, 1, 1), thickness=2.0, normal=(0, 0, 1),
            points=[(0, 0, 0), (1, 1, 1)],
        )
        append_stroke(
            settings, layer_uid=layer.uid, editor="VIEW_3D", tool="LINE",
            color=(1, 1, 1, 1), thickness=2.0, normal=(0, 0, 1),
            points=[(2, 2, 2), (3, 3, 3)],
        )
        self.assertEqual(bpy.ops.better_anotate.delete_last(), {"FINISHED"})
        self.assertEqual(len(settings.strokes), 1)
        self.assertEqual(settings.strokes[0].uid, first.uid)

    # -- eraser snapshot/restore -----------------------------------------

    def test_snapshot_restore_roundtrip(self):
        from better_anotate.operators import _restore_stroke, _snapshot_stroke
        from better_anotate.props import append_stroke, ensure_layer

        settings = self._settings()
        settings.layers.clear()
        settings.strokes.clear()
        layer = ensure_layer(settings)
        stroke = append_stroke(
            settings, layer_uid=layer.uid, editor="VIEW_3D", tool="ARROW",
            color=(0.5, 0.1, 0.9, 1.0), thickness=6.0, normal=(0, 1, 0),
            points=[(1, 2, 3), (4, 5, 6)],
        )
        original_uid = stroke.uid
        snap = _snapshot_stroke(settings, stroke)
        settings.strokes.remove(0)
        self.assertEqual(len(settings.strokes), 0)
        _restore_stroke(settings, snap)
        self.assertEqual(len(settings.strokes), 1)
        restored = settings.strokes[0]
        self.assertEqual(restored.uid, original_uid)
        self.assertEqual(restored.tool, "ARROW")
        self.assertEqual(tuple(restored.normal), (0, 1, 0))
        self.assertEqual(len(restored.points), 2)
        self.assertAlmostEqual(restored.color[2], 0.9, places=5)

    # -- conversion ---------------------------------------------------------

    def test_convert_to_grease_pencil(self):
        from better_anotate.props import append_stroke, ensure_layer

        settings = self._settings()
        settings.layers.clear()
        settings.strokes.clear()
        # remove leftovers from previous runs
        for ob in [o for o in bpy.data.objects if o.name.startswith("Better Annotate")]:
            bpy.data.objects.remove(ob)

        layer = ensure_layer(settings)
        append_stroke(
            settings, layer_uid=layer.uid, editor="VIEW_3D", tool="FREEHAND",
            color=(1, 0, 0, 1), thickness=4.0, normal=(0, 0, 1),
            points=[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        )
        append_stroke(
            settings, layer_uid=layer.uid, editor="VIEW_3D", tool="CIRCLE",
            color=(0, 1, 0, 1), thickness=3.0, normal=(0, 0, 1),
            points=[(0, 0, 1), (2, 0, 1)],
        )
        # a 2D stroke must be skipped
        append_stroke(
            settings, layer_uid=layer.uid, editor="IMAGE_EDITOR", tool="LINE",
            color=(0, 0, 1, 1), thickness=2.0, normal=(0, 0, 1),
            points=[(5, 5, 0), (15, 5, 0)],
        )

        self.assertEqual(bpy.ops.better_anotate.convert_to_gp.poll(), True)
        self.assertEqual(bpy.ops.better_anotate.convert_to_gp(), {"FINISHED"})

        ob = bpy.data.objects.get("Better Annotate")
        self.assertIsNotNone(ob)
        gp = ob.data
        self.assertEqual(len(gp.layers), 1)
        drawing = gp.layers[0].frames[0].drawing
        # freehand (1 polyline) + circle (1 polyline) — 2D stroke skipped
        self.assertEqual(len(drawing.strokes), 2)
        freehand = drawing.strokes[0]
        self.assertEqual(len(freehand.points), 3)
        self.assertEqual(tuple(freehand.points[0].position), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(freehand.points[0].vertex_color[0], 1.0, places=5)
        circle = drawing.strokes[1]
        self.assertGreaterEqual(len(circle.points), 33)

        # conversion is disabled again once only 2D strokes remain
        settings.strokes.remove(1)  # remove circle (index 1)
        settings.strokes.remove(0)  # remove freehand
        self.assertEqual(len(settings.strokes), 1)
        self.assertEqual(bpy.ops.better_anotate.convert_to_gp.poll(), False)

        bpy.data.objects.remove(ob)

    # -- draw/erase operators are safely unavailable headless ---------------

    def test_modal_ops_poll_gracefully_without_windows(self):
        # background mode has no window/region context
        self.assertFalse(bpy.ops.better_anotate.draw.poll())
        self.assertFalse(bpy.ops.better_anotate.erase.poll())


if __name__ == "__main__":
    unittest.main()
