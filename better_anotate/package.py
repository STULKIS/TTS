#!/usr/bin/env python3
"""Build release archives for Better Anotate.

Creates two zips in ``dist/``:

* ``better_anotate-<version>.zip``       — extension format (Blender 4.2+),
  files at the archive root next to ``blender_manifest.toml``.
* ``better_anotate-legacy-<version>.zip`` — classic add-on format (folder
  wrapper + ``bl_info``), for *Install from Disk* as a legacy add-on.

Usage:  python3 better_anotate/package.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DIST = REPO / "dist"

FILES = (
    "__init__.py",
    "blender_manifest.toml",
    "props.py",
    "state.py",
    "math_utils.py",
    "gp_convert.py",
    "overlay.py",
    "operators.py",
    "ui.py",
    "keymap.py",
    "README.md",
)


def read_version() -> str:
    text = (HERE / "blender_manifest.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("version not found in blender_manifest.toml")
    return match.group(1)


def main() -> None:
    version = read_version()
    DIST.mkdir(exist_ok=True)

    ext_zip = DIST / f"better_anotate-{version}.zip"
    with zipfile.ZipFile(ext_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            path = HERE / name
            if path.exists():
                zf.write(path, name)  # extension zips: files at archive root

    legacy_zip = DIST / f"better_anotate-legacy-{version}.zip"
    with zipfile.ZipFile(legacy_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            path = HERE / name
            if path.exists():
                zf.write(path, f"better_anotate/{name}")  # folder wrapper

    for z in (ext_zip, legacy_zip):
        print(f"built {z.relative_to(REPO)} ({z.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
