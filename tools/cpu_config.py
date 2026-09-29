#!/usr/bin/env python3
"""Set GPT-SoVITS to CPU mode — run this BEFORE the first launch on a
machine without an NVIDIA GPU.

The default tts_infer.yaml assumes CUDA (`device: cuda:0`, `is_half: true`).
On a CPU-only box that breaks the api_v2 / type_ui path (half precision is
not a CPU mode). This script rewrites the config idempotently:

    python tools/cpu_config.py --gsv-root D:\\GSV

Effect per section:  `device: cuda:*` -> `cpu`  and  `is_half: true` -> `false`
The original is backed up once to `tts_infer.yaml.orig`. Everything else is
untouched (line-based edit, no yaml dependency). Stdlib only.

The WebUI (go-webui.bat) has its own Config tab that does the same thing —
run this script for the API / type_ui path, or set it in the tab. Both are
safe to mix.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_DEVICE = re.compile(r"^(\s*device:\s*)\S+")
_HALF = re.compile(r"^(\s*is_half:\s*)true\b")


def convert_lines(lines: list[str]) -> tuple[list[str], int]:
    out: list[str] = []
    changed = 0
    for line in lines:
        new = _DEVICE.sub(r"\g<1>cpu", line)
        if new != line:
            changed += 1
        new2 = _HALF.sub(r"\g<1>false", new)
        if new2 != new:
            changed += 1
        out.append(new2)
    return out, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gsv-root", type=Path, default=Path("."),
                    help="GPT-SoVITS repo root (the folder containing GPT_SoVITS/)")
    args = ap.parse_args()

    config = args.gsv_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    if not config.is_file():
        raise SystemExit(
            f"[abort] not found: {config}\n"
            f"        pass --gsv-root pointing at the GPT-SoVITS folder "
            f"(the one containing GPT_SoVITS/).")

    text = config.read_text(encoding="utf-8")
    new_text, changed = convert_lines(text.splitlines(True))
    new_text = "".join(new_text)

    if new_text != text:
        backup = config.with_suffix(".yaml.orig")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
            print(f"[ok] backed up original -> {backup}")
        config.write_text(new_text, encoding="utf-8")
        print(f"[ok] {config}: {changed} value(s) set to CPU mode "
              f"(device: cpu, is_half: false)")
    else:
        print(f"[ok] {config} already in CPU mode — nothing to change")


if __name__ == "__main__":
    main()
