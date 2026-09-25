#!/usr/bin/env python3
"""Turn a plain-text dialogue script into a render manifest (TSV) — companion to
tools/render_batch.py and tools/render_batch_gsv.py.

Write a script the natural way, one line per speaker:

    # a scene
    dragon: The fate of this world was never already written.
    drake: You kept me waiting. That was your first mistake.
    dragon: 这个世界的命运，还没有定数。
    drake [ja]: 待たせたな。君の最初の失敗だ。

and get the 4-column manifest (id, char, lang, text) the renderers consume:

    python tools/dialogue2tsv.py --script scene.txt               # manifest to stdout
    python tools/dialogue2tsv.py --script scene.txt --output lines.tsv

  * ids are auto-numbered per character: dragon-01, dragon-02, drake-01, …
  * lang is inferred from the text (kana → ja, Hangul → ko, Han without kana → zh,
    anything else → en); override per line with `char [xx]: …`
  * `#` comments and blank lines are ignored; full-width colons (：) are accepted
  * for the CosyVoice3 path, pipe ja output through tools/ja_katakana.py

Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LANGS = {"en", "zh", "ja", "ko", "yue"}
LINE_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*(?:\[([A-Za-z]+)\])?\s*[:：]\s*(.+)$")


def infer_lang(text: str) -> str:
    """Heuristic from Unicode ranges: kana → ja, Hangul → ko, Han-only → zh, else en.

    Kana/Hangul are checked over the WHOLE text first, because Japanese text also
    contains Han — a line with any kana is ja even if it is mostly kanji.
    """
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:  # kana, halfwidth forms
            return "ja"
        if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF:  # Hangul syllables/jamo
            return "ko"
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:  # Han
            return "zh"
    return "en"


def parse_line(line_no: int, raw: str) -> tuple[str, str, str]:
    m = LINE_RE.match(raw.strip())
    if not m:
        raise SystemExit(f"{line_no}: expected '<char>[:lang]: <text>', got {raw!r}")
    char_id, bracket, text = m.group(1), m.group(2), m.group(3).strip()
    lang = bracket.lower() if bracket else infer_lang(text)
    if lang not in LANGS:
        raise SystemExit(f"{line_no}: lang {lang!r} not in {sorted(LANGS)}")
    return char_id, lang, text


def script_to_rows(text: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        rows.append(parse_line(line_no, raw))
    return rows


def to_tsv(rows: list[tuple[str, str, str]]) -> str:
    counters: dict[str, int] = {}
    out: list[str] = []
    for char_id, lang, text in rows:
        counters[char_id] = counters.get(char_id, 0) + 1
        out.append(f"{char_id}-{counters[char_id]:02d}\t{char_id}\t{lang}\t{text}")
    return "\n".join(out) + ("\n" if out else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--script", type=Path, required=True,
                    help="plain text: one '<char>: <line>' per speaker")
    ap.add_argument("--output", type=Path, default=None,
                    help="write the manifest here (default: stdout)")
    args = ap.parse_args()

    rows = script_to_rows(args.script.read_text(encoding="utf-8"))
    body = to_tsv(rows)
    if args.output:
        args.output.write_text(body, encoding="utf-8")
        print(f"[ok] {len(rows)} line(s) -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
