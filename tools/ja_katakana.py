#!/usr/bin/env python3
"""Convert Japanese lines to spaced katakana for CosyVoice3 — companion to
my-tts-plan.corrected.md ('Japanese: upstream recommends spaced KATAKANA') and
tools/render_batch.py.

CosyVoice3's known-weak path is raw kanji input for Japanese; upstream recommends feeding
spaced katakana (e.g. レキシ テキ セカイ ニ オイ テ ワ、…). This tool rewrites the `ja`
rows of a manifest — or plain text — into that form using pykakasi:

    pip install pykakasi
    python tools/ja_katakana.py --manifest lines.tsv                # prints converted manifest
    python tools/ja_katakana.py --manifest lines.tsv --output lines.kana.tsv
    python tools/ja_katakana.py --text この世界の運命は、まだ決まっていない。

Manifest mode keeps every non-`ja` row byte-identical (comments and blank lines included),
so the output is a drop-in replacement for tools/render_batch.py.
GPT-SoVITS does NOT need this — it handles ordinary Japanese text (see local-tts-guide.md).

Punctuation is glued to the preceding token (… ワ、マダ … スギナイ。), matching the
repo's example lines in tools/lines.sample.tsv. Caveat: pykakasi is a heuristic converter —
particle readings are not disambiguated (は → ハ, not ワ) and token boundaries may differ from
hand-written kana; treat the output as a draft you can hand-tweak before rendering.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path


def _is_punct(tok: str) -> bool:
    return all(unicodedata.category(c).startswith(("P", "S")) for c in tok)


def to_spaced_katakana(text: str, kakasi) -> str:
    out: list[str] = []
    for item in kakasi.convert(text):
        tok = item["kana"]
        if not tok or not tok.strip():
            continue  # whitespace input tokens are separators, not content
        if out and _is_punct(tok):
            out[-1] += tok  # glue 、。!? etc. to the preceding word
        else:
            out.append(tok)
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", type=Path, help="TSV manifest (id/char/lang/text)")
    g.add_argument("--text", help="a single Japanese string")
    ap.add_argument("--output", type=Path, help="write here instead of stdout")
    args = ap.parse_args()

    try:
        import pykakasi
    except ImportError:
        raise SystemExit("[abort] pykakasi is required: pip install pykakasi")
    kakasi = pykakasi.kakasi()

    lines_out: list[str] = []
    converted = 0
    if args.text is not None:
        print(to_spaced_katakana(args.text, kakasi))
        return

    with args.manifest.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if raw.strip().startswith("#") or not raw.strip():
                lines_out.append(raw.rstrip("\n"))
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise SystemExit(f"{args.manifest}:{lineno}: expected 4 tab-separated fields, "
                                 f"got {len(parts)}")
            line_id, char_id, lang, text = parts
            if lang == "ja":
                text = to_spaced_katakana(text, kakasi)
                converted += 1
            lines_out.append(f"{line_id}\t{char_id}\t{lang}\t{text}")

    body = "\n".join(lines_out) + "\n"
    if args.output:
        args.output.write_text(body, encoding="utf-8")
        print(f"[ok] converted {converted} ja line(s) -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
