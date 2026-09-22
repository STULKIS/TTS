#!/usr/bin/env python3
"""Generate a self-contained HTML player for a seed-clip pack — companion to
seeds/ and tools/check_seeds.py.

Walks seeds/<char>/<lang>*.wav (+ verbatim .txt transcripts) and writes ONE html
page: a card per character, language sections in canonical order (en, zh, ja, ko),
and per clip an <audio> player, the transcript, and duration/sample-rate meta.
The page references clips by RELATIVE path, so it works opened straight from the
repo checkout — no server, no build step:

    python tools/make_showcase.py                       # seeds/ -> seeds/showcase.html
    python tools/make_showcase.py --seeds seeds --out seeds/showcase.html \
        --title "Dragon & Drake — seed voices"

Stdlib only, deterministic output (no timestamps) — safe to commit and regenerate.
Unreadable WAVs are skipped with a stderr warning; use tools/check_seeds.py for
full pack validation (30 s cap, rates, transcripts).
"""
from __future__ import annotations

import argparse
import html
import re
import string
import sys
import wave
from pathlib import Path

LANG_ORDER = {"en": 0, "zh": 1, "ja": 2, "ko": 3, "yue": 4}
DEFAULT_TITLE = "Seed voices"
_CLIP_RE = re.compile(r"^([a-z]+)(\d+)?$")

PAGE = string.Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
       max-width: 54rem; padding: 0 1rem; line-height: 1.55; }
.sub { opacity: .7; margin-top: .25rem; }
.card { border: 1px solid rgba(128,128,128,.35); border-radius: .75rem;
        padding: .5rem 1.25rem 1rem; margin: 1.5rem 0; }
.card h2 { margin: .5rem 0; }
.lang h3 { margin: 1.25rem 0 .25rem; font-size: .85rem; text-transform: uppercase;
           letter-spacing: .08em; opacity: .6; }
.clip { padding: .5rem 0; border-top: 1px dashed rgba(128,128,128,.3); }
.lang .clip:first-of-type { border-top: 0; }
.text { margin: .5rem 0 .25rem; }
.meta { opacity: .55; font-size: .8rem; }
audio { width: 100%; }
</style>
</head>
<body>
<h1>$title</h1>
<p class="sub">$summary</p>
$cards
</body>
</html>
""")


def clip_sort_key(wav: Path) -> tuple[int, int]:
    """Order clips as en, en2, … en9, zh, zh2, … (language blocks, then number)."""
    m = _CLIP_RE.match(wav.stem)
    if m:
        return (LANG_ORDER.get(m.group(1), 99), int(m.group(2) or 0))
    return (99, 0)


def clip_info(wav: Path) -> dict | None:
    try:
        with wave.open(str(wav), "rb") as w:
            rate, nch, frames = w.getframerate(), w.getnchannels(), w.getnframes()
    except (wave.Error, EOFError, OSError):
        return None
    return {"rate": rate, "channels": nch, "seconds": frames / rate if rate else 0.0}


def collect(seeds: Path) -> list[dict]:
    """chars -> {id, clips: [{file, lang, text, rate, channels, seconds}]}"""
    chars: list[dict] = []
    for char_dir in sorted(p for p in seeds.iterdir() if p.is_dir()):
        wavs = sorted(char_dir.glob("*.wav"), key=clip_sort_key)
        clips: list[dict] = []
        for wav in wavs:
            info = clip_info(wav)
            if info is None:
                print(f"[warn] {wav}: unreadable WAV — skipped", file=sys.stderr)
                continue
            m = _CLIP_RE.match(wav.stem)
            txt = wav.with_suffix(".txt")
            text = (txt.read_text(encoding="utf-8", errors="replace").strip()
                    if txt.exists() else "")
            clips.append({
                "file": f"{char_dir.name}/{wav.name}",
                "lang": m.group(1) if m else wav.stem,
                "text": text,
                **info,
            })
        if clips:
            chars.append({"id": char_dir.name, "clips": clips})
    return chars


def render_page(chars: list[dict], title: str) -> str:
    cards: list[str] = []
    for char in chars:
        by_lang: dict[str, list[dict]] = {}
        for clip in char["clips"]:
            by_lang.setdefault(clip["lang"], []).append(clip)
        sections: list[str] = []
        for lang in sorted(by_lang, key=lambda l: LANG_ORDER.get(l, 99)):
            clips = by_lang[lang]
            rows = []
            for n, clip in enumerate(clips, 1):
                text = html.escape(clip["text"]) or "(no transcript)"
                rows.append(
                    '<div class="clip">'
                    f'<audio controls preload="none" '
                    f'src="{html.escape(clip["file"], quote=True)}"></audio>'
                    f'<p class="text">{text}</p>'
                    f'<span class="meta">{n}/{len(clips)} · {clip["seconds"]:.2f} s · '
                    f'{clip["rate"]} Hz</span>'
                    "</div>"
                )
            sections.append(f'<div class="lang"><h3>{html.escape(lang)}</h3>{"".join(rows)}</div>')
        cards.append(
            f'<section class="card"><h2>{html.escape(char["id"])}</h2>{"".join(sections)}</section>'
        )
    total = sum(len(c["clips"]) for c in chars)
    summary = (f"{len(chars)} character(s) · {total} clips · "
               f"generated by tools/make_showcase.py")
    return PAGE.substitute(title=html.escape(title), summary=summary, cards="".join(cards))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=Path, default=Path("seeds"))
    ap.add_argument("--out", type=Path, default=None,
                    help="output html (default: <seeds>/showcase.html)")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    args = ap.parse_args()

    if not args.seeds.is_dir():
        raise SystemExit(f"[abort] seeds dir not found: {args.seeds}")
    chars = collect(args.seeds)
    if not chars:
        raise SystemExit(f"[abort] no readable .wav clips under {args.seeds}")

    out = args.out or (args.seeds / "showcase.html")
    out.write_text(render_page(chars, args.title), encoding="utf-8")
    total = sum(len(c["clips"]) for c in chars)
    print(f"[ok] wrote {out} ({len(chars)} character(s), {total} clip(s))")


if __name__ == "__main__":
    main()
