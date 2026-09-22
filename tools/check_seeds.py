#!/usr/bin/env python3
"""Validate a seed-clip pack against the verified prompt-clip rules — companion to
tts-plan-review.md ('Prompt-clip rules (verified in cosyvoice/cli/frontend.py)') and
local-tts-guide.md §8.

Rules enforced (CosyVoice2/3; GPT-SoVITS refs live in the same envelope):
  * readable PCM WAV                                   -> error otherwise
  * duration <= 30 s                                   -> ERROR (frontend asserts:
                                                          'assert speech.shape[1] / 16000 <= 30')
  * duration in 3-10 s sweet spot                      -> warnings outside (<2 s clones poorly,
                                                          >10 s gains nothing)
  * sample rate >= 16 kHz                              -> ERROR below (tokens/embedding path);
                                                          <24 kHz -> warning (prompt features are
                                                          also loaded at 24 kHz; export 24 kHz+)
  * mono                                               -> warning if multi-channel
  * transcript <clip>.txt must exist, be non-empty     -> ERROR otherwise

Usage:
    python tools/check_seeds.py                          # scan seeds/
    python tools/check_seeds.py --seeds seeds --manifest tools/lines.sample.tsv
    python tools/check_seeds.py --seeds seeds --require-lang dragon zh   # spot checks

Exit status: 0 = pack usable (warnings allowed), 1 = at least one error.
Stdlib only. Pair with tools/render_batch.py / tools/render_batch_gsv.py.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

LANGS = {"en", "zh", "ja", "ko", "yue"}


def check_clip(wav: Path, txt: Path | None, errors: list[str], warnings: list[str]) -> None:
    try:
        with wave.open(str(wav), "rb") as w:
            rate, nch, frames = w.getframerate(), w.getnchannels(), w.getnframes()
    except (wave.Error, EOFError, OSError) as e:
        errors.append(f"{wav}: not a readable PCM WAV ({e})")
        return
    dur = frames / rate if rate else 0.0

    if dur > 30:
        errors.append(f"{wav}: {dur:.1f}s > 30s HARD LIMIT — CosyVoice frontend asserts; trim it")
    elif dur < 2:
        warnings.append(f"{wav}: only {dur:.1f}s — clips under ~2s clone poorly")
    elif dur > 10:
        warnings.append(f"{wav}: {dur:.1f}s — usable, but the sweet spot is 3-10s")

    if rate < 16000:
        errors.append(f"{wav}: {rate} Hz < 16 kHz — speech-tokenizer path needs >= 16 kHz")
    elif rate < 24000:
        warnings.append(f"{wav}: {rate} Hz — prompt features also load at 24 kHz; export >= 24 kHz")
    if nch != 1:
        warnings.append(f"{wav}: {nch} channels — mono preferred")

    if txt is None or not txt.exists():
        errors.append(f"{wav}: transcript missing ({(txt or wav.with_suffix('.txt'))}) — "
                      f"the verbatim text is mandatory")
    elif not txt.read_text(encoding="utf-8", errors="replace").strip():
        errors.append(f"{txt}: transcript is empty")


def read_manifest_langs(path: Path) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if raw.strip().startswith("#") or not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise SystemExit(f"{path}:{lineno}: expected 4 tab-separated fields, got {len(parts)}")
            _, char_id, lang, _ = parts
            if lang not in LANGS:
                raise SystemExit(f"{path}:{lineno}: lang {lang!r} not in {sorted(LANGS)}")
            used.setdefault(char_id, set()).add(lang)
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=Path, default=Path("seeds"))
    ap.add_argument("--manifest", type=Path, default=None,
                    help="TSV manifest (id/char/lang/text); every char/lang it uses must have "
                         "a validated seed clip")
    args = ap.parse_args()

    if not args.seeds.is_dir():
        raise SystemExit(f"[abort] seeds dir not found: {args.seeds}")

    used = read_manifest_langs(args.manifest) if args.manifest else None
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    for char_dir in sorted(p for p in args.seeds.iterdir() if p.is_dir()):
        wavs = sorted(char_dir.glob("*.wav"))
        if not wavs:
            warnings.append(f"{char_dir}/: no .wav clips")
            continue
        for wav in wavs:
            txt = wav.with_suffix(".txt")
            check_clip(wav, txt, errors, warnings)
            checked += 1

    if used is not None:
        for char_id, langs in sorted(used.items()):
            for lang in sorted(langs):
                wav = args.seeds / char_id / f"{lang}.wav"
                if not wav.exists():
                    errors.append(f"manifest needs {wav} — missing (add a 3-10s clip + .txt)")

    for w in warnings:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[error] {e}")
    print(f"[{'FAIL' if errors else 'ok'}] checked {checked} clip(s) under {args.seeds}: "
          f"{len(errors)} error(s), {len(warnings)} warning(s)")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
