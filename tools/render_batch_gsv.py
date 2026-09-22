#!/usr/bin/env python3
"""Batch character-line renderer for GPT-SoVITS v2Pro/v2ProPlus — companion to local-tts-guide.md.

Renders the SAME manifest format as render_batch.py (CosyVoice path), but through a running
GPT-SoVITS api_v2 server, hot-swapping the per-character model pair between characters:

    id<TAB>char_id<TAB>lang<TAB>text        (UTF-8, '#' = comment)

Layout expected:
    seeds/<char_id>/<lang>.wav      3-10 s reference clip (reuses the CosyVoice seed pack)
    seeds/<char_id>/<lang>.txt      exact transcript of that clip (prompt_text)
    gsv-models/<char_id>/gpt.ckpt               fine-tuned GPT weights for the character
    gsv-models/<char_id>/sovits.safetensors     fine-tuned SoVITS weights (.pth also accepted)

Server setup (see local-tts-guide.md §7):
    python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
    with GPT_SoVITS/configs/tts_infer.yaml set to: device: cpu / is_half: false

Notes:
  * stdlib only — no torch needed where this script runs
  * one RNG seed per character (same reproducibility convention as render_batch.py)
  * reference paths are resolved to absolute paths; the api_v2 server reads them server-side
  * --dry-run validates manifest + files without touching the server

Not executed end-to-end here (no GPT-SoVITS weights in this sandbox) — endpoints and params
verified against api_v2.py on RVC-Boss/GPT-SoVITS@main 2026-09 (/tts, /set_gpt_weights,
/set_sovits_weights, /control; text, text_lang, ref_audio_path, prompt_text, prompt_lang,
seed, speed_factor, repetition_penalty, streaming_mode, media_type).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

LANGS = {"en", "zh", "ja", "ko", "yue"}  # GPT-SoVITS language codes (README language dictionary)


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if raw.strip().startswith("#") or not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise SystemExit(f"{path}:{lineno}: expected 4 tab-separated fields, got {len(parts)}")
            line_id, char_id, lang, text = parts
            if lang not in LANGS:
                raise SystemExit(f"{path}:{lineno}: lang {lang!r} not in {sorted(LANGS)}")
            rows.append({"id": line_id, "char": char_id, "lang": lang, "text": text})
    return rows


def find_model(models_dir: Path, char_id: str, names: tuple[str, ...]) -> Path | None:
    d = models_dir / char_id
    if not d.is_dir():
        return None
    for n in names:
        p = d / n
        if p.exists():
            return p
    return None


class Api:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, **params) -> str:
        url = f"{self.base}/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode({k: str(v) for k, v in params.items()})
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def tts(self, out_path: Path, **params) -> None:
        url = f"{self.base}/tts?" + urllib.parse.urlencode({k: str(v) for k, v in params.items()})
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read()
        if "json" in ctype:  # api_v2 returns JSON error bodies with HTTP 400; 200+json = still wrong
            raise RuntimeError(f"server returned JSON instead of audio: {body[:300]!r}")
        out_path.write_bytes(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", default="http://127.0.0.1:9880", help="api_v2.py base URL")
    ap.add_argument("--lines", type=Path, default=Path("lines.tsv"))
    ap.add_argument("--seeds", type=Path, default=Path("seeds"))
    ap.add_argument("--models_dir", type=Path, default=Path("gsv-models"),
                    help="dir holding <char_id>/gpt.ckpt + <char_id>/sovits.safetensors|.pth")
    ap.add_argument("--out", type=Path, default=Path("wavs-gsv"))
    ap.add_argument("--seed", type=int, default=42, help="base RNG seed, pinned per character")
    ap.add_argument("--speed", type=float, default=1.0, help="speed_factor passed to /tts")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-request timeout in seconds (CPU rendering is slow)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate manifest, seeds and model files; do not call the server")
    args = ap.parse_args()

    rows = read_manifest(args.lines)
    by_char: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_char[r["char"]].append(r)

    # 1) resolve per-character assets up front
    assets: dict[str, dict] = {}
    problems = 0
    for char_id in by_char:
        wav_txt = {}
        for lang in {r["lang"] for r in by_char[char_id]}:
            wav = args.seeds / char_id / f"{lang}.wav"
            txt = args.seeds / char_id / f"{lang}.txt"
            if wav.exists() and txt.exists():
                wav_txt[lang] = (wav.resolve(), txt.read_text(encoding="utf-8").strip())
            else:
                print(f"[warn] {char_id}/{lang}: missing {wav}{'/txt' if wav.exists() else ''} "
                      f"— lines for this language will be skipped")
        gpt = find_model(args.models_dir, char_id, ("gpt.ckpt",))
        sovits = find_model(args.models_dir, char_id, ("sovits.safetensors", "sovits.pth"))
        if gpt is None or sovits is None:
            print(f"[error] {char_id}: need {args.models_dir}/{char_id}/gpt.ckpt and "
                  f"sovits.safetensors|.pth — train per local-tts-guide.md §4-6")
            problems += 1
        if not wav_txt:
            problems += 1
        assets[char_id] = {"gpt": gpt, "sovits": sovits, "refs": wav_txt}
    if problems:
        raise SystemExit(f"[abort] {problems} character(s) missing assets (dry-run={args.dry_run})")

    total = sum(len(v) for v in by_char.values())
    print(f"[info] manifest={args.lines} lines={total} characters={len(by_char)}")
    if args.dry_run:
        print("[dry-run] manifest, seeds and model files OK — nothing rendered")
        return

    api = Api(args.api, args.timeout)
    try:  # cheap liveness check: a bare GET /tts must answer (4xx = up, no route = down)
        urllib.request.urlopen(f"{args.api}/tts", timeout=5)
    except urllib.error.HTTPError:
        pass  # any HTTP status means the server answered
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SystemExit(
            f"[abort] cannot reach GPT-SoVITS api_v2 at {args.api} ({e}).\n"
            f"        Start it first (repo root): python api_v2.py -a 127.0.0.1 -p 9880 "
            f"-c GPT_SoVITS/configs/tts_infer.yaml")

    # 2) render, hot-swapping the model pair when the character changes
    loaded: dict[str, bool] = {}
    rendered, skipped = 0, 0
    for char_id, char_rows in by_char.items():
        a = assets[char_id]
        if char_id not in loaded:
            api._get("set_gpt_weights", weights_path=str(a["gpt"]))
            api._get("set_sovits_weights", weights_path=str(a["sovits"]))
            loaded[char_id] = True
            print(f"[ok] loaded character model {char_id} ({a['gpt'].name} + {a['sovits'].name})")
        seed = args.seed + sum(ord(c) for c in char_id) % 1000  # stable per-character seed
        for r in char_rows:
            out = args.out / char_id / f"{r['id']}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                print(f"[skip] {out} exists")
                skipped += 1
                continue
            ref = a["refs"].get(r["lang"])
            if ref is None:
                print(f"[skip] {r['id']}: no reference clip for {char_id}/{r['lang']}")
                skipped += 1
                continue
            wav, prompt_text = ref
            t0 = time.time()
            try:
                api.tts(out, text=r["text"], text_lang=r["lang"],
                        ref_audio_path=str(wav), prompt_text=prompt_text,
                        prompt_lang=r["lang"], seed=seed, speed_factor=args.speed,
                        media_type="wav", streaming_mode="false")
            except (urllib.error.HTTPError, RuntimeError) as e:
                print(f"[error] {r['id']}: {e}")
                skipped += 1
                continue
            dur = time.time() - t0
            rendered += 1
            print(f"[wav] {out}  ({dur:.1f}s)  '{r['text'][:36]}'")
    print(f"[done] rendered={rendered} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()
