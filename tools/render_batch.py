#!/usr/bin/env python3
"""Batch character-line renderer for CosyVoice2/3 on CPU — companion to my-tts-plan.corrected.md.

Layout expected (relative to CosyVoice repo root):
    seeds/<char_id>/<lang>.wav     3-10 s reference clip for that character+language  (<=30 s HARD LIMIT)
    seeds/<char_id>/<lang>.txt     exact transcript of that clip (required)
    lines.tsv                      id<TAB>char_id<TAB>lang<TAB>text        (UTF-8, '#' = comment)

What it does:
  * registers every seed clip ONCE via add_zero_shot_spk() so the speech-tokenizer /
    campplus pass is not repeated per line (the big CPU win), then renders all lines
  * pins one RNG seed per character so a 200-line batch does not drift between takes
  * optional --instruct for style (dialect/emotion/rate) — uses inference_instruct2,
    which needs the clip present at call time
  * writes wavs/<char_id>/<id>.wav at 24 kHz mono (CV3) / 24 kHz (CV2)
  * --dry-run validates the manifest + seed clips and prints the plan (speakers,
    missing clips, CPU time estimate) WITHOUT importing torch — run it first

Notes for Japanese: upstream recommends spaced katakana for CV3. Pre-convert the text
rather than feeding raw kanji — tools/ja_katakana.py rewrites the `ja` rows of a manifest
(needs pykakasi). Validate the seed pack itself with tools/check_seeds.py.

Not executed here (no CUDA/model weights in this sandbox) — API signatures verified against
CosyVoice@main 2026-09: inference_zero_shot(tts_text, prompt_text, prompt_wav, zero_shot_spk_id,
stream, speed), add_zero_shot_spk(prompt_text, prompt_wav, zero_shot_spk_id), save_spkinfo().
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append("third_party/Matcha-TTS")  # noqa: E402  (CosyVoice expects this)

LANGS = {"en", "zh", "ja", "ko", "yue"}


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


def dry_run_plan(rows: list[dict], seeds: Path, instruct: str) -> dict:
    """What a real run would do, minus torch: speakers, missing clips, time estimate.

    The estimate assumes ~5 s of audio per line at RTF 2–5 on CPU (see the plan's
    'What to expect on CPU' section).
    """
    speakers: dict[str, dict] = {}
    missing: list[dict] = []
    for r in rows:
        spk_id = f"{r['char']}__{r['lang']}"
        wav = seeds / r["char"] / f"{r['lang']}.wav"
        txt = wav.with_suffix(".txt")
        if wav.exists() and txt.exists():
            spk = speakers.setdefault(spk_id, {"char": r["char"], "lang": r["lang"], "lines": 0})
            spk["lines"] += 1
        else:
            absent = [str(p) for p in (wav, txt) if not p.exists()]
            item = dict(r)
            item["missing"] = " + ".join(absent)
            missing.append(item)
    per_line_s = 5.0
    return {
        "rows": len(rows),
        "speakers": [speakers[k] for k in sorted(speakers)],
        "missing": missing,
        "est_low_s": len(rows) * per_line_s * 2.0,
        "est_high_s": len(rows) * per_line_s * 5.0,
        "instruct": bool(instruct),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="pretrained_models/Fun-CosyVoice3-0.5B")
    ap.add_argument("--lines", type=Path, default=Path("lines.tsv"))
    ap.add_argument("--seeds", type=Path, default=Path("seeds"))
    ap.add_argument("--out", type=Path, default=Path("wavs"))
    ap.add_argument("--seed", type=int, default=42, help="RNG seed, pinned per character")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--instruct", default="",
                    help="style string for inference_instruct2, e.g. "
                         "'You are a helpful assistant. 请用四川话表达。<|endofprompt|>'")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all)")
    ap.add_argument("--save_spkinfo", action="store_true",
                    help="persist registered speakers into the model dir (spk2info.pt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate manifest + seed clips and print the plan; no torch, "
                         "no rendering (exit 1 if any line lacks a seed clip and no "
                         "--instruct fallback is given)")
    args = ap.parse_args()

    rows = read_manifest(args.lines)

    by_char = defaultdict(list)
    for r in rows:
        by_char[r["char"]].append(r)

    if args.dry_run:
        plan = dry_run_plan(rows, args.seeds, args.instruct)
        print(f"[dry-run] {args.lines}: {plan['rows']} line(s)")
        if plan["speakers"]:
            print("[dry-run] speakers to register (clip encoded ONCE per char/lang):")
            for spk in plan["speakers"]:
                wav = args.seeds / spk["char"] / f"{spk['lang']}.wav"
                print(f"  {spk['char']}__{spk['lang']}  ({spk['lines']} line(s)) <- {wav}")
        else:
            print("[dry-run] no speakers can be registered — no seed clips match the manifest")
        if plan["missing"]:
            for item in plan["missing"]:
                print(f"[error] {item['id']}: missing {item['missing']}")
            if plan["instruct"]:
                print(f"[warn] {len(plan['missing'])} line(s) will fall back to --instruct + a raw clip")
            else:
                print(f"[fail] {len(plan['missing'])} line(s) lack a seed clip — add seeds/<char>/<lang>.wav + .txt")
        renderable = plan["rows"] - len(plan["missing"])
        if renderable > 0:
            print(f"[dry-run] estimate: {renderable} line(s) ≈ "
                  f"{plan['est_low_s'] / 60:.0f}-{plan['est_high_s'] / 60:.0f} min "
                  f"at RTF 2-5 (5 s of audio per line, CPU)")
        if plan["missing"] and not plan["instruct"]:
            raise SystemExit(1)
        print("[ok] ready to render — re-run without --dry-run")
        return

    # Heavy deps only for real rendering, so --dry-run works anywhere.
    import torch  # noqa: E402
    import torchaudio  # noqa: E402
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
    from cosyvoice.utils.common import set_all_random_seed  # noqa: E402

    if args.threads:
        torch.set_num_threads(args.threads)
    if torch.cuda.is_available() is False:
        print("[info] CPU mode: expect RTF 2-5, i.e. tens of seconds per line")

    cv = AutoModel(model_dir=args.model_dir)
    print(f"[info] model={args.model_dir} sample_rate={cv.sample_rate} lines={len(rows)}")

    # 1) register every character's seed clip once
    registered: dict[str, str] = {}
    missing: list[dict] = []
    for r in rows:
        spk_id = f"{r['char']}__{r['lang']}"
        if spk_id in registered:
            continue
        wav = args.seeds / r["char"] / f"{r['lang']}.wav"
        txt = args.seeds / r["char"] / f"{r['lang']}.txt"
        if wav.exists() and txt.exists():
            dur = torchaudio.info(wav).num_frames / max(1, torchaudio.info(wav).sample_rate)
            if dur > 30:
                raise SystemExit(f"{wav}: {dur:.1f}s > 30s hard limit — trim the clip")
            if dur < 2:
                print(f"[warn] {wav}: only {dur:.1f}s — short references clone poorly")
            cv.add_zero_shot_spk(txt.read_text(encoding="utf-8").strip(), str(wav), spk_id)
            registered[spk_id] = spk_id
            print(f"[ok] registered speaker {spk_id}")
        else:
            missing.append(r)
    if missing:
        note = "use inference_instruct2 with --instruct" if args.instruct else \
               "be SKIPPED (fix: add a seed clip at seeds/<char>/<lang>.wav + .txt)"
        print(f"[warn] {len(missing)} line(s) have no seed clip; they will {note}")
    if args.save_spkinfo and registered:
        cv.save_spkinfo()
        print("[ok] spk2info written to model dir — future runs skip re-registration")

    # 2) render
    total, skipped = 0, 0
    for char_id, char_rows in by_char.items():
        set_all_random_seed(args.seed)  # per-character reproducibility
        for r in char_rows:
            spk_id = f"{r['char']}__{r['lang']}"
            out = args.out / char_id / f"{r['id']}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                print(f"[skip] {out} exists")
                skipped += 1
                continue
            if spk_id in registered:
                gen = cv.inference_zero_shot(r["text"], "", "", zero_shot_spk_id=spk_id,
                                             stream=False, speed=args.speed)
            elif args.instruct:  # fall back to a raw reference + instruction
                wav = sorted(Path(args.seeds / char_id).glob("*.wav"))
                if not wav:
                    print(f"[skip] {r['id']}: no seed clip and no instruct fallback available")
                    skipped += 1
                    continue
                gen = cv.inference_instruct2(r["text"], args.instruct, str(wav[0]),
                                             stream=False, speed=args.speed)
            else:
                skipped += 1
                continue
            chunks = list(gen)  # one entry per text chunk the splitter produced
            if not chunks:
                print(f"[warn] {r['id']}: model returned no audio")
                skipped += 1
                continue
            audio = torch.cat([c["tts_speech"] for c in chunks], dim=-1)
            torchaudio.save(str(out), audio, cv.sample_rate)
            secs = audio.shape[-1] / cv.sample_rate
            total += 1
            print(f"[wav] {out}  {secs:.2f}s  '{r['text'][:36]}'")
    print(f"[done] rendered={total} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()
