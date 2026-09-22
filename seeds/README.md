# seeds/ — original character voice packs

One directory per character. Each character ships **5 lines per language** (en · zh · ja · ko),
generated in-chat as an *original* designed voice (no VA cloning — see `my-tts-plan.corrected.md`
Part 2, "The seed trick").

## Layout

```
seeds/<char>/<lang>.wav    # line 1 — the canonical prompt clip (register as the named speaker)
seeds/<char>/<lang>.txt    # verbatim transcript (mandatory, CosyVoice prompt rule)
seeds/<char>/<lang2..5>.wav/.txt   # extra lines (same voice) — training fodder
```

`<lang>` ∈ {en, zh, ja, ko}. `tools/check_seeds.py` validates every clip; the cross-check with a
manifest requires at least `<lang>.wav` per char/lang the manifest uses.

## Clip rules (enforced by tools/check_seeds.py)

* PCM WAV, **mono**, **≥ 24 kHz** (16 kHz is the hard floor; the 24 kHz path loads prompt feats)
* **3–10 s** sweet spot (30 s is a hard frontend assert; <2 s clones poorly)
* **verbatim** transcript next to every clip; clean audio, no music, no silence padding,
  no lossy round-trips (export straight from the generator as WAV)

## Characters (voice briefs — for re-auditioning in a fresh chat session)

| char | Brief | Lines |
|---|---|---|
| `dragon` | Ancient dragon spirit. Female, low, warm, unhurried — slow and timeless, never loud. | 5 × en · zh · ja · ko |
| `drake` | Sardonic wit. Male, dry, cutting, impatient — speaks in verdicts, never in small talk. | 5 × en · zh · ja · ko |

Session notes: voices were picked by audition (English, `use_case: characters`), then the same
speakers were reused for zh/ja/ko via the per-call language override.

## Use

```bash
# 1) validate the pack
python tools/check_seeds.py --seeds seeds --manifest tools/lines.sample.tsv

# 2) CosyVoice path — register <lang>.wav as a named speaker once, then batch-render
python tools/render_batch.py --lines lines.tsv --model_dir pretrained_models/Fun-CosyVoice3-0.5B
#    (ja lines: pre-convert to spaced katakana — tools/ja_katakana.py)

# 3) GPT-SoVITS path — fine-tune per-character models on these lines (see local-tts-guide.md),
#    then render the same manifest with per-character model hot-swap
python tools/render_batch_gsv.py --lines lines.tsv
```

## Status

- [x] en — dragon ×5, drake ×5 (44.1 kHz mono, 3.0–6.5 s, 0 errors/warnings)
- [x] zh — dragon ×5, drake ×5
- [x] ja — dragon ×5, drake ×5
- [x] ko — dragon ×5, drake ×5
