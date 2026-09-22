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

## Voice features (design vocabulary) — 58 scales

Features are properties of the **timbre/manner** — set *at design time* (in-chat audition) and
baked into the reference clips. Render-time channels can only steer *style* (rate/volume/emotion),
so "a drier voice" = regenerate the clips, not a render flag. Say any value out loud —
*"make drake drier"*, *"dragon but younger"* — and the affected char/lang set gets re-auditioned.

**Physical / texture** (baked into the voice):
age (child → ancient) · gender presentation · voice weight (thin ↔ full) · pitch ·
pitch variety (monotone ↔ big intonation) · wetness↔dryness · roughness (silky ↔ gravelly) ·
smokiness · breathiness · reediness · velvety · vocal fry · voice cracks · nasality ·
resonance (head ↔ chest) · brightness (dark ↔ crystalline) · placement (front ↔ back) ·
vowel openness

**Delivery / dynamics** (how it's spoken):
rate · volume · intimacy (distant ↔ close-mic) · energy · steadiness (wobbly ↔ rock-steady) ·
articulation (slurred ↔ crisp) · mumble · rhythm (staccato ↔ flowing) · pausing (rattling ↔
thoughtful) · emphasis · sentence endings (falling ↔ rising ↔ trailing) · drawn-out vowels ·
dynamics (flat ↔ big swells)

**Emotion / personality** (mood coloring):
emotion baseline · warmth · smile tone · seductiveness · authority (timid ↔ commanding) ·
playfulness · menace · composure · softness (hard ↔ vulnerable) · sharpness · confidence ·
anxiety · tiredness (wide-awake ↔ sleepy) · melancholy · smugness · sarcasm ·
deadpan · theatricality · formality (slang ↔ stiff) · politeness (blunt ↔ ultra-polite / keigo)

**Quirks & special** (character flavor):
laugh style (giggle · chuckle · cackle · snort) · sigh style · verbal tics ("heh", "hmph",
catchphrases) · speech quirks (light stutter · lisp) · accent (none · regional — 18 CN dialects
validated on CosyVoice · foreign/L2) · muttering tendency · whisper register

Render-time nudges (CosyVoice `--instruct`, the *validated* subset only): rate
(fastest/slowest), volume ("as loudly as possible" / "very soft voice"), emotion
(very happy/sad/angry), dialects, robot style. Everything else above is design-time.

## Voice presets — 110 named archetypes

**Full catalog: [`VOICE-PRESETS.md`](../VOICE-PRESETS.md)** — 110 presets in 10 groups:
parental & familial (10), romance (12), gacha hero & combat (14), mystic & supernatural (12),
work & society (12), comedy (10), fantasy races & roles (10), CJK culture (10),
special & weird (10), film & radio (10).

Request any of them in chat: *"new character with a **mommy** voice"*,
*"re-voice drake as **noir detective**"*, or mix: *"**daddy** but **drier**, a touch **gravelly**."*
Not on the list? Describe it with the features (or in plain words) → it becomes a custom preset.

Current values:

- **dragon** — ancient · dry-warm (dry voice *with* warmth, not wet) · solid · low · glacial · calm
- **drake** — middle-aged · dry/papery · solid · mid-low · brisk · volatile-but-controlled

Want a different value? Say it in chat ("make drake drier", "dragon but younger, a touch
wetter", "new character with a mommy voice") → re-audition + regenerate. The feature sticks across
the whole pack because every clip comes from the same voice.

## Use

```bash
# 0) hear the pack
python tools/make_showcase.py   # -> seeds/showcase.html (open in any browser)

# 1) validate the pack
python tools/check_seeds.py --seeds seeds --manifest tools/lines.sample.tsv

# 1b) write new lines as plain text, then pre-flight the render
#     dragon: 这句话…  /  drake [ja]: 待たせたな。
python tools/dialogue2tsv.py --script scene.txt --output lines.tsv
python tools/render_batch.py --lines lines.tsv --dry-run

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
