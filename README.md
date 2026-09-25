# TTS — original character voices, Windows, no NVIDIA GPU

Working notes for a fully-local, legally-clean character-voice pipeline (en · zh · ja · ko).

| File | What it is |
|---|---|
| `my-tts-plan.corrected.md` | The plan itself, fact-checked and command-by-command corrected. Start here. |
| `tts-plan-review.md` | Every diff vs. the original draft, with severity, evidence, and sources. |
| `local-tts-guide.md` | General reference — GPT-SoVITS: version map, free-Colab training, dataset prep, CPU inference (+CPUFast fork). |
| `tools/render_batch.py` | CPU batch renderer (CosyVoice path): registers each character's seed clip as a named speaker, renders a `.tsv` manifest. |
| `tools/render_batch_gsv.py` | Batch renderer (GPT-SoVITS path): same manifest, hot-swaps per-character fine-tuned models via `api_v2`. |
| `tools/check_seeds.py` | Validates a seed pack against the verified prompt-clip rules (30 s hard cap, ≥16 kHz, mono, 3–10 s, transcripts). |
| `tools/ja_katakana.py` | Converts `ja` manifest lines to spaced katakana for CosyVoice3 (pykakasi). |
| `tools/dialogue2tsv.py` | Plain-text script → manifest TSV (auto ids, lang inferred, `char [xx]:` override). |
| `tools/make_showcase.py` | One-click HTML player for the seed pack (`seeds/showcase.html`). |
| `tools/type_ui.py` | Type-and-speak WebUI for GPT-SoVITS: pick a character (or upload any reference clip), choose Natural / Alive / Dramatic delivery, type text, hear it — runs inside your GPT-SoVITS install. |
| `tools/cpu_config.py` | One-shot CPU-mode fixer for no-NVIDIA boxes (`device: cpu`, `is_half: false` in tts_infer.yaml, idempotent, backed up). Run before first launch. |
| `tools/preflight.py` | Pre-flight checker: GPT-SoVITS root, CPU config, pretrained weights, seed-pack integrity, ports, disk — one command before your first render. |
| `tools/presets_ingest.py` | Validates a `presets.csv` voice catalog (numbering, enums, base/variant blocks, dupes) and generates `PRESET-CATALOG.md` + `PRESET-CATALOG.html` with every line mapped to the 58-feature vocabulary. |
| `tools/audio_fx.py` | Per-line delivery controls applied post-render: pitch (semitones, WSOLA), speed, volume (dB), fx (including timing-safe `alive` humanize/lift) — backs the optional `lines.tsv` columns 5–8. |
| `INSTALL.bat` | One-click Windows installer (CPU route): copies the repo to `D:\TTS`, unpacks the GPT-SoVITS package to `D:\GSV`, sets CPU mode, runs pre-flight, creates a desktop icon, opens the WebUI. |
| `REPAIR.bat` / `UNINSTALL.bat` | Companion one-clicks: re-apply CPU mode + pre-flight, or remove icon/folders. |
| `tools/lines.sample.tsv` | Example manifest (2 characters × en/zh/ja/ko) — works with both renderers. |
| `seeds/` | The seed packs: dragon & drake, 5 lines × en/zh/ja/ko each (44.1 kHz mono WAV + verbatim transcripts). Voice briefs in `seeds/README.md`. |
| `VOICE-PRESETS.md` | 110 named voice presets in 10 groups (mommy, daddy, noir detective, movie-trailer voice, …) — request any by name in chat. |
| `presets.csv` | The full gacha voice catalog (~1000 lines, 25 classes: base + texture/pace variants). Drop it at the repo root and run `python tools/presets_ingest.py` → validated + `PRESET-CATALOG.md/.html`. |

Fast path: design voices in chat → save 3–10 s clips as `seeds/<char>/<lang>.wav` (the
packs for **dragon** and **drake** already ship in `seeds/`) → validate with
`tools/check_seeds.py` → hear them in `seeds/showcase.html` (`tools/make_showcase.py`) →
write lines as plain text (`tools/dialogue2tsv.py`) → `render_batch.py --dry-run`, then render.
For a dedicated per-character model instead (better consistency over hundreds of lines), fine-tune
GPT-SoVITS v2ProPlus on free Colab and render with `render_batch_gsv.py` — see `local-tts-guide.md`.
Hardware rule: reference-clip cloning runs on CPU (slowly); text→timbre *design* (Qwen3-TTS
1.7B-VoiceDesign) needs ~8 GB of CUDA VRAM.

## Making a render sound alive

The Studio's **🔥 Make it alive** button is intentionally a natural-performance preset, not a
loud-effects button. It uses moderate sampling (`temperature 1.05`, `top_p 0.96`,
`repetition_penalty 1.14`), a small speed nudge, and timing-safe micro-dynamics. It no longer
stacks chorus, synthetic breath, or extreme “fever” sampling, which can make a cloned voice
sound metallic or robotic. The reference clip supplies timbre; choose a **Performance** take
when you want the emotion/energy to carry into the new line.

For the best results:

1. Start with a clean mono 3–10 second reference whose transcript is exact.
2. Click **Make it alive**, then render three takes with **Life lottery** enabled; it ranks
   phrase dynamics while penalizing clipping and broadband noise.
3. Use punctuation and sentence-sized lines. Keep `Balanced` for calm dialogue and `Alive` for
   expressive dialogue; `Dramatic` is a nudge, not a guarantee of an emotion the source clip
   never performed.
4. Keep **robot**, **phone**, and **chorus** off for human speech. The optional `alive` audio-FX
   macro expands only to the restrained `humanize,lift` pair.

The backend now validates sampling controls, forwards repetition penalty correctly, and joins all
non-streaming chunks instead of silently keeping only the first chunk of a split sentence.
