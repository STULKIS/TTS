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
| `tools/type_ui.py` | Type-and-speak WebUI for GPT-SoVITS: pick a character (or upload any reference clip), type text, hear it — runs inside your GPT-SoVITS install. |
| `tools/cpu_config.py` | One-shot CPU-mode fixer for no-NVIDIA boxes (`device: cpu`, `is_half: false` in tts_infer.yaml, idempotent, backed up). Run before first launch. |
| `tools/preflight.py` | Pre-flight checker: GPT-SoVITS root, CPU config, pretrained weights, seed-pack integrity, ports, disk — one command before your first render. |
| `tools/lines.sample.tsv` | Example manifest (2 characters × en/zh/ja/ko) — works with both renderers. |
| `seeds/` | The seed packs: dragon & drake, 5 lines × en/zh/ja/ko each (44.1 kHz mono WAV + verbatim transcripts). Voice briefs in `seeds/README.md`. |
| `VOICE-PRESETS.md` | 110 named voice presets in 10 groups (mommy, daddy, noir detective, movie-trailer voice, …) — request any by name in chat. |

Fast path: design voices in chat → save 3–10 s clips as `seeds/<char>/<lang>.wav` (the
packs for **dragon** and **drake** already ship in `seeds/`) → validate with
`tools/check_seeds.py` → hear them in `seeds/showcase.html` (`tools/make_showcase.py`) →
write lines as plain text (`tools/dialogue2tsv.py`) → `render_batch.py --dry-run`, then render.
For a dedicated per-character model instead (better consistency over hundreds of lines), fine-tune
GPT-SoVITS v2ProPlus on free Colab and render with `render_batch_gsv.py` — see `local-tts-guide.md`.
Hardware rule: reference-clip cloning runs on CPU (slowly); text→timbre *design* (Qwen3-TTS
1.7B-VoiceDesign) needs ~8 GB of CUDA VRAM.
