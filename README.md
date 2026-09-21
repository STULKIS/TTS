# TTS — original character voices, Windows, no NVIDIA GPU

Working notes for a fully-local, legally-clean character-voice pipeline (en · zh · ja · ko).

| File | What it is |
|---|---|
| `my-tts-plan.corrected.md` | The plan itself, fact-checked and command-by-command corrected. Start here. |
| `tts-plan-review.md` | Every diff vs. the original draft, with severity, evidence, and sources. |
| `tools/render_batch.py` | CPU batch renderer: registers each character's seed clip as a named speaker, renders a `.tsv` manifest. |
| `tools/lines.sample.tsv` | Example manifest (2 characters × en/zh/ja/ko). |
| `local-tts-guide.md` | *TODO* — general reference (GPT-SoVITS Colab training → CPU inference, dataset prep). |

Fast path: design voices in chat → save 3–10 s clips as `seeds/<char>/<lang>.wav` → `render_batch.py`.
Hardware rule: reference-clip cloning runs on CPU (slowly); text→timbre *design* (Qwen3-TTS
1.7B-VoiceDesign) needs ~8 GB of CUDA VRAM.
