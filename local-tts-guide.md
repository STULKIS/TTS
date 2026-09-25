# Local TTS Guide — GPT-SoVITS: free-GPU training → CPU inference

Written **2026-09-22**; every command and number verified against upstream sources listed at the
bottom (GPT-SoVITS `main` branch + README, the official wiki, the `lj1995` Hugging Face repos,
the `api_v2.py` source, and the CPUFast fork README).
Companion documents: `my-tts-plan.corrected.md` (the plan) and `tts-plan-review.md` (the fact-check).

The plan's "Now" tier is CosyVoice 3 + seed clips (`tools/render_batch.py`). This guide covers the
plan's "Later / dedicated character model" tier: **GPT-SoVITS** — train a per-character model on a
free cloud GPU (Colab), then run it at home on a **CPU-only Windows box**, using the same seed
clips you already saved under `seeds/<char>/<lang>.wav`.

---

## 1. What GPT-SoVITS is, as of Sept 2026

- Open-source TTS + voice cloning WebUI. **MIT license, ~62k stars**, actively maintained
  (main branch pushed 2026-08-18; changelog entries through 2026-04).
- **Zero-shot from a 5 s sample; fine-tune from ~1 minute** of training data (official README).
- Languages: **zh, en, ja, ko, yue (Cantonese)** with cross-lingual inference (train on one
  language, speak another).
- WebUI tools bundled: vocal/accompaniment separation (UVR5), silence slicer, multilingual ASR,
  transcript proofreading (subfix), one-click dataset formatting, training, inference.

### Version map (official wiki, checked 2026-09-22)

| Version | Released | Languages | Params (GPT+SoVITS) | Notes |
|---|---|---|---|---|
| v1 | 2024-01 | zh ja en | 90M+77M | baseline |
| v2 | 2024-08 | + ko yue | 90M+77M | speed control, reference-free mode, 5k-h pretrain |
| v3 | 2025-02 | same | 330M+77M | much higher zero-shot similarity; 24 kHz BigVGAN; metallic-artifact complaints |
| v4 | 2025-04 | same | 330M+77M | fixes v3 artifacts; native 48 kHz; author calls it "v3's direct replacement" |
| **v2Pro** | 2025-06 | same | 133M+77M | v3/v4-level zero-shot similarity at v2's hardware cost & speed |
| **v2ProPlus** | 2025-06 | same | 152M+77M | best SeedTTS-eval CN similarity of any release; **current recommendation** |

SeedTTS-eval Chinese zero-shot benchmark from the wiki (lower WER / higher SIM = better;
GT = ground truth): v2ProPlus **WER 0.016 / SIM 0.737** vs v4 0.013 / 0.735 vs GT — / 0.750.

**Which one to use: v2ProPlus.** The current WebUI's version selector offers v1, v2, v4, v2Pro,
v2ProPlus — v3 was removed. The wiki's own guidance: "there is no need to continue using v3/v4"
because the v2Pro series matches their similarity at v2's hardware cost. Nuance: v1/v2/v2Pro-family
models are more forgiving of *average-quality* training audio; v3/v4 lean harder on the reference
clip and want cleaner sources. For seed clips recorded from a chat (decent but not studio), that's
another vote for v2ProPlus.

### Speed (official README benchmarks, v2ProPlus)

| Hardware | RTF | Reading |
|---|---|---|
| RTX 4090 | 0.014 | ~1400 words (≈4 min speech) in 3.4 s |
| RTX 4060 Ti | 0.028 | ~36× realtime |
| **Apple M4 CPU** | **0.526** | **~2× realtime — CPU batch rendering is realistic** |

Compare CosyVoice3-0.5B on CPU at RTF 2–5 (`tts-plan-review.md` finding 5): GPT-SoVITS is the
*faster* CPU path for pre-rendered lines, at the cost of training a model per character.

---

## 2. Which tool for which job (ties into the plan)

| Need | Use |
|---|---|
| Many characters, one model, zero training, instruct-style styling | CosyVoice 3 + `tools/render_batch.py` |
| One hero character, maximum consistency over hundreds of lines | **GPT-SoVITS v2ProPlus fine-tune (this guide)** |
| Text→brand-new timbre, no reference audio, GPU available | Qwen3-TTS 1.7B-VoiceDesign (plan Part 3) |

They compose: design/audition the voice in chat → save the seed pack → either register it as a
CosyVoice named speaker *or* fine-tune a GPT-SoVITS character model from it.

---

## 3. Install (pick one)

Verified install paths from the README (2026-09-22):

**A. Windows integrated package (GPU-first; includes weights)** — latest package in
`lj1995/GPT-SoVITS-windows-package` is **`GPT-SoVITS-v2pro-20250604.7z`** (~8.2 GB;
`-nvidia50` variant for RTX 50xx). Extract, double-click `go-webui.bat`.
⚠️ The README's own download button still links the older v3lora-20250228 package — use the
HF repo tree, not the button.

**B. conda + installer script (Windows/Linux/macOS; supports CPU)**

```bash
conda create -n GPTSoVits python=3.10 && conda activate GPTSoVits
# Windows PowerShell:
pwsh -F install.ps1 --Device <CU126|CU128|CPU> --Source <HF|HF-Mirror|ModelScope> [--DownloadUVR5]
# Linux:
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
# macOS: --device <MPS|CPU>  (README: Mac GPU training quality is poor, CPU is used for training)
```

The installer downloads pretrained models; also installs ffmpeg (Ubuntu: `sudo apt install ffmpeg
libsox-dev`). Tested envs include a pure-CPU one (Python 3.9 + PyTorch 2.2.2).

**C. Docker** — `xxxxrt666/gpt-sovits` on Docker Hub (CU126/CU128, full/`Lite` tags; `Lite` omits
UVR5/ASR weights), or build with `bash docker_build.sh --cuda <12.6|12.8> [--lite]`.

**Manual model placement** (only if the installer failed): weights from
`huggingface.co/lj1995/GPT-SoVITS` → `GPT_SoVITS/pretrained_models/`; G2PWModel (Chinese TTS) →
`GPT_SoVITS/text/`; UVR5 weights → `tools/uvr5/uvr5_weights/`; faster-whisper-large-v3 (en/ja ASR)
→ `tools/asr/models/`. FunASR-family models auto-download on first use.

---

## 4. Train in Colab (free GPU), infer at home (CPU)

The repo ships an official notebook: **`Colab-WebUI.ipynb`** (README badge "Train In Colab").
Verified contents: it clones the repo, installs condacolab (Anaconda3-2024.10-1), creates the
`GPTSoVITS` py3.10 env, runs `bash install.sh --device CU126 --source HF --download-uvr5`, then
launches `python webui.py` with `is_share=True` (public gradio link). There is also a
`Colab-Inference.ipynb` for inference-only sessions.

Free-tier reality (check current Colab policy yourself): free GPU is typically T4-class (~16 GB
VRAM), sessions cap around 12 h, and availability varies. v2Pro training needs roughly 12 GB VRAM,
so it fits a free T4 at batch size 1–2 — slow, but fine for a 1–10 minute dataset.

**Workflow:**

1. Runtime → Change runtime type → GPU.
2. Run the two "Env Setup (Run Once Only)" cells. Mount your Google Drive
   (`from google.colab import drive; drive.mount('/content/drive')`) — put your source audio there
   and save trained weights there, or they die with the session.
3. Run the launch cell; open the `*.gradio.live` link; do dataset prep + training in the WebUI
   (Sections 5–6 below).
4. Download your artifacts from `GPT_weights_v2/<exp>/*.ckpt` and `SoVITS_weights_v2/<exp>/*.safetensors`
   (v2Pro/v2ProPlus naming) to your Windows box.

Home machine = CPU-only, so install there with `--Device CPU` / `--device CPU` (option B above)
and use the artifacts in Section 7.

---

## 5. Dataset preparation (the "0" tab tools)

Same seed-pack philosophy as the plan: **quality over quantity; 1–30 minutes of clean speech**,
no music, no reverb, no mp3 round-trips. The WebUI's dataset tab is a guided pipeline
(README order): *fill audio path → slice → denoise (optional) → ASR → proofread transcripts →
next tab*.

| Step | Tool (repo path) | Notes |
|---|---|---|
| Vocal separation | UVR5 (`tools/uvr5/`) | only if your source has music/backs; optional for clean recordings |
| Slicing | `tools/slice_audio.py` (+`slicer2.py`) | auto-splits on silence into ~3–15 s clips; defaults work |
| Denoise | `tools/cmd-denoise.py` | optional; prefer clean sources over aggressive denoising |
| ASR (auto-label) | `tools/asr/funasr_asr.py`, `tools/asr/fasterwhisper_asr.py` | engine choice matters (below) |
| Proofreading | `tools/subfix_webui.py` | fix ASR mistakes by ear; bad transcripts poison the model |
| Super-resolution | `tools/audio_sr.py`, `tools/AP_BWE_main/` | optional 24k→48k upsampling for v2-family outputs |

ASR engines currently offered by the WebUI (README): **Fun-ASR-Nano** (multilingual + dialects),
**SenseVoice** (fast), classic **FunASR** Paraformer/UniASR (zh/yue), and **faster-whisper
large-v3** for en/ja. Community rule of thumb: faster-whisper for English/Japanese, FunASR for
Chinese.

Output of this pipeline is the annotation file, one line per clip:

```
vocal_path|speaker_name|language|text        # language ∈ zh, ja, en, ko, yue
```

Then "one-click formatting" (training tab → dataset formatting) runs tokenization → HuBERT feature
extraction → speaker verification → semantic-token extraction, writing everything the trainer needs
under `logs/<experiment>/`.

---

## 6. Fine-tuning (training tab)

Train **SoVITS first, then GPT** (two separate one-click buttons). Settings that recur across
official-adjacent guides for small (1–10 min) datasets — treat as starting points, not gospel:

- **SoVITS:** batch size 2 (1 if ≤6 GB VRAM), total epochs ≈ 8, text-LR weighting ≤ 0.4,
  save every few epochs so you can pick the best checkpoint.
- **GPT:** higher epoch counts (~15) are common; v2Pro/v2ProPlus also offer a **DPO** toggle
  (changelog 2026-04 still fixes DPO-training bugs, so it's a live feature).
- Watch for overfitting on 1-minute datasets: loss still falling ≠ audio still improving; listen
  to the saved checkpoints.

Outputs per experiment: one GPT `.ckpt` + one SoVITS `.safetensors` (or `.pth`) — that pair **is**
your character model.

---

## 7. CPU inference at home (the part this repo cares about)

Four verified entry points, in order of usefulness for batch work:

**A. `api_v2.py` — HTTP server (recommended for scripted rendering).**
Launch (defaults verified in source: binds `127.0.0.1:9880`):

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

CPU setup lives in `GPT_SoVITS/configs/tts_infer.yaml` (`device: cpu`, `is_half: false`).
Endpoints: `GET/POST /tts`, `GET /set_refer_audio`, `GET /set_gpt_weights`,
`GET /set_sovits_weights`, `GET /control?command=restart|exit`. Key `/tts` params:
`text`, `text_lang`, `ref_audio_path`, `prompt_text`, `prompt_lang`, `seed`, `speed_factor`,
`repetition_penalty` (default 1.35), `top_k`, `top_p`, `temperature`, `streaming_mode`, `media_type=wav`. The Studio uses moderate sampling by default; avoid extreme temperature if you want clean, human phonemes.
Smoke test:

```bash
curl -G "http://127.0.0.1:9880/tts" \
  --data-urlencode "text=这个世界的命运，还没有定数。" \
  --data-urlencode "text_lang=zh" \
  --data-urlencode "ref_audio_path=/abs/path/seeds/dragon/zh.wav" \
  --data-urlencode "prompt_text=<verbatim transcript of the clip>" \
  --data-urlencode "prompt_lang=zh" \
  --data-urlencode "seed=42" -o line.wav
```

**B. `tools/render_batch_gsv.py` (this repo)** — manifest-driven batch renderer against the API
above, mirroring `tools/render_batch.py`: same `id⇥char⇥lang⇥text` TSV, per-character model
hot-swap via `/set_gpt_weights`+`/set_sovits_weights`, per-character RNG seed, reuses
`seeds/<char>/<lang>.wav|.txt` as reference clips. Stdlib-only.

**C. `GPT_SoVITS/inference_cli.py`** — one-shot CLI (verified args):
`--gpt_model --sovits_model --ref_audio --ref_text --ref_language --target_text
--target_language --output_path`. Fine for single lines; no batching.

**D. WebUI** — `python webui.py` then `1-GPT-SoVITS-TTS → 1C-inference`, or
`python GPT_SoVITS/inference_webui.py` directly (`inference_gui.py` also ships). Best for
auditioning settings interactively.

**Make the CPU faster:** the official README itself links a CPU-optimized inference fork,
**`baicai-1145/GPT-SoVITS-CPUFast`** (checked 2026-09-22). It strips training/dataset tooling,
keeps `webui.py`, `inference_webui_fast.py`, `api.py`/`api_v2.py`, and has CPU-only installers
(`bash install.sh --source ModelScope --version v2ProPlus`). Reported wins with **no quantization
and no quality tradeoffs**: ≈32% end-to-end, ≈45% on pure-Chinese text. If CPU speed becomes the
bottleneck, train upstream, render with this fork.

Expect on an ordinary CPU: RTF ≈ 0.5 official (M4) → a 20 s line costs ~10 s; 150 lines of
dialogue ≈ 25–40 min unattended. That's roughly 4–10× quicker per line than CosyVoice3 on CPU.

---

## 8. Reference-clip rules (carry over from the plan)

- Validate a pack mechanically with `tools/check_seeds.py` (30 s hard cap, ≥ 16 kHz, mono,
  transcript present) — same envelope applies to both engines.
- **3–10 s**, mono, ≥ 32 kHz source preferred (v2-family output rate), clean, no music.
- Transcript **verbatim**, in the clip's language; `prompt_lang` must match.
- Same seed + same reference = reproducible takes (`seed` param; the batch renderer pins it
  per character, same convention as `render_batch.py --seed`).
- One reference per language if the character speaks several; cross-lingual synthesis works but
  a target-language reference is always safer (plan Part 2 caveat applies unchanged).
- Legally unchanged from the plan: MIT code, but train only on voices you own or have rights to.
  "Original character" does not excuse cloning a real performer.

---

## Sources (checked 2026-09-22)

- Upstream repo, README (features, RTF table, installs, pretrained models, dataset format):
  https://github.com/RVC-Boss/GPT-SoVITS — MIT, ~62k stars, pushed 2026-08-18
- Version table + SeedTTS-eval numbers + version-selection guidance:
  https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7)
- Changelog (v2Pro series 2025-06; DPO fixes 2026-04):
  https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/en/Changelog_EN.md
- Official Colab: `Colab-WebUI.ipynb` / `Colab-Inference.ipynb` in repo root
- Windows packages tree (`GPT-SoVITS-v2pro-20250604.7z` latest):
  https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main
- Pretrained weights: https://huggingface.co/lj1995/GPT-SoVITS
- `api_v2.py` endpoints/params and `GPT_SoVITS/inference_cli.py` args: file sources on `main`
- CPU fork (linked from upstream README): https://github.com/baicai-1145/GPT-SoVITS-CPUFast
- Dataset/training community guidance (epochs, batch sizes, 3–10 s refs): docs.aihub.gg GPT-SoVITS
  guide, the rentry English guide linked from the README — community docs, not upstream law
