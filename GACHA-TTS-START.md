# 🎮 Gacha-Voice TTS — turnkey start (with the dragon/drake pack)

> **Goal: a TTS you TYPE INTO — on your own PC — that can make (almost) any voice you
> want, in en · zh · ja · ko (+ yue).**
> Engine: **GPT-SoVITS** (MIT, RVC-Boss). This doc merges the community gacha-Voice
> setup guide with the verified facts in `local-tts-guide.md` and the seed packs that
> ship in this repo (`seeds/`).

**The "type text" part, up front:** GPT-SoVITS *is* a full TTS — a WebUI where you
upload a reference clip (or load a trained character model), **type your text**, pick
the language, and get speech. This repo's `seeds/` pack is the reference-clip side of
that: dragon & drake, 5 lines × 4 languages each, with verbatim transcripts.

---

## 1. Install on Windows (closest thing to an .exe)

**A. Integrated package (GPU-first, weights included):**
1. HuggingFace `lj1995/GPT-SoVITS-windows-package` → latest **`GPT-SoVITS-v2pro-20250604.7z`** (~8.2 GB).
   ⚠️ The README's own download button still links an older package — use the repo tree
   (verified 2026-09-22, see `local-tts-guide.md` §3A).
2. Extract → **double-click `go-webui.bat`** → WebUI opens in your browser. Done.

**B. No NVIDIA GPU? (your box)** — the package above is GPU-first; the CPU-tested route:
```powershell
# conda + installer script, Windows PowerShell (local-tts-guide.md §3B)
conda create -n GPTSoVits python=3.10 && conda activate GPTSoVits
pwsh -F install.ps1 --Device CPU --Source HF
python webui.py en_US
```
A pure-CPU env is explicitly tested upstream (Python 3.9 + PyTorch 2.2.2).
Recommendation: 16 GB RAM helps; CPU generation is usable, CPU *training* is not —
train in free Colab (route B below) or skip training and live on zero-shot.

**Version:** start with **v2Pro** (near-v4 quality at v2 cost). v3/v4 want more VRAM.

---

## 2. Zero-shot — type text today (5 minutes, no training)

WebUI → **Inference** tab (zero-shot mode):

1. Reference audio: one clip from `seeds/` (longest per char/lang — 5 s+ is ideal):

   | char   | en         | zh         | ja         | ko         |
   |--------|------------|------------|------------|------------|
   | dragon | `en2.wav` 6.5 s | `zh5.wav` 5.8 s | `ja5.wav` 7.4 s | `ko5.wav` 5.5 s |
   | drake  | `en2.wav` 5.3 s | `zh2.wav` 5.3 s | `ja3.wav` 6.7 s | `ko2.wav` 6.0 s |

2. Reference text: paste the matching `<clip>.txt` **verbatim**.
3. Text: **type your line** · language: `zh_cn` / `ja` / `en` / `ko` → generate.

Cross-lingual works: reference in zh, text in en (or vice versa) — v2Pro/Plus handles it.
CPU expectation: RTF ~0.5 (guide §8: 0.526 on an Apple M4 CPU) → a 5 s line takes ~3 s.
Great for pre-rendered lines; not live chat on CPU.

**Scripting / mods / games:** `python api_v2.py -p 9880` → `POST /tts`
(`text`, `text_lang`, `ref_audio_path`, `prompt_text`, `prompt_lang`).
This repo's `tools/render_batch_gsv.py` renders a whole TSV manifest against that API
with per-character model hot-swap.

### 2b. One-command type-a-text WebUI (this repo)

`tools/type_ui.py` — a single-file WebUI wrapping the same zero-shot engine:
pick a character from the seed pack (transcript auto-filled), **type your text**,
click **🔊 Speak** → audio. A second tab takes *any* reference clip you upload
(5–10 s wav + verbatim transcript) — i.e. any voice you want, not just the pack.

```powershell
# in your GPT-SoVITS folder (integrated package: cd into the extracted 7z;
# conda route: conda activate GPTSoVits first), from a terminal:
python C:\TTS\tools\type_ui.py --seeds C:\TTS\seeds
# → open http://127.0.0.1:7861
```
(If `python` isn't recognized, use the `python.exe` inside the integrated-package
folder. The UI reads `GPT_SoVITS/configs/tts_infer.yaml`, so it also picks up a
trained checkpoint automatically once you set `t2s_weights_path`/`vits_weights_path`.)

---

## 3. Fine-tune — make the character *yours* (where "close" becomes "spot-on")

Zero-shot clones the reference's quirks but stays slightly generic on personality.
A dedicated model fixes that — and this repo feeds the pipeline end to end:

1. **Corpus:** the pack gives 25 s per char/lang as a seed. Grow it in-chat:
   "100 in-character lines for dragon, zh" → 100 wav+transcript pairs, same voice,
   3–10 s each, all inside the verified prompt-clip envelope. 1–15 min per voice is
   the sweet spot (15+ min = scary good).
2. **TTS Formatting tab:** drop the lines in — auto-slicing at silences, UVR5 strips
   any BGM.
3. **ASR tab:** auto-transcription (EN/JA/KR → faster-whisper large-v3, ZH → Damonto)
   → `.list` (`wav|speaker|lang|text`).
4. **1-GPT training tab** → GPT stage, then SoVITS stage. Free Colab notebook
   (`Colab-WebUI.ipynb`) for GPU-less boxes; ~30–60 min on an RTX 3060.
5. **Inference tab:** load the checkpoint, pick a punchy reference line, type, generate.

`tools/render_batch_gsv.py` then batch-renders any `lines.tsv` manifest against the
API with the trained model.

---

## 4. Voice features (age, wetness/dryness, breath…)

Features are **timbre properties** — set at design time, baked into the reference
clips, not render-time flags. The design vocabulary (scales + which render-time
knobs exist) lives in `seeds/README.md`. Current values:

- **dragon** — ancient · dry-warm · solid · low · glacial · calm
- **drake** — middle-aged · dry/papery · solid · mid-low · brisk · volatile-but-controlled

Different values? Ask in chat ("dragon but younger, a touch wetter") → the affected
char/lang set gets re-auditioned and regenerated, then fed back into zero-shot or the
training corpus.

---

## 5. Voices: original (this repo) vs ripped (community)

- **Original / OC voices (this repo):** designed in-chat, Apache-2.0 engine, 100% clean.
  The dragon/drake pack is that, working.
- **Ripped gacha VO** (AssetStudio for Unity titles, FModel for Unreal, YouTube
  line compilations): grey area — fine for personal use, don't redistribute. The
  pipeline is identical (rip → UVR5 → ASR → train/zero-shot). Your call.
- **Real VA voices without consent:** don't, especially if published/monetized
  (right of publicity). See `my-tts-plan.corrected.md` Part 2 caveats.

---

## 6. Engine comparison (the short version)

| Want | Use |
|---|---|
| Full TTS + fine-tune, en/zh/ja/ko/yue, MIT, community standard | **GPT-SoVITS** (this doc) |
| 9 langs + 18 CN dialects, zero-shot, Apache-2.0, streaming/emotion | CosyVoice 3 (WSL2 path — `my-tts-plan.corrected.md` Part 1) |
| Text → brand-new timbre, no reference clip | Qwen3-TTS 1.7B-VoiceDesign (needs ~8 GB VRAM) |

**Hard requirement = en + zh + ja + ko with fine-tuning → GPT-SoVITS, no contest.**
