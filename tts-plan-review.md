# Fact-check of "My TTS Plan — Original Voices, Windows, No NVIDIA GPU"

Reviewed **2026-09-20** against upstream sources (CosyVoice `main`, `example.py`, `cosyvoice/cli/cosyvoice.py`,
`webui.py`, `requirements.txt`, CosyVoice3 model card, Qwen3-TTS repo, HF model trees).
Companion: `my-tts-plan.corrected.md`. General reference: `local-tts-guide.md` (written 2026-09-22 — see bottom).

## TL;DR verdict

Your **strategy is right**; your **install/run commands are about 80% right**, and **three of them would fail outright**:

| # | Severity | Claim in your plan | Reality |
|---|---|---|---|
| 1 | 🔴 Breaks | `model.inference_instruct(text, '中文女', persona)` with a CosyVoice **3** model | `inference_instruct` is **v1-only** (`assert self.__class__.__name__ == 'CosyVoice'`). CV3 needs **`inference_instruct2(tts_text, instruct_text, prompt_wav)`** — and there is **no `'中文女'` spk in CV3**. |
| 2 | 🔴 Breaks | `git clone https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct pretrained_models/...` | Without `git lfs pull` you get **pointer files → crash on load**. Also CV2/CV3 dirs need the **`CosyVoice-BlankEN/`** subfolder, which `git clone` of *your own* target dir into `pretrained_models/Foo` can strand in a nested folder. Use `snapshot_download` (HF or ModelScope). |
| 3 | 🔴 Breaks | "Gradio UI opens at `http://127.0.0.1:7860`" after `python webui.py` | `webui.py --port` **defaults to 8000**, not 7860, and `--model_dir` **defaults to `pretrained_models/CosyVoice2-0.5B`** — you'd silently be running CV2. |
| 4 | 🟠 Misleading | "Instruct mode = describe a voice → it speaks" | **Timbre comes from the reference clip, not from the persona text.** CV3's instruct channel is *style/dialect/emotion/rate* only, and the officially-tested instruction set is a **fixed 26-entry list** (`cosyvoice/utils/common.py::instruct_list`). Free-form English persona prose = best-effort, unvalidated. |
| 5 | 🟠 Wrong-by-optimism | "several seconds per sentence on a modern CPU" | CV3-0.5B on CPU is commonly reported at **RTF 2–5 → tens of seconds per line** (one interactive-app project measured **60+ s/utterance**). Budget ~1–2 h for a 100–200-line batch, not "a few minutes". |
| 6 | 🟠 Incomplete | Japanese works with `<\|jp\|>` tag | Upstream note: **for CV3 Japanese you should feed spaced katakana**, not raw kanji/kana (e.g. `レキシ テキ セカイ ニ オイ テ ワ、…`). Your JP sample line needs conversion. |
| 7 | 🟡 Outdated | "grab the newer Fun-CosyVoice3-0.5B" | The current checkpoint is **`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`** — and the RL variant is **not a separate repo**, it's **`llm.rl.pt` inside that same repo** (copy it over `llm.pt` to use it). RL is measurably better on CV3-Eval (zh CER 0.81 vs 1.21; en WER 1.68 vs 2.24; hard CER 5.44 vs 6.71). You can also skip `flow.decoder.estimator.*` (1.3 GB, TensorRT-only). |
| 8 | 🟡 Unnecessary | WSL2 is required because `ttsfrd` is Linux-only | `ttsfrd` is **optional**: the repo pins `wetext==0.0.4` and falls back automatically ("if you do not install ttsfrd, we will use wetext by default"). Keep WSL2 anyway (deps behave), but drop the "must" framing. Missing from your apt line: **`sox libsox-dev`**. |
| 9 | 🟡 Half-right | "Qwen3-TTS 0.6B ≈ 4GB" for VoiceDesign | There is **no 0.6B VoiceDesign build** — released: `1.7B-VoiceDesign`, `1.7B/0.6B-CustomVoice`, `1.7B/0.6B-Base`. VoiceDesign = **1.7B only** (plan for ~8GB bf16; third-party guides say 8–12GB). 0.6B ≈ 4GB applies to CustomVoice/Base. |
| 10 | 🟡 Half-right | CV3 webui modes: "Instruct, Zero-shot, Cross-lingual, **Voice Conversion**" | Open-source `webui.py` exposes exactly 4 modes: 预训练音色 (SFT), 3s极速复刻 (zero-shot), 跨语种复刻 (cross-lingual), 自然语言控制 (instruct-v1). **No VC tab**, and the "自然语言控制" tab calls v1 `inference_instruct` → it does **not** drive CV3 the way the hosted demo does. Script the API instead. |
| 11 | 🟢 Correct | 9 langs (zh en ja ko de es fr it ru) + 18+ CN dialects, zh/en/ja/ko | ✅ matches the CV3 model card. |
| 12 | 🟢 Correct | `pip install -U qwen-tts`, Apache-2.0, 10 langs, VoiceDesign from text, no reference audio needed | ✅ all true (released 2026-01-22). `instruct` param exists; batch + `max_new_tokens` pass-through. |
| 13 | 🟢 Correct | Seed-clip trick as a real technique | ✅ Stronger than you thought: Qwen3-TTS ships an **official "Voice Design then Clone"** workflow (`create_voice_clone_prompt`) and CosyVoice2/3 ship **`add_zero_shot_spk()` + `save_spkinfo()`** = your seed clip becomes a *named, cached speaker*. |
| 14 | 🟢 Correct (with a caveat) | "AI voice → 100% legally clean" | Model + code are **Apache-2.0** (commercial OK; keep NOTICE if you redistribute). Caveat: **`'中文女'`/`'中文男'` are the vendor's pretrained voices** — for "original, no-VA" work you should only ever use *your own* reference clips. And the *character* you write may still be someone's IP even if the *voice* isn't. |

## Two upgrades you should copy into the plan

1. **Permanence: one reference → a saved speaker.** Do this once per character, then the 100+
   later lines skip prompt re-encoding (meaningful on CPU):

```python
cosyvoice.add_zero_shot_spk(prompt_text, 'seeds/dragon_en.wav', 'dragon')
cosyvoice.save_spkinfo()                      # writes spk2info.pt in the model dir
cosyvoice.inference_zero_shot(text, '', '', zero_shot_spk_id='dragon')
```

2. **Consistency: pin the seed.** `webui.py` has the 🎲 button + `set_all_random_seed(seed)`. In scripts,
   call `from cosyvoice.utils.common import set_all_random_seed; set_all_random_seed(42)` before each
   generation for one character — it's how you stop a 200-line batch drifting into three different voices.

## Prompt-clip rules (verified in `cosyvoice/cli/frontend.py`)

- **Hard cap 30 s**: `assert speech.shape[1] / 16000 <= 30` in `_extract_speech_token`. Longer = crash, not warning. Sweet spot **3–10 s**.
- Loaded at 16 kHz for tokens/embedding and **24 kHz** for `prompt_speech_feat`, so a 24 kHz mono source covers both; anything under 16 kHz is refused by the webui and degrades the API path.
- Transcript must match the audio **verbatim**. CV3's prompt form is `You are a helpful assistant.<|endofprompt|>` + transcript.
- Gotcha in `inference_instruct2`: `frontend_instruct2` passes your **`instruct_text` straight through as `prompt_text`** and drops the LLM prompt *speech* tokens (flow keeps them → that's why the clip is mandatory for timbre). So **you** must include the `You are a helpful assistant…<|endofprompt|>` wrapper in the instruct string; nothing adds it for you. Omitting `<|endofprompt|>` = the instruction gets read aloud / prosody leaks.
- Skip text-normalization when you hand-write tags: `text_normalize` auto-disables the frontend if the string contains `<|` … `|>` — that's why tags survive.
- Keep lines ≥ half the prompt length or CV logs "too short than prompt text" and quality drops —
  write lines of ~6+ words, or merge micro-lines into one render.
- Export from this chat at the highest sample rate offered; **do not** re-encode to mp3/ogg before use
  (artifacts teach the vocoder to reproduce artifacts).

## What I would actually run first

1. Seed pack from this chat (2 voices × 4 langs × 5 lines) → `seeds/` — free, today, no setup.
2. WSL2 + **`Fun-CosyVoice3-0.5B-2512_RL`** + `inference_instruct2` **or** saved `zero_shot_spk_id` → 60–90 min setup.
3. If CPU hurts: `CosyVoice2-0.5B` (faster, better CPU RTF) for bulk lines, CV3 for hero lines.
4. Only buy an RTX 3060 12GB if you also want Qwen3-TTS **1.7B-VoiceDesign** (that's the one that
   designs a voice from text alone) or GPT-SoVITS fine-tunes.

## Sources

- CosyVoice upstream: `README.md`, `example.py`, `cosyvoice/cli/cosyvoice.py`, `cosyvoice/utils/common.py`, `webui.py`, `requirements.txt` — https://github.com/FunAudioLLM/CosyVoice
- Fun-CosyVoice3-0.5B(-2512) model card / eval table — https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- CosyVoice2-0.5B file tree (shows `CosyVoice-BlankEN/`, no `spk2info.pt`) — https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/tree/main
- Qwen3-TTS (release 2026-01-22, model table, quickstart, "Voice Design then Clone") — https://github.com/QwenLM/Qwen3-TTS
- CPU/RTF datapoints — CosyVoice issues #1927 (RTF 1.44–3.28 on a cloud box), casehubio/blocks #223 ("60+ s per utterance on CPU")
- GPT-SoVITS (MIT, 5 s zero-shot / 1 min few-shot, en·ja·ko·yue·zh) — https://github.com/RVC-Boss/GPT-SoVITS
- CV3 GGUF community runtime (CPU-friendly, quantized) — https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF

## Follow-up (added 2026-09-22)

The `local-tts-guide.md` flagged as missing above has been written: GPT-SoVITS version map (why
v2ProPlus over v3/v4), free-Colab training workflow, dataset prep (slicer/ASR/subfix), CPU
inference incl. the official README's RTF numbers (0.526 on an Apple M4 CPU) and the
CPU-optimized `GPT-SoVITS-CPUFast` fork. Complemented by `tools/render_batch_gsv.py`, which
renders the same TSV manifest as `render_batch.py` against a GPT-SoVITS `api_v2` server with
per-character model hot-swap. All claims re-verified against GPT-SoVITS `main` on 2026-09-22.
