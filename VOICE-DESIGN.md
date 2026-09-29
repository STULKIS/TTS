# Voice Designer — experimental model-driven traits and acting

This adds a separate **Qwen3-TTS-12Hz-1.7B-VoiceDesign** mode. It sends voice and
acting descriptions to the speech model's real `generate_voice_design(...,
instruct=...)` API. It does **not** turn traits into pitch/noise effects, secretly
use chat-generated audio, or treat GPT-SoVITS as an instruction-following engine.

## Start on your Windows PC

1. Close Classic Studio to free RAM.
2. Double-click **VOICE-DESIGN.bat** from the updated branch zip. It offers to
   install/update `D:\TTS` if you launched from another folder. Alternatively,
   choose **2 — Voice Designer** from the existing Studio desktop launcher.
3. First launch installs an isolated Python 3.12 CPU environment, downloads the
   model, then attempts a **real speech render**. Only success creates its ready
   marker. This can take a long time; keep the window open.
4. It creates a **Gacha Voice Designer** desktop icon and opens the controls at
   port 7862. Subsequent starts reuse the installed environment/model files.

No manual Python/pip/model commands, NVIDIA GPU, administrator rights or API
key are required by this installer. This remains a BAT/PowerShell launcher, not
an EXE. It does not modify your GPT-SoVITS dependencies or system Python/PATH.
Your installed Classic Studio remains available as option 1.

Downloads and setup audio live under `D:\TTS\.voice-design` (or the folder you
chose). The bootstrap archive has a pinned SHA256 check; downloads use HTTPS.
Only one launcher session can own this folder at a time. The first setup asks
for at least **16 GB free disk** and **9 GiB currently available RAM**. The RAM
check is a conservative guard, **not a measured guarantee** that every prompt
fits a 16 GB PC. Close games, Classic Studio and large browser sessions first.
If setup fails, the window keeps the error visible and does not mark the model
ready. Retry via the same launcher; don't reinstall GPT-SoVITS.

## Actual control panel

Open **Voice Designer · traits + acting** in section 2. It contains all 58 axes
from the earlier voice vocabulary, plus free-form design and acting fields:

- **Voice / texture (18):** age, gender presentation, vocal weight, register,
  intonation, wet/dry tone, rasp, smokiness, breathiness, reediness, velvet,
  vocal fry, cracks, nasality, resonance, brightness, placement, vowel openness.
- **Delivery / dynamics (13):** pace, acting loudness, intimacy, energy,
  steadiness, articulation, mumble, rhythm, pausing, emphasis, sentence endings,
  drawn-out vowels, loudness dynamics.
- **Emotion / personality (20):** emotion, warmth, smile, flirtatious tone,
  authority, playfulness, menace, composure, softness, sharpness, confidence,
  anxiety, tiredness, melancholy, smugness, sarcasm, deadpan, theatricality,
  formality, politeness.
- **Quirks / special (7):** laugh, sigh, verbal mannerisms, speech quirks,
  accent request, muttering, whisper.
- **Additional fields:** free-form voice brief, acting direction, specific
  dialogue words to emphasize, experimental tearfulness/crying.

**Every field is a soft instruction, not a calibrated physical slider.**
Unspecified fields add nothing to the prompt. Fine textures, accents,
laughing/sighing, stutters/lisps and crying are especially uncertain. You can
request them, but audible adherence has NOT been certified individually.
Contradictory traits may be blended or ignored. Start with two or three traits.

**Show exact model instruction** displays the server-built text passed into
`instruct`. That instruction is separate from the spoken `text`; we do not add
stage directions to the dialogue. The WAV response includes an instruction
SHA256 for tracing. Model hallucinations/omissions are still possible.

Applying a catalog preset in this mode sends its description as the voice
brief; it does not apply that preset's DSP pitch/pace. Saved setups include the
voice instructions. Each prepared script row gets its own editable copy of the
traits/directions. Changing top controls does not change prepared rows.

## What is / isn't supported

| Capability | Status |
|---|---|
| Natural-language voice design and acting input | Official model API; adapter wired and contract-tested |
| Independent exact physical/anatomical sliders | **No** — descriptive instructions only |
| The 58 trait requests | Reach the model prompt; individual audible adherence unverified |
| Laughing, sighing, crying, accent details, speech quirks | **Experimental requests**, not reliable timed events |
| Text emphasis | Soft instruction; not word-level timing/SSML control |
| Reference cloning / preserving Dragon or Drake identity | **Not in this mode**; a new design render may change identity |
| English, Chinese, Japanese, Korean, Portuguese, Spanish, French, German, Italian, Russian | Published model language set; not all languages locally auditioned |
| Cantonese | **Not enabled** in this backend; Classic still offers its existing route |
| Temperature, top-k/p, repetition penalty, seed | Passed to the model |
| GPT-SoVITS speed factor, splitting, phrase gap, sentence batching | Disabled here; use acting pace/pausing requests instead |
| Cosmetic pitch/volume/FX | Still optional post-processing, not the trait/acting implementation |
| Reuse matching take / fresh take | Cache includes the complete instruction; fresh bypasses reuse |
| Script editor | Up to 20 independent rows; max 600 characters per design render; no silent text truncation |

To keep a designed identity for repeated dialogue, you can download a short
successful design take and use it as a reference in Classic Studio. That does
not give Classic this instruction interface; it remains reference-driven.

## Validation and limits — read before expecting production quality

- Verified the documented API against the actual **qwen-tts 0.1.1 PyPI wheel**.
- Automated tests exercise all 58 instruction mappings, the CPU load options,
  language/parameter validation, prompt preview, request routing, cache
  invalidation, profile persistence and independent per-line UI requests.
- Those tests use **fake model output** for contract testing. They are NOT
  voice-quality, emotion-adherence or Ryzen performance benchmarks.
- This development sandbox has only about **4 GB RAM**. It cannot run this full
  float32 model. Its CPU-wheel download was also blocked by the sandbox network.
  No successful full-model synthesis is claimed here.
- The Windows installer/parser is tested separately in CI; an actual Windows
  install, weight download, model render and listening test still need to run on
  your machine. The installer automates the inference smoke test and saves
  `.voice-design/setup-test.wav` with a JSON report, but cannot judge acting.
- CPU float32 + SDPA is chosen for compatibility, not a speed promise. This
  model can be considerably slower than Classic on a 5600G. Cache reuse helps
  repeated text; new text/changed directions require fresh model work.

## Upstream basis

The official API and model capability/language table are documented by Qwen:
[1](https://github.com/QwenLM/Qwen3-TTS). The package is pinned at `qwen-tts==0.1.1`
with its published Transformers/Accelerate versions. CPU Torch/Torchaudio 2.8.0
are installed separately from PyTorch's CPU wheel index; FlashAttention/CUDA are
not installed by this setup.
