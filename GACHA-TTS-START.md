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

**The one-click way (start here):** double-click **`INSTALL.bat`** in this
folder. It copies the repo to `D:\TTS`, asks where the ~8 GB GPT-SoVITS
package is (and gives you the download link if you don't have it yet),
unpacks it to `D:\GSV` with 7-Zip, switches GPT-SoVITS to CPU mode (no
NVIDIA), runs the pre-flight checks, creates a **"Gacha TTS" desktop icon**,
and opens the WebUI. The only manual step left in the whole install is the
one-time ~8 GB package download — no terminal, no typing paths.
Companions: **`REPAIR.bat`** (re-applies CPU mode + pre-flight when
something stops working) and **`UNINSTALL.bat`** (icon + optional
folder removal).

**The manual way (only if the installer misbehaves):**

**Step 0 — 30-second pre-flight** (do this before any download):
- [ ] **30 GB free disk** (the extracted package is big)
- [ ] **16 GB RAM** (8 GB minimum) — yours: 16 GB ✅
- [ ] **7-Zip** installed (https://www.7-zip.org) — needed to open the .7z
- [ ] Pick a **short extract path**: `D:\GSV` (not `C:\Users\YourName\Downloads\…` —
      the folder paths inside are very long and Windows will choke)

**Step 1 — Get the engine (pick one):**

**A. Integrated package (recommended — closest to double-click):**
1. HuggingFace `lj1995/GPT-SoVITS-windows-package` → **`GPT-SoVITS-v2pro-20250604.7z`** (~8.2 GB).
   ⚠️ Use the repo tree, NOT the README's download button (it links an older package;
   verified 2026-09-22, see `local-tts-guide.md` §3A).
2. Extract to `D:\GSV` → folder `D:\GSV` containing `go-webui.bat`.

**B. CPU-first route (use if A misbehaves on a no-GPU box)** — the explicitly CPU-tested path:
```powershell
# needs Miniconda (docs.conda.io/miniconda), then:
conda create -n GPTSoVits python=3.10 && conda activate GPTSoVits
cd D:\GSV
pwsh -F install.ps1 --Device CPU --Source HF
python webui.py en_US
```
(A pure-CPU env is tested upstream: Python 3.9 + PyTorch 2.2.2.)

**Step 2 — Pre-fix the CPU config (the #1 silent failure on no-NVIDIA boxes):**
Open a terminal **inside** `D:\GSV` (type `cmd` in the folder's address bar → Enter), then:
```
python D:\TTS\tools\cpu_config.py --gsv-root D:\GSV
```
Sets `device: cpu` / `is_half: false` in `tts_infer.yaml` (backed up first, idempotent —
safe to re-run). The WebUI's Config tab does the same thing; this covers the API/type_ui path.

**Step 3 — Run the pre-flight checker (finds problems before they bite):**
```
python D:\TTS\tools\preflight.py --gsv-root D:\GSV --seeds D:\TTS\seeds --manifest D:\TTS\tools\lines.sample.tsv
```
All green → proceed. Any `[fail]` → the line tells you exactly what to fix.

**Step 4 — Start:** `go-webui.bat` (full WebUI) and/or `type_ui.py` (see §2b).

**Troubleshooting (if you see → do this):**

| If you see this | Do this |
|---|---|
| SmartScreen "Windows protected your PC" on `go-webui.bat` | **More info → Run anyway** (it's a local script, not malware) |
| 7z "archive is corrupt" | Re-download; file should be ~8.2 GB; extract to a short path |
| `python` is not recognized in the terminal | The terminal must be opened *inside* `D:\GSV` (package's python); or run `go-webui.bat` once first |
| CUDA / `is_half` / `half` error on first generation | Step 2 not done → run `cpu_config.py`; or WebUI → Config tab → device: `cpu`, is_half: `false` |
| "Port 9874 already in use" | Close the other server window, or change the port in the Config tab |
| type_ui: "does not look like a GPT-SoVITS root" | `--gsv-root` must point at the folder containing `GPT_SoVITS\` |
| type_ui: "pipeline failed to load" | Weights missing → Step 1B's installer, or `local-tts-guide.md` §3; run preflight to see exactly what |
| Generation takes seconds per line | **Normal on CPU** (RTF ~0.5): 5 s of audio ≈ 3 s of waiting. Don't close the tab |
| "Can't reach this page" in browser | The server's terminal window must stay open; open `http://127.0.0.1:7861` by hand |
| Defender quarantines a `.bat` (rare) | Allow the notification, or add an exclusion for `D:\GSV` |

**Version:** start with **v2Pro** (near-v4 quality at v2 cost). v3/v4 want more VRAM.
CPU generation is usable; CPU *training* is not — train in free Colab (§3) or live on zero-shot.

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
# terminal opened INSIDE D:\GSV (address bar → cmd → Enter):
python D:\TTS\tools\type_ui.py --seeds D:\TTS\seeds --open
# → browser opens http://127.0.0.1:7861 automatically
# (conda route: conda activate GPTSoVits first; --gsv-root D:\GSV if run elsewhere)
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

## 4. Voice features (age, wetness/dryness, breath…) + presets

Features are **timbre properties** — set at design time, baked into the reference
clips, not render-time flags. The full **58-feature vocabulary** (physical/texture,
delivery, emotion/personality, quirks — age, wetness↔dryness, smokiness, vocal fry,
nasality, resonance, smile tone, deadpan, sarcasm, keigo-level politeness, laugh
style, stutter, accents, …) plus **110 named presets in 10 groups** —
mommy, daddy, cold CEO, tsundere, ancient dragon, noir detective, movie-trailer
voice, … — live in `seeds/README.md` (features) and **`VOICE-PRESETS.md`** (catalog).
Request any of them in chat: *"a mommy voice for a new character"*,
*"make drake drier"*.

### 4a. The control menu — everything you can tune

**A. The voice itself** (design-time, in chat — re-audition regenerates the
reference clips and the new timbre sticks across every line):
all 58 features from `seeds/README.md` — age · voice weight · pitch ·
intonation · wetness↔dryness · roughness · smokiness · breathiness ·
reediness · velvet · vocal fry · voice cracks · nasality · resonance
(head↔chest) · brightness · placement · vowel openness · rate · volume ·
intimacy · energy · steadiness · articulation · mumble · rhythm · pausing ·
emphasis · sentence endings · drawn-out vowels · dynamics · emotion baseline
· warmth · smile tone · seductiveness · authority · playfulness · menace ·
composure · softness · sharpness · confidence · anxiety · tiredness ·
melancholy · smugness · sarcasm · deadpan · theatricality · formality ·
politeness/keigo · laugh style · sigh style · verbal tics · stutter/lisp ·
accents/dialects · muttering · whisper — plus any of the 110 named presets
by name, plus lines from your `presets.csv` catalog.

**B. Per line** (no redesign — the take you already have stays):

| control | where |
|---|---|
| text | type anything — en · zh · ja · ko · yue, cross-lingual OK |
| voice | any seed character, or any 3–10 s reference clip you point it at |
| **pitch** | manifest column 5 (semitones, +3 = one tone up) or `--pitch` |
| **speed** | manifest column 6 (1.1 = 10% faster) or `--speed` |
| **volume** | manifest column 7 (dB, -2 = quieter) or `--volume` |
| **fx** | manifest column 8 or `--fx`: `robot` · `phone` · `reverb` · `normalize` (comma list, e.g. `reverb,normalize`) |
| take | re-render for a fresh take; `--seed` changes the roll |
| style | `--instruct` (CosyVoice3): rate / volume / emotion / dialect / robot |

**C. Per batch:** TSV manifest · `tools/dialogue2tsv.py` (script → manifest) ·
resume (existing outputs are skipped) · `--threads` · `--dry-run` first.

**D. Per character:** zero-shot (any clip, instant) → fine-tuned model (free
Colab, spot-on consistency) — hot-swapped per character in the batch.

**Your own catalog (`presets.csv`)** — the full gacha voice list (~1000 lines,
25 classes, each character = a base row + texture/pace variants). Drop the CSV at
the repo root and run:

```bash
python tools/presets_ingest.py
```

→ validates every line (sequential numbering, enums, base/variant blocks,
duplicate names, unescapes `&amp;`), then generates **`PRESET-CATALOG.md`** and
**`PRESET-CATALOG.html`** — browse/filter by class, pitch, pace, rarity; click a
row for its 58-feature bundle. Any line is orderable by number, name, or class:
*"new character with a **Villain & Mastermind** voice — line 361"* /
*"*Crimson Script*, but slower".

**Gacha roll / search** (no catalog needed):
```bash
python tools/presets_ingest.py --roll 3 --rarity 5 --class "Idol & Stage"   # roll 3 rarity-5 idols
python tools/presets_ingest.py --roll 5 --seed 7                            # reproducible roll
python tools/presets_ingest.py --find gravel                                # search any axis
``` Current values:

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
