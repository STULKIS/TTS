```
🎯 My TTS Plan v2 — Original Voices, Windows, No NVIDIA GPU
(v2 = fact-checked 2026-09-20; see tts-plan-review.md for every diff and why)

Your requirements: local TTS · original character voices (no VA cloning) ·
English, Chinese, Japanese, Korean · Windows PC without NVIDIA GPU.
Companion to local-tts-guide.md (the general reference — still TODO).

The stack at a glance
Tier    Tool                      What it gives you                                        Needs GPU?
Now     CosyVoice 3 (2512)        Style a voice by instruction — but the *timbre* comes     ❌ CPU works,
                                  from a reference clip, not from prose. 9 langs (zh en       ~2-5x realtime
                                  ja ko de es fr it ru) + 18 CN dialects, 24 kHz
Now+    Seed clips + this chat    Design voices by audition → save clips as a NAMED          ❌
                                  zero-shot speaker (add_zero_shot_spk → spk2info.pt)
Later   Qwen3-TTS 1.7B-VoiceDesign  Text → brand-new voice, NO reference audio needed.       ✅ bf16 ≈ 8GB
                                  (there is no 0.6B VoiceDesign; 0.6B CustomVoice/Base        (0.6B ≈ 4GB,
                                   = the cheap local option)                                   no VoiceDesign)
Later   GPT-SoVITS                Dedicated character model: 5 s zero-shot, 1 min fine-tune  ✅ train, or
                                  · en/ja/ko/yue/zh · MIT                                     ❌ CPU inference

────────────────────────────────────────────────────────────────────────
Part 0 — The correction that matters most
CosyVoice 3 is NOT "text → new voice". It is "reference clip → that voice,
styled by instruction". Two separate knobs:

  · WHO it sounds like  → prompt_wav (the clip)        ← your originality comes from here
  · HOW it says it       → instruct_text (dialect/emotion/rate/volume)

Officially-tested instructions are a fixed set (26) in cosyvoice/utils/common.py:
the 18 CN dialects, "as loudly as possible", "very soft voice", fastest/slowest rate,
very happy / sad / angry, Peppa-Pig style, robot style. Free-form English persona prose
("ancient dragon spirit, velvety…") often steers style but is unvalidated and will not
give you a specific timbre. Qwen3-TTS-VoiceDesign IS text→timbre, if that's what you want.

Part 1 — CosyVoice 3 on Windows (via WSL2)
Why WSL2: cleanest deps + sox/onnxruntime behave. It is NOT because ttsfrd is
required — ttsfrd is optional (repo falls back to wetext==0.0.4, pure Python).
Alternatives: Docker Desktop on the WSL2 backend; or a native Windows port
(you'll hit deepspeed==0.15.1 [linux-only marker] and the cu121 torch index).
~60-90 min total incl. downloads (~5-8 GB).

1. Enable WSL2 (PowerShell as Admin)
PowerShell
    wsl --install
Reboot, open Ubuntu, create user.

2. Install inside Ubuntu
Bash
    sudo apt update && sudo apt install -y git git-lfs sox libsox-dev unzip python3.10-venv
    git lfs install

    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    git submodule update --init --recursive      # Matcha-TTS etc. — required

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash
    exec bash

    conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice
    pip install -r requirements.txt
    # ⚠️ pinned torch is the cu121 index; on a GPU-less box CPU wheels install fine,
    #    but if resolution hiccups: pip install torch==2.3.1 torchaudio==2.3.1 --index-url
    #    https://download.pytorch.org/whl/cpu
    # optional, better zh/en text normalization (skip it the first pass):
    #   cd pretrained_models/CosyVoice-ttsfrd/ && unzip resource.zip -d .
    #   pip install ttsfrd_dependency-0.1-py3-none-any.whl ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

3. Download a model — use the SDK, not git clone
Bash
    python - <<'PY'
    from huggingface_hub import snapshot_download
    d = 'pretrained_models/Fun-CosyVoice3-0.5B'
    # ~5.2 GB. Saves 1.3 GB: that TRT flow-decoder onnx is GPU-only, useless on CPU.
    snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir=d,
                      ignore_patterns=['flow.decoder.estimator.*'])
    PY
    # The "RL" checkpoint is NOT a separate repo — it's llm.rl.pt inside that same repo.
    # To use it (better CER/WER on CV3-Eval), swap it into place; the loader wants 'llm.pt':
    cd pretrained_models/Fun-CosyVoice3-0.5B
    cp llm.pt llm.base.pt && cp llm.rl.pt llm.pt && cd -
    # cheap + faster fallback for bulk lines:
    python -c "from huggingface_hub import snapshot_download as s; \
               s('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
    # Why not `git clone https://huggingface.co/...`: git-lfs pointers crash the loader,
    # and CosyVoice2/3 dirs must contain CosyVoice-BlankEN/ (the code overrides that path).
    # If you must git-clone, clone the *exact* repo id into pretrained_models/<same-name>,
    # or use ModelScope: https://www.modelscope.cn/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

4. WebUI — optional, and not great for voice design
Bash
    python webui.py --port 7860 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
Windows browser → http://localhost:7860   (webui binds 0.0.0.0; default port is 8000!)
    Modes are in Chinese: 预训练音色(SFT) · 3s极速复刻(zero-shot) · 跨语种复刻(cross-lingual)
    · 自然语言控制(instruct). NOTE: there is no VC tab, and 自然语言控制 calls the v1
    inference_instruct with a preset-voice dropdown — that path needs CosyVoice-300M-Instruct.
    With Fun-CosyVoice3-0.5B the dropdown is empty. For CV3 voice work, use the script below.
    (Better zero-setup CV3 demo: HF/ModelScope Space "FunAudioLLM/Fun-CosyVoice3-0.5B",
     or a local GGUF runtime for CPU, e.g. cstr/cosyvoice3-0.5b-2512-GGUF.)

5. Script it — the API that actually exists on CV3
Python
    import sys; sys.path.append('third_party/Matcha-TTS')
    from cosyvoice.cli.cosyvoice import AutoModel
    import torchaudio
    from cosyvoice.utils.common import set_all_random_seed

    cv = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')   # CosyVoice3 subclasses CV2
    print(cv.sample_rate)                                                # 24000

    # A) styled speech in a voice you own (reference clip = timbre, instruct = style)
    set_all_random_seed(42)                                   # pin per character
    persona = 'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>'
    for i, chunk in enumerate(cv.inference_instruct2(
            '这个世界的命运，还没有定数。', persona, 'seeds/dragon_zh.wav', stream=False)):
        torchaudio.save(f'line_{i}.wav', chunk['tts_speech'], cv.sample_rate)

    # B) register the seed clip once → reuse as a named speaker, no prompt re-encoding
    cv.add_zero_shot_spk('You are a helpful assistant.<|endofprompt|>' + 'この世界の運命は、まだ決まっていない。',
                         'seeds/dragon_ja.wav', 'dragon')
    cv.save_spkinfo()          # persists into the model dir as spk2info.pt
    for i, chunk in enumerate(cv.inference_zero_shot('まだ終わっていない。', '', '',
                                                     zero_shot_spk_id='dragon', stream=False)):
        torchaudio.save(f'ja_{i}.wav', chunk['tts_speech'], cv.sample_rate)

Why (B) matters on CPU: frontend_zero_shot re-runs whisper-log-mel + speech_tokenizer_v3.onnx +
campplus on the clip for EVERY call; add_zero_shot_spk does it once and caches into spk2info.pt.
Also: the CV3 repo ships no spk2info.pt → the SFT dropdown is empty, which is exactly why
'choose a preset voice' is not a thing on CV3. (If you want text-described personas with NO
reference clip at all, that exists only on v1: CosyVoice-300M-Instruct + inference_instruct
+ '中文男'/'中文女' — old, weaker on ja/ko, but zero-reference.)

Language notes (CV3):
  · Chinese/English/Korean: feed text as-is.
  · Japanese: upstream recommends spaced KATAKANA (レキシ テキ セカイ ニ オイ テ ワ、…) —
    raw kanji text is a known weak path. Run your JP lines through a kana converter first.
  · <|zh|><|en|><|ja|><|ko|><|yue|> tags are the *v1* inference_cross_lingual API
    (example.py still documents them). CV3 cross-lingual = "You are a helpful assistant.<|endofprompt|>"
    + target-language text; it also does cross-lingual zero-shot cloning natively.
  · Bonus: CV3 supports pronunciation inpainting — [j][ǐ] pinyin / CMU for EN. Useful for
    names and constructed languages.

What to expect on CPU (real numbers, not vibes)
  ~0.5B model, RTF ≈ 2–5 → 20-90 s per spoken line depending on CPU single-core perf.
  Plan a batch run: 150 lines ≈ 1-2.5 h overnight. Fine for pre-rendered dialogue for
  videos/mods/games; not for live chat. Speedups that are CPU-legal: fp16=False (CPU has no
  fp16 path benefit), stream=False, CosyVoice2-0.5B instead of 3, quantized GGUF runtimes,
  and (B) above so prompts aren't re-encoded per line.

Part 2 — The seed trick 🌱 (this chat as voice designer)
Works exactly as you hoped, and it's now the officially-shaped workflow on both ends:

  1. Audition voices here → pick one per character.
  2. Ask for a seed pack: 10-20 in-character lines × (en/zh/ja/ko), same voice, mono
     WAV ≥ 24 kHz, no music, 3-10 s clips for prompts + longer ones for training.
  3. Locally:
     · CosyVoice2/3 → add_zero_shot_spk() + save_spkinfo() → permanent named speaker
     · Qwen3-TTS    → create_voice_clone_prompt(ref_audio, ref_text) (their README literally
                      documents "Voice Design then Clone" for character consistency across lines)
     · GPT-SoVITS   → 3-10 s reference for zero-shot, or ~1 min of clips → fine-tuned model
  4. Reproducibility: same seed + same reference clip = same timbre every run.

Caveats worth knowing:
  · The designed voice inherits the reference clip's quirks (breathiness, room tone, artifacts).
    Keep source clips clean; trim silence; don't mp3-round-trip them.
  · Timbre fidelity across languages is the hard part: one voice per language is safer than
    one voice × 4 languages. If you need one-performer-in-4-languages, generate the reference in
    the *target* language each time (CV3's cross-lingual cloning is built for this).
  · Legally: AI-generated voice ≠ anyone's protected voice. Apache-2.0 code/weights (keep the
    NOTICE if you ship the code). Still avoid the vendor's preset voices ('中文女' etc.) and
    never fine-tune on a clip you don't have rights to. Original voice ≠ original character.

Part 3 — Upgrade paths (when hardware changes)
Any NVIDIA GPU (used RTX 3060 12GB is the sweet spot)
  → Qwen3-TTS:  conda create -n qwen3-tts python=3.12 -y && pip install -U qwen-tts
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
                                      device_map='cuda:0', dtype=torch.bfloat16)
    wavs, sr = m.generate_voice_design(text='…', language='Chinese',
                                       instruct='低く落ち着きのある、古龍の化身のような女性')
    · VoiceDesign (text→timbre, no reference) = 1.7B ONLY, ~8GB bf16 (8-12GB reported w/o FA2)
    · 0.6B-CustomVoice / 0.6B-Base ≈ 4GB — Base = 3-s clone from ref_audio+ref_text, and it is
      the CPU-tolerant option if you'd rather skip the GPU (slow, but it runs)
    · 9 built-in speakers incl. Ono_Anna (JP) and Sohee (KO) — fine for temp lines, but they're
      vendor voices: for "original", use VoiceDesign → your own clip → clone prompt
    · flash-attn optional (sdpa works); fine-tuning + multi-speaker podcast gen in repo

Dedicated character models
  → GPT-SoVITS (MIT, ~60k stars): zero-shot from 5 s, fine-tune from ~1 min of audio,
    en/ja/ko/yue/zh; train in free Colab, infer at home on CPU (RTF ~0.03 on GPU; CPU is
    usable for batch). UI: python webui.py en_US. Pick v2Pro/v2ProPlus for cross-lingual quality.
    See local-tts-guide.md for the full dataset/slicer/ASR-labeling workflow (TODO).

Quick decision cheat-sheet
Want a voice NOW, zero setup........ ask this chat (audition + seed pack)
Want it local, free, CPU........... CosyVoice 3 via WSL2, using a seed clip as the speaker
                                     (and CV2-0.5B or GGUF when CPU speed bites)
Want text→timbre with no clip...... Qwen3-TTS 1.7B-VoiceDesign (needs NVIDIA)
Want a dedicated character model... GPT-SoVITS (train in Colab)
Want live/real-time dialogue....... none of the above on CPU — that needs a GPU + vLLM/TRT
```
