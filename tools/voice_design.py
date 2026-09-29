"""Instruction-driven voice design, separate from GPT-SoVITS and cosmetic DSP.

All 58 vocabulary axes are soft language instructions, NOT validated independent
acoustic parameters. Qwen3 VoiceDesign may ignore, blend or conflict on traits.
A new render is a new designed performance, not an identity-preserving clone.
No model dependencies are imported until the first actual inference.
"""
import math
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
             "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German",
             "it": "Italian", "ru": "Russian"}


def _field(key, label, group, choices=None, experimental=False):
    return {"key": key, "label": label, "group": group, "choices": choices,
            "experimental": experimental, "control_type": "soft_instruction"}


def _choices(key, label, group, values, experimental=False):
    return _field(key, label, group, values.split("|"), experimental)


# First choice in the UI is always Unspecified: no instruction is added.
SCHEMA = []
for group, rows in [
    ("Voice / texture", [
        ("age", "Apparent age", "young adult|middle-aged|elderly|ancient fantasy character"),
        ("gender", "Gender presentation", "feminine|masculine|androgynous"),
        ("weight", "Vocal weight", "thin and light|medium weight|full-bodied and heavy"),
        ("register", "Vocal pitch / register", "low register|mid register|high register"),
        ("intonation", "Pitch variety", "nearly monotone|subtle pitch variation|wide expressive intonation"),
        ("dryness", "Wet / dry tone", "dry papery timbre|neutral vocal texture|rounded liquid timbre, without added mouth noises"),
        ("roughness", "Roughness / rasp", "silky smooth|slightly raspy|gravelly"),
        ("smokiness", "Smokiness", "clear nonsmoky tone|slightly smoky|strongly smoky"),
        ("breathiness", "Breathiness", "clear supported phonation, no added breath noises|lightly breathy phonation|airy breathy phonation, not inserted noise"),
        ("reediness", "Reediness", "rounded non-reedy timbre|slightly reedy|distinctly reedy"),
        ("velvet", "Velvety tone", "plain clean tone|soft velvety tone|rich velvety tone"),
        ("fry", "Vocal fry", "no vocal fry|occasional light vocal fry|prominent creaky vocal fry"),
        ("cracks", "Voice cracks", "steady unbroken voice|occasional subtle voice cracks|emotionally cracking voice"),
        ("nasality", "Nasality", "low nasality|lightly nasal resonance|strongly nasal resonance"),
        ("resonance", "Resonance", "head-dominant resonance|balanced resonance|chest-dominant resonance"),
        ("brightness", "Brightness", "dark rounded timbre|neutral brightness|bright crystalline timbre"),
        ("placement", "Vocal placement", "forward placement|central placement|back-placed tone"),
        ("vowels", "Vowel openness", "compact closed vowels|natural vowels|open spacious vowels"),
    ]),
    ("Delivery / dynamics", [
        ("pace", "Acting pace", "slow and unhurried|conversational pace|brisk delivery"),
        ("loudness", "Acting loudness", "quiet supported speech|conversational loudness|projected loud speech"),
        ("intimacy", "Intimacy", "distant formal address|conversational distance|intimate close-mic delivery"),
        ("energy", "Energy", "low energy|moderate energy|high energetic delivery"),
        ("steadiness", "Steadiness", "slightly wavering delivery|natural steadiness|rock-steady delivery"),
        ("articulation", "Articulation", "relaxed articulation|clear natural articulation|crisp precise articulation"),
        ("mumble", "Mumbling", "no mumbling|occasional understated mumble|mumbled but intelligible delivery"),
        ("rhythm", "Rhythm", "clipped staccato rhythm|natural conversational rhythm|smooth flowing rhythm"),
        ("pausing", "Pausing", "few brief pauses|natural phrasing pauses|thoughtful pauses"),
        ("emphasis", "Emphasis", "restrained emphasis|natural word emphasis|strong expressive word emphasis"),
        ("endings", "Sentence endings", "falling sentence endings|rising sentence endings|trailing-off sentence endings"),
        ("drawl", "Drawn-out vowels", "short clean vowels|subtle vowel lengthening|drawn-out vowels"),
        ("dynamics", "Loudness dynamics", "even loudness|subtle loudness variation|large expressive loudness swells"),
    ]),
    ("Emotion / personality", [
        ("emotion", "Emotion", "neutral|happy|excited|sad|angry|afraid|surprised|tender|disgusted"),
        ("warmth", "Warmth", "cool detached tone|neutral warmth|warm reassuring tone"),
        ("smile", "Smile tone", "unsmiling tone|slight smile in the voice|bright smiling tone"),
        ("seductiveness", "Flirtatious tone", "non-flirtatious|subtly flirtatious adult tone|confident flirtatious adult tone"),
        ("authority", "Authority", "timid delivery|self-assured delivery|commanding authority"),
        ("playfulness", "Playfulness", "serious delivery|lightly playful|mischievously playful"),
        ("menace", "Menace", "non-threatening|subtly ominous|menacing theatrical delivery"),
        ("composure", "Composure", "flustered delivery|mostly composed|fully composed and controlled"),
        ("softness", "Softness", "hard guarded delivery|gentle delivery|soft vulnerable delivery"),
        ("sharpness", "Sharpness", "rounded gentle delivery|pointed clear delivery|sharp biting delivery"),
        ("confidence", "Confidence", "hesitant and uncertain|comfortable and assured|strongly confident"),
        ("anxiety", "Anxiety", "relaxed|slightly nervous|anxious and tense"),
        ("tiredness", "Tiredness", "wide-awake|weary|sleepy"),
        ("melancholy", "Melancholy", "no melancholy|wistful|deeply melancholy"),
        ("smugness", "Smugness", "unassuming|slightly smug|self-satisfied and smug"),
        ("sarcasm", "Sarcasm", "sincere and literal|subtly sarcastic|clearly sarcastic"),
        ("deadpan", "Deadpan", "emotionally responsive|restrained deadpan|flat dry deadpan"),
        ("theatricality", "Theatricality", "naturalistic acting|heightened acting|theatrical stage delivery"),
        ("formality", "Formality", "casual delivery without changing the words|neutral formality|formal delivery without changing the words"),
        ("politeness", "Politeness", "blunt delivery|courteous delivery|deferential delivery without rewriting the text"),
    ]),
    ("Quirks / special — experimental", [
        ("laugh", "Laugh style", "no laughter|a brief soft giggle before speaking|a brief warm chuckle before speaking|a brief theatrical cackle before speaking"),
        ("sigh", "Sigh style", "no sighing|a brief weary sigh before speaking|a brief relieved sigh before speaking"),
        ("tics", "Verbal tics", "none|slight hesitant vocal mannerisms without adding words|dry amused vocal mannerisms without adding words"),
        ("quirks", "Speech quirks", "none|a light stutter|a light lisp"),
        ("accent", "Accent / dialect request", None),
        ("muttering", "Muttering", "no muttering|slightly under-the-breath delivery|muttered but intelligible speech"),
        ("whisper", "Whisper register", "normal voiced speech|soft near-whisper|whispered speech"),
    ]),
]:
    for key, label, values in rows:
        experimental = group.startswith("Quirks") or key in {"dryness", "placement", "vowels", "cracks", "reediness"}
        SCHEMA.append(_choices(key, label, group, values, experimental) if values else
                      _field(key, label, group, experimental=True))

FIELDS = {f["key"]: f for f in SCHEMA}


def compile_instruction(design):
    """Validate UI values and build the EXACT instruction given to the model."""
    if not isinstance(design, dict):
        raise ValueError("design must be an object")
    unknown = set(design) - set(FIELDS) - {"description", "direction", "emphasize", "crying"}
    if unknown:
        raise ValueError("Unknown design controls: " + ", ".join(sorted(unknown)))
    lines = ["Speak the supplied dialogue, not these instructions. Preserve the dialogue words."]
    for field in SCHEMA:
        value = design.get(field["key"], "")
        if value in (None, ""):
            continue
        if not isinstance(value, str) or len(value) > 160:
            raise ValueError(field["label"] + " must be a short text value")
        if field["choices"] is not None and value not in field["choices"]:
            raise ValueError("Invalid choice for " + field["label"])
        lines.append(field["label"] + ": " + value + ".")
    for key, label, limit in [("description", "Voice design brief", 1200),
                               ("direction", "Acting direction", 1000),
                               ("emphasize", "Emphasize these dialogue words", 240)]:
        value = design.get(key, "")
        if value:
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"{label} must be at most {limit} characters")
            lines.append(label + ": " + value.strip())
    crying = design.get("crying", "")
    if crying not in ("", "none", "tearful voice", "speaking through restrained sobs"):
        raise ValueError("Invalid crying request")
    if crying and crying != "none":
        lines.append("Experimental acting request: " + crying + ".")
    if len(lines) == 1:
        lines.append("A clear natural adult voice with conversational delivery.")
    return "\n".join(lines)


def generation_options(req):
    """Only pass documented generation options. No hidden DSP or truncation knob."""
    limits = {"temperature": (.1, 1.5, 1.), "top_k": (1, 50, 15),
              "top_p": (.5, 1., 1.), "repetition_penalty": (1., 2., 1.2)}
    result = {}
    for key, (low, high, default) in limits.items():
        try:
            value = float(req.get(key, default))
        except (ValueError, TypeError):
            raise ValueError("Invalid " + key)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be in {low}..{high}")
        if key == "top_k" and not value.is_integer():
            raise ValueError("top_k must be an integer")
        result[key] = int(value) if key == "top_k" else value
    return result


class VoiceDesigner:
    """Official Qwen adapter. One model; load lazily; caller serializes requests."""
    def __init__(self, threads=None):
        self.model = None
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)

    def load(self):
        if self.model is not None:
            return
        import psutil
        if psutil.virtual_memory().available < 9 * 1024 ** 3:
            raise RuntimeError("Voice Designer CPU mode needs at least 9 GiB currently available RAM before loading. "
                               "Close Classic Studio and other large apps, then retry. Nothing was downloaded by this check.")
        import torch
        from qwen_tts import Qwen3TTSModel
        torch.set_num_threads(self.threads)
        print("[design] Loading/downloading " + MODEL_ID + " on CPU. This can take several minutes.", flush=True)
        model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32,
                                             attn_implementation="sdpa")
        self.model = model

    def generate(self, req):
        import numpy as np
        text = req.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 600:
            raise ValueError("Voice Designer: use 1–600 characters per line; split longer dialogue into script rows.")
        language = req.get("text_lang", "en")
        if language not in LANGUAGES:
            raise ValueError("This VoiceDesign model does not offer that language; Cantonese is not enabled.")
        instruction = compile_instruction(req.get("design", {}))
        options = generation_options(req)
        try:
            seed = int(req.get("seed", -1))
        except (ValueError, TypeError):
            raise ValueError("Seed must be an integer")
        if seed < -1 or seed > 2**32 - 1:
            raise ValueError("Seed must be -1 or 0..4294967295")
        self.load()
        import torch
        import secrets
        torch.manual_seed(secrets.randbelow(2**32) if seed == -1 else seed)
        # Real voice-model conditioning: no breath synthesis, pitch modulation,
        # instruction tags added to the spoken text, or assistant cloud TTS.
        with torch.inference_mode():
            wavs, sr = self.model.generate_voice_design(text=text, language=LANGUAGES[language],
                                                       instruct=instruction, **options)
        raw = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if raw.size < sr // 10 or not np.isfinite(raw).all() or np.max(np.abs(raw)) < 1e-6:
            raise RuntimeError("Voice Designer produced empty, silent or invalid audio; try a shorter direction.")
        return sr, raw.copy(), instruction


def self_test(output):
    """Run REAL local inference before marking a Windows installation ready."""
    import json
    import soundfile as sf
    sr, wav, instruction = VoiceDesigner().generate({"text": "Hello. This is my voice test.",
        "text_lang": "en", "seed": 17, "design": {"warmth": "warm reassuring tone"}})
    sf.write(output, wav, sr)
    report = {"model": MODEL_ID, "device": "cpu", "sample_rate": sr, "seconds": len(wav)/sr,
              "instruction": instruction, "note": "Inference succeeded; this is not a trait-adherence or quality certification."}
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[ok] Actual model produced the setup test. Listen to it to judge quality.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", type=Path, required=True)
    args = parser.parse_args()
    self_test(args.self_test)
