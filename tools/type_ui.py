#!/usr/bin/env python3
"""Gacha TTS Studio — the type-and-speak WebUI for GPT-SoVITS.

The full front end: gacha preset catalog (this repo's presets.csv, 1000 lines),
per-line controls (pitch / speed / volume / FX), character seed packs
(seeds/: dragon & drake x en/zh/ja/ko) or any uploaded reference clip,
type your text, click Speak, play + download.

Runs IN your GPT-SoVITS install (same conda env / integrated package):
    cd D:\\GSV\\GPT-SoVITS-v2pro-20250604
    runtime\\python.exe D:\\TTS\\tools\\type_ui.py --seeds D:\\TTS\\seeds
Then open http://127.0.0.1:7861 in your browser.
(Or just double-click the "Gacha TTS Studio" desktop icon.)

Options:
    --seeds DIR       seed pack root (dirs: <char>/<lang>[N].wav + .txt)
    --presets FILE    preset catalog csv (default: <repo>/presets.csv)
    --gsv-root DIR    GPT-SoVITS repo root (default: current directory)
    --port N          default 7861 (stock webui 9874, api_v2 9880)
    --bind ADDR       default 127.0.0.1 (local machine tool)
    --tts-config P    default GPT_SoVITS/configs/tts_infer.yaml (v2/v2Pro)
    --open            open the browser automatically when the server starts

How it works: the same in-process tts_pipeline.run(req) call api_v2.py uses.
Pitch/volume/FX are applied as post-processing with tools/audio_fx.py
(the tested DSP: pitch = WSOLA-style shift, speed is engine-side
speed_factor, FX = robot/phone/reverb/normalize).

Depends only on the GPT-SoVITS environment (fastapi/uvicorn/numpy/soundfile).
"""
# NOTE: no `from __future__ import annotations` here - FastAPI must see the
# real Request type object on the handler signature (string annotations on a
# function defined inside build_app do not resolve, and FastAPI would treat
# `request` as a required query field -> HTTP 422 on every call).
import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# CPU tuning: half the logical threads (= physical cores). torch defaults to
# every logical thread and thrashes on SMT CPUs like the Ryzen 5 5600G.
_THREADS = max(1, (os.cpu_count() or 4) // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

# Set BLAS/OpenMP limits BEFORE audio_fx imports numpy.
import audio_fx  # noqa: E402
from studio_cache import TakeCache, ReferenceCache, file_version  # noqa: E402

LANGS = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}
_CLIP_RE = re.compile(r"^([a-z]+)(\d+)?$")

# preset catalog axis -> engine control value
PITCH_SHIFT = {  # semitones for the post-hoc pitch shifter
    "High": 4, "Soft": 2, "Mid": 1, "Neutral": 0, "Low": -3, "Deep": -5, "Synth": 0,
}
PACE_SPEED = {  # engine-side speed_factor, from PACE_FEATURES semantics
    "Dash": 1.25, "Burst": 1.20, "Lilt": 1.15, "Tick": 1.10, "Pulse": 1.00,
    "Tale": 0.95, "Measure": 0.85, "Drift": 0.80, "Lounge": 0.75, "Pause": 0.70,
}


def scan_pack(seeds: Path) -> list[dict]:
    """[{char, lang, clip, transcript, seconds}] for every <char>/<lang>*.wav + .txt."""
    items: list[dict] = []
    if not seeds.is_dir():
        return items
    for char_dir in sorted(p for p in seeds.iterdir() if p.is_dir()):
        wavs = sorted(char_dir.glob("*.wav"), key=lambda w: (w.stem, w.name))
        for wav in wavs:
            try:
                m = _CLIP_RE.match(wav.stem)
                lang = m.group(1) if m else wav.stem
                if lang not in LANGS:
                    continue
                txt = wav.with_suffix(".txt")
                transcript = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
                seconds = 0.0
                try:
                    with wave.open(str(wav), "rb") as w:
                        seconds = w.getnframes() / max(1, w.getframerate())
                except (wave.Error, EOFError, OSError):
                    pass
                items.append({"char": char_dir.name, "lang": lang,
                              "clip": wav.name, "transcript": transcript, "seconds": round(seconds, 2)})
            except Exception:  # one bad/locked file must never kill the whole scan
                continue
    return items


def load_presets(path: Path) -> list[dict]:
    """rows of presets.csv -> [{no, name, class, gender, rarity, pitch, pace,
    signature, description, pitch_shift, speed}]. Bad rows are skipped."""
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                pitch = (r.get("pitch") or "").strip()
                pace = (r.get("pace") or "").strip()
                rows.append({
                    "no": int(r["no"]),
                    "name": (r.get("name") or "").strip(),
                    "class": (r.get("class") or "").strip(),
                    "gender": (r.get("gender") or "").strip(),
                    "rarity": int(r["rarity"]),
                    "pitch": pitch,
                    "pace": pace,
                    "signature": (r.get("signature") or "").strip().lower() == "yes",
                    "description": (r.get("description") or "").strip(),
                    "pitch_shift": PITCH_SHIFT.get(pitch, 0),
                    "speed": PACE_SPEED.get(pace, 1.0),
                })
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def _life_score(x, sr) -> float:
    """Experimental acoustic-variation score, NOT a naturalness/acting score.
    Noise can score highly too; never use this as a quality guarantee."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return 0.0
    fl = max(8, int(0.04 * sr))
    if x.size < fl:
        return 0.0
    nf = x.size // fl
    fr = x[: nf * fl].reshape(nf, fl)
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-9
    zcr = (np.diff(np.signbit(fr), axis=1) != 0).mean(1) * float(sr)
    peak = float(np.max(np.abs(x))) + 1e-9
    crest = 20.0 * np.log10(peak / (float(np.sqrt(np.mean(x ** 2))) + 1e-9))
    dyn = float(np.std(rms) / np.mean(rms))
    zdyn = float(np.std(zcr)) / 1500.0
    return round(max(0.0, min(10.0, (crest / 6.0 + dyn * 3.0 + zdyn) * 1.6)), 2)


def build_app(gsv_root: Path, seeds: Path, presets_path: Path, tts_config_path: str):
    # GPT-SoVITS imports (must run inside its environment)
    sys.path.insert(0, str(gsv_root))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

    from fastapi import FastAPI, Request  # noqa: E402
    from starlette.concurrency import run_in_threadpool
    import numpy as np  # noqa: E402
    import soundfile as sf  # noqa: E402
    from fastapi.responses import HTMLResponse, Response, JSONResponse  # noqa: E402

    print(f"[studio] loading GPT-SoVITS pipeline (config={tts_config_path}) ...")
    try:
        tts_config = TTS_Config(tts_config_path)
        try:
            import torch
            torch.set_num_threads(_THREADS)
        except Exception:
            pass
        tts_pipeline = TTS(tts_config)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"[abort] GPT-SoVITS pipeline failed to load: {e}\n"
            "  · Weights missing/incomplete? See local-tts-guide.md §3 (manual model placement).\n"
            "  · CUDA error on a no-GPU box? Run: python tools/cpu_config.py --gsv-root <GPT-SoVITS folder>")
    pack = scan_pack(seeds)
    presets_rows = load_presets(presets_path)
    print(f"[studio] ready: version={tts_config.version} languages={tts_config.languages} "
          f"pack={len(pack)} clip(s) under {seeds} · {len(presets_rows)} presets from {presets_path.name}")

    take_cache = TakeCache()
    reference_cache = ReferenceCache()
    render_lock = threading.Lock()  # GPT-SoVITS has mutable prompt/RNG state.
    last_prompt = None

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            reference_cache.close()

    APP = FastAPI(lifespan=lifespan)
    uploads = Path(tempfile.gettempdir()) / "type_ui_uploads"
    uploads.mkdir(exist_ok=True)

    @APP.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE_HTML.replace("__PACK__", json.dumps(pack, ensure_ascii=False))

    @APP.get("/api/pack")
    async def pack_list():
        # live re-scan: recovers from a startup scan that raced an antivirus
        # copy, and tells the UI exactly where it looked when empty
        items = scan_pack(seeds)
        wav_files = len(list(seeds.rglob("*.wav"))) if seeds.is_dir() else 0
        return {"path": str(seeds), "exists": seeds.is_dir(),
                "wav_files": wav_files, "count": len(items), "items": items}

    @APP.get("/api/status")
    async def status():
        return {"ready": True, "version": tts_config.version,
                "languages": tts_config.languages, "pack": len(pack),
                "presets": len(presets_rows), "cpu_threads": _THREADS,
                "take_cache_mb": take_cache.max_bytes // (1024 * 1024)}

    @APP.get("/api/presets")
    async def presets():
        return {"count": len(presets_rows),
                "classes": sorted({p["class"] for p in presets_rows}),
                "presets": presets_rows}

    @APP.post("/api/tts")
    async def tts(request: Request):
        req = await request.json()
        # Do not block status/UI requests during CPU inference. Serialize model
        # calls: extra concurrent model jobs waste RAM and race its prompt cache.
        return await run_in_threadpool(render_locked, req)

    def render_locked(req):
        queued = time.perf_counter()
        with render_lock:
            response = render(req)
            response.headers["X-Total-Seconds"] = f"{time.perf_counter() - queued:.3f}"
            return response

    def render(req):
        nonlocal last_prompt
        started = time.perf_counter()
        # resolve reference audio: a pack clip (relative to the seeds dir),
        # a plain absolute path, or an uploaded temp file
        ref = req.get("ref_audio_path", "")
        if ref.startswith("perf:"):
            # performance bank take: shipped under <repo>/perf with its transcript
            perf_dir = seeds.parent / "perf"
            perf_id = Path(ref[len("perf:"):]).name
            ref_path = perf_dir / f"{perf_id}.wav"
            if not ref_path.is_file():
                return JSONResponse(status_code=400, content={"message": f"performance take not found: {perf_id} (REPAIR.bat restores it)"})
            side = perf_dir / f"{perf_id}.txt"
            if side.is_file():
                req["prompt_text"] = side.read_text(encoding="utf-8").strip()
            if not req.get("prompt_lang"):
                req["prompt_lang"] = "en"
        elif ref.startswith("upload:"):
            ref_path = Path(ref[len("upload:"):])
        else:
            ref_path = Path(ref)
            if not ref_path.is_absolute() and not ref_path.is_file():
                cand = seeds / ref
                if cand.is_file():
                    ref_path = cand
        if not ref_path.is_file():
            return JSONResponse(status_code=400, content={"message": f"ref_audio_path not found: {ref}"})
        ref = str(ref_path.resolve())
        # A stable fitted path lets GPT-SoVITS reuse reference encoding on each
        # script line; the old code made a new temp path on every request.
        try:
            source_version = file_version(ref)
            ref = reference_cache.fit(ref)
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"cannot read reference audio: {e}"})
        try:
            engine_batch = int(req.get("batch_size", 1))
        except (ValueError, TypeError):
            engine_batch = 1
        engine_batch = engine_batch if engine_batch in (1, 2, 4) else 1
        prompt_text = req.get("prompt_text", "")
        if not prompt_text:
            return JSONResponse(status_code=400,
                                content={"message": "prompt_text (verbatim transcript of the reference clip) is required"})
        text_lang = req.get("text_lang", "en")
        payload = {
            "text": req.get("text", ""),
            "text_lang": text_lang,
            "ref_audio_path": ref,
            "prompt_text": prompt_text,
            "prompt_lang": req.get("prompt_lang", "") or text_lang,
            "top_k": req.get("top_k", 15),
            "top_p": req.get("top_p", 1.0),
            "temperature": req.get("temperature", 1.0),
            "text_split_method": req.get("text_split_method", "cut5"),
            "fragment_interval": req.get("fragment_interval", 0.3),
            "batch_size": engine_batch,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": req.get("speed_factor", 1.0),
            "seed": req.get("seed", -1),
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": req.get("repetition_penalty", 1.2),
        }
        if not payload["text"].strip():
            return JSONResponse(status_code=400, content={"message": "text is required"})
        # Key every inference setting, exact text/transcript and source version.
        # DSP controls intentionally are NOT in this key: reuse raw speech when
        # tweaking FX. A random seed only reuses a take with explicit UI consent.
        key_payload = dict(payload, ref_audio_path=source_version)
        cache_key = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        reuse = req.get("reuse_take", False) is True
        cached = take_cache.get(cache_key) if reuse else None
        hit = cached is not None
        infer_seconds = 0.0
        if cached is None:
            # Upstream caches by path/text, not file version or prompt language.
            prompt_identity = (source_version, prompt_text, payload["prompt_lang"])
            prompt_cache = getattr(tts_pipeline, "prompt_cache", None)
            if isinstance(prompt_cache, dict) and last_prompt != prompt_identity:
                if last_prompt is None or last_prompt[0] != source_version:
                    prompt_cache["ref_audio_path"] = None
                prompt_cache["prompt_text"] = None
            before_infer = time.perf_counter()
            generator = None
            try:
                generator = tts_pipeline.run(payload)
                sr, audio = next(generator)
                # Own this buffer: model/DSP mutations must not corrupt a take.
                raw = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
            except Exception as e:
                last_prompt = None
                return JSONResponse(status_code=500, content={"message": f"tts failed: {e}"})
            finally:
                if generator is not None and hasattr(generator, "close"):
                    generator.close()
            infer_seconds = time.perf_counter() - before_infer
            last_prompt = prompt_identity
            raw_score = _life_score(raw, sr)
            cached = (sr, raw, raw_score)
            take_cache.put(cache_key, cached, raw.nbytes)
        sr, raw, raw_score = cached
        # per-line controls: pitch shift + volume (post-hoc DSP), speed was engine-side
        x = raw.copy()
        try:
            pitch = float(req.get("pitch", 0.0) or 0.0)
            volume = float(req.get("volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            pitch, volume = 0.0, 0.0
        if pitch:
            x = audio_fx.pitch_shift(x, sr, pitch)
        if volume:
            x = audio_fx.apply_volume(x, volume)
        for name in audio_fx.parse_fx(str(req.get("fx", "") or "")):
            x = audio_fx.apply_fx(x, sr, name)
        # trim the engine's trailing inter-fragment gap (~0.3 s of silence),
        # keeping only a short tail after the last real sound
        nz = np.nonzero(np.abs(x) > 1e-4)[0]
        if nz.size:
            x = x[:min(len(x), int(nz[-1]) + int(0.05 * sr))]
        x = audio_fx._peak_limit(x)  # final safety: never write clipped samples
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV")
        return Response(buf.getvalue(), media_type="audio/wav",
                        headers={"X-Life-Score": str(raw_score),
                                 "X-Take-Cache": "hit" if hit else "miss",
                                 "X-Inference-Seconds": f"{infer_seconds:.3f}",
                                 "X-Render-Seconds": f"{time.perf_counter() - started:.3f}"})

    @APP.post("/api/upload")
    async def upload(request: Request):
        form = await request.form()
        file = form.get("file")
        if file is None or not getattr(file, "filename", ""):
            return JSONResponse(status_code=400, content={"message": "no file uploaded"})
        data = await file.read()
        name = f"{uuid.uuid4().hex}.wav"
        (uploads / name).write_bytes(data)
        return {"path": str(uploads / name)}

    return APP


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gacha TTS Studio</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
       max-width: 72rem; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: .25rem; }
.sub { opacity: .7; }
.card { border: 1px solid rgba(128,128,128,.35); border-radius: .75rem;
        padding: 1rem 1.25rem; margin: 1.25rem 0; }
h2 { font-size: 1rem; margin: 0 0 .5rem 0; }
.row { display: flex; flex-wrap: wrap; gap: .75rem; margin: .75rem 0; align-items: center; }
label { font-size: .85rem; opacity: .8; display: block; margin-bottom: .2rem; }
select, input[type=number], input[type=text] { font: inherit; padding: .3rem .5rem; border-radius: .4rem;
        border: 1px solid rgba(128,128,128,.5); background: transparent; color: inherit; }
textarea { width: 100%; box-sizing: border-box; font: inherit; min-height: 5.5rem;
        padding: .5rem .7rem; border-radius: .5rem; border: 1px solid rgba(128,128,128,.5);
        background: transparent; color: inherit; resize: vertical; }
button { font: inherit; padding: .55rem 1.4rem; border-radius: .5rem; border: 0;
        background: #4f7cff; color: #fff; cursor: pointer; }
button.ghost { background: transparent; color: inherit;
        border: 1px solid rgba(128,128,128,.5); }
button:disabled { opacity: .5; cursor: wait; }
.clipinfo { font-size: .85rem; opacity: .75; }
audio { width: 100%; margin-top: .75rem; }
.err { color: #ff6b6b; }
.ok { color: #51cf66; }
#tabs button { background: transparent; color: inherit; border: 0; border-bottom: 2px solid transparent;
        border-radius: 0; padding: .4rem .8rem; }
#tabs button.on { border-bottom-color: #4f7cff; }
.chip { display: inline-block; font-size: .72rem; padding: .05rem .5rem; margin: .1rem .15rem .1rem 0;
        border: 1px solid rgba(128,128,128,.5); border-radius: 1rem; opacity: .85; }
.prow { border: 1px solid rgba(128,128,128,.3); border-radius: .5rem; padding: .4rem .6rem;
        margin: .35rem 0; cursor: pointer; }
.prow:hover { border-color: #4f7cff; }
.pbrief { font-size: .8rem; opacity: .7; margin-top: .15rem; }
#plist { max-height: 19rem; overflow-y: auto; }
.apreset { border-top: 1px dashed rgba(128,128,128,.4); margin-top: .6rem; padding-top: .5rem;
        font-size: .9rem; }
input[type=range] { width: 9rem; }
.fx { display: flex; gap: .9rem; flex-wrap: wrap; align-items: center; }
.fx label { display: flex; align-items: center; gap: .3rem; margin: 0; font-size: .85rem; }
a.dl { font-size: .85rem; }
.controlnav { display:flex; flex-wrap:wrap; gap:1rem; padding:.8rem; border:1px solid #8886; border-radius:.5rem; }
.controlnav a { color:inherit; }
fieldset { border:1px solid #8886; border-radius:.5rem; padding:1rem; min-width:0; }
legend { font-weight:600; }
#expr { flex:1 1 100%; }
#expr > div:not(.row) { display:inline-block; margin:.4rem .8rem .4rem 0; vertical-align:top; }
.controlgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:.7rem; }
.controlgrid input, .controlgrid select { box-sizing:border-box; max-width:100%; width:100%; }
#queue > li { margin:1rem 0; padding:1rem; border:1px solid #8886; border-radius:.5rem; }
#queue textarea { min-height:4rem; }
#queue audio { max-width:35rem; }

</style>
</head>
<body>
<h1>🎮 Gacha TTS Studio</h1>
<p class="sub">Controls edition · GPT-SoVITS zero-shot · preset catalog · per-line controls. <span id="ver"></span></p>
<p id="demobanner" class="err" style="display:none">DEMO ENGINE — this preview renders a test tone instead of a real voice (the real models run only on your PC). Good for trying the controls; audio quality says nothing about the real renders.</p>

<nav class="controlnav" aria-label="Control menu">
  <a href="#preset-panel">Preset catalog</a><a href="#voice-panel">Reference &amp; voice traits</a>
  <a href="#render-panel">Pitch / pace / volume</a><a href="#expr">Delivery &amp; sampling</a>
  <a href="#fx-panel">FX</a><a href="#custom-panel">My setups</a><a href="#script-panel">Per-line editor</a>
</nav>

<div class="card" id="preset-panel">
  <h2>1 · Gacha preset — pick the voice design</h2>
  <div class="row">
    <div style="flex:1"><label>Search name or brief</label><input type="text" id="q" placeholder="e.g. gravel, whisper, idol..."></div>
    <div><label>Class</label><select id="cls"></select></div>
    <div><label>Gender (catalog)</label><select id="genderfilter"><option value="all">all</option></select></div>
    <div><label>Pitch family</label><select id="pitchfilter"><option value="all">all</option></select></div>
    <div><label>Pace family</label><select id="pacefilter"><option value="all">all</option></select></div>
    <div><label>Rarity</label>
      <select id="rar"><option value="all">any</option><option value="3">★★★</option><option value="4">★★★★</option><option value="5">★★★★★</option></select></div>
    <div style="align-self:flex-end"><button id="roll" class="ghost">🎲 ROLL</button></div>
  </div>
  <div class="clipinfo" id="pcount">loading catalog…</div>
  <p class="clipinfo">Catalog entries are design briefs, not 1,000 separately trained voices. Applying one sets pitch, pace and an available performance reference.</p>
  <div id="plist"></div>
  <button id="morepresets" type="button" class="ghost">Show more presets</button>
  <div class="apreset" id="apreset" style="display:none"></div>
</div>

<div class="card" id="voice-panel">
  <h2>2 · Voice — the reference it is cloned from</h2>
  <p><strong>Voice traits — reference-dependent, not synthesis sliders:</strong> age, weight, wet/dry tone, rasp, smokiness, breathiness, fry, nasality, resonance, brightness, accent, whisper and personality. Choose or upload a reference with those qualities. A catalog description does not make this engine redesign a voice.</p>
  <div id="tabs"><button id="tab-preset" class="on" onclick="switchTab('preset')">Preset characters</button>
  <button id="tab-custom" onclick="switchTab('custom')">Any reference clip</button></div>
  <div id="pane-preset">
    <div class="row">
      <div><label>Character</label><select id="char"></select></div>
      <div><label>Language</label><select id="lang"></select></div>
      <div><label>Clip</label><select id="clip"></select></div>
    </div>
    <div class="row">
      <div style="flex:1"><label>Performance reference — influences delivery; emotion transfer is not guaranteed</label>
        <select id="perf">
          <option value="">Use selected character/language clip</option>
          <optgroup label="dragon — performance bank">
            <option value="excite-dragon">🔥 Excited — bouncing off the walls</option>
            <option value="warm-dragon">🫖 Warm — soft fireside</option>
            <option value="dramatic-dragon">🎭 Dramatic — stage tragedy</option>
            <option value="cheer-dragon">🌸 Cheerful — bright greetings</option>
            <option value="dark-dragon">🌑 Dark — purring menace</option>
          </optgroup>
          <optgroup label="drake — performance bank">
            <option value="excite-drake">🔥 Excited — bouncing off the walls</option>
            <option value="warm-drake">🫖 Warm — soft fireside</option>
            <option value="dramatic-drake">🎭 Dramatic — stage tragedy</option>
            <option value="cheer-drake">🌸 Cheerful — bright greetings</option>
            <option value="dark-drake">🌑 Dark — purring menace</option>
          </optgroup>
        </select></div>
    </div>
    <div class="clipinfo" id="clipinfo"></div>
  </div>
  <div id="pane-custom" style="display:none">
    <div class="row">
      <div><label>Reference clip (5–10 s wav)</label><input type="file" id="ref" accept=".wav,audio/*"></div>
      <div><label>Reference language</label>
        <select id="reflang"><option>en</option><option>zh</option><option>ja</option><option>ko</option><option>yue</option></select></div>
    </div>
    <div><label>Transcript of the clip (verbatim)</label><textarea id="reftext" style="min-height:3.5rem"></textarea></div>
  </div>
</div>

<div class="card" id="render-panel">
  <h2>3 · Your text + controls</h2>
  <div class="row">
    <div style="flex:1"><label>Your text</label><textarea id="text" placeholder="Type anything…"></textarea></div>
  </div>
  <div class="row">
    <div><label>Text language</label>
      <select id="textlang"><option>en</option><option>zh</option><option>ja</option><option>ko</option><option>yue</option></select></div>
    <div><label>Pitch <span id="pitchv">0</span> st</label>
      <input type="range" id="pitch" min="-12" max="12" step="1" value="0"></div>
    <div><label>Speed</label><input type="number" id="speed" value="1.0" min="0.5" max="1.5" step="0.05"></div>
    <div><label>Volume <span id="volv">0</span> dB</label>
      <input type="range" id="vol" min="-12" max="12" step="1" value="0"></div>
    <div><label>Seed (-1 = random)</label><input type="number" id="seed" value="-1"></div>
    <div><label>Sentence batching (render speed, not speaking pace)</label>
      <select id="enginebatch"><option value="1" selected>1 · lowest RAM</option>
        <option value="2">2 · try for longer text</option><option value="4">4 · higher RAM</option></select></div>
    <fieldset id="expr"><legend>Delivery &amp; sampling — all controls</legend>
      <div class="row">
        <button type="button" id="style-subtle">Subtle</button>
        <button type="button" id="style-balanced">Balanced</button>
        <button type="button" id="style-animated">Animated</button>
        <button type="button" id="style-fever">Fever</button>
      </div>
      <div><label>Temperature (higher = more variation, also more errors)</label>
        <input type="number" id="temperature" min="0.1" max="1.5" step="0.05" value="1.0"></div>
      <div><label>top_k</label><input type="number" id="top_k" min="1" max="50" step="1" value="15"></div>
      <div><label>top_p</label><input type="number" id="top_p" min="0.5" max="1" step="0.05" value="1"></div>
      <div><label>Repetition penalty</label>
        <input type="number" id="rep" min="1" max="2" step="0.05" value="1.2"></div>
      <div><label>Phrasing</label><select id="splitm">
        <option value="cut0">continuous delivery (no splits)</option>
        <option value="cut1">group four sentences</option>
        <option value="cut2">group about 50 characters</option>
        <option value="cut3">split at Chinese full stops</option>
        <option value="cut4">split at English periods</option>
        <option value="cut5" selected>split at punctuation</option>
      </select></div>
      <div><label>Pause between phrases (s)</label>
        <input type="number" id="gap" value="0.3" min="0.05" max="1" step="0.05"></div>
    </fieldset>
  </div>
  <p class="clipinfo">Natural delivery uses the reference performance, not added breaths. Optional FX below alter the sound; they do not create acting.</p>
  <label><input type="checkbox" id="reuse" checked> Reuse matching take — repeat text or adjust FX without resynthesizing</label>
  <p class="clipinfo">Fresh take / per-line reroll always synthesize again. Sentence batching only helps text with multiple split segments; continuous phrasing stays one segment.</p>
  <div class="row fx" id="fx-panel">
    <label><input type="checkbox" id="fx-robot"> robot</label>
    <label><input type="checkbox" id="fx-phone"> phone</label>
    <label><input type="checkbox" id="fx-reverb"> reverb</label>
    <label><input type="checkbox" id="fx-chorus"> chorus</label>
    <label><input type="checkbox" id="fx-echo"> echo</label>
    <label><input type="checkbox" id="fx-humanize"> humanize</label>
    <label><input type="checkbox" id="fx-lift"> lift</label>
    <label><input type="checkbox" id="fx-breath"> synthetic breath noise (effect only)</label>
    <label><input type="checkbox" id="fx-sparkle"> sparkle</label>
    <label><input type="checkbox" id="fx-normalize"> normalize</label>
  </div>
  <div class="row">
    <button id="go">🔊 Speak</button>
    <button id="fresh" type="button">🎲 Fresh take</button>
    <button id="alive" type="button" title="Clear artificial effects and pitch shift; use a performed reference with moderate sampling">🔥 Natural delivery</button>
    <button id="sample" type="button" class="ghost">↺ Use sample line</button>
    <a id="dl" class="dl" style="display:none">⬇ download wav</a>
    <span id="msg"></span>
  </div>
  <audio id="out" controls style="display:none"></audio>
</div>

<div class="card" id="custom-panel">
  <h2>4 · My custom setups</h2>
  <p class="clipinfo">Save working delivery controls + FX in this browser. Reference selection is separate. Your description is a design note, not an engine instruction; no voice is generated from it.</p>
  <div class="row">
    <div><label for="setupname">Setup name</label><input id="setupname" type="text" maxlength="80" placeholder="My dialogue voice"></div>
    <div><label for="savedsetup">Saved setups</label><select id="savedsetup"></select></div>
    <button id="savesetup" type="button">Save current controls</button>
    <button id="loadsetup" type="button">Load</button><button id="deletesetup" type="button" class="ghost">Delete</button>
  </div>
  <label for="designnote">Voice design / free-form description (notes only)</label>
  <textarea id="designnote" placeholder="e.g. mature, dry, warm, slightly gravelly — choose a matching reference separately"></textarea>
  <span id="setupmsg" role="status"></span>
</div>

<div class="card" id="script-panel">
  <h2>5 · Script queue — paste up to 20 lines, each its own clip</h2>
  <div class="row">
    <div style="flex:1"><label>Paste one line per row, then prepare editable rows (up to 20)</label>
      <textarea id="scriptq" placeholder="Hi. I'm just testing my voice to see how it sounds.&#10;I want to make sure everything sounds natural and clear."></textarea></div>
  </div>
  <div class="row">
    <label><input type="checkbox" id="lottery"> 🎲 Experimental variation lottery — 3 takes (slower; not a naturalness judge)</label>
    <button id="prepare" type="button">1 · Prepare / replace rows</button>
    <button id="brender">2 · Render rows</button>
    <button id="bplayall" type="button" style="display:none">▶▶ Play all</button>
  </div>
  <p class="clipinfo">Each prepared row has its own voice, text, language, pitch, pace, volume, sampling, phrasing, seed and FX. Top controls are copied when preparing, not read again during rendering. Editing pasted text requires preparing again.</p>
  <ol id="queue"></ol>
</div>

<script>
const PACK = __PACK__;
const byChar = {};
function rebuildByChar() {
  for (const k of Object.keys(byChar)) delete byChar[k];
  PACK.forEach(p => (byChar[p.char] = byChar[p.char] || {})[p.lang] =
    (byChar[p.char][p.lang] || []).concat(p));
}
rebuildByChar();

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const charsSel = $("char"), langSel = $("lang"), clipSel = $("clip");

function fillChars() {
  charsSel.innerHTML = Object.keys(byChar).map(c => `<option>${esc(c)}</option>`).join("");
  if (charsSel.options.length) fillLangs();
  else $("clipinfo").textContent = "(no seed pack found" +
    (window.__PACKDIAG ? " at " + window.__PACKDIAG : "") +
    " — copy the seeds folder there, or use Any reference clip)";
}
async function loadPack() {
  try {
    const r = await fetch("/api/pack");
    const j = await r.json();
    window.__PACKDIAG = (j.path || "?") + " — exists=" + j.exists + ", wav files=" + j.wav_files;
    if (j.items && j.items.length) {
      PACK.length = 0;
      j.items.forEach(p => PACK.push(p));
      rebuildByChar();
      fillChars();
    } else {
      fillChars();
    }
  } catch (e) { /* keep the embedded pack */ }
}
loadPack();
function fillLangs() {
  const langs = Object.keys(byChar[charsSel.value] || {});
  langSel.innerHTML = langs.map(l => `<option>${l}</option>`).join("");
  if (langs.length) fillClips();
}
function fillClips() {
  const clips = byChar[charsSel.value][langSel.value];
  clipSel.innerHTML = clips.map((c, i) =>
    `<option value="${i}">${esc(c.clip)} (${c.seconds}s)</option>`).join("");
  showClipInfo();
}
function currentClip() {
  const clips = byChar[charsSel.value]?.[langSel.value] || [];
  return clips[+clipSel.value];
}
function showClipInfo() {
  const c = currentClip();
  $("clipinfo").textContent = c ? `“${c.transcript}”` : "";
  if (c) {
    if (!$("text").value.trim()) $("text").value = c.transcript;
    $("textlang").value = c.lang;
  }
}
charsSel.onchange = fillLangs; langSel.onchange = fillClips; clipSel.onchange = showClipInfo;
$("sample").onclick = () => {
  const c = currentClip();
  if (c) {
    $("text").value = c.transcript;
    $("textlang").value = c.lang;
    msg("sample line loaded", false);
  } else { msg("pick a character + language first", true); }
};

let customPath = null;
$("ref").onchange = async () => {
  const f = $("ref").files[0];
  if (!f) return;
  const form = new FormData(); form.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: form });
  const j = await r.json();
  customPath = j.path;
};

function switchTab(t) {
  $("pane-preset").style.display = t === "preset" ? "" : "none";
  $("pane-custom").style.display = t === "custom" ? "" : "none";
  $("tab-preset").classList.toggle("on", t === "preset");
  $("tab-custom").classList.toggle("on", t === "custom");
}

// ---- preset catalog ----
let PRESETS = [];
let activePreset = null;
function presetMatches(p) {
  const q = $("q").value.trim().toLowerCase();
  const cls = $("cls").value;
  const rar = $("rar").value;
  if (cls !== "all" && p["class"] !== cls) return false;
  if (rar !== "all" && String(p.rarity) !== rar) return false;
  for (const [id, key] of [['genderfilter', 'gender'], ['pitchfilter', 'pitch'], ['pacefilter', 'pace']]) {
    if ($(id).value !== 'all' && p[key] !== $(id).value) return false;
  }
  if (q && !(p.name.toLowerCase().includes(q) || p.description.toLowerCase().includes(q))) return false;
  return true;
}
let visiblePresets = 24;
function renderPresets(limit = 24) {
  visiblePresets = typeof limit === "number" ? limit : 24;
  const matches = PRESETS.filter(presetMatches);
  $("pcount").textContent = matches.length + " / " + PRESETS.length + " presets match";
  const box = $("plist");
  box.innerHTML = matches.slice(0, visiblePresets).map(p =>
    `<div class="prow" data-no="${p.no}"><b>#${p.no} ${esc(p.name)}</b> ` +
    `<span class="chip">${esc(p["class"])}</span><span class="chip">${"★".repeat(p.rarity)}</span>` +
    `<div class="pbrief">${esc(p.description.slice(0, 120))}…</div></div>`).join("") ||
    `<div class="pbrief">no matches</div>`;
  $("morepresets").style.display = matches.length > visiblePresets ? "" : "none";
  box.querySelectorAll(".prow").forEach(el => el.onclick = () => applyPreset(+el.dataset.no));
}
const PERF_FAM = [
  ["dark", /dark|yandere|kuudere|tsundere|demon|vampire|villain|edgy|shadow|curse|death|goth|cold|rival|sly|fox|snake|mafia|assassin/i],
  ["dramatic", /knight|hero|warrior|samurai|lord|king|prince|paladin|ruler|general|legend|epic|battle|boss|commander|captain|viking|noble/i],
  ["excite", /genki|energetic|hyper|idol|excited|upbeat|punchy|rookie|speed|cheerful|genki|shonen|pirate/i],
  ["cheer", /cute|sweet|idol|princess|fairy|magical|school|maid|bubbly|smile|bright|moe|kawaii|idol|teen|idolgirl/i],
  ["warm", /gentle|soft|warm|kind|healer|mother|cozy|bard|narrator|storyteller|mature|calm|senpai|onee|neechan|lady|madam|milf/i],
];
function familyFor(p) {
  const s = (p.name + " " + p["class"] + " " + (p.description || "") + " " + (p.signature || "")).toLowerCase();
  for (const fam of PERF_FAM) if (fam[1].test(s)) return fam[0];
  return "warm";
}
function applyPreset(no) {
  const p = PRESETS.find(x => x.no === no);
  if (!p) return;
  activePreset = p;
  $("pitch").value = p.pitch_shift;
  $("pitchv").textContent = (p.pitch_shift > 0 ? "+" : "") + p.pitch_shift;
  $("speed").value = p.speed;
  let perfNote = "";
  if ($("perf") && (charsSel.value === "dragon" || charsSel.value === "drake")) {
    const want = familyFor(p) + "-" + charsSel.value;
    if ($("perf").querySelector('option[value="' + want + '"]')) {
      $("perf").value = want;
      perfNote = `performance: ${$("perf").selectedOptions[0].textContent} · `;
    }
  }
  $("apreset").style.display = "";
  $("apreset").innerHTML = `<b>#${p.no} ${esc(p.name)}</b> ` +
    `<span class="chip">${esc(p["class"])}</span><span class="chip">${esc(p.gender)}</span>` +
    `<span class="chip">${"★".repeat(p.rarity)}</span><span class="chip">pitch: ${esc(p.pitch)}</span>` +
    `<span class="chip">pace: ${esc(p.pace)}</span>` +
    (p.signature ? `<span class="chip">signature</span>` : "") +
    `<div class="pbrief">${esc(p.description)}</div>` +
    `<div class="pbrief">${perfNote}controls set to its pitch (${(p.pitch_shift > 0 ? "+" : "") + p.pitch_shift} st) ` +
    `and pace (speed ${p.speed}). FX is yours to add.</div>`;
  msg("preset applied (pitch, pace and reference mood only)" + (perfNote ? " · " + $("perf").selectedOptions[0].textContent : ""), false);
}
$("roll").onclick = () => {
  const matches = PRESETS.filter(presetMatches);
  if (!matches.length) { msg("no presets match the current filter", true); return; }
  applyPreset(matches[Math.floor(Math.random() * matches.length)].no);
};
$("q").oninput = renderPresets;
$("cls").onchange = renderPresets;
$("rar").onchange = renderPresets;
['genderfilter', 'pitchfilter', 'pacefilter'].forEach(id => { $(id).onchange = () => renderPresets(); });
$("morepresets").onclick = () => renderPresets(visiblePresets + 24);
fetch("/api/presets").then(r => r.json()).then(j => {
  PRESETS = j.presets;
  for (const [id, key] of [['genderfilter', 'gender'], ['pitchfilter', 'pitch'], ['pacefilter', 'pace']]) {
    $(id).replaceChildren(new Option('all', 'all'));
    [...new Set(PRESETS.map(p => p[key]).filter(Boolean))].sort().forEach(v => $(id).add(new Option(v, v)));
  }
  $("cls").innerHTML = `<option value="all">all</option>` +
    j.classes.map(c => `<option>${esc(c)}</option>`).join("");
  $("pcount").textContent = j.count + " presets loaded";
  renderPresets();
}).catch(() => { $("pcount").textContent = "catalog not found"; });

$("pitch").oninput = e => $("pitchv").textContent =
  (e.target.value > 0 ? "+" : "") + e.target.value;
$("vol").oninput = e => $("volv").textContent =
  (e.target.value > 0 ? "+" : "") + e.target.value;

// ---- speak ----
let lastURL = null;
async function speak(fresh = false) {
  const tab = $("tab-preset").classList.contains("on") ? "preset" : "custom";
  let ref = null, promptText = "", promptLang = "";
  if (tab === "preset") {
    const pf = ($("perf") && $("perf").value) || "";
    if (pf) {
      ref = "perf:" + pf; promptText = "(auto from performance bank)"; promptLang = "en";
    } else {
      const c = currentClip();
      if (!c) { msg("pick a character + language first", true); return; }
      ref = c.char + "/" + c.clip; promptText = c.transcript; promptLang = c.lang;
    }
  } else {
    if (!customPath) { msg("upload a reference clip first", true); return; }
    ref = "upload:" + customPath; promptText = $("reftext").value.trim();
    promptLang = $("reflang").value;
  }
  const text = $("text").value.trim();
  if (!text) { msg("type some text first", true); return; }
  if (!promptText) { msg("the reference transcript is missing", true); return; }

  const fx = ["robot", "phone", "reverb", "chorus", "echo", "humanize", "lift", "breath", "sparkle", "normalize"]
    .filter(f => $("fx-" + f).checked).join(",");
  const btn = $("go"); btn.disabled = true; $("fresh").disabled = true;
  msg("rendering… (first use loads reference features)", false);
  try {
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref_audio_path: ref, prompt_text: promptText, prompt_lang: promptLang,
        text, text_lang: $("textlang").value, speed_factor: +$("speed").value, seed: fresh ? -1 : +$("seed").value,
        pitch: +$("pitch").value, volume: +$("vol").value, fx,
        top_k: +$("top_k").value, top_p: +$("top_p").value,
        temperature: +$("temperature").value, repetition_penalty: +$("rep").value,
        text_split_method: $("splitm").value, fragment_interval: +$("gap").value,
        batch_size: +$("enginebatch").value, reuse_take: $("reuse").checked && !fresh }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      msg(j.message || ("HTTP " + r.status), true);
      return;
    }
    const blob = await r.blob();
    if (lastURL) URL.revokeObjectURL(lastURL);
    lastURL = URL.createObjectURL(blob);
    const out = $("out");
    out.src = lastURL;
    out.style.display = "";
    out.play();
    const dl = $("dl");
    dl.href = lastURL;
    dl.download = "gacha-" + (activePreset ? activePreset.no + "-" : "") + Date.now() + ".wav";
    dl.style.display = "";
    msg(renderTiming(r), false);
  } catch (e) {
    msg("error: " + e, true);
  } finally {
    btn.disabled = false; $("fresh").disabled = false;
  }
}
$("go").onclick = () => speak(false);
$("fresh").onclick = () => speak(true);
function renderTiming(r) {
  const seconds = r.headers.get("X-Total-Seconds") || r.headers.get("X-Render-Seconds") || "?";
  return (r.headers.get("X-Take-Cache") === "hit" ? "reused take" : "new take") + " · " + seconds + "s";
}
function msg(t, isErr) { const m = $("msg"); m.textContent = t; m.className = isErr ? "err" : "ok"; }

const STYLES = {
  subtle:   { temperature: 0.4, top_k: 10, top_p: 0.8, rep: 1.35, speed: 0.95 },
  balanced: { temperature: 1.0, top_k: 15, top_p: 1.0, rep: 1.2,  speed: 1.0 },
  animated: { temperature: 1.15, top_k: 20, top_p: 1.0, rep: 1.15, speed: 1.05 },
  fever:    { temperature: 1.4, top_k: 30, top_p: 1.0, rep: 1.1,  speed: 1.1 },
};
Object.keys(STYLES).forEach(name => {
  const b = $("style-" + name);
  if (b) b.onclick = () => {
    const s = STYLES[name];
    $("temperature").value = s.temperature; $("top_k").value = s.top_k;
    $("top_p").value = s.top_p; $("rep").value = s.rep; $("speed").value = s.speed;
    msg("style: " + name, false);
  };
});
$("alive").onclick = () => {
  // Start from the actual performance, not noise, pitch wobble or fever sampling.
  $("style-balanced").click();
  ["robot", "phone", "reverb", "chorus", "echo", "humanize", "lift", "breath", "sparkle", "normalize"]
    .forEach(f => { $("fx-" + f).checked = false; });
  $("pitch").value = 0; $("pitchv").textContent = "0";
  $("splitm").value = "cut0";
  $("gap").value = 0.2;
  $("lottery").checked = false;
  if ($("tab-preset").classList.contains("on") && !$("perf").value &&
      ["dragon", "drake"].includes(charsSel.value)) {
    $("perf").value = (activePreset ? familyFor(activePreset) : "warm") + "-" + charsSel.value;
  }
  msg("Natural delivery: FX off, pitch reset, normal pace, continuous phrasing. Reference performance drives expression; hit 🔊 Speak.", false);
};

// ---- script queue: render N lines one by one, play as they finish ----
function voiceState() {
  let ref = null, promptText = "", promptLang = "";
  const pf = ($("perf") && $("perf").value) || "";
  if ($("tab-preset").classList.contains("on") && pf) {
    ref = "perf:" + pf; promptText = "(auto from performance bank)"; promptLang = "en";
  } else {
    const c = currentClip();
    if ($("tab-preset").classList.contains("on") && c) {
      ref = c.char + "/" + c.clip; promptText = c.transcript; promptLang = c.lang;
    }
  }
  if (!ref && customPath) {
    ref = "upload:" + customPath; promptText = $("reftext").value.trim();
    promptLang = $("reflang").value;
  }
  return { ref, promptText, promptLang };
}
function fxCsv() {
  return ["robot", "phone", "reverb", "chorus", "echo", "humanize", "lift", "breath", "sparkle", "normalize"]
    .filter(f => $("fx-" + f).checked).join(",");
}
// ---- Explicit control snapshots, saved setups and editable script rows ----
const CONTROL_FIELDS = [
  ['textlang', 'text_lang', 'Text language'], ['pitch', 'pitch', 'Pitch (semitones)'],
  ['speed', 'speed_factor', 'Speaking pace'], ['vol', 'volume', 'Volume (dB)'],
  ['seed', 'seed', 'Seed (-1 random)'], ['temperature', 'temperature', 'Temperature'],
  ['top_k', 'top_k', 'Top K'], ['top_p', 'top_p', 'Top P'],
  ['rep', 'repetition_penalty', 'Repetition penalty'], ['splitm', 'text_split_method', 'Phrasing'],
  ['gap', 'fragment_interval', 'Phrase pause (seconds)'], ['enginebatch', 'batch_size', 'Sentence batching'],
  ['reuse', 'reuse_take', 'Reuse matching take'], ['lottery', 'lottery', 'Three-take lottery'],
];
const FX_NAMES = ['robot','phone','reverb','chorus','echo','humanize','lift','breath','sparkle','normalize'];
function controlValue(el) {
  return el.type === 'checkbox' ? el.checked :
    ['number', 'range'].includes(el.type) ? Number(el.value) : el.value;
}
function captureControls() {
  const values = Object.fromEntries(CONTROL_FIELDS.map(([id, key]) => [key, controlValue($(id))]));
  values.fx = fxCsv();
  return values;
}
function captureRequest() {
  const v = voiceState();
  return { ...captureControls(), ref_audio_path: v.ref, prompt_text: v.promptText, prompt_lang: v.promptLang };
}
function applyControls(values) {
  for (const [id, key] of CONTROL_FIELDS) {
    if (!(key in values)) continue;
    const el = $(id);
    if (el.type === 'checkbox') el.checked = values[key] === true;
    else el.value = values[key];
  }
  const fx = String(values.fx || '').split(',');
  FX_NAMES.forEach(f => { $('fx-' + f).checked = fx.includes(f); });
  $('pitchv').textContent = $('pitch').value; $('volv').textContent = $('vol').value;
}
let savedSetups = {};
try {
  const stored = JSON.parse(localStorage.getItem('gacha.delivery.setups.v1') || '{}');
  if (stored && typeof stored === 'object' && !Array.isArray(stored)) savedSetups = stored;
} catch (_) { /* Storage can be disabled; saving reports that below. */ }
function listSetups() {
  $('savedsetup').replaceChildren(new Option('Choose a saved setup', ''));
  Object.keys(savedSetups).sort().forEach(name => $('savedsetup').add(new Option(name, name)));
}
function persistSetups() {
  localStorage.setItem('gacha.delivery.setups.v1', JSON.stringify(savedSetups));
  listSetups();
}
$('savesetup').onclick = () => {
  const name = $('setupname').value.trim();
  if (!name) { $('setupmsg').textContent = 'Give this setup a name first.'; return; }
  try {
    // Define a plain own key even for a name such as "__proto__".
    Object.defineProperty(savedSetups, name, {value: {controls: captureControls(), note: $('designnote').value},
      configurable: true, enumerable: true, writable: true});
    persistSetups(); $('savedsetup').value = name;
    $('setupmsg').textContent = 'Saved in this browser. Reference voice unchanged.';
  } catch (e) { $('setupmsg').textContent = 'Could not save: browser storage unavailable or full.'; }
};
$('loadsetup').onclick = () => {
  const name = $('savedsetup').value;
  if (!Object.hasOwn(savedSetups, name) || !savedSetups[name]?.controls) return;
  const s = savedSetups[name]; applyControls(s.controls);
  activePreset = null; $('apreset').style.display = 'none';
  $('setupname').value = name; $('designnote').value = s.note || '';
  $('setupmsg').textContent = 'Delivery controls loaded. Choose your reference separately; prepared rows are unchanged.';
};
$('deletesetup').onclick = () => {
  const name = $('savedsetup').value;
  if (!name) return;
  try { delete savedSetups[name]; persistSetups(); $('setupmsg').textContent = 'Setup deleted.'; }
  catch (_) { $('setupmsg').textContent = 'Browser storage unavailable.'; }
};
listSetups();

async function batchOne(line, fresh = false, settings = null) {
  const snapshot = settings || captureRequest();
  if (!snapshot.ref_audio_path || !snapshot.prompt_text) throw new Error('pick a reference voice first');
  const n = snapshot.lottery ? 3 : 1;
  const {lottery, ...payload} = snapshot;
  let best = null;
  for (let k = 0; k < n; k++) {
    const r = await fetch('/api/tts', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...payload, text: line,
        seed: n > 1 || fresh ? -1 : payload.seed,
        reuse_take: payload.reuse_take && !fresh && n === 1}),
    });
    if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.message || ('HTTP ' + r.status)); }
    const score = parseFloat(r.headers.get('X-Life-Score') || '0');
    const blob = await r.blob();
    if (!best || score > best.score) best = {blob, score, timing: renderTiming(r)};
  }
  return best;
}
let queueRows = [];
let queueBusy = false;
function addOption(select, value, text) { select.add(new Option(text, value)); }
function prepareRows() {
  if (queueBusy) return false;
  const lines = $('scriptq').value.split('\n').map(s => s.trim()).filter(Boolean);
  if (!lines.length || lines.length > 20) { msg('Paste between 1 and 20 lines; nothing was discarded.', true); return false; }
  const snapshot = captureRequest();
  if (!snapshot.ref_audio_path || !snapshot.prompt_text) { msg('Choose a reference voice first.', true); return false; }
  queueRows.forEach(row => { if (row.url) URL.revokeObjectURL(row.url); });
  queueRows = []; $('queue').replaceChildren(); $('bplayall').style.display = 'none';
  lines.forEach((text, index) => {
    const li = document.createElement('li');
    const fieldset = document.createElement('fieldset');
    const legend = document.createElement('legend'); legend.textContent = 'Line ' + (index + 1);
    const textLabel = document.createElement('label'); textLabel.textContent = 'Text';
    const input = document.createElement('textarea'); input.value = text; textLabel.append(input);
    const voiceLabel = document.createElement('label'); voiceLabel.textContent = 'Reference voice for this line';
    const voice = document.createElement('select');
    addOption(voice, 'base', 'Copied reference: ' + snapshot.ref_audio_path);
    PACK.forEach((p, i) => addOption(voice, 'pack:' + i, p.char + ' / ' + p.lang + ' / ' + p.clip));
    Array.from($('perf').options).filter(o => o.value).forEach(o => addOption(voice, 'perf:' + o.value, o.textContent));
    voiceLabel.append(voice);
    const grid = document.createElement('div'); grid.className = 'controlgrid';
    const controls = {};
    CONTROL_FIELDS.forEach(([id, key, title]) => {
      const label = document.createElement('label'); label.textContent = title;
      const el = $(id).cloneNode(true); el.removeAttribute('id');
      if (el.type === 'range') el.type = 'number';
      if (el.type === 'number') el.required = true;
      if (el.type === 'checkbox') el.checked = snapshot[key]; else el.value = snapshot[key];
      el.dataset.control = key; controls[key] = el; label.append(el); grid.append(label);
    });
    const effects = document.createElement('div'); effects.className = 'row fx';
    const fxChecks = {};
    FX_NAMES.forEach(f => {
      const label = document.createElement('label'); label.textContent = f === 'breath' ? 'synthetic breath noise' : f;
      const el = document.createElement('input'); el.type = 'checkbox'; el.checked = snapshot.fx.split(',').includes(f);
      el.dataset.fx = f; fxChecks[f] = el; label.prepend(el); effects.append(label);
    });
    const actions = document.createElement('div'); actions.className = 'row';
    const render = document.createElement('button'); render.type = 'button'; render.textContent = 'Render this line';
    const fresh = document.createElement('button'); fresh.type = 'button'; fresh.textContent = 'Fresh take';
    const dl = document.createElement('a'); dl.textContent = 'Download WAV'; dl.style.display = 'none'; dl.download = 'line' + (index + 1) + '.wav';
    const status = document.createElement('span'); status.setAttribute('role', 'status'); status.textContent = 'Ready — editable before rendering';
    const audio = document.createElement('audio'); audio.controls = true; audio.style.display = 'none';
    actions.append(render, fresh, dl); fieldset.append(legend, textLabel, voiceLabel, grid, effects, actions);
    li.append(fieldset, status, audio); $('queue').append(li);
    const row = {input, voice, controls, fxChecks, fieldset, status, audio, dl, url:null, base:{...snapshot}};
    queueRows.push(row);
    fieldset.oninput = () => { status.textContent = row.url ? 'Settings edited — render to update the audio below.' : 'Ready'; };
    render.onclick = () => runOneRow(row, false);
    fresh.onclick = () => runOneRow(row, true);
  });
  msg('Rows prepared. Each has its own controls; top controls no longer change them.', false);
  return true;
}
function rowSettings(row) {
  const settings = {...row.base};
  for (const [key, el] of Object.entries(row.controls)) {
    if (!el.reportValidity()) throw new Error('Check the highlighted line control.');
    settings[key] = controlValue(el);
  }
  settings.fx = FX_NAMES.filter(f => row.fxChecks[f].checked).join(',');
  const selected = row.voice.value;
  if (selected.startsWith('pack:')) {
    const p = PACK[Number(selected.slice(5))];
    settings.ref_audio_path = p.char + '/' + p.clip; settings.prompt_text = p.transcript; settings.prompt_lang = p.lang;
  } else if (selected.startsWith('perf:')) {
    settings.ref_audio_path = selected; settings.prompt_text = '(auto from performance bank)'; settings.prompt_lang = 'en';
  }
  return settings;
}
function setQueueBusy(busy) {
  queueBusy = busy; $('prepare').disabled = busy; $('brender').disabled = busy;
  queueRows.forEach(row => { row.fieldset.disabled = busy; });
}
async function renderRow(row, fresh) {
  try {
    const text = row.input.value.trim();
    if (!text) throw new Error('This line needs text.');
    const wasDisabled = row.fieldset.disabled;
    row.fieldset.disabled = false;
    let settings;
    try { settings = rowSettings(row); } finally { row.fieldset.disabled = wasDisabled; }
    row.status.textContent = 'Rendering…';
    const best = await batchOne(text, fresh, settings);
    if (row.url) URL.revokeObjectURL(row.url);
    row.url = URL.createObjectURL(best.blob);
    row.audio.src = row.url; row.audio.style.display = '';
    row.dl.href = row.url; row.dl.style.display = '';
    row.status.textContent = best.timing + ' · variation ' + best.score.toFixed(1);
    $('bplayall').style.display = '';
    return true;
  } catch (e) { row.status.textContent = 'Render failed (previous audio, if any, is unchanged): ' + e.message; return false; }
}
async function runOneRow(row, fresh) {
  if (queueBusy) return;
  setQueueBusy(true);
  try { await renderRow(row, fresh); } finally { setQueueBusy(false); }
}
$('prepare').onclick = prepareRows;
$('brender').onclick = async () => {
  if (queueBusy || (!queueRows.length && !prepareRows())) return;
  setQueueBusy(true);
  let completed = 0;
  try {
    for (const row of queueRows) if (await renderRow(row, false)) completed++;
  } finally { setQueueBusy(false); }
  msg('Script done — ' + completed + '/' + queueRows.length + ' rendered. Failed rows remain editable.', completed !== queueRows.length);
};
$('bplayall').onclick = () => {
  const urls = queueRows.filter(row => row.url).map(row => row.url);
  let index = 0;
  const next = () => { if (index < urls.length) { const audio = new Audio(urls[index++]); audio.onended = next; audio.play(); } };
  next();
};

fetch("/api/status").then(r => r.json()).then(j => {
  $("ver").textContent = `v${j.version} · langs: ${j.languages.join(", ")} · ${j.pack} pack clip(s) · ${j.presets} presets · ${j.cpu_threads} CPU threads`;
  if (String(j.version).toLowerCase().indexOf("stub") >= 0) {
    const b = $("demobanner"); if (b) b.style.display = "";
  }
}).catch(() => {});
fillChars();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=Path, default=Path(__file__).resolve().parent.parent / "seeds",
                    help="seed pack root (<char>/<lang>[N].wav + .txt; default: <repo>/seeds)")
    ap.add_argument("--presets", type=Path, default=Path(__file__).resolve().parent.parent / "presets.csv",
                    help="preset catalog csv (default: <repo>/presets.csv)")
    ap.add_argument("--gsv-root", type=Path, default=Path("."),
                    help="GPT-SoVITS repo root (default: cwd)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--tts-config", default="GPT_SoVITS/configs/tts_infer.yaml")
    ap.add_argument("--open", action="store_true",
                    help="open the browser automatically when the server starts")
    args = ap.parse_args()

    if not (args.gsv_root / "GPT_SoVITS").is_dir():
        raise SystemExit(
            f"[abort] {args.gsv_root} does not look like a GPT-SoVITS root "
            f"(no GPT_SoVITS/ dir) — run from the GPT-SoVITS folder or pass --gsv-root")

    # GPT-SoVITS code measures several paths from the CURRENT working directory
    # (e.g. GPT_SoVITS/sv.py does sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")
    # and loads pretrained_models relative to cwd). The stock go-webui.bat therefore
    # always `cd /d` into the package root first — do the same, whatever folder
    # this script was launched from.
    os.chdir(args.gsv_root.resolve())

    # resolve the tts config against the GSV root as well (default is CWD-relative)
    tts_cfg = Path(args.tts_config)
    if not tts_cfg.is_absolute() and not tts_cfg.is_file():
        cand = args.gsv_root / tts_cfg
        if cand.is_file():
            tts_cfg = cand

    import socket  # noqa: E402

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((args.bind, args.port))
    except OSError:
        raise SystemExit(
            f"[abort] port {args.port} is already in use — "
            f"close the other server or try: --port {args.port + 1}")
    finally:
        probe.close()

    import uvicorn  # noqa: E402

    app = build_app(args.gsv_root.resolve(), args.seeds.resolve(),
                    args.presets.resolve(), str(tts_cfg))
    url = f"http://{'127.0.0.1' if args.bind in ('0.0.0.0', '::') else args.bind}:{args.port}"
    if args.open:
        import threading  # noqa: E402
        import webbrowser  # noqa: E402
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"[studio] serving {url}")
    uvicorn.run(app, host=args.bind, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
