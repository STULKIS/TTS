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
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_fx  # noqa: E402

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


def build_app(gsv_root: Path, seeds: Path, presets_path: Path, tts_config_path: str):
    # GPT-SoVITS imports (must run inside its environment)
    sys.path.insert(0, str(gsv_root))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

    from fastapi import FastAPI, Request  # noqa: E402
    import numpy as np  # noqa: E402
    import soundfile as sf  # noqa: E402
    from fastapi.responses import HTMLResponse, Response, JSONResponse  # noqa: E402

    print(f"[studio] loading GPT-SoVITS pipeline (config={tts_config_path}) ...")
    try:
        tts_config = TTS_Config(tts_config_path)
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

    APP = FastAPI()
    uploads = Path(tempfile.gettempdir()) / "type_ui_uploads"
    uploads.mkdir(exist_ok=True)

    @APP.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE_HTML.replace("__PACK__", json.dumps(pack, ensure_ascii=False))

    @APP.get("/api/status")
    async def status():
        return {"ready": True, "version": tts_config.version,
                "languages": tts_config.languages, "pack": len(pack),
                "presets": len(presets_rows)}

    @APP.get("/api/presets")
    async def presets():
        return {"count": len(presets_rows),
                "classes": sorted({p["class"] for p in presets_rows}),
                "presets": presets_rows}

    @APP.post("/api/tts")
    async def tts(request: Request):
        req = await request.json()
        # resolve reference audio: a pack clip (relative to the seeds dir),
        # a plain absolute path, or an uploaded temp file
        ref = req.get("ref_audio_path", "")
        if ref.startswith("upload:"):
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
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": req.get("speed_factor", 1.0),
            "seed": req.get("seed", -1),
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": False,
            "repetition_penalty": 1.35,
        }
        if not payload["text"].strip():
            return JSONResponse(status_code=400, content={"message": "text is required"})
        try:
            generator = tts_pipeline.run(payload)
            sr, audio = next(generator)
        except Exception as e:  # surface the real error in the UI
            return JSONResponse(status_code=500, content={"message": f"tts failed: {e}"})
        # per-line controls: pitch shift + volume (post-hoc DSP), speed was engine-side
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
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
        return Response(buf.getvalue(), media_type="audio/wav")

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


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gacha TTS Studio</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
       max-width: 50rem; padding: 0 1rem; line-height: 1.5; }
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
</style>
</head>
<body>
<h1>🎮 Gacha TTS Studio</h1>
<p class="sub">GPT-SoVITS zero-shot · preset catalog · per-line controls. <span id="ver"></span></p>
<p id="demobanner" class="err" style="display:none">DEMO ENGINE — this preview renders a test tone instead of a real voice (the real models run only on your PC). Good for trying the controls; audio quality says nothing about the real renders.</p>

<div class="card">
  <h2>1 · Gacha preset — pick the voice design</h2>
  <div class="row">
    <div style="flex:1"><label>Search name or brief</label><input type="text" id="q" placeholder="e.g. gravel, whisper, idol..."></div>
    <div><label>Class</label><select id="cls"></select></div>
    <div><label>Rarity</label>
      <select id="rar"><option value="all">any</option><option value="3">★</option><option value="4">★★</option><option value="5">★★★</option></select></div>
    <div style="align-self:flex-end"><button id="roll" class="ghost">🎲 ROLL</button></div>
  </div>
  <div class="clipinfo" id="pcount">loading catalog…</div>
  <div id="plist"></div>
  <div class="apreset" id="apreset" style="display:none"></div>
</div>

<div class="card">
  <h2>2 · Voice — the reference it is cloned from</h2>
  <div id="tabs"><button id="tab-preset" class="on" onclick="switchTab('preset')">Preset characters</button>
  <button id="tab-custom" onclick="switchTab('custom')">Any reference clip</button></div>
  <div id="pane-preset">
    <div class="row">
      <div><label>Character</label><select id="char"></select></div>
      <div><label>Language</label><select id="lang"></select></div>
      <div><label>Clip</label><select id="clip"></select></div>
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

<div class="card">
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
  </div>
  <div class="row fx">
    <label><input type="checkbox" id="fx-robot"> robot</label>
    <label><input type="checkbox" id="fx-phone"> phone</label>
    <label><input type="checkbox" id="fx-reverb"> reverb</label>
    <label><input type="checkbox" id="fx-normalize"> normalize</label>
  </div>
  <div class="row">
    <button id="go">🔊 Speak</button>
    <button id="sample" type="button" class="ghost">↺ Use sample line</button>
    <a id="dl" class="dl" style="display:none">⬇ download wav</a>
    <span id="msg"></span>
  </div>
  <audio id="out" controls style="display:none"></audio>
</div>

<script>
const PACK = __PACK__;
const byChar = {};
PACK.forEach(p => (byChar[p.char] = byChar[p.char] || {})[p.lang] =
  (byChar[p.char][p.lang] || []).concat(p));

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const charsSel = $("char"), langSel = $("lang"), clipSel = $("clip");

function fillChars() {
  charsSel.innerHTML = Object.keys(byChar).map(c => `<option>${esc(c)}</option>`).join("");
  if (charsSel.options.length) fillLangs();
  else $("clipinfo").textContent = "(no seed pack found — pass --seeds or use Any reference clip)";
}
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
    $("text").value = c.transcript;   // known-good line for this voice/lang — one-click test
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
  if (q && !(p.name.toLowerCase().includes(q) || p.description.toLowerCase().includes(q))) return false;
  return true;
}
function renderPresets() {
  const matches = PRESETS.filter(presetMatches);
  $("pcount").textContent = matches.length + " / " + PRESETS.length + " presets match";
  const box = $("plist");
  box.innerHTML = matches.slice(0, 12).map(p =>
    `<div class="prow" data-no="${p.no}"><b>#${p.no} ${esc(p.name)}</b> ` +
    `<span class="chip">${esc(p["class"])}</span><span class="chip">${"★".repeat(p.rarity)}</span>` +
    `<div class="pbrief">${esc(p.description.slice(0, 120))}…</div></div>`).join("") ||
    `<div class="pbrief">no matches</div>`;
  box.querySelectorAll(".prow").forEach(el => el.onclick = () => applyPreset(+el.dataset.no));
}
function applyPreset(no) {
  const p = PRESETS.find(x => x.no === no);
  if (!p) return;
  activePreset = p;
  $("pitch").value = p.pitch_shift;
  $("pitchv").textContent = (p.pitch_shift > 0 ? "+" : "") + p.pitch_shift;
  $("speed").value = p.speed;
  $("apreset").style.display = "";
  $("apreset").innerHTML = `<b>#${p.no} ${esc(p.name)}</b> ` +
    `<span class="chip">${esc(p["class"])}</span><span class="chip">${esc(p.gender)}</span>` +
    `<span class="chip">${"★".repeat(p.rarity)}</span><span class="chip">pitch: ${esc(p.pitch)}</span>` +
    `<span class="chip">pace: ${esc(p.pace)}</span>` +
    (p.signature ? `<span class="chip">signature</span>` : "") +
    `<div class="pbrief">${esc(p.description)}</div>` +
    `<div class="pbrief">controls set to its pitch (${(p.pitch_shift > 0 ? "+" : "") + p.pitch_shift} st) ` +
    `and pace (speed ${p.speed}). FX is yours to add.</div>`;
  msg("preset applied", false);
}
$("roll").onclick = () => {
  const matches = PRESETS.filter(presetMatches);
  if (!matches.length) { msg("no presets match the current filter", true); return; }
  applyPreset(matches[Math.floor(Math.random() * matches.length)].no);
};
$("q").oninput = renderPresets;
$("cls").onchange = renderPresets;
$("rar").onchange = renderPresets;
fetch("/api/presets").then(r => r.json()).then(j => {
  PRESETS = j.presets;
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
$("go").onclick = async () => {
  const tab = $("tab-preset").classList.contains("on") ? "preset" : "custom";
  let ref = null, promptText = "", promptLang = "";
  if (tab === "preset") {
    const c = currentClip();
    if (!c) { msg("pick a character + language first", true); return; }
    ref = c.char + "/" + c.clip; promptText = c.transcript; promptLang = c.lang;
  } else {
    if (!customPath) { msg("upload a reference clip first", true); return; }
    ref = "upload:" + customPath; promptText = $("reftext").value.trim();
    promptLang = $("reflang").value;
  }
  const text = $("text").value.trim();
  if (!text) { msg("type some text first", true); return; }
  if (!promptText) { msg("the reference transcript is missing", true); return; }

  const fx = ["robot", "phone", "reverb", "normalize"]
    .filter(f => $("fx-" + f).checked).join(",");
  const btn = $("go"); btn.disabled = true;
  msg("synthesizing… (CPU: a few seconds)", false);
  try {
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref_audio_path: ref, prompt_text: promptText, prompt_lang: promptLang,
        text, text_lang: $("textlang").value, speed_factor: +$("speed").value, seed: +$("seed").value,
        pitch: +$("pitch").value, volume: +$("vol").value, fx }),
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
    msg("ok", false);
  } catch (e) {
    msg("error: " + e, true);
  } finally {
    btn.disabled = false;
  }
};
function msg(t, isErr) { const m = $("msg"); m.textContent = t; m.className = isErr ? "err" : "ok"; }

fetch("/api/status").then(r => r.json()).then(j => {
  $("ver").textContent = `v${j.version} · langs: ${j.languages.join(", ")} · ${j.pack} pack clip(s) · ${j.presets} presets`;
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
    ap.add_argument("--seeds", type=Path, default=Path("seeds"),
                    help="seed pack root (<char>/<lang>[N].wav + .txt)")
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
