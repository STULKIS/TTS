#!/usr/bin/env python3
"""Type-and-speak WebUI for GPT-SoVITS — the "I can type text" front end.

Runs IN your GPT-SoVITS install (same conda env / integrated package), serving a
single-page app: pick a character from a seed pack (this repo's seeds/ ships
dragon & drake × en/zh/ja/ko), or upload ANY 5-10 s reference clip, type your
text, choose the language, click Speak.

Usage (from the GPT-SoVITS repo root, in its environment):
    python /path/to/TTS/tools/type_ui.py --seeds /path/to/TTS/seeds
    # or: copy tools/type_ui.py + seeds/ into your GPT-SoVITS folder first:
    python type_ui.py --seeds seeds
Then open http://127.0.0.1:7861 in your browser.

Options:
    --seeds DIR       seed pack root (dirs: <char>/<lang>[N].wav + .txt)
    --gsv-root DIR    GPT-SoVITS repo root (default: current directory)
    --port N          default 7861 (api_v2.py uses 9880)
    --bind ADDR       default 127.0.0.1 (local machine tool)
    --tts-config P    default GPT_SoVITS/configs/tts_infer.yaml (v2/v2Pro)
    --version-tag V   v1/v2/v2Pro/v2ProPlus (default: read from tts_infer.yaml)

Why this exists: api_v2.py is a raw JSON API; this wraps the same
tts_pipeline.run(req) call with a text box, character picker, and audio player —
zero-shot cloning of the reference clip, no training needed. Trained
per-character models work too: set t2s_weights_path/vits_weights_path in
tts_infer.yaml and the UI uses them automatically.

Depends only on the GPT-SoVITS environment (fastapi/uvicorn/numpy/soundfile).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
import uuid
import wave
from pathlib import Path

LANGS = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}
_CLIP_RE = re.compile(r"^([a-z]+)(\d+)?$")


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


def build_app(gsv_root: Path, seeds: Path, tts_config_path: str):
    # GPT-SoVITS imports (must run inside its environment)
    sys.path.insert(0, str(gsv_root))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

    from fastapi import FastAPI, Request  # noqa: E402
    import numpy as np  # noqa: E402
    import soundfile as sf  # noqa: E402
    from fastapi.responses import HTMLResponse, Response, JSONResponse  # noqa: E402

    print(f"[type-ui] loading GPT-SoVITS pipeline (config={tts_config_path}) …")
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
    print(f"[type-ui] ready: version={tts_config.version} languages={tts_config.languages} "
          f"pack={len(pack)} clip(s) under {seeds}")

    APP = FastAPI()
    uploads = Path(tempfile.gettempdir()) / "type_ui_uploads"
    uploads.mkdir(exist_ok=True)

    @APP.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE_HTML.replace("__PACK__", json.dumps(pack, ensure_ascii=False))

    @APP.get("/api/status")
    async def status():
        return {"ready": True, "version": tts_config.version,
                "languages": tts_config.languages, "pack": len(pack)}

    @APP.post("/api/tts")
    async def tts(request: Request):
        req = await request.json()
        # resolve reference audio: server-side pack path, or an uploaded temp file
        ref = req.get("ref_audio_path", "")
        if ref.startswith("upload:") and Path(ref[len("upload:"):]).is_file():
            ref = str(Path(ref[len("upload:"):]).resolve())
        elif not Path(ref).is_file():
            return JSONResponse(status_code=400, content={"message": f"ref_audio_path not found: {ref}"})
        prompt_text = req.get("prompt_text", "")
        if not prompt_text:
            return JSONResponse(status_code=400,
                                content={"message": "prompt_text (verbatim transcript of the reference clip) is required"})
        payload = {
            "text": req.get("text", ""),
            "text_lang": req.get("text_lang", "en"),
            "ref_audio_path": ref,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang or req.get("text_lang", "en"),
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
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1), sr, format="WAV")
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
<title>Type &amp; Speak — GPT-SoVITS</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
       max-width: 46rem; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: .25rem; }
.sub { opacity: .7; }
.card { border: 1px solid rgba(128,128,128,.35); border-radius: .75rem;
        padding: 1rem 1.25rem; margin: 1.25rem 0; }
.row { display: flex; flex-wrap: wrap; gap: .75rem; margin: .75rem 0; align-items: center; }
label { font-size: .85rem; opacity: .8; display: block; margin-bottom: .2rem; }
select, input[type=number] { font: inherit; padding: .3rem .5rem; border-radius: .4rem;
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
</style>
</head>
<body>
<h1>Type &amp; Speak</h1>
<p class="sub">GPT-SoVITS zero-shot — reference clip + your text. <span id="ver"></span></p>

<div id="tabs"><button id="tab-preset" class="on" onclick="switchTab('preset')">Preset characters</button>
<button id="tab-custom" onclick="switchTab('custom')">Any reference clip</button></div>

<div id="pane-preset" class="card">
  <div class="row">
    <div><label>Character</label><select id="char"></select></div>
    <div><label>Language</label><select id="lang"></select></div>
    <div><label>Clip</label><select id="clip"></select></div>
  </div>
  <div class="clipinfo" id="clipinfo"></div>
</div>

<div id="pane-custom" class="card" style="display:none">
  <div class="row">
    <div><label>Reference clip (5–10 s wav)</label><input type="file" id="ref" accept=".wav,audio/*"></div>
    <div><label>Reference language</label>
      <select id="reflang"><option>en</option><option>zh</option><option>ja</option><option>ko</option><option>yue</option></select></div>
  </div>
  <div><label>Transcript of the clip (verbatim)</label><textarea id="reftext" style="min-height:3.5rem"></textarea></div>
</div>

<div class="card">
  <div class="row">
    <div style="flex:1"><label>Your text</label><textarea id="text" placeholder="Type anything…"></textarea></div>
  </div>
  <div class="row">
    <div><label>Text language</label>
      <select id="textlang"><option>en</option><option>zh</option><option>ja</option><option>ko</option><option>yue</option></select></div>
    <div><label>Speed</label><input type="number" id="speed" value="1.0" min="0.5" max="2" step="0.05"></div>
    <div><label>Seed (-1 = random)</label><input type="number" id="seed" value="-1"></div>
  </div>
  <div class="row">
    <button id="go">🔊 Speak</button>
    <button id="sample" type="button" class="ghost">↺ Use sample line</button>
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
const charsSel = $("char"), langSel = $("lang"), clipSel = $("clip");

function fillChars() {
  charsSel.innerHTML = Object.keys(byChar).map(c => `<option>${c}</option>`).join("");
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
    `<option value="${i}">${c.clip} (${c.seconds}s)</option>`).join("");
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

  const btn = $("go"); btn.disabled = true; msg("synthesizing… (CPU: a few seconds)");
  try {
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref_audio_path: ref, prompt_text: promptText, prompt_lang: promptLang,
        text, text_lang: $("textlang").value, speed_factor: +$("speed").value, seed: +$("seed").value }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      msg(j.message || ("HTTP " + r.status), true);
      return;
    }
    const blob = await r.blob();
    const out = $("out");
    out.src = URL.createObjectURL(blob);
    out.style.display = "";
    out.play();
    msg("ok", false);
  } catch (e) {
    msg("error: " + e, true);
  } finally {
    btn.disabled = false;
  }
};
function msg(t, isErr) { const m = $("msg"); m.textContent = t; m.className = isErr ? "err" : "ok"; }

fetch("/api/status").then(r => r.json()).then(j => {
  $("ver").textContent = `v${j.version} · langs: ${j.languages.join(", ")} · ${j.pack} pack clip(s)`;
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

    app = build_app(args.gsv_root.resolve(), args.seeds.resolve(), args.tts_config)
    url = f"http://{'127.0.0.1' if args.bind in ('0.0.0.0', '::') else args.bind}:{args.port}"
    if args.open:
        import threading  # noqa: E402
        import webbrowser  # noqa: E402
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"[type-ui] serving {url}")
    uvicorn.run(app, host=args.bind, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
