#!/usr/bin/env python3
"""Post-render audio controls for TTS lines: pitch, speed, volume.

Pure post-processing on the rendered WAV — no model, no re-render, so a line's
delivery can be tuned after the fact ("same take, a touch higher"). This is
what backs the per-line manifest columns in tools/render_batch.py:

    id<TAB>char<TAB>lang<TAB>text<TAB>[pitch_semitones]<TAB>[speed]<TAB>[volume_db]

Standalone use (tune any wav):

    python tools/audio_fx.py --in line.wav --out line_hi.wav \
        --pitch 3 --speed 1.05 --volume -2

Controls:
  --pitch    semitones, +3 = one tone higher, -12 = one octave lower (WSOLA,
             duration is preserved)
  --speed    1.1 = 10% faster (resample)
  --volume   dB, -2 = quieter, +6 = about twice as loud (soft-clips instead
             of distorting)

Needs numpy (present in the GPT-SoVITS conda env; every other tool in this
repo is stdlib-only). Reads/writes 16-bit PCM WAV (32-bit float WAV input
accepted); multi-channel input is folded to mono.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - env without numpy
    np = None


def _require_numpy() -> None:
    if np is None:
        raise SystemExit(
            "audio_fx needs numpy — it is part of the GPT-SoVITS conda env; "
            "run it from that env (or: pip install numpy)"
        )


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV as (sample_rate, float64 mono in [-1, 1])."""
    _require_numpy()
    with wave.open(str(path), "rb") as w:
        nch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported {sw * 8}-bit sample width (need 16-bit PCM or 32-bit float)")
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return sr, a


def save_wav(path: Path, sr: int, x: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(x, -1.0, 1.0) * 32767.0
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.astype(np.int16).tobytes())


def speed_change(x: np.ndarray, sr: int, speed: float) -> tuple[np.ndarray, int]:
    """Change tempo by resampling (1.1 = 10% faster). Rate stays the same."""
    if speed == 1.0:
        return x, sr
    n_out = max(1, int(round(len(x) / speed)))
    t = np.linspace(0.0, len(x) - 1, n_out)
    return np.interp(t, np.arange(len(x)), x), sr


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by semitones, preserving duration (WSOLA overlap-add).

    Time-stretches by 1/2^(s/12) with 20 ms grains / 5 ms hop / 5 ms search,
    then resamples back to the original length.
    """
    if semitones == 0:
        return x
    grain = int(round(0.020 * sr))
    if len(x) < 4 * grain:
        # too short for grains — fall back to a plain resample (duration changes)
        t = np.linspace(0.0, len(x) - 1, max(1, int(round(len(x) * 2 ** (-semitones / 12.0)))))
        return np.interp(t, np.arange(len(x)), x)
    hop = int(round(0.005 * sr))
    search = int(round(0.005 * sr))
    stretch = 2.0 ** (semitones / 12.0)  # >1 = lengthen (then resample back => pitch up)
    n_out = int(round(len(x) * stretch))
    out = np.zeros(n_out + grain)
    weight = np.zeros(n_out + grain)
    win = np.hamming(grain)
    out_hop = hop * stretch
    pos = 0.0
    while True:
        o = int(round(pos))
        if o >= n_out:
            break
        c = int(round(pos / stretch))
        max_cand = len(x) - grain
        lo, hi = max(0, c - search), min(max_cand, c + search)
        if lo > hi:
            lo = hi = max(0, min(c, max_cand))
        window = out[o:o + grain]
        best, best_corr = min(c, max_cand), -np.inf
        for cand in range(lo, hi + 1):
            corr = float(np.dot(window, x[cand:cand + grain]))
            if corr > best_corr:
                best_corr, best = corr, cand
        out[o:o + grain] += x[best:best + grain] * win
        weight[o:o + grain] += win
        pos += out_hop
    out, weight = out[:n_out], weight[:n_out]
    np.maximum(weight, 1e-6, out=weight)
    stretched = out / weight
    if stretched.size != len(x):
        t = np.linspace(0.0, stretched.size - 1, len(x))
        stretched = np.interp(t, np.arange(stretched.size), stretched)
    return stretched


def apply_volume(x: np.ndarray, db: float) -> np.ndarray:
    """Gain in dB; soft-clips with tanh instead of distorting when it peaks."""
    if db == 0.0:
        return x
    y = x * (10.0 ** (db / 20.0))
    if float(np.max(np.abs(y))) > 1.0:
        y = np.tanh(y) / np.tanh(1.0)
    return y


def process(in_path, out_path, pitch: float = 0.0, speed: float = 1.0,
            volume_db: float = 0.0) -> dict:
    """Apply pitch/speed/volume to one WAV file. Returns a small report."""
    _require_numpy()
    in_path, out_path = Path(in_path), Path(out_path)
    if in_path.resolve() == out_path.resolve():
        raise ValueError("in and out must be different files (no in-place mode)")
    if not (pitch or speed != 1.0 or volume_db):
        save_wav(out_path, *load_wav(in_path))
        return {"seconds": 0.0, "applied": []}
    sr, x = load_wav(in_path)
    applied = []
    if pitch:
        x = pitch_shift(x, sr, float(pitch))
        applied.append(f"pitch {float(pitch):+g} st")
    if speed != 1.0:
        x, sr = speed_change(x, sr, float(speed))
        applied.append(f"speed {float(speed):g}x")
    if volume_db:
        x = apply_volume(x, float(volume_db))
        applied.append(f"volume {float(volume_db):+g} dB")
    save_wav(out_path, sr, x)
    return {"sr": sr, "seconds": len(x) / sr, "applied": applied}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pitch / speed / volume controls for rendered WAVs")
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--pitch", type=float, default=0.0, help="semitones (+3 = one tone up)")
    ap.add_argument("--speed", type=float, default=1.0, help="tempo (1.1 = 10%% faster)")
    ap.add_argument("--volume", type=float, default=0.0, help="dB (+6 ~ 2x louder)")
    args = ap.parse_args(argv)
    if not args.src.exists():
        print(f"error: {args.src} not found", file=sys.stderr)
        return 2
    try:
        rep = process(args.src, args.dst, args.pitch, args.speed, args.volume)
    except Exception as e:  # noqa: BLE001 - report, don't traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"[fx] {args.dst}: {rep['seconds']:.2f}s — " + (", ".join(rep["applied"]) or "no-op (copy)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
