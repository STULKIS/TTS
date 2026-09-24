#!/usr/bin/env python3
"""Post-render audio controls for TTS lines: pitch, speed, volume.

Pure post-processing on the rendered WAV — no model, no re-render, so a line's
delivery can be tuned after the fact ("same take, a touch higher"). This is
what backs the per-line manifest columns in tools/render_batch.py:

    id<TAB>char<TAB>lang<TAB>text<TAB>[pitch_semitones]<TAB>[speed]<TAB>[volume_db]<TAB>[fx]

Standalone use (tune any wav):

    python tools/audio_fx.py --in line.wav --out line_hi.wav \
        --pitch 3 --speed 1.05 --volume -2 --fx "reverb,normalize"

Controls:
  --pitch    semitones, +3 = one tone higher, -12 = one octave lower (WSOLA,
             duration is preserved)
  --speed    1.1 = 10% faster (resample)
  --volume   dB, -2 = quieter, +6 = about twice as loud (a limiter keeps the
             waveform intact at the top — loud, never distorted)
  --fx       comma tokens: robot · phone · reverb · chorus · echo · humanize
             · sparkle · normalize (normalize = level to -20 dBFS RMS — put it
             last in the chain; humanize/sparkle = the 'alive' pair)

Needs numpy (present in the GPT-SoVITS conda env; every other tool in this
repo is stdlib-only). Reads/writes 16-bit PCM WAV (32-bit float WAV input
accepted); multi-channel input is folded to mono.
"""
from __future__ import annotations

import argparse
import re
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


def _peak_limit(x: np.ndarray, ceiling: float = 0.999) -> np.ndarray:
    """Scale the whole signal down if needed so the peak sits at the ceiling.

    Waveform shape (and dynamics) are preserved — nothing is squared off. Loud
    settings stay loud; they can no longer destroy a take.
    """
    p = float(np.max(np.abs(x)))
    if p > ceiling > 0:
        return x * (ceiling / p)
    return x


def apply_volume(x: np.ndarray, db: float) -> np.ndarray:
    """Gain in dB. If the result would clip, the take is scaled back to just
    under full scale (a limiter) instead of being distorted."""
    if db == 0.0:
        return x
    return _peak_limit(x * (10.0 ** (db / 20.0)))


def _bandpass_fft(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """FFT band-pass (stylized FX use; sharp edges, slight ringing)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x))


def robot(x: np.ndarray, sr: int) -> np.ndarray:
    """Classic robot voice: hard square drive + low-pass + 55 Hz ring-mod warble."""
    t = np.arange(len(x)) / sr
    y = _bandpass_fft(np.sign(x), sr, 0.0, 4000.0)
    return y * np.sign(np.sin(2 * np.pi * 55.0 * t))


def phone(x: np.ndarray, sr: int) -> np.ndarray:
    """Telephone effect: band-pass 300-3400 Hz."""
    return _bandpass_fft(x, sr, 300.0, 3400.0)


def reverb(x: np.ndarray, sr: int, wet: float = 0.4) -> np.ndarray:
    """Cheap decaying-tap reverb (FIR comb) — space without a long tail model."""
    delays = (0.025, 0.041, 0.059, 0.077, 0.097, 0.121)
    gains = (0.5, 0.4, 0.32, 0.25, 0.18, 0.12)
    y = np.zeros_like(x)
    for d_s, g in zip(delays, gains):
        d = int(round(d_s * sr))
        if d >= len(x):
            break
        y[d:] += g * x[:-d]
    out = x + wet * y
    return _peak_limit(out)


def normalize(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """RMS-normalize to a target loudness (use as the LAST fx in the chain)."""
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-9:
        return x
    y = x * (10.0 ** (target_dbfs / 20.0) / rms)
    return _peak_limit(y)


def humanize(x: np.ndarray, sr: int) -> np.ndarray:
    """The 'alive' effect: micro tempo drift per 120 ms block + breathing gain.

    Cloned speech often sounds flat; this reintroduces the tiny timing and
    loudness wander of a living speaker. Duration preserved; deterministic.
    """
    _require_numpy()
    rng = np.random.default_rng((len(x) * 2654435761) & 0xFFFFFFFF)
    n = len(x)
    blk = max(2, int(round(0.12 * sr)))
    parts = []
    for i0 in range(0, n, blk):
        seg = x[i0:i0 + blk]
        if len(seg) < 2:
            parts.append(seg)
            continue
        f = 1.0 + rng.uniform(-0.025, 0.025)
        m = max(2, int(round(len(seg) * f)))
        parts.append(np.interp(np.linspace(0.0, len(seg) - 1, m),
                               np.arange(len(seg)), seg))
    y = np.concatenate(parts) if parts else x.copy()
    if len(y) != n:
        y = np.interp(np.linspace(0.0, len(y) - 1, n), np.arange(len(y)), y)
    t = np.arange(n) / sr
    breathe = 1.0 + 0.08 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0.0, 6.28)) \
        + 0.05 * np.sin(2 * np.pi * 2.1 * t)
    return _peak_limit(y * breathe)


def sparkle(x: np.ndarray, sr: int) -> np.ndarray:
    """Air + presence shelf (+2.5 dB at 3-6 kHz, +3.5 dB above) — cuts the
    'muffled clone' feel and makes diction glitter."""
    _require_numpy()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    g = 1.0 + 0.33 * ((f >= 3000) & (f < 6000)) + 0.45 * (f >= 6000)
    return _peak_limit(np.fft.irfft(X * g, n=len(x)))


def chorus(x: np.ndarray, sr: int, wet: float = 0.5) -> np.ndarray:
    """Two detuned delayed copies (14/23 ms, ±0.4%) — thickens a thin take."""
    _require_numpy()
    y = x.copy()
    for d_s, g, det in ((0.014, 0.30, 1.004), (0.023, 0.22, 0.996)):
        d = int(round(d_s * sr))
        if d >= len(x):
            continue
        src = x[d:]
        m = max(2, int(round(len(src) * det)))
        delayed = np.interp(np.linspace(0.0, len(src) - 1, m), np.arange(len(src)), src)
        k = min(len(y), m)
        y[:k] += (wet * g) * delayed[:k]
    return _peak_limit(y)


def echo(x: np.ndarray, sr: int, wet: float = 0.45) -> np.ndarray:
    """Three decaying taps (90/180/280 ms) — a small room behind the words."""
    _require_numpy()
    y = np.zeros_like(x)
    for d_s, g in ((0.09, 0.4), (0.18, 0.22), (0.28, 0.12)):
        d = int(round(d_s * sr))
        if d >= len(x):
            break
        y[d:] += g * x[:-d]
    return _peak_limit(x + wet * y)


FX_TOKENS = ("robot", "phone", "reverb", "chorus", "echo", "humanize", "sparkle", "normalize")


def parse_fx(spec: str) -> list[str]:
    """'robot, reverb' -> ['robot', 'reverb']. Unknown tokens warn + skip
    (a typo should never kill a whole batch)."""
    out: list[str] = []
    for tok in re.split(r"[,\s]+", (spec or "").strip()):
        if not tok:
            continue
        t = tok.lower()
        if t in FX_TOKENS:
            if t not in out:
                out.append(t)
        else:
            print(f"[warn] unknown fx token {tok!r} — skipped "
                  f"(known: {', '.join(FX_TOKENS)})", file=sys.stderr)
    return out


def apply_fx(x: np.ndarray, sr: int, name: str) -> np.ndarray:
    if name == "robot":
        return robot(x, sr)
    if name == "phone":
        return phone(x, sr)
    if name == "reverb":
        return reverb(x, sr)
    if name == "chorus":
        return chorus(x, sr)
    if name == "echo":
        return echo(x, sr)
    if name == "humanize":
        return humanize(x, sr)
    if name == "sparkle":
        return sparkle(x, sr)
    if name == "normalize":
        return normalize(x)
    raise ValueError(f"unknown fx {name!r}")


def process(in_path, out_path, pitch: float = 0.0, speed: float = 1.0,
            volume_db: float = 0.0, fx: str = "") -> dict:
    """Apply pitch/speed/volume/fx to one WAV file. Returns a small report."""
    _require_numpy()
    in_path, out_path = Path(in_path), Path(out_path)
    if in_path.resolve() == out_path.resolve():
        raise ValueError("in and out must be different files (no in-place mode)")
    fx_names = parse_fx(fx)
    if not (pitch or speed != 1.0 or volume_db or fx_names):
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
    for name in fx_names:
        x = apply_fx(x, sr, name)
        applied.append(f"fx {name}")
    save_wav(out_path, sr, x)
    return {"sr": sr, "seconds": len(x) / sr, "applied": applied}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pitch / speed / volume / fx controls for rendered WAVs")
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--pitch", type=float, default=0.0, help="semitones (+3 = one tone up)")
    ap.add_argument("--speed", type=float, default=1.0, help="tempo (1.1 = 10%% faster)")
    ap.add_argument("--volume", type=float, default=0.0, help="dB (+6 ~ 2x louder)")
    ap.add_argument("--fx", default="",
                    help="comma-separated tokens: " + ",".join(FX_TOKENS) +
                         " (e.g. 'robot,reverb' — put normalize last)")
    args = ap.parse_args(argv)
    if not args.src.exists():
        print(f"error: {args.src} not found", file=sys.stderr)
        return 2
    try:
        rep = process(args.src, args.dst, args.pitch, args.speed, args.volume, args.fx)
    except Exception as e:  # noqa: BLE001 - report, don't traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"[fx] {args.dst}: {rep['seconds']:.2f}s — " + (", ".join(rep["applied"]) or "no-op (copy)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
