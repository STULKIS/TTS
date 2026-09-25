#!/usr/bin/env python3
"""Pre-flight check for the Windows install — run BEFORE your first render.

One command that finds the problems while they're still cheap:

    python tools/preflight.py --gsv-root D:\\GSV --seeds D:\\TTS\\seeds

Checks (stdlib only, read-only — nothing is modified):
  1. GPT-SoVITS root: GPT_SoVITS/ dir + configs/tts_infer.yaml
  2. CPU correctness: no-NVIDIA machines need `device: cpu` / `is_half: false`
     (fix: tools/cpu_config.py)
  3. Pretrained weights present (v2final or v2Pro + hubert + roberta)
  4. Seed pack integrity (same rules as tools/check_seeds.py, incl. the
     sample-manifest cross-check when the manifest is passed)
  5. Ports: 9874 (WebUI) / 7861 (type_ui) free
  6. Free disk space (warns below 10 GB)

Prints [ok] / [warn] / [fail] per check, ends with the exact next command.
Exit 0 = ready (warnings allowed), 1 = at least one [fail].
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import socket
import sys
from pathlib import Path

WEIGHT_SETS = {
    "v2final": ["gpt.pth", "s2G2333.pth", "s2D2333.pth"],
    "v2Pro": ["s2Gv2Pro.pth", "s2Dv2Pro.pth"],
    "v2ProPlus": ["s2Gv2ProPlus.pth", "s2Dv2ProPlus.pth"],
}
SHARED = ["chinese-hubert-base", "chinese-roberta-wwm-ext-large"]

fails: list[str] = []
warns: list[str] = []


def ok(msg: str) -> None:
    print(f"[ok]   {msg}")


def warn(msg: str) -> None:
    warns.append(msg)
    print(f"[warn] {msg}")


def fail(msg: str) -> None:
    fails.append(msg)
    print(f"[fail] {msg}")


def check_gsv(root: Path) -> bool:
    pkg = root / "GPT_SoVITS"
    if not pkg.is_dir():
        fail(f"{root} is not a GPT-SoVITS root (no GPT_SoVITS/ dir)")
        return False
    ok(f"GPT-SoVITS root: {root}")

    config = pkg / "configs" / "tts_infer.yaml"
    if not config.is_file():
        fail(f"missing {config}")
        return False
    ok("tts_infer.yaml present")

    text = config.read_text(encoding="utf-8", errors="replace")
    devices = {line.split(":", 1)[1].strip() for line in text.splitlines()
               if line.strip().startswith("device:")}
    halves = {line.split(":", 1)[1].strip() for line in text.splitlines()
              if line.strip().startswith("is_half:")}
    if devices and devices <= {"cuda", "cuda:0"}:
        fail("config still targets CUDA — no-NVIDIA box: run "
             f"`python tools/cpu_config.py --gsv-root {root}`")
    elif "true" in halves:
        warn("is_half: true found — CPU machines need it false "
             "(run tools/cpu_config.py)")
    else:
        ok("config is CPU-correct")

    pm = pkg / "pretrained_models"
    if not pm.is_dir():
        fail(f"missing pretrained models dir: {pm} (install step skipped?)")
        return False
    have = [p.name for p in pm.iterdir() if p.is_dir()]
    model_set = next((name for name, files in WEIGHT_SETS.items()
                      if all((pm / "gsv-v2final-pretrained" / f).is_file()
                             or (pm / name / f).is_file() for f in files)), None)
    if model_set is None:
        fail(f"no GPT/SoVITS weight set found under {pm} "
             f"(need v2final, v2Pro or v2ProPlus weights + s2G/s2D pairs)")
    else:
        ok(f"weight set found: {model_set}")
    for name in SHARED:
        if (pm / name).is_dir():
            ok(f"{name} present")
        else:
            fail(f"missing {pm / name} (required for zero-shot)")
    return True


def check_seeds(seeds: Path, manifest: Path | None) -> None:
    if not seeds.is_dir():
        fail(f"seeds dir not found: {seeds}")
        return
    tools_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("check_seeds", tools_dir / "check_seeds.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        fail("cannot load tools/check_seeds.py")
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    for char_dir in sorted(p for p in seeds.iterdir() if p.is_dir()):
        for wav in sorted(char_dir.glob("*.wav")):
            module.check_clip(wav, wav.with_suffix(".txt"), errors, warnings)
            checked += 1
    for w in warnings:
        warn(f"seeds: {w}")
    for e in errors:
        fail(f"seeds: {e}")
    if not errors:
        ok(f"seed pack: {checked} clip(s) valid under {seeds}")

    if manifest is not None:
        if not manifest.is_file():
            fail(f"manifest not found: {manifest}")
        else:
            used = module.read_manifest_langs(manifest)
            missing = [f"{seeds}/{c}/{l}.wav" for c, langs in sorted(used.items())
                       for l in sorted(langs)
                       if not (seeds / c / f"{l}.wav").is_file()]
            if missing:
                fail("manifest needs missing seed clips: " + ", ".join(missing))
            else:
                ok(f"manifest cross-check: {manifest.name} fully covered")


def check_port(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        ok(f"port {port} free")
    except OSError:
        warn(f"port {port} already in use — pick another with --port / the WebUI Config tab")
    finally:
        s.close()


def check_disk(root: Path) -> None:
    try:
        usage = shutil.disk_usage(str(root))
    except OSError:
        warn("could not check disk space")
        return
    free_gb = usage.free / 1024 ** 3
    if free_gb < 10:
        warn(f"only {free_gb:.1f} GB free on {root.anchor} — 10 GB+ recommended")
    else:
        ok(f"disk: {free_gb:.1f} GB free on {root.anchor}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gsv-root", type=Path, default=Path("."),
                    help="GPT-SoVITS repo root (folder containing GPT_SoVITS/)")
    ap.add_argument("--seeds", type=Path, default=Path("seeds"))
    ap.add_argument("--manifest", type=Path, default=None,
                    help="TSV manifest to cross-check against the pack")
    ap.add_argument("--skip-gsv", action="store_true",
                    help="only check seeds/ports/disk (e.g. before GPT-SoVITS is installed)")
    args = ap.parse_args()

    fails.clear()
    warns.clear()
    print("== pre-flight check ==")
    gsv_ok = check_gsv(args.gsv_root) if not args.skip_gsv else None
    check_seeds(args.seeds, args.manifest)
    check_port(9874)
    check_port(7861)
    check_disk(args.gsv_root if args.gsv_root.exists() else Path.cwd())

    print()
    if fails:
        print(f"[FAIL] {len(fails)} problem(s) — fix the [fail] lines above, re-run.")
        raise SystemExit(1)
    if gsv_ok is None:
        print("[ready] seeds + ports + disk OK (GPT-SoVITS not checked — --skip-gsv).")
    elif gsv_ok:
        print(f"[ready] all checks passed. Start the TTS:\n"
              f"    python tools/type_ui.py --seeds {args.seeds} --open\n"
              f"  (or double-click go-webui.bat for the full WebUI)")
    else:
        print("[ready-with-caveats] see [fail] lines — seeds are fine, GPT-SoVITS setup needs work.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
