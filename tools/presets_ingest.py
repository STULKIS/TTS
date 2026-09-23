#!/usr/bin/env python3
"""Validate the gacha preset catalog (presets.csv) and generate a browsable catalog.

Input (CSV with header, one voice per line):

    no,name,class,gender,rarity,pitch,pace,signature,description

  no          1..N, sequential
  name        base:  "Genki Spark"              variant: "Genki Spark · Honey Lilt"
  class       archetype class (HTML entities are unescaped: "&amp;" -> "&")
  gender      Feminine | Masculine | Androgynous | Mixed
  rarity      3 | 4 | 5   (base rows are 5)
  pitch       High | Mid | Low | Deep | Neutral | Soft | Synth   (other -> warning)
  pace        Dash | Lilt | Burst | Drift | Tick | Pulse | Measure | Lounge | Tale | Pause
  signature   yes (base row) | no (variant)
  description one line of flavor text (quoted if it contains commas)

Outputs (written next to the CSV):

    PRESET-CATALOG.md    grouped class -> character; every line mapped to the
                         58-feature vocabulary in seeds/README.md, so any line
                         is directly orderable ("new character with a ... voice")
    PRESET-CATALOG.html  same data, single file, search + class/pitch/pace/rarity
                         filters in the browser

Exit codes: 0 = valid (warnings allowed) · 1 = hard validation errors
            2 = input file not found

Usage:
    python tools/presets_ingest.py                      # presets.csv in cwd
    python tools/presets_ingest.py --csv mine.csv --no-html
"""
from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

HEADER = ["no", "name", "class", "gender", "rarity", "pitch",
          "pace", "signature", "description"]
GENDERS = {"Feminine", "Masculine", "Androgynous", "Mixed"}
RARITIES = {3, 4, 5}
SIGNATURES = {"yes", "no"}
EXPECTED_PITCHES = {"High", "Mid", "Low", "Deep", "Neutral", "Soft", "Synth"}
EXPECTED_PACES = {"Dash", "Lilt", "Burst", "Drift", "Tick", "Pulse",
                  "Measure", "Lounge", "Tale", "Pause"}
EXPECTED_TEXTURES = {"Sunbeam", "Honey", "Bell", "Reed", "Crystal", "Husk",
                     "Silver", "Velvet", "Resonant", "Gravel", "Wisp", "Chrome"}
VARIANT_SEP = " \u00b7 "  # " · "

# ---- axis -> 58-feature vocabulary (seeds/README.md) ------------------------
# pitch axis -> Physical/texture group
PITCH_FEATURES = {
    "High":    "high pitch · bright · forward placement",
    "Mid":     "mid pitch · even intonation",
    "Low":     "low pitch · chest resonance · full weight",
    "Deep":    "very low pitch · chest resonance · heavy, unhurried weight",
    "Neutral": "mid pitch · level pitch variety · no tilt",
    "Soft":    "soft register · low volume · slight breathiness · close-mic intimacy",
    "Synth":   "synthetic register · metallic brightness · rock-steady · neutral affect",
}
# pace axis -> Delivery/dynamics group
PACE_FEATURES = {
    "Dash":    "very fast rate · high energy · flowing rhythm",
    "Lilt":    "fast rate · big intonation · flowing upbeat rhythm",
    "Burst":   "fast rate · staccato bursts · punchy emphasis",
    "Drift":   "slow rate · loose rhythm · slight mumble, trailing phrases",
    "Tick":    "medium-fast rate · staccato · crisp articulation · short pauses",
    "Pulse":   "steady metronomic rhythm · medium rate · rock-steady",
    "Measure": "slow rate · even rhythm · level emphasis · clear articulation",
    "Lounge":  "slow rate · low energy · relaxed formality · easy confidence",
    "Tale":    "medium rate · narrative swells · shaped emphasis · mid intimacy",
    "Pause":   "long thoughtful pauses · trailing sentence endings · slow rate",
}
# texture axis (variant names) -> Physical/texture group
TEXTURE_FEATURES = {
    "Sunbeam":   "bright · dry-warm · smile tone · warmth",
    "Honey":     "warm wetness · slight smokiness · velvety · sweet smile tone",
    "Bell":      "crystalline brightness · head resonance · dry · crisp articulation",
    "Reed":      "reedy · dry · light breathiness",
    "Crystal":   "crystalline · dry · head resonance · rock-steady",
    "Husk":      "mid-high roughness · dry · hollow weight",
    "Silver":    "cold brightness · dry · head resonance · low warmth",
    "Velvet":    "velvety · warm wetness · chest resonance · soft",
    "Resonant":  "chest resonance · full weight · authoritative",
    "Gravel":    "gravelly roughness · dry · mature age",
    "Wisp":      "breathy · thin weight · low volume · back placement",
    "Chrome":    "metallic brightness · flat resonance · crisp · rock-steady · neutral affect",
}


def clean(value: str) -> str:
    """Unescape HTML entities (&amp; -> &) and strip surrounding whitespace."""
    return html_mod.unescape(value or "").strip()


def feature_bundle(row: dict) -> str:
    """Map one catalog line onto the 58-feature vocabulary."""
    parts = []
    pitch = PITCH_FEATURES.get(row["pitch"])
    parts.append(pitch if pitch else f"pitch: {row['pitch']} (no mapping — extend PITCH_FEATURES)")
    texture = row.get("texture") or ""
    tex = TEXTURE_FEATURES.get(texture)
    if tex:
        parts.append(f"texture: {tex}")
    pace = PACE_FEATURES.get(row["pace"])
    parts.append(pace if pace else f"pace: {row['pace']} (no mapping — extend PACE_FEATURES)")
    return " · ".join(parts)


def parse_csv(path: Path):
    """Read the CSV. Returns (rows, errors, warnings)."""
    errors, warnings, rows = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        errors.append("file is empty")
        return rows, errors, warnings
    header = [clean(c) for c in raw[0]]
    if header != HEADER:
        errors.append(f"line 1: header mismatch — got {header!r}, want {HEADER!r}")
    for line, rec in enumerate(raw[1:], start=2):
        if not rec or all(not clean(c) for c in rec):
            continue  # trailing blank line
        if len(rec) != len(HEADER):
            errors.append(f"line {line}: expected {len(HEADER)} fields, got {len(rec)}")
            continue
        row = dict(zip(HEADER, (clean(c) for c in rec)))
        row["_line"] = line
        rows.append(row)
    return rows, errors, warnings


def validate(rows):
    """Structural checks. Returns (errors, warnings); annotates rows in place."""
    errors, warnings = [], []
    seen_names, bases = {}, {}
    for pos, row in enumerate(rows, start=1):
        line = row["_line"]
        # sequential numbering
        if not row["no"].isdigit() or int(row["no"]) != pos:
            errors.append(f"line {line}: no={row['no']!r}, expected {pos}")
            row["no"] = -1
        else:
            row["no"] = int(row["no"])
        for field in HEADER[1:]:
            if not row[field]:
                errors.append(f"line {line} ({row['no']}): empty {field!r}")
        # enums
        if row["gender"] not in GENDERS:
            errors.append(f"line {line} ({row['no']}): unknown gender {row['gender']!r}")
        if row["rarity"].isdigit() and int(row["rarity"]) in RARITIES:
            row["rarity"] = int(row["rarity"])
        else:
            errors.append(f"line {line} ({row['no']}): rarity must be 3|4|5, got {row['rarity']!r}")
        if row["signature"] not in SIGNATURES:
            errors.append(f"line {line} ({row['no']}): signature must be yes|no, got {row['signature']!r}")
        if row["pitch"] not in EXPECTED_PITCHES:
            warnings.append(f"line {line} ({row['no']}): pitch {row['pitch']!r} outside expected set")
        if row["pace"] not in EXPECTED_PACES:
            warnings.append(f"line {line} ({row['no']}): pace {row['pace']!r} outside expected set")
        # duplicate names
        key = row["name"].casefold()
        if key in seen_names:
            errors.append(f"line {line} ({row['no']}): duplicate name {row['name']!r} (first at line {seen_names[key]})")
        else:
            seen_names[key] = line
        # base / variant structure
        is_base = row["signature"] == "yes"
        if VARIANT_SEP in row["name"]:
            left, _, right = row["name"].partition(VARIANT_SEP)
            if is_base:
                errors.append(f"line {line} ({row['no']}): base row has a variant-style name")
            else:
                if left not in bases:
                    errors.append(f"line {line} ({row['no']}): variant of unknown base {left!r}")
                elif bases[left]["class"] != row["class"]:
                    errors.append(f"line {line} ({row['no']}): variant of {left!r} appears in class {row['class']!r}, base is in {bases[left]['class']!r}")
                parts = right.split()
                if len(parts) == 2:
                    tex, pace_word = parts
                    row["texture"] = tex
                    if tex not in EXPECTED_TEXTURES:
                        warnings.append(f"line {line} ({row['no']}): texture {tex!r} outside the 12 expected families")
                    if pace_word != row["pace"]:
                        warnings.append(f"line {line} ({row['no']}): name says pace {pace_word!r} but column says {row['pace']!r}")
                else:
                    row["texture"] = ""
                    warnings.append(f"line {line} ({row['no']}): variant suffix {right!r} is not '<Texture> <Pace>'")
        else:
            row["texture"] = ""
            if not is_base:
                errors.append(f"line {line} ({row['no']}): non-base row name lacks {VARIANT_SEP!r}")
            elif isinstance(row["rarity"], int) and row["rarity"] != 5:
                warnings.append(f"line {line} ({row['no']}): base row has rarity {row['rarity']} (expected 5)")
        if is_base:
            bases[row["name"]] = row
    # block sizes — a short final block usually means a truncated file
    if bases:
        counts = Counter()
        for row in rows:
            if row["signature"] == "no" and VARIANT_SEP in row["name"]:
                counts[row["name"].partition(VARIANT_SEP)[0]] += 1
        sizes = [counts.get(b, 0) for b in bases]
        mode = Counter(sizes).most_common(1)[0][0]
        for name, n in zip(bases, sizes):
            if n < mode:
                warnings.append(f"character {name!r}: only {n} variant(s) (catalog mode is {mode}) — block may be truncated")
    # class contiguity — a class that opens, closes, and re-opens
    seen = []
    for row in rows:
        if not seen or seen[-1] != row["class"]:
            if row["class"] in seen:
                warnings.append(f"line {row['_line']} ({row['no']}): class {row['class']!r} appears in two separate blocks")
            seen.append(row["class"])
    return errors, warnings


def summarize(rows):
    classes = OrderedDict()
    for row in rows:
        classes.setdefault(row["class"], []).append(row)
    base_n = sum(1 for r in rows if r["signature"] == "yes")
    summary = {
        "rows": len(rows),
        "bases": base_n,
        "variants": len(rows) - base_n,
        "classes": len(classes),
        "class_table": [
            (c, sum(1 for r in v if r["signature"] == "yes"),
             sum(1 for r in v if r["signature"] == "no"), len(v))
            for c, v in classes.items()
        ],
        "pitch_vocab": dict(Counter(r["pitch"] for r in rows).most_common()),
        "pace_vocab": dict(Counter(r["pace"] for r in rows).most_common()),
        "texture_vocab": dict(Counter(r["texture"] for r in rows if r.get("texture")).most_common()),
        "rarity_vocab": dict(sorted(Counter(r["rarity"] for r in rows if isinstance(r["rarity"], int)).items(), reverse=True)),
        "gender_vocab": dict(Counter(r["gender"] for r in rows).most_common()),
    }
    return summary, classes


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_md(path: Path, rows, classes, summary, warnings):
    L = []
    L.append("# Gacha Voice Presets — full catalog")
    L.append("")
    L.append(f"> **Generated by `tools/presets_ingest.py` from `presets.csv`** — do not edit by hand.")
    L.append("> Edit the CSV, then re-run: `python tools/presets_ingest.py`")
    L.append("")
    L.append("## Validation summary")
    L.append("")
    L.append(f"- **{summary['rows']} lines** = {summary['bases']} bases (signature=yes) + "
             f"{summary['variants']} variants, in **{summary['classes']} classes**")
    L.append(f"- warnings: {len(warnings)}" + ("" if warnings else " (clean)"))
    L.append(f"- pitch vocabulary: {', '.join(f'{k} ({v})' for k, v in summary['pitch_vocab'].items())}")
    L.append(f"- pace vocabulary: {', '.join(f'{k} ({v})' for k, v in summary['pace_vocab'].items())}")
    L.append(f"- texture vocabulary: {', '.join(f'{k} ({v})' for k, v in summary['texture_vocab'].items())}")
    L.append(f"- rarity: {', '.join(f'{k} ({v})' for k, v in summary['rarity_vocab'].items())}")
    L.append(f"- gender: {', '.join(f'{k} ({v})' for k, v in summary['gender_vocab'].items())}")
    if warnings:
        L.append("")
        L.append("Warnings (not blocking — inspect before ordering these lines):")
        for w in warnings:
            L.append(f"- {w}")
    L.append("")
    L.append("## How to order any line")
    L.append("")
    L.append("Each line maps 1:1 onto the **58-feature vocabulary** (`seeds/README.md`):")
    L.append("")
    L.append("- **pitch column** → pitch / resonance / weight / brightness (Physical group)")
    L.append("- **pace column** → rate / rhythm / emphasis / pausing (Delivery group)")
    L.append("- **texture** (2nd word of variant names) → roughness / wetness / velvety / breathiness / metallic (Physical group)")
    L.append("")
    L.append("Prompt template for chat (or a seed design brief):")
    L.append("")
    L.append("> new character with a **{class}** voice — {pitch} pitch, {texture} texture, {pace} delivery: “{description}”")
    L.append("")
    L.append("Class index:")
    L.append("")
    L.append("| # | class | bases | variants | lines |")
    L.append("|---|-------|-------|----------|-------|")
    for i, (c, b, v, n) in enumerate(summary["class_table"], 1):
        L.append(f"| {i} | {c} | {b} | {v} | {n} |")
    L.append("")
    for cls, clines in classes.items():
        L.append(f"## {cls} ({len(clines)})")
        L.append("")
        for row in clines:
            if row["signature"] == "yes":
                L.append(f"### {row['no']}. {row['name']} — *base* (rarity {row['rarity']} · {row['pitch']} · {row['pace']} · {row['gender']})")
                L.append("")
                L.append(f"> {row['description']}")
                L.append("")
                L.append(f"Feature bundle: {feature_bundle(row)}")
                L.append("")
            else:
                tex = f" · {row['texture']}" if row.get("texture") else ""
                L.append(f"- **{row['no']}. {row['name']}** (rarity {row['rarity']} · {row['pitch']}{tex} · {row['pace']}) — {row['description']}")
                L.append(f"  - bundle: {feature_bundle(row)}")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_html(path: Path, rows, classes, summary, warnings):
    data = [{
        "no": r["no"], "name": r["name"], "class": r["class"], "gender": r["gender"],
        "rarity": r["rarity"], "pitch": r["pitch"], "pace": r["pace"],
        "signature": r["signature"], "description": r["description"],
        "texture": r.get("texture", ""), "bundle": feature_bundle(r),
    } for r in rows]
    classes_list = list(classes.keys())
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Gacha Voice Presets</title>
<style>
:root{--bg:#0f1115;--fg:#e8e6e3;--dim:#9a97a0;--acc:#ffb454;--base:#5cc8ff;--card:#181b22;--line:#262a33}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;padding:18px}
h1{font-size:20px;margin:0 0 4px}h1 b{color:var(--acc)}
.meta{color:var(--dim);margin-bottom:14px}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.controls input,.controls select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px}
.controls input{flex:1;min-width:220px}
.row{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer}
.row:hover{border-color:var(--acc)}
.row .top{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.row .no{color:var(--dim);min-width:44px}
.row .name{font-weight:600}.row .name.base{color:var(--base)}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:20px;padding:1px 8px}
.tag.r5{color:var(--acc);border-color:var(--acc)}
.row .desc{color:var(--dim);font-style:italic;margin-top:4px}
.row .bundle{display:none;margin-top:6px;color:#bfe3ff;font-size:12px}
.row.open .bundle{display:block}
.warn{color:#ffd08a;background:#2a2113;border:1px solid #55401e;border-radius:10px;padding:10px 12px;margin-bottom:14px;font-size:13px}
.warn b{color:#ff9d5c}
.hidden{display:none}
</style></head><body>
<h1><b>1000</b> Gacha Voice Presets — full catalog <span style="color:var(--dim);font-weight:400">(generated from presets.csv)</span></h1>
<div class="meta">click a row for its 58-feature bundle · order any of them in chat</div>
<div class="warn hidden" id="warn"></div>
<div class="controls">
<input id="q" type="search" placeholder="search name / class / description…">
<select id="cls"><option value="">all classes</option></select>
<select id="pit"><option value="">all pitches</option></select>
<select id="pac"><option value="">all paces</option></select>
<select id="rar"><option value="">all rarities</option></select>
<select id="sig"><option value="">base + variants</option><option value="yes">bases only</option><option value="no">variants only</option></select>
</div>
<div id="list"></div>
<script>
const DATA = __DATA__;
const CLASSES = __CLASSES__;
const WARN = __WARN__;
const $ = id => document.getElementById(id);
function fill(sel, values){ for (const v of values) { const o = document.createElement("option"); o.value = v; o.textContent = v; $(sel).appendChild(o); } }
fill("cls", CLASSES);
fill("pit", [...new Set(DATA.map(d => d.pitch))].sort());
fill("pac", [...new Set(DATA.map(d => d.pace))].sort());
fill("rar", [...new Set(DATA.map(d => d.rarity))].sort((a,b)=>b-a));
if (WARN.length){ $("warn").classList.remove("hidden"); $("warn").innerHTML = "<b>" + WARN.length + " validation warning(s)</b><br>" + WARN.join("<br>"); }
function render(){
  const q = $("q").value.trim().toLowerCase(), c = $("cls").value, p = $("pit").value, k = $("pac").value, r = $("rar").value, s = $("sig").value;
  $("list").innerHTML = "";
  let shown = 0;
  for (const d of DATA){
    if (c && d.class !== c) continue;
    if (p && d.pitch !== p) continue;
    if (k && d.pace !== k) continue;
    if (r && String(d.rarity) !== r) continue;
    if (s && d.signature !== s) continue;
    if (q && !(d.name + " " + d.class + " " + d.description + " " + d.texture).toLowerCase().includes(q)) continue;
    shown++;
    const el = document.createElement("div");
    el.className = "row";
    const tex = d.texture ? d.texture + " · " : "";
    el.innerHTML = `<div class="top"><span class="no">#${d.no}</span><span class="name ${d.signature==="yes"?"base":""}">${d.name}</span>` +
      `<span class="tag r${d.rarity}">★${d.rarity}</span><span class="tag">${d.pitch}</span>` +
      (tex ? `<span class="tag">${d.texture}</span>` : "") +
      `<span class="tag">${d.pace}</span><span class="tag">${d.gender}</span><span class="tag">${d.class}</span></div>` +
      `<div class="desc">“${d.description}”</div><div class="bundle">${d.bundle}</div>`;
    el.onclick = () => el.classList.toggle("open");
    $("list").appendChild(el);
  }
  $("list").insertAdjacentHTML("beforeend", shown === 0 ? '<div class="meta">no rows match</div>' : "");
}
["q","cls","pit","pac","rar","sig"].forEach(id => $(id).addEventListener("input", render));
render();
</script></body></html>"""
    page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__CLASSES__", json.dumps(classes_list, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__WARN__", json.dumps(warnings, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("<b>1000</b> Gacha Voice Presets",
                        f"<b>{summary['rows']}</b> Gacha Voice Presets")
    path.write_text(page, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate presets.csv and generate PRESET-CATALOG.md/.html")
    ap.add_argument("--csv", default="presets.csv", help="input CSV (default: presets.csv in cwd)")
    ap.add_argument("--no-html", action="store_true", help="skip the HTML catalog")
    args = ap.parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found — drop the CSV at the repo root, or pass --csv PATH",
              file=sys.stderr)
        return 2
    rows, errors, warnings = parse_csv(csv_path)
    v_errors, v_warnings = validate(rows)
    errors += v_errors
    warnings += v_warnings
    if errors:
        print(f"presets_ingest: {len(errors)} error(s) in {csv_path.name} — catalog NOT generated:", file=sys.stderr)
        for e in errors:
            print(f"  error: {e}", file=sys.stderr)
        return 1
    summary, classes = summarize(rows)
    out_md = csv_path.parent / "PRESET-CATALOG.md"
    write_md(out_md, rows, classes, summary, warnings)
    out_html = None
    if not args.no_html:
        out_html = csv_path.parent / "PRESET-CATALOG.html"
        write_html(out_html, rows, classes, summary, warnings)
    print(f"presets_ingest: OK — {summary['rows']} lines "
          f"({summary['bases']} bases + {summary['variants']} variants) "
          f"in {summary['classes']} classes; {len(warnings)} warning(s)")
    for w in warnings:
        print(f"  warning: {w}")
    print(f"  wrote {out_md}")
    if out_html:
        print(f"  wrote {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
