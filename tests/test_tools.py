from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, filename: str):
    """Load a standalone script from tools/ without making tools a package."""
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib failure
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK_SEEDS = load_tool("check_seeds", "check_seeds.py")
JA_KATAKANA = load_tool("ja_katakana", "ja_katakana.py")
GSV = load_tool("render_batch_gsv", "render_batch_gsv.py")
# Loading render_batch proves it stays importable without torch/cosyvoice installed
# (heavy deps are lazy — see --dry-run).
SHOWCASE = load_tool("make_showcase", "make_showcase.py")
DIALOGUE = load_tool("dialogue2tsv", "dialogue2tsv.py")
RENDER_BATCH = load_tool("render_batch", "render_batch.py")
CPU_CONFIG = load_tool("cpu_config", "cpu_config.py")
PREFLIGHT = load_tool("preflight", "preflight.py")


def write_wav(path: Path, *, rate: int = 24_000, seconds: float = 4,
              channels: int = 1) -> None:
    """Write a small, valid PCM WAV suitable for seed validation tests."""
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * frames * channels)


class CheckSeedsTests(unittest.TestCase):
    def test_valid_clip_has_no_errors_or_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "en.wav"
            transcript = root / "en.txt"
            write_wav(wav)
            transcript.write_text("A valid transcript.\n", encoding="utf-8")

            errors: list[str] = []
            warnings: list[str] = []
            CHECK_SEEDS.check_clip(wav, transcript, errors, warnings)

            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])

    def test_clip_rules_report_duration_rate_channels_and_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "short.wav"
            transcript = root / "short.txt"
            write_wav(wav, rate=8_000, seconds=1, channels=2)
            transcript.write_text("", encoding="utf-8")

            errors: list[str] = []
            warnings: list[str] = []
            CHECK_SEEDS.check_clip(wav, transcript, errors, warnings)

            self.assertTrue(any("< 16 kHz" in error for error in errors))
            self.assertTrue(any("transcript is empty" in error for error in errors))
            self.assertTrue(any("under ~2s" in warning for warning in warnings))
            self.assertTrue(any("channels" in warning for warning in warnings))

    def test_overlong_clip_and_missing_transcript_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "long.wav"
            write_wav(wav, seconds=31)

            errors: list[str] = []
            warnings: list[str] = []
            CHECK_SEEDS.check_clip(wav, None, errors, warnings)

            self.assertTrue(any("HARD LIMIT" in error for error in errors))
            self.assertTrue(any("transcript missing" in error for error in errors))

    def test_corrupt_wav_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "broken.wav"
            wav.write_bytes(b"not a wave file")
            errors: list[str] = []
            warnings: list[str] = []

            CHECK_SEEDS.check_clip(wav, wav.with_suffix(".txt"), errors, warnings)

            self.assertEqual(warnings, [])
            self.assertTrue(any("not a readable PCM WAV" in error for error in errors))

    def test_manifest_language_index_skips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "lines.tsv"
            manifest.write_text(
                "# comment\n\n"
                "one\tRin\ten\tHello\n"
                "two\tRin\tja\tこんにちは\n"
                "three\tMio\tzh\t你好\n",
                encoding="utf-8",
            )

            self.assertEqual(
                CHECK_SEEDS.read_manifest_langs(manifest),
                {"Rin": {"en", "ja"}, "Mio": {"zh"}},
            )

    def test_manifest_rejects_wrong_field_count_and_unknown_language(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "lines.tsv"
            manifest.write_text("bad\trow\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "expected 4 tab-separated fields"):
                CHECK_SEEDS.read_manifest_langs(manifest)

            manifest.write_text("one\tRin\tfr\tBonjour\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not in"):
                CHECK_SEEDS.read_manifest_langs(manifest)

    def test_main_cross_checks_manifest_assets_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = root / "seeds" / "Rin"
            seeds.mkdir(parents=True)
            write_wav(seeds / "en.wav")
            (seeds / "en.txt").write_text("Hello", encoding="utf-8")
            manifest = root / "lines.tsv"
            manifest.write_text(
                "one\tRin\ten\tHello\n"
                "two\tRin\tja\tこんにちは\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with patch.object(
                sys,
                "argv",
                ["check_seeds.py", "--seeds", str(root / "seeds"),
                 "--manifest", str(manifest)],
            ), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    CHECK_SEEDS.main()

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("manifest needs", output.getvalue())
            self.assertIn("checked 1 clip(s)", output.getvalue())


class FakeKakasi:
    def __init__(self, readings: dict[str, list[str]]):
        self.readings = readings

    def convert(self, text: str):
        return [{"kana": token} for token in self.readings[text]]


class JaKatakanaTests(unittest.TestCase):
    def test_to_spaced_katakana_skips_whitespace_and_glues_punctuation(self):
        kakasi = FakeKakasi({
            "ignored": ["コノ", " ", "セカイ", "。", "!"],
        })

        self.assertEqual(
            JA_KATAKANA.to_spaced_katakana("ignored", kakasi),
            "コノ セカイ。!",
        )

    def test_manifest_mode_only_converts_ja_rows(self):
        readings = {
            "この世界。": ["コノ", "セカイ", "。"],
        }
        fake_pykakasi = types.SimpleNamespace(kakasi=lambda: FakeKakasi(readings))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "lines.tsv"
            output = root / "converted.tsv"
            manifest.write_text(
                "# keep this comment\n"
                "one\tRin\tja\tこの世界。\n"
                "two\tRin\ten\tHello\n"
                "\n",
                encoding="utf-8",
            )

            with patch.dict(sys.modules, {"pykakasi": fake_pykakasi}):
                with patch.object(
                    sys,
                    "argv",
                    ["ja_katakana.py", "--manifest", str(manifest),
                     "--output", str(output)],
                ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    JA_KATAKANA.main()

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "# keep this comment\n"
                "one\tRin\tja\tコノ セカイ。\n"
                "two\tRin\ten\tHello\n"
                "\n",
            )
            self.assertIn("converted 1 ja line(s)", stderr.getvalue())

    def test_manifest_rejects_wrong_field_count(self):
        fake_pykakasi = types.SimpleNamespace(kakasi=lambda: FakeKakasi({}))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bad.tsv"
            manifest.write_text("not\ta\tmanifest\n", encoding="utf-8")
            with patch.dict(sys.modules, {"pykakasi": fake_pykakasi}):
                with patch.object(sys, "argv", ["ja_katakana.py", "--manifest", str(manifest)]):
                    with self.assertRaisesRegex(SystemExit, "expected 4 tab-separated fields"):
                        JA_KATAKANA.main()


class GsvToolTests(unittest.TestCase):
    def test_manifest_reader_handles_comments_and_rejects_bad_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "lines.tsv"
            manifest.write_text(
                "# comment\n"
                "one\tRin\ten\tHello\n"
                "two\tRin\tja\tこんにちは\n",
                encoding="utf-8",
            )
            self.assertEqual(
                GSV.read_manifest(manifest),
                [
                    {"id": "one", "char": "Rin", "lang": "en", "text": "Hello"},
                    {"id": "two", "char": "Rin", "lang": "ja", "text": "こんにちは"},
                ],
            )

            manifest.write_text("bad\tRin\tfr\tBonjour\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not in"):
                GSV.read_manifest(manifest)

    def test_find_model_accepts_supported_sovits_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            character = models / "Rin"
            character.mkdir()
            (character / "sovits.pth").touch()

            self.assertEqual(
                GSV.find_model(models, "Rin", ("sovits.safetensors", "sovits.pth")),
                character / "sovits.pth",
            )
            self.assertIsNone(GSV.find_model(models, "Mio", ("gpt.ckpt",)))


class MakeShowcaseTests(unittest.TestCase):
    def _pack(self, seeds: Path) -> None:
        rin = seeds / "Rin"
        rin.mkdir(parents=True)
        write_wav(rin / "en.wav")
        (rin / "en.txt").write_text("Hello world", encoding="utf-8")
        write_wav(rin / "en2.wav", seconds=5)
        (rin / "en2.txt").write_text("Second line", encoding="utf-8")
        write_wav(rin / "ja.wav")
        (rin / "ja.txt").write_text("こんにちは", encoding="utf-8")
        mio = seeds / "Mio"
        mio.mkdir()
        write_wav(mio / "ko.wav")
        (mio / "ko.txt").write_text("안녕", encoding="utf-8")

    def test_page_has_players_transcripts_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = root / "seeds"
            self._pack(seeds)
            out = root / "showcase.html"
            with patch.object(
                sys, "argv", ["make_showcase.py", "--seeds", str(seeds),
                              "--out", str(out)]
            ), contextlib.redirect_stdout(io.StringIO()):
                SHOWCASE.main()

            page = out.read_text(encoding="utf-8")
            for src in ("Rin/en.wav", "Rin/en2.wav", "Rin/ja.wav", "Mio/ko.wav"):
                self.assertIn(f'src="{src}"', page)
            self.assertIn("Hello world", page)
            self.assertIn("こんにちは", page)
            self.assertIn("<h2>Rin</h2>", page)
            self.assertIn("<h2>Mio</h2>", page)
            # clip 1 before clip 2; language blocks in canonical order
            self.assertLess(page.index('src="Rin/en.wav"'), page.index('src="Rin/en2.wav"'))
            self.assertLess(page.index('src="Rin/en.wav"'), page.index('src="Rin/ja.wav"'))
            self.assertLess(page.index("<h3>en</h3>"), page.index("<h3>ja</h3>"))

    def test_no_readable_clips_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            seeds = Path(directory) / "seeds"
            (seeds / "Rin").mkdir(parents=True)
            (seeds / "Rin" / "en.wav").write_bytes(b"junk")
            (seeds / "Rin" / "en.txt").write_text("x", encoding="utf-8")
            with patch.object(sys, "argv", ["make_showcase.py", "--seeds", str(seeds)]), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaisesRegex(SystemExit, "no readable"):
                    SHOWCASE.main()
            self.assertIn("unreadable", err.getvalue())

    def test_escapes_transcript_markup(self):
        with tempfile.TemporaryDirectory() as directory:
            seeds = Path(directory) / "seeds"
            (seeds / "Rin").mkdir(parents=True)
            write_wav(seeds / "Rin" / "en.wav")
            (seeds / "Rin" / "en.txt").write_text("<script>alert(1)</script>", encoding="utf-8")
            out = Path(directory) / "s.html"
            with patch.object(
                sys, "argv", ["make_showcase.py", "--seeds", str(seeds),
                              "--out", str(out)]
            ), contextlib.redirect_stdout(io.StringIO()):
                SHOWCASE.main()
            page = out.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", page)
            self.assertIn("&lt;script&gt;", page)


class Dialogue2TsvTests(unittest.TestCase):
    def test_infer_lang_by_unicode_ranges(self):
        self.assertEqual(DIALOGUE.infer_lang("The fate of this world."), "en")
        self.assertEqual(DIALOGUE.infer_lang("这个世界的命运。"), "zh")
        self.assertEqual(DIALOGUE.infer_lang("この世界の運命は。"), "ja")  # kana beats kanji
        self.assertEqual(DIALOGUE.infer_lang("세상의 운명은 아직."), "ko")

    def test_script_to_tsv_with_ids_override_and_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "scene.txt"
            script.write_text(
                "# a scene\n"
                "\n"
                "dragon: The fate of this world was never already written.\n"
                "drake: You kept me waiting.\n"
                "dragon: 这个世界的命运，还没有定数。\n"
                "drake [ja]: 待たせたな。君の最初の失敗だ。\n"
                "dragon：세상의 운명은 아직 정해지지 않았다.\n",
                encoding="utf-8",
            )
            out = Path(directory) / "lines.tsv"
            with patch.object(
                sys, "argv", ["dialogue2tsv.py", "--script", str(script),
                              "--output", str(out)]
            ), contextlib.redirect_stderr(io.StringIO()):
                DIALOGUE.main()
            self.assertEqual(
                out.read_text(encoding="utf-8"),
                "dragon-01\tdragon\ten\tThe fate of this world was never already written.\n"
                "drake-01\tdrake\ten\tYou kept me waiting.\n"
                "dragon-02\tdragon\tzh\t这个世界的命运，还没有定数。\n"
                "drake-02\tdrake\tja\t待たせたな。君の最初の失敗だ。\n"
                "dragon-03\tdragon\tko\t세상의 운명은 아직 정해지지 않았다.\n",
            )

    def test_rejects_unknown_bracket_lang_and_malformed_line(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "bad.txt"
            script.write_text("dragon [fr]: Bonjour\n", encoding="utf-8")
            with patch.object(sys, "argv", ["dialogue2tsv.py", "--script", str(script)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "not in"):
                    DIALOGUE.main()
            script.write_text("no colon here\n", encoding="utf-8")
            with patch.object(sys, "argv", ["dialogue2tsv.py", "--script", str(script)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "expected"):
                    DIALOGUE.main()


class RenderBatchDryRunTests(unittest.TestCase):
    def test_plan_reports_speakers_missing_and_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = root / "seeds" / "Rin"
            seeds.mkdir(parents=True)
            write_wav(seeds / "en.wav")
            (seeds / "en.txt").write_text("Hi", encoding="utf-8")
            rows = [
                {"id": "one", "char": "Rin", "lang": "en", "text": "Hello"},
                {"id": "two", "char": "Rin", "lang": "en", "text": "Again"},
                {"id": "three", "char": "Rin", "lang": "zh", "text": "你好"},
            ]
            plan = RENDER_BATCH.dry_run_plan(rows, root / "seeds", "")
            self.assertEqual(plan["rows"], 3)
            self.assertEqual(
                [f"{s['char']}__{s['lang']}" for s in plan["speakers"]],
                ["Rin__en"],
            )
            self.assertEqual(plan["speakers"][0]["lines"], 2)
            self.assertEqual([r["id"] for r in plan["missing"]], ["three"])
            self.assertIn("zh.wav", plan["missing"][0]["missing"])
            self.assertEqual(plan["est_low_s"], 3 * 5 * 2)
            self.assertEqual(plan["est_high_s"], 3 * 5 * 5)

    def test_module_has_no_module_level_heavy_deps(self):
        # load_tool() at import time already proves render_batch.py loads without
        # torch/cosyvoice installed; keep the explicit assertion too.
        heavy = ("torch", "torchaudio", "AutoModel", "set_all_random_seed")
        self.assertFalse(any(name in vars(RENDER_BATCH) for name in heavy))


class CpuConfigTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        configs = root / "GPT_SoVITS" / "configs"
        configs.mkdir(parents=True)
        cfg = configs / "tts_infer.yaml"
        cfg.write_text(
            "v2:\n"
            "  device: cuda:0\n"
            "  is_half: true\n"
            "  t2s_weights_path: x.ckpt\n",
            encoding="utf-8",
        )
        return cfg

    def test_rewrites_device_and_half_and_backs_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self._write_config(root)
            with patch.object(sys, "argv", ["cpu_config.py", "--gsv-root", str(root)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                CPU_CONFIG.main()
            text = cfg.read_text(encoding="utf-8")
            self.assertIn("device: cpu", text)
            self.assertIn("is_half: false", text)
            self.assertIn("t2s_weights_path: x.ckpt", text)
            self.assertIn("cuda:0", cfg.with_suffix(".yaml.orig").read_text(encoding="utf-8"))

    def test_idempotent_and_keeps_first_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self._write_config(root)
            argv = ["cpu_config.py", "--gsv-root", str(root)]
            for _ in range(2):
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    CPU_CONFIG.main()
            text = cfg.read_text(encoding="utf-8")
            self.assertEqual(text.count("device: cpu"), 1)
            self.assertIn("is_half: false", text)
            self.assertIn("cuda:0", cfg.with_suffix(".yaml.orig").read_text(encoding="utf-8"))

    def test_missing_config_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["cpu_config.py", "--gsv-root", directory]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "not found"):
                    CPU_CONFIG.main()


class PreflightTests(unittest.TestCase):
    def _make_gsv(self, root: Path, *, weights: bool = True, cpu: bool = True) -> None:
        pkg = root / "GPT_SoVITS"
        (pkg / "configs").mkdir(parents=True)
        dev, half = ("cuda:0", "true") if not cpu else ("cpu", "false")
        (pkg / "configs" / "tts_infer.yaml").write_text(
            f"v2:\n  device: {dev}\n  is_half: {half}\n", encoding="utf-8")
        pm = pkg / "pretrained_models"
        if weights:
            (pm / "gsv-v2final-pretrained").mkdir(parents=True)
            for f in ("gpt.pth", "s2G2333.pth", "s2D2333.pth"):
                (pm / "gsv-v2final-pretrained" / f).touch()
            (pm / "chinese-hubert-base").mkdir(parents=True)
            (pm / "chinese-roberta-wwm-ext-large").mkdir(parents=True)

    def _make_seeds(self, root: Path) -> None:
        seeds = root / "seeds" / "Rin"
        seeds.mkdir(parents=True)
        write_wav(seeds / "en.wav")
        (seeds / "en.txt").write_text("Hi", encoding="utf-8")

    def test_ready_when_all_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_gsv(root)
            self._make_seeds(root)
            output = io.StringIO()
            with patch.object(sys, "argv", ["preflight.py", "--gsv-root", str(root),
                                            "--seeds", str(root / "seeds")]), \
                    contextlib.redirect_stdout(output):
                PREFLIGHT.main()
            out = output.getvalue()
            self.assertIn("[ready] all checks passed", out)
            self.assertNotIn("[fail]", out)

    def test_cuda_config_and_missing_weights_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_gsv(root, weights=False, cpu=False)
            self._make_seeds(root)
            with patch.object(sys, "argv", ["preflight.py", "--gsv-root", str(root),
                                            "--seeds", str(root / "seeds")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    PREFLIGHT.main()
            self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
