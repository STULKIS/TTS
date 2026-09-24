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
PRESETS = load_tool("presets_ingest", "presets_ingest.py")
PRESETS_AUDIO = load_tool("audio_fx", "audio_fx.py")


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

    def test_manifest_parses_optional_control_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.tsv"
            path.write_text(
                "a1\tRin\ten\tHello there\n"
                "a2\tRin\ten\tLouder higher\t3\t1.1\t-2\n"
                "a3\tRin\ten\tDefaults\n",
                encoding="utf-8",
            )
            rows = RENDER_BATCH.read_manifest(path)
            self.assertEqual([r["id"] for r in rows], ["a1", "a2", "a3"])
            self.assertEqual(rows[0]["pitch"], None)
            self.assertEqual(rows[0]["speed"], None)
            self.assertEqual(rows[0]["volume"], None)
            self.assertEqual(rows[1]["pitch"], 3.0)
            self.assertEqual(rows[1]["speed"], 1.1)
            self.assertEqual(rows[1]["volume"], -2.0)

    def test_manifest_rejects_bad_field_count_and_bad_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.tsv"
            path.write_text("a1\tRin\ten\tHi\t1\t1\t2\t9\textra\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                RENDER_BATCH.read_manifest(path)
            path.write_text("a1\tRin\ten\tHi\tnotanumber\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                RENDER_BATCH.read_manifest(path)

    def test_manifest_parses_fx_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.tsv"
            path.write_text(
                "a1\tRin\ten\tHi\t2\t1.0\t0\trobot,reverb\n"
                "a2\tRin\ten\tPlain\n",
                encoding="utf-8",
            )
            rows = RENDER_BATCH.read_manifest(path)
            self.assertEqual(rows[0]["fx"], "robot,reverb")
            self.assertEqual(rows[1]["fx"], None)


try:
    import numpy as _NP  # noqa: F401
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class AudioFxTests(unittest.TestCase):
    def _sine(self, path: Path, sr: int = 24_000, seconds: float = 2.0, hz: float = 440.0):
        import numpy as np
        t = np.arange(int(sr * seconds)) / sr
        x = 0.5 * np.sin(2 * np.pi * hz * t)
        PRESETS_AUDIO.save_wav(path, sr, x)
        return x

    @staticmethod
    def _peak_hz(path: Path, sr: int) -> float:
        import numpy as np
        _, a = PRESETS_AUDIO.load_wav(path)
        spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
        freqs = np.fft.rfftfreq(len(a), 1 / sr)
        return float(freqs[int(np.argmax(spec))])

    def test_pitch_shift_octave_up_preserves_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            x = self._sine(src)
            out = Path(directory) / "out.wav"
            PRESETS_AUDIO.process(src, out, pitch=12)
            _, hi = PRESETS_AUDIO.load_wav(out)
            self.assertEqual(len(hi), len(x))
            self.assertAlmostEqual(self._peak_hz(out, 24_000), 880, delta=10)

    def test_pitch_shift_octave_down(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            self._sine(src)
            out = Path(directory) / "out.wav"
            PRESETS_AUDIO.process(src, out, pitch=-12)
            self.assertAlmostEqual(self._peak_hz(out, 24_000), 220, delta=10)

    def test_speed_halves_duration(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            x = self._sine(src)
            out = Path(directory) / "out.wav"
            PRESETS_AUDIO.process(src, out, speed=2.0)
            _, fast = PRESETS_AUDIO.load_wav(out)
            self.assertEqual(len(fast), len(x) // 2)

    def test_volume_gain_and_soft_clip(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            self._sine(src)
            out = Path(directory) / "out.wav"
            PRESETS_AUDIO.process(src, out, volume_db=6)
            _, vol = PRESETS_AUDIO.load_wav(out)
            self.assertAlmostEqual(float(np.max(np.abs(vol))), 0.5 * 10 ** (6 / 20), places=3)
            # loud input + gain must stay in range (soft clip, not distortion)
            loud_src = Path(directory) / "loud.wav"
            _, x = PRESETS_AUDIO.load_wav(src)
            PRESETS_AUDIO.save_wav(loud_src, 24_000, x * 1.9)
            loud_out = Path(directory) / "loud.wav.out"
            PRESETS_AUDIO.process(loud_src, loud_out, volume_db=10)
            _, clipped = PRESETS_AUDIO.load_wav(loud_out)
            self.assertLessEqual(float(np.max(np.abs(clipped))), 1.0)

    def test_parse_fx_tokens(self):
        self.assertEqual(PRESETS_AUDIO.parse_fx("robot, reverb"), ["robot", "reverb"])
        self.assertEqual(PRESETS_AUDIO.parse_fx("ROBOT,Robot"), ["robot"])
        self.assertEqual(PRESETS_AUDIO.parse_fx("  "), [])
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertEqual(PRESETS_AUDIO.parse_fx("robot,warp"), ["robot"])
        self.assertIn("unknown fx token 'warp'", buf.getvalue())

    def test_robot_spectrum_is_ring_modulated(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            self._sine(src, hz=440.0)
            _, x = PRESETS_AUDIO.load_wav(src)
            y = PRESETS_AUDIO.robot(x, 24_000)
            spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            freqs = np.fft.rfftfreq(len(y), 1 / 24_000)
            peaks = set()
            for i in np.argsort(spec)[::-1][:2]:
                if freqs[i] > 100:
                    peaks.add(int(round(freqs[i] / 5) * 5))
            self.assertEqual(peaks, {385, 495})  # 440 +/- 55 Hz ring-mod

    def test_phone_bandpass(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            self._sine(src, hz=440.0)
            _, mid = PRESETS_AUDIO.load_wav(src)
            y_mid = PRESETS_AUDIO.phone(mid, 24_000)
            self.assertGreater(np.sqrt(np.mean(y_mid ** 2)) / np.sqrt(np.mean(mid ** 2)), 0.9)
            low = 0.5 * np.sin(2 * np.pi * 100 * np.arange(len(mid)) / 24_000)
            y_low = PRESETS_AUDIO.phone(low, 24_000)
            self.assertLess(np.sqrt(np.mean(y_low ** 2)), 0.01)

    def test_reverb_impulse_has_decaying_echos(self):
        import numpy as np
        x = np.zeros(24_000)
        x[0] = 1.0
        y = PRESETS_AUDIO.reverb(x, 24_000)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0 + 1e-9)
        for delay_ms, floor in ((25, 0.15), (41, 0.10), (59, 0.08)):
            self.assertGreater(float(y[int(delay_ms * 24_000 / 1000)]), floor)

    def test_normalize_rms(self):
        import numpy as np
        rng = np.random.default_rng(1)
        x = rng.normal(size=24_000)
        y = PRESETS_AUDIO.normalize(x)
        rms_db = 20 * np.log10(float(np.sqrt(np.mean(y ** 2))))
        self.assertAlmostEqual(rms_db, -20.0, delta=0.5)

    def test_process_with_fx_token_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.wav"
            self._sine(src)
            out = Path(directory) / "out.wav"
            rep = PRESETS_AUDIO.process(src, out, fx="phone,normalize")
            self.assertEqual(rep["applied"], ["fx phone", "fx normalize"])


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


# A minimal valid catalog: one class, two bases, mixed rarity, an entity-escaped class name.
PRESETS_GOOD = (
    "no,name,class,gender,rarity,pitch,pace,signature,description\n"
    '1,Genki Spark,Genki &amp; Sunshine,Feminine,5,High,Dash,yes,"A little comet that laughs first."\n'
    '2,Genki Spark · Honey Lilt,Genki &amp; Sunshine,Feminine,4,High,Lilt,no,"Softer sugar, still fast."\n'
    '3,Genki Spark · Gravel Tale,Genki &amp; Sunshine,Feminine,3,High,Tale,no,"Scrappy storyteller energy."\n'
    '4,Snowglass,Kuudere Ice,Feminine,5,Soft,Pulse,yes,"Quiet, clear, perfectly still."\n'
    '5,Snowglass · Bell Burst,Kuudere Ice,Feminine,4,High,Burst,no,"Crisp ice chime."\n'
)


class PresetsIngestTests(unittest.TestCase):
    def _write(self, content: str) -> Path:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            fh.write(content)
            return Path(fh.name)

    def _run(self, content: str):
        import tempfile
        csv_path = self._write(content)
        tmp = str(csv_path)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = PRESETS.main(["--csv", tmp, "--no-html"])
            return rc, buf.getvalue(), csv_path
        finally:
            for extra in ("PRESET-CATALOG.md", "PRESET-CATALOG.html"):
                p = csv_path.with_name(extra)
                if p.exists():
                    p.unlink()
            csv_path.unlink(missing_ok=True)

    def test_wellformed_catalog_ok(self):
        rc, out, _ = self._run(PRESETS_GOOD)
        self.assertEqual(rc, 0)
        self.assertIn("OK", out)
        self.assertIn("5 lines", out)
        # entity-escaped class name is unescaped in the report
        self.assertNotIn("&amp;", out)

    def test_html_generated_by_default(self):
        import tempfile
        csv_path = self._write(PRESETS_GOOD)
        tmp = str(csv_path)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = PRESETS.main(["--csv", tmp])
            self.assertEqual(rc, 0)
            html_path = csv_path.with_name("PRESET-CATALOG.html")
            self.assertTrue(html_path.exists())
            text = html_path.read_text(encoding="utf-8")
            self.assertIn("Genki & Sunshine", text)
            self.assertIn("Genki Spark", text)
            # md also written
            self.assertTrue(csv_path.with_name("PRESET-CATALOG.md").exists())
        finally:
            for extra in ("PRESET-CATALOG.md", "PRESET-CATALOG.html"):
                p = csv_path.with_name(extra)
                if p.exists():
                    p.unlink()
            csv_path.unlink(missing_ok=True)

    def test_gap_in_numbering_is_error(self):
        # row 2 missing -> expected 2, got 3
        bad = PRESETS_GOOD.replace("\n2,Genki Spark · Honey Lilt,", "\n")
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 1)

    def test_empty_field_is_error(self):
        bad = PRESETS_GOOD.replace('"Softer sugar, still fast."', '""')
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 1)

    def test_unknown_gender_is_error(self):
        bad = PRESETS_GOOD.replace("Feminine", "Alien", 1)
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 1)

    def test_variant_of_unknown_base_is_error(self):
        bad = (
            "no,name,class,gender,rarity,pitch,pace,signature,description\n"
            '1,Genki Spark,Genki &amp; Sunshine,Feminine,5,High,Dash,yes,"A comet."\n'
            '2,Other Base · Honey Lilt,Genki &amp; Sunshine,Feminine,4,High,Lilt,no,"Orphan variant."\n'
        )
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 1)

    def test_duplicate_name_is_error(self):
        bad = PRESETS_GOOD.replace(
            '4,Snowglass,', '4,Genki Spark,'
        ).replace('5,Snowglass · Bell Burst,', '5,Genki Spark · Bell Burst,')
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 1)

    def test_out_of_expected_pitch_warns_but_passes(self):
        # "Wet" is not in the expected pitch set -> warning, still exit 0
        bad = PRESETS_GOOD.replace("1,Genki Spark,", "1,Genki Spark,", 1).replace(
            "Feminine,5,High,Dash", "Feminine,5,Wet,Dash", 1)
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 0)
        self.assertIn("warning", out.lower())

    def test_truncated_block_warns(self):
        # Snowglass has 1 variant vs mode of 2 -> truncation warning
        bad = (
            "no,name,class,gender,rarity,pitch,pace,signature,description\n"
            '1,Genki Spark,Genki &amp; Sunshine,Feminine,5,High,Dash,yes,"A comet."\n'
            '2,Genki Spark · Honey Lilt,Genki &amp; Sunshine,Feminine,4,High,Lilt,no,"Sugar."\n'
            '3,Genki Spark · Gravel Tale,Genki &amp; Sunshine,Feminine,3,High,Tale,no,"Scrappy."\n'
            '4,Snowglass,Kuudere Ice,Feminine,5,Soft,Pulse,yes,"Still."\n'
            '5,Snowglass · Bell Burst,Kuudere Ice,Feminine,4,High,Burst,no,"Chime."\n'
        )
        rc, out, _ = self._run(bad)
        self.assertEqual(rc, 0)
        self.assertIn("truncated", out.lower())

    def test_missing_file_returns_2(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = PRESETS.main(["--csv", "/nonexistent/presets.csv"])
        self.assertEqual(rc, 2)

    def test_roll_is_deterministic_with_seed(self):
        csv_path = self._write(PRESETS_GOOD)
        tmp = str(csv_path)
        outs = []
        try:
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    rc = PRESETS.main(["--csv", tmp, "--roll", "2", "--seed", "7"])
                self.assertEqual(rc, 0)
                outs.append(buf.getvalue())
            self.assertEqual(outs[0], outs[1])
            self.assertIn("roll: 2 voice(s) from 5", outs[0])
            # no catalog written by a roll
            self.assertFalse(csv_path.with_name("PRESET-CATALOG.md").exists())
        finally:
            csv_path.unlink(missing_ok=True)

    def test_roll_respects_rarity_filter(self):
        csv_path = self._write(PRESETS_GOOD)
        tmp = str(csv_path)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = PRESETS.main(["--csv", tmp, "--roll", "5", "--rarity", "5", "--seed", "1"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("from 2 matching line(s)", out)  # only the two bases
            self.assertIn("Genki Spark", out)
            self.assertIn("Snowglass", out)
            self.assertNotIn("Honey Lilt", out)
        finally:
            csv_path.unlink(missing_ok=True)

    def test_find_searches_description(self):
        csv_path = self._write(PRESETS_GOOD)
        tmp = str(csv_path)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = PRESETS.main(["--csv", tmp, "--find", "sugar"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("1 match(es)", out)
            self.assertIn("Honey Lilt", out)
        finally:
            csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
