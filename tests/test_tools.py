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


if __name__ == "__main__":
    unittest.main()
