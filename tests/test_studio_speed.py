"""Speed-path correctness with a counted fake engine (NOT a voice benchmark)."""
import concurrent.futures
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from studio_cache import ReferenceCache, TakeCache

HAS_AUDIO = all(importlib.util.find_spec(m) for m in ("numpy", "soundfile"))
HAS_API = HAS_AUDIO and all(importlib.util.find_spec(m) for m in ("fastapi", "httpx"))


class TakeCacheTests(unittest.TestCase):
    def test_lru_byte_bound_replacement_and_oversize(self):
        c = TakeCache(max_bytes=10, max_entries=2)
        c.put("a", "one", 4)
        c.put("b", "two", 4)
        self.assertEqual(c.get("a"), "one")
        c.put("c", "three", 4)
        self.assertIsNone(c.get("b"))
        c.put("a", "replacement", 7)
        self.assertEqual(c.bytes, 7)
        self.assertIsNone(c.get("c"))
        c.put("a", "oversize", 11)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.bytes, 0)

    def test_entry_bound(self):
        c = TakeCache(max_entries=2)
        for i in range(5):
            c.put(i, i, 1)
        self.assertEqual(list(c.entries), [3, 4])


@unittest.skipUnless(HAS_AUDIO, "numpy/soundfile not installed")
class ReferenceCacheTests(unittest.TestCase):
    def test_stable_fit_invalidation_eviction_and_cleanup(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "input.wav"
            cache = ReferenceCache(max_entries=1)
            try:
                sf.write(source, np.zeros(16000), 16000)
                first = cache.fit(source)
                self.assertEqual(first, cache.fit(source))
                self.assertAlmostEqual(sf.info(first).duration, 3.2)
                sf.write(source, np.zeros(15 * 16000), 16000)
                second = cache.fit(source)
                self.assertNotEqual(first, second)
                self.assertFalse(Path(first).exists())
                self.assertAlmostEqual(sf.info(second).duration, 9.5)
                sf.write(source, np.zeros(5 * 16000), 16000)
                self.assertEqual(cache.fit(source), str(source.resolve()))
            finally:
                cache.close()
            self.assertFalse(Path(second).exists())


@unittest.skipUnless(HAS_API, "API test dependencies not installed")
class StudioSpeedTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import soundfile as sf
        from fastapi.testclient import TestClient
        from type_ui import build_app
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "ref.wav"
        sf.write(self.source, np.sin(np.arange(4 * 16000) * .07) * .2, 16000)
        self.payloads = []
        self.closed = 0
        self.active = 0
        self.max_active = 0
        self.fail = False
        self.wait_for_release = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.prompt_resets = []
        outer = self

        class Config:
            version = "TEST-DOUBLE"
            languages = ["en", "ja"]
            def __init__(self, path):
                pass

        class Engine:
            def __init__(self, config):
                self.prompt_cache = {}

            def run(self, payload):
                outer.payloads.append(dict(payload))
                outer.prompt_resets.append(dict(self.prompt_cache))
                self.prompt_cache.update(ref_audio_path=payload["ref_audio_path"],
                                         prompt_text=payload["prompt_text"])
                outer.active += 1
                outer.max_active = max(outer.max_active, outer.active)
                try:
                    outer.entered.set()
                    if outer.wait_for_release:
                        if not outer.release.wait(5):
                            raise RuntimeError("test release timed out")
                    if outer.fail:
                        raise RuntimeError("intentional test failure")
                    yield 16000, (np.sin(np.arange(16000) * .1) * .25).astype("float32")
                finally:
                    outer.active -= 1
                    outer.closed += 1

        module = types.ModuleType("GPT_SoVITS.TTS_infer_pack.TTS")
        module.TTS = Engine
        module.TTS_Config = Config
        with patch.dict(sys.modules, {module.__name__: module}):
            app = build_app(self.root, self.root / "seeds", self.root / "presets.csv", "test.yaml")
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.request = dict(ref_audio_path=str(self.source), prompt_text="Reference speech.",
                            prompt_lang="en", text="Hello there.", text_lang="en", seed=-1,
                            reuse_take=True)

    def post(self, **changes):
        return self.client.post("/api/tts", json=dict(self.request, **changes))

    def test_reuse_identical_and_reprocess_without_mutating_raw(self):
        first = self.post()
        second = self.post()
        fx = self.post(volume=-6, fx="normalize")
        third = self.post()
        self.assertEqual(first.status_code, 200, first.text[:200])
        self.assertEqual(first.headers["x-take-cache"], "miss")
        self.assertEqual(second.headers["x-take-cache"], "hit")
        self.assertEqual(second.headers["x-inference-seconds"], "0.000")
        self.assertEqual(first.content, second.content)
        self.assertEqual(first.content, third.content)
        self.assertNotEqual(first.content, fx.content)
        self.assertEqual(first.headers["x-life-score"], fx.headers["x-life-score"])
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(self.closed, 1)
        self.assertGreaterEqual(float(first.headers["x-total-seconds"]), 0)

    def test_fresh_and_api_default_never_reuse_random(self):
        self.post()
        self.post(reuse_take=False)
        request = dict(self.request)
        del request["reuse_take"]
        self.client.post("/api/tts", json=request)
        self.assertEqual(len(self.payloads), 3)

    def test_inference_controls_are_all_in_cache_key(self):
        self.post()
        changes = [dict(text="Different."), dict(prompt_text="Different ref."),
                   dict(prompt_lang="ja"), dict(text_lang="ja"), dict(seed=123),
                   dict(temperature=.8), dict(top_k=20), dict(top_p=.8),
                   dict(repetition_penalty=1.1), dict(speed_factor=1.1),
                   dict(fragment_interval=.5), dict(text_split_method="cut0"),
                   dict(batch_size=2)]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(self.post(**change).headers["x-take-cache"], "miss")
        self.assertEqual(len(self.payloads), 1 + len(changes))
        self.assertEqual(self.payloads[-1]["batch_size"], 2)
        self.assertTrue(self.payloads[-1]["parallel_infer"])

    def test_source_change_invalidates_take_and_engine_prompt(self):
        self.post()
        stamp = self.source.stat().st_mtime_ns + 1_000_000_000
        os.utime(self.source, ns=(stamp, stamp))
        self.assertEqual(self.post().headers["x-take-cache"], "miss")
        self.assertIsNone(self.prompt_resets[-1]["ref_audio_path"])
        self.assertIsNone(self.prompt_resets[-1]["prompt_text"])

    def test_reference_fit_reused_between_new_lines(self):
        import numpy as np
        import soundfile as sf
        sf.write(self.source, np.zeros(16000), 16000)
        self.post(text="One.")
        self.post(text="Two.")
        paths = [p["ref_audio_path"] for p in self.payloads]
        self.assertEqual(paths[0], paths[1])
        self.assertAlmostEqual(sf.info(paths[0]).duration, 3.2)

    def test_errors_not_cached_and_batch_limit(self):
        self.fail = True
        self.assertEqual(self.post().status_code, 500)
        self.fail = False
        self.assertEqual(self.post().headers["x-take-cache"], "miss")
        self.post(batch_size=999, reuse_take=False)
        self.assertEqual(self.payloads[-1]["batch_size"], 1)
        self.assertEqual(self.post(ref_audio_path="missing.wav").status_code, 400)

    def test_model_serialized_but_status_remains_responsive(self):
        self.wait_for_release = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(self.post, reuse_take=False)
            self.assertTrue(self.entered.wait(3))
            second = pool.submit(self.post, text="Second.", reuse_take=False)
            try:
                status = pool.submit(self.client.get, "/api/status").result(timeout=3)
                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json()["take_cache_mb"], 64)
            finally:
                self.release.set()
            self.assertEqual(first.result(timeout=3).status_code, 200)
            self.assertEqual(second.result(timeout=3).status_code, 200)
        self.assertEqual(self.max_active, 1)


class SpeedUIContractTests(unittest.TestCase):
    def test_batch_reuse_and_reroll_payloads(self):
        import re
        import shutil
        if not shutil.which("node"):
            self.skipTest("node unavailable")
        src = (ROOT / "tools" / "type_ui.py").read_text(encoding="utf-8")
        script = src.split('PAGE_HTML = r"""', 1)[1]
        batch = re.search(r'async function batchOne\(.*?\n}', script, re.S).group(0)
        timing = re.search(r'function renderTiming\(.*?\n}', script, re.S).group(0)
        controls = re.search(r'const CONTROL_FIELDS = .*?\n];', script, re.S).group(0)
        for name in ('controlValue', 'captureControls', 'captureRequest'):
            controls += re.search(r'function ' + name + r'\(.*?\n}', script, re.S).group(0)
        prelude = r'''
const assert = require('node:assert/strict');
const elements = {};
const $ = id => elements[id] ||= {value: 1, checked: false, type: 'number'};
const voiceState = () => ({ref:'dragon/en.wav', promptText:'Reference.', promptLang:'en'});
const fxCsv = () => '';
const sent = [];
const fetch = async (url, opts) => { sent.push(JSON.parse(opts.body)); return {
  ok:true, headers:{get: () => '1'}, blob: async () => 'wav'
}; };
'''
        assertions = r'''
(async () => {
  $('reuse').type = 'checkbox'; $('lottery').type = 'checkbox'; $('splitm').type = 'select-one';
  $('reuse').checked = true; $('seed').value = 12; $('enginebatch').value = 2;
  $('splitm').value = 'cut5';
  await batchOne('Hello');
  assert.equal(sent[0].reuse_take, true);
  assert.equal(sent[0].seed, 12);
  assert.equal(sent[0].batch_size, 2);
  assert.equal(sent[0].text_split_method, 'cut5');
  await batchOne('Hello', true);
  assert.equal(sent[1].reuse_take, false);
  assert.equal(sent[1].seed, -1);
  $('lottery').checked = true;
  await batchOne('Hello');
  assert.equal(sent.length, 5);
  for (const p of sent.slice(2)) {
    assert.equal(p.reuse_take, false); assert.equal(p.seed, -1);
  }
})().catch(e => { console.error(e); process.exit(1); });
'''
        result = subprocess.run(['node', '-e', prelude + controls + timing + batch + assertions],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('seed: fresh ? -1 : +$("seed").value', script)
        self.assertIn('reuse_take: $("reuse").checked && !fresh', script)
        self.assertIn('await batchOne(text, fresh, settings)', script)


if __name__ == "__main__":
    unittest.main()
