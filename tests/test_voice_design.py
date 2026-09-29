"""Voice Designer contracts. Fake model tests prove wiring, NOT audible adherence."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import voice_design as vd

HAS_API = all(importlib.util.find_spec(m) for m in ('numpy', 'soundfile', 'fastapi', 'httpx'))


class DesignInstructionTests(unittest.TestCase):
    def test_all_58_axes_reach_instruction(self):
        self.assertEqual(len(vd.SCHEMA), 58)
        self.assertEqual(len(vd.FIELDS), 58)
        for f in vd.SCHEMA:
            with self.subTest(field=f['key']):
                value = f['choices'][-1] if f['choices'] else 'Brazilian Portuguese accent'
                instruction = vd.compile_instruction({f['key']: value})
                self.assertIn(value, instruction)
                self.assertEqual(f['control_type'], 'soft_instruction')

    def test_acting_and_free_form_are_model_instructions(self):
        instruction = vd.compile_instruction({'description':'Warm dry adult voice.',
            'direction':'Start confident, then sound worried.', 'emphasize':'not today',
            'crying':'tearful voice', 'laugh':'a brief warm chuckle before speaking'})
        for phrase in ('Warm dry adult voice.', 'Start confident', 'not today', 'tearful', 'chuckle'):
            self.assertIn(phrase, instruction)
        self.assertIn('not these instructions', instruction)

    def test_validation_and_language_boundaries(self):
        for bad in (None, [], {'fake':'yes'}, {'dryness':'magic'}, {'description':'x'*1201},
                    {'crying':'uncontrolled'}, {'emotion':12}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vd.compile_instruction(bad)
        self.assertIn('pt', vd.LANGUAGES)
        self.assertNotIn('yue', vd.LANGUAGES)
        self.assertEqual(len(vd.LANGUAGES), 10)
        self.assertIn('natural adult voice', vd.compile_instruction({}))
        for key, value in [('temperature', float('nan')), ('top_k', 3.5), ('top_p', 4)]:
            with self.assertRaises(ValueError):
                vd.generation_options({key:value})

    def test_installer_is_isolated_and_requires_real_smoke(self):
        ps = (ROOT/'tools/start_voice_design.ps1').read_text()
        self.assertIn('Get-FileHash $Zip -Algorithm SHA256', ps)
        self.assertIn('https://download.pytorch.org/whl/cpu', ps)
        self.assertIn('9GB', ps)
        self.assertIn('uv-x86_64-pc-windows-msvc.zip', ps)
        self.assertLess(ps.index("'--self-test'"), ps.index('Set-Content -Encoding UTF8 $Ready'))
        self.assertNotIn('flash-attn', (ROOT/'tools/requirements-voice-design.txt').read_text())
        for filename in ('VOICE-DESIGN.bat', 'GACHA-STUDIO.bat'):
            data = (ROOT/filename).read_bytes()
            data.decode('ascii')
            self.assertNotIn(b'\n', data.replace(b'\r\n', b''))


@unittest.skipUnless(HAS_API, 'API test dependencies not installed')
class DesignAdapterTests(unittest.TestCase):
    def test_official_method_receives_real_instruction_separate_from_text(self):
        import numpy as np
        calls = []
        fake_torch = types.SimpleNamespace(manual_seed=lambda x: calls.append(('seed', x)),
                                           inference_mode=contextlib.nullcontext)
        class Model:
            def generate_voice_design(self, **kwargs):
                calls.append(kwargs)
                return [np.sin(np.arange(16000)*.1)*.2], 16000
        designer = vd.VoiceDesigner()
        designer.model = Model()
        with patch.dict(sys.modules, {'torch':fake_torch}):
            sr, raw, prompt = designer.generate({'text':'Only these words.', 'text_lang':'pt', 'seed':42,
                'design':{'roughness':'slightly raspy', 'whisper':'whispered speech'}})
        self.assertEqual(sr, 16000)
        self.assertEqual(raw.size, 16000)
        self.assertEqual(calls[0], ('seed',42))
        self.assertEqual(calls[1]['text'], 'Only these words.')
        self.assertEqual(calls[1]['language'], 'Portuguese')
        self.assertEqual(calls[1]['instruct'], prompt)
        self.assertIn('slightly raspy', prompt)
        self.assertNotIn('pitch', calls[1])
        self.assertNotIn('ref_audio', calls[1])

    def test_cpu_loader_options_and_memory_gate(self):
        calls = []
        fake_torch = types.SimpleNamespace(float32='float32', set_num_threads=lambda x: calls.append(x))
        fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=12*1024**3))
        class SDK:
            @staticmethod
            def from_pretrained(model, **kwargs):
                calls.append((model,kwargs))
                return object()
        with patch.dict(sys.modules, {'torch':fake_torch, 'psutil':fake_psutil,
                                     'qwen_tts':types.SimpleNamespace(Qwen3TTSModel=SDK)}):
            d = vd.VoiceDesigner(6); d.load(); d.load()
            self.assertEqual(calls, [6, (vd.MODEL_ID, {'device_map':'cpu', 'dtype':'float32', 'attn_implementation':'sdpa'})])
            fake_psutil.virtual_memory = lambda: types.SimpleNamespace(available=3*1024**3)
            with self.assertRaisesRegex(RuntimeError, 'available RAM'):
                vd.VoiceDesigner().load()
            self.assertEqual(len(calls),2)

    def test_no_silent_output_or_unbounded_input(self):
        import numpy as np
        class Model:
            def generate_voice_design(self, **kwargs):
                return [np.zeros(16000)], 16000
        d = vd.VoiceDesigner(); d.model = Model()
        with patch.dict(sys.modules, {'torch':types.SimpleNamespace(manual_seed=lambda x: None,
                                                                    inference_mode=contextlib.nullcontext)}):
            with self.assertRaisesRegex(RuntimeError, 'silent'):
                d.generate({'text':'hello'})
            for change in ({'text':'x'*601}, {'text_lang':'yue'}, {'seed':-2}):
                with self.assertRaises(ValueError):
                    d.generate(dict({'text':'hello'}, **change))


@unittest.skipUnless(HAS_API, 'API test dependencies not installed')
class DesignAPITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from type_ui import build_app
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = Path(self.tmp.name)
        self.calls = []
        outer = self
        class FakeDesigner:
            model = None
            def __init__(self, threads):
                pass
            def generate(self, req):
                import numpy as np
                outer.calls.append(req)
                return 16000, (np.sin(np.arange(16000)*.1)*.2).astype('float32'), vd.compile_instruction(req['design'])
        with patch.object(vd, 'VoiceDesigner', FakeDesigner):
            app = build_app(p, p/'seeds', p/'presets.csv', '', engine='qwen-design')
        self.client = TestClient(app); self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.req = {'engine':'qwen-design', 'design':{'emotion':'angry'}, 'text':'Not today.',
                    'text_lang':'en', 'reuse_take':True, 'seed':2}

    def test_design_without_reference_preview_cache_and_invalidation(self):
        schema = self.client.get('/api/design/schema').json()
        self.assertTrue(schema['available'])
        self.assertEqual(len(schema['fields']),58)
        preview = self.client.post('/api/design/preview', json={'design':self.req['design']}).json()['instruction']
        first = self.client.post('/api/tts', json=self.req)
        self.assertEqual(first.status_code,200, first.text[:100])
        self.assertEqual(first.headers['x-design-instruction-sha256'], hashlib.sha256(preview.encode()).hexdigest())
        self.assertEqual(first.headers['x-take-cache'], 'miss')
        self.assertEqual(self.client.post('/api/tts', json=self.req).headers['x-take-cache'], 'hit')
        changed = dict(self.req, design={'emotion':'happy'})
        self.assertEqual(self.client.post('/api/tts', json=changed).headers['x-take-cache'], 'miss')
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.calls[1]['design']['emotion'],'happy')
        self.assertNotIn('ref_audio_path',self.calls[1])

    def test_unsupported_controls_are_rejected_not_silently_ignored(self):
        for changes in ({'engine':'gsv'}, {'speed_factor':1.2}, {'text_split_method':'cut5'},
                        {'batch_size':4}, {'design':{'madeup':'yes'}}):
            with self.subTest(changes=changes):
                response = self.client.post('/api/tts', json=dict(self.req, **changes))
                self.assertEqual(response.status_code,400,response.text)
        self.assertEqual(len(self.calls),0)


if __name__ == '__main__':
    unittest.main()
