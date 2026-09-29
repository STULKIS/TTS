"""Bounded, session-only Studio caches. No model dependencies.

Call from the Studio's single render lock: these helpers are not thread safe.
Audio stays in RAM; fitted references use an automatically cleaned temp dir.
"""
from collections import OrderedDict
from pathlib import Path
import tempfile
import uuid


def file_version(path):
    p = Path(path).resolve()
    s = p.stat()
    return str(p), s.st_size, s.st_mtime_ns, s.st_ctime_ns


class TakeCache:
    """LRU bounded by both audio bytes and entry count; never caches failures."""
    def __init__(self, max_bytes=64 * 1024 * 1024, max_entries=24):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key):
        item = self.entries.get(key)
        if item is None:
            return None
        self.entries.move_to_end(key)
        return item[0]

    def put(self, key, value, size):
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        if size > self.max_bytes or self.max_entries < 1:
            return
        self.entries[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed


class ReferenceCache:
    """Keep fitted references at stable paths so the engine can reuse its prompt.

    In-range references are untouched. Changes to a source file invalidate the
    fit. A 16-file LRU bounds disk usage; files disappear at normal shutdown.
    """
    def __init__(self, max_entries=16):
        self.max_entries = max_entries
        self.directory = tempfile.TemporaryDirectory(prefix="studio_refs_")
        self.entries = OrderedDict()

    def fit(self, source):
        import numpy as np
        import soundfile as sf
        key = file_version(source)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        info = sf.info(source)
        duration = info.frames / info.samplerate
        if 3.0 <= duration <= 10.0:
            return str(Path(source).resolve())
        x, sr = sf.read(source, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if duration > 10.0:
            # Existing Studio fitting policy; no new speech/tempo modification.
            win = int(9.5 * sr)
            energy = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
            start = int(np.argmax(energy[win:] - energy[:-win]))
            x = x[start:start + win]
        else:
            x = np.pad(x, (0, max(0, int(3.2 * sr) - len(x))))
        dest = Path(self.directory.name) / (uuid.uuid4().hex + ".wav")
        sf.write(dest, x, sr)
        self.entries[key] = str(dest)
        while len(self.entries) > self.max_entries:
            _, old = self.entries.popitem(last=False)
            Path(old).unlink(missing_ok=True)
        return str(dest)

    def close(self):
        self.entries.clear()
        self.directory.cleanup()
