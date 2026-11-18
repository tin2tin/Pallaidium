"""Runtime for the Robin TTS plugin (FixedSeed/Robin).

Kept free of ``bpy`` / Pallaidium imports so it can be exercised standalone.

Robin is not a transformers/diffusers model: the HF repo ships a small ``robin``
python package under ``package/`` next to the weights. Nothing is pip-installed:
the repo is fetched into the HF cache (weights, VAE, tokenizer, speech tokenizer
ONNX and the package source) and ``package/`` is put on ``sys.path``.

The one dependency conflict is s3tokenizer: the add-on pins 0.2.0 (used by other
plugins) but Robin's speech tokenizer is the v3 25 Hz model, which needs >= 0.3.0.
0.3.0 is pure Python, so its wheel is unpacked into
``%LOCALAPPDATA%/Pallaidium/robin`` and swapped into ``sys.modules`` only while the
speech tokenizer is built; the pinned 0.2.0 modules are restored afterwards.

Weights: FixedSeed/Robin (7.2 GB model + 1 GB tokenizer + 0.5 GB VAE, bf16 ~8-10 GB
VRAM, Fixed Seed Open License: Apache-2.0 based, commercial use limited above a
revenue threshold).
"""

import glob
import io
import json
import os
import sys
import urllib.request
import zipfile

REPO_ID = "FixedSeed/Robin"
LORA_FILE = "robin_character_lora.safetensors"
S3_VERSION = "0.3.0"
SAMPLE_RATE = 48000

HOME = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Pallaidium", "robin")
S3_DIR = os.path.join(HOME, f"s3tokenizer-{S3_VERSION}")

_CORE_FILES = ["robin_v1.safetensors", "robin_vae.safetensors", "speech_tokenizer_v3_25hz.onnx",
               "tokenizer.json", "config.json", "package/*", "package/**"]


def cancelled_types():
    """The exception Robin raises when a progress callback returns False."""
    from robin.pipeline import Cancelled
    return (Cancelled,)


def ensure_repo(local_files_only=False) -> str:
    """Download the Robin repo (weights + package source, not ComfyUI/images/LoRA) -> local dir."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(REPO_ID, allow_patterns=_CORE_FILES, local_files_only=local_files_only)
    pkg = os.path.join(path, "package")
    if not os.path.isfile(os.path.join(pkg, "robin", "pipeline.py")):
        raise RuntimeError(f"Robin: package source missing in {path}")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    return path


def ensure_lora(local_files_only=False) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, LORA_FILE, local_files_only=local_files_only)


def ensure_s3tokenizer() -> str:
    """Unpack the pure-python s3tokenizer 0.3.0 wheel into HOME once; return the dir holding ``s3tokenizer/``."""
    if os.path.isfile(os.path.join(S3_DIR, "s3tokenizer", "model_v3.py")):
        return S3_DIR
    print(f"Robin: downloading s3tokenizer {S3_VERSION} (speech tokenizer v3 support) ...")
    with urllib.request.urlopen(f"https://pypi.org/pypi/s3tokenizer/{S3_VERSION}/json", timeout=60) as r:
        meta = json.load(r)
    wheel = next(u["url"] for u in meta["urls"] if u["filename"].endswith(".whl"))
    with urllib.request.urlopen(wheel, timeout=120) as r:
        data = r.read()
    os.makedirs(S3_DIR, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(S3_DIR)
    return S3_DIR


def load_robin(device, dtype, local_files_only=False):
    """Build the Robin pipeline with its speech tokenizer (v3) already constructed."""
    repo = ensure_repo(local_files_only)
    s3_dir = ensure_s3tokenizer()
    from robin.pipeline import Robin

    robin = Robin.from_pretrained(repo, device=device, dtype=dtype)

    # Swap s3tokenizer 0.3.0 in just long enough to build the speech tokenizer; it keeps its own
    # module reference (self.s3), so the pinned 0.2.0 can be restored right after.
    saved = {k: v for k, v in sys.modules.items() if k == "s3tokenizer" or k.startswith("s3tokenizer.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, s3_dir)
    try:
        robin.speech            # lazy property: builds SpeechTokenizer (imports s3tokenizer 0.3.0)
    finally:
        try:
            sys.path.remove(s3_dir)
        except ValueError:
            pass
        for k in [k for k in sys.modules if k == "s3tokenizer" or k.startswith("s3tokenizer.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return robin


class Progress:
    """pieces.* progress object: maps (stage, step, total) onto one 0..1 bar and carries cancel."""

    BAR = 1000

    def __init__(self, progress_fn=None, should_cancel=None, pieces=1):
        self.fn, self.cancel, self.pieces = progress_fn, should_cancel, max(1, pieces)
        self.done = 0            # finished takes (a retry counts as extra work, clamped)
        self._last = -1

    def start(self, seconds):
        pass

    def finish(self, extra=0):
        self.done += 0 if extra else 1

    def __call__(self, stage, step, total):
        if self.cancel and self.cancel():
            return False
        # plan = first ~half, decode = second half of one take
        frac = (0.5 * step / total) if stage == "plan" else (0.5 + 0.5 * step / total)
        v = min(self.BAR - 1, int(self.BAR * (min(self.done, self.pieces - 1) + frac) / self.pieces))
        if self.fn is not None and v != self._last:
            self._last = v
            self.fn(v, self.BAR)
        return True


def synthesize(robin, text, *, voice, lang, seed, cfg, steps, reference, reference_text, revoice,
               lora_scale, progress_fn=None, should_cancel=None):
    """Return (float32 mono waveform at 48 kHz, number of chunks)."""
    import numpy as np
    import soundfile as sf
    from robin import pieces

    if revoice:
        if not reference:
            raise ValueError("Robin Re-voice needs a Speaker Ref. clip (the recording to perform again).")
        x, rate = sf.read(reference, dtype="float32", always_2d=True)
        if len(x) / rate > 45:
            raise ValueError(f"Re-voice source is {len(x) / rate:.0f} s long; use at most 45 s.")
        prog = Progress(progress_fn, should_cancel, 1)
        wav = pieces.change(robin, x.mean(1), rate, text, voice=voice, lang=lang, seed=seed, steps=steps, progress=prog)
        return np.asarray(wav, dtype=np.float32), 1

    ref = None
    if reference:
        x, rate = sf.read(reference, dtype="float32", always_2d=True)
        ref = (x.mean(1), rate)
    n = len(pieces.split(text, 35, lang))
    prog = Progress(progress_fn, should_cancel, n)
    wav = pieces.read(
        robin, text, voice,
        reference=ref[0] if ref else None, reference_rate=ref[1] if ref else None,
        reference_text=reference_text or None, seed=seed, cfg=cfg, lang=lang, steps=steps, progress=prog,
    )
    return np.asarray(wav, dtype=np.float32), n
