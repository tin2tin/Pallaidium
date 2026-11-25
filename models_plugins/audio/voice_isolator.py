"""Voice isolation / denoising via Inworld Voice Isolator v1 (Apache-2.0).

Keeps the foreground voice of a SOUND strip and removes noise, music, other
(quieter) voices and room tail. Returns two SOUND strips at 48 kHz: the isolated
voice and the background (the input minus the voice).

The model is a 64M-param band-split dual-path network (offline checkpoint
``denoiser.safetensors``, 256 MB from inworld/voice-isolator). Long clips are
processed in 8 s chunks overlapping by 2 s by the library itself. It is a
denoiser, not a speaker selector and not a restoration model.
"""

import json
import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path
from .stem_split import _to_wav

_REPO              = "inworld/voice-isolator"
_MULTI_STEM_PREFIX = "MULTI_STEM:"


class VoiceIsolatorPlugin(ModelPlugin):
    MODEL_ID     = _REPO
    DISPLAY_NAME = "Voice Isolation: Inworld Voice Isolator"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Remove noise, music and room tail from a SOUND strip; outputs the clean voice and the background (48 kHz)"

    # The SOUND strip is the input; no prompt needed.
    INPUTS       = InputSpec.AUDIO_REF
    UI_SECTIONS  = []
    PARAMS       = ParamSpec()
    # inworld.voice_isolation is a namespace package -> checked via find_spec in is_available().
    REQUIRED_PACKAGES = []

    supports_inpaint     = False
    supports_img2img     = False
    show_enhance         = False
    requires_input_strip = True
    supports_batch       = False   # deterministic

    def is_available(self) -> bool:
        import importlib.util as ilu
        try:
            return ilu.find_spec("inworld.voice_isolation") is not None
        except (ImportError, ValueError):
            return False

    def load(self, prefs, scene, **kw):
        import torch

        if bool(getattr(prefs, "local_files_only", False)):
            os.environ["HF_HUB_OFFLINE"] = "1"
        from inworld import voice_isolation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Inworld Voice Isolator on {device}...")
        isolator = voice_isolation.VoiceIsolator(_REPO, device=device, mode="offline")
        return {"pipe": None, "model": isolator, "vocoder": None, "feature_extractor": None,
                "device": device}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import numpy as np
        import soundfile as sf
        import torch

        isolator = pipe_obj["model"]
        audio_path = inputs.audio_ref
        if not audio_path:
            raise ValueError("No audio input — select a SOUND strip (the audio strip of a movie, not the MOVIE strip).")

        self.set_phase(inputs, "Extracting audio…")
        wav_path, is_temp = _to_wav(audio_path)
        try:
            wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        finally:
            if is_temp:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
        mono = np.ascontiguousarray(wav.mean(axis=1))
        sample_rate = isolator.sample_rate  # 48000

        self.set_phase(inputs, "Isolating voice…")
        print(f"Voice Isolator | {mono.shape[0] / sr:.1f}s @ {sr} Hz")
        with torch.inference_mode():
            # Library resamples to 48 kHz internally and returns at the requested rate.
            voice = isolator.isolate(mono, sr, output_sample_rate=sample_rate)
        voice = np.asarray(voice.cpu().numpy() if hasattr(voice, "cpu") else voice, dtype=np.float32).reshape(-1)

        # Background = input at 48 kHz minus the voice (the library aligns lengths).
        from inworld.voice_isolation import audio as _va
        ref = _va.resample(torch.from_numpy(mono), sr, sample_rate).numpy() if sr != sample_rate else mono
        n = min(len(ref), len(voice))
        voice, background = voice[:n], ref[:n] - voice[:n]

        self.set_phase(inputs, "Saving…")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        paths = {}
        for name, data in (("voice", voice), ("background", background)):
            out = solve_path(f"{name}_{base}.wav")
            sf.write(out, data, sample_rate, subtype="FLOAT")
            paths[name] = out
        print(f"Voice Isolator saved: {paths}")
        return _MULTI_STEM_PREFIX + json.dumps(paths)
