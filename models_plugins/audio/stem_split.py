"""Stem splitter via demucs-onnx — separates a source audio/video into stems."""

import json

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs
from ...utils.helpers import solve_path


_MULTI_STEM_PREFIX = "MULTI_STEM:"

STEM_NAMES_4 = ["vocals", "drums", "bass", "other"]
STEM_NAMES_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]


def _to_wav(audio_path):
    """Return (wav_path, is_temp).

    If soundfile can read the file directly, return it unchanged.
    Otherwise extract audio via PyAV and write a temp WAV.
    PyAV (av==16.0.1) is already installed and handles MP4/AAC/MP3/etc.
    """
    import soundfile as sf
    try:
        sf.info(audio_path)
        return audio_path, False
    except Exception:
        pass

    import tempfile
    import av
    import numpy as np

    chunks = []
    sr = 44100
    with av.open(audio_path) as container:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError(f"No audio stream found in: {audio_path}")
        stream = audio_streams[0]
        sr = stream.codec_context.sample_rate or 44100
        # Resample to float32 planar stereo so we always get a consistent layout
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())       # (2, n) float32
        for out in resampler.resample(None):          # flush
            chunks.append(out.to_ndarray())

    if not chunks:
        raise ValueError(f"No audio frames decoded from: {audio_path}")

    audio = np.concatenate(chunks, axis=1)            # (2, total_samples)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio.T, sr)                   # soundfile: (samples, channels)
    return tmp.name, True


class StemSplitterPlugin(ModelPlugin):
    MODEL_ID     = "StemSplitter"
    DISPLAY_NAME = "Stem Splitter (demucs-onnx)"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Split an audio/video strip into vocals, drums, bass, other (+ guitar & piano with 6-stem model)"

    # Audio reference is the input strip; no text prompt needed.
    INPUTS       = InputSpec.AUDIO_REF
    UI_SECTIONS  = []
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["demucs_onnx", "soundfile"]

    supports_inpaint      = False
    supports_img2img      = False
    show_enhance          = False
    requires_input_strip  = True

    def load(self, prefs, scene, **kw):
        # demucs-onnx loads its ONNX model lazily on first separate() call.
        return {"pipe": None, "model": None, "vocoder": None, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        audio_path = inputs.audio_ref
        if not audio_path:
            raise ValueError("No audio reference path — select a SOUND or MOVIE strip as input.")

        model = getattr(scene, "stem_split_model", "htdemucs_ft")
        all_stems = STEM_NAMES_6 if model == "htdemucs_6s" else STEM_NAMES_4
        selected  = [s for s in all_stems if getattr(scene, f"stem_split_{s}", True)]
        if not selected:
            raise ValueError("No stems selected — check at least one stem checkbox.")

        import os
        import shutil
        import tempfile

        # Convert to WAV if the source is a video or unsupported format
        self.set_phase(inputs, "Extracting audio…")
        wav_path, is_temp = _to_wav(audio_path)

        orig_base = os.path.splitext(os.path.basename(audio_path))[0]

        import tqdm.std as _tqdm_std
        _orig_init   = _tqdm_std.tqdm.__init__
        _orig_update = _tqdm_std.tqdm.update
        _bars: dict  = {}

        _DOWNLOAD_EXTS = (".onnx", ".pt", ".bin", ".safetensors")

        def _p_init(self2, *a, **kw2):
            _orig_init(self2, *a, **kw2)
            if not getattr(self2, "disable", False):
                _bars[id(self2)] = [self2.n or 0, self2.total or 0]
                desc = getattr(self2, "desc", "") or ""
                if any(ext in desc for ext in _DOWNLOAD_EXTS):
                    self.set_phase(inputs, "Downloading model…")
                else:
                    self.set_phase(inputs, "Separating stems…")

        def _p_update(self2, n=1):
            res = _orig_update(self2, n)
            entry = _bars.get(id(self2))
            if entry is not None:
                entry[0] = self2.n or 0
                entry[1] = self2.total or 0
            total_b = sum(v[1] for v in _bars.values() if v[1] > 0)
            done_b  = sum(v[0] for v in _bars.values())
            if inputs.progress_fn and total_b > 0:
                inputs.progress_fn(int(done_b / total_b * 100), 100)
            return res

        _tqdm_std.tqdm.__init__ = _p_init
        _tqdm_std.tqdm.update   = _p_update

        # For htdemucs_6s: all 6 stems come from one model; no per-stem specialists.
        # For 4-stem models: each stem has its own specialist — pass only the
        # selected list so demucs-onnx runs only those specialists (much faster).
        _stems_arg = None if model == "htdemucs_6s" else selected

        import glob as _glob
        stem_paths = {}

        try:
            if len(selected) == 1 and model != "htdemucs_6s":
                # Single specialist: 4× faster than the full bag model.
                # Returns a numpy array (channels, samples) at the input file's SR.
                import soundfile as sf
                sr = sf.info(wav_path).samplerate
                from demucs_onnx import separate_stem
                self.set_phase(inputs, "Downloading model…")
                arr = separate_stem(wav_path, selected[0],
                                    providers="cpu", progress=True)
                self.set_phase(inputs, "Saving stems…")
                stem_name = selected[0]
                out_path = solve_path(f"{stem_name}_{orig_base}.wav")
                sf.write(out_path, arr.T, sr)
                stem_paths[stem_name] = out_path
            else:
                # Multiple stems (or 6-stem model): use output_dir so demucs-onnx
                # writes files at the correct sample rate without manual SR handling.
                tmp_out = tempfile.mkdtemp()
                try:
                    from demucs_onnx import separate
                    self.set_phase(inputs, "Downloading model…")
                    separate(wav_path, output_dir=tmp_out, model=model,
                             stems=_stems_arg, providers="cpu", progress=True)
                    self.set_phase(inputs, "Saving stems…")
                    for stem_name in selected:
                        matches = _glob.glob(
                            os.path.join(tmp_out, "**", f"{stem_name}.wav"),
                            recursive=True,
                        )
                        if not matches:
                            print(f"[Stem Splitter] Output not found for '{stem_name}' in {tmp_out!r}")
                            for root, _, files in os.walk(tmp_out):
                                for f in files:
                                    print(f"  found: {os.path.join(root, f)}")
                            continue
                        dst = solve_path(f"{stem_name}_{orig_base}.wav")
                        shutil.copy2(matches[0], dst)
                        stem_paths[stem_name] = dst
                finally:
                    shutil.rmtree(tmp_out, ignore_errors=True)
        finally:
            _tqdm_std.tqdm.__init__ = _orig_init
            _tqdm_std.tqdm.update   = _orig_update
            if is_temp:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

        # Return a multi-stem sentinel that _queue_insert_strip understands.
        return _MULTI_STEM_PREFIX + json.dumps(stem_paths)

    def draw_custom_ui(self, layout, context):
        scene = context.scene
        layout.prop(scene, "stem_split_model", text="Variant")
        col = layout.column(align=True)
        col.use_property_split = True
        col.prop(scene, "stem_split_vocals")
        col.prop(scene, "stem_split_drums")
        col.prop(scene, "stem_split_bass")
        col.prop(scene, "stem_split_other")
        if getattr(scene, "stem_split_model", "") == "htdemucs_6s":
            col.prop(scene, "stem_split_guitar")
            col.prop(scene, "stem_split_piano")

