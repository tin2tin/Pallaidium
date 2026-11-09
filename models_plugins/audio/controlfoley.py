"""Controllable video-to-audio (Foley) via ControlFoley (Xiaomi, ACM MM 2026).

One 44.1 kHz model, five tasks that follow the inputs you give it:

  MOVIE strip + prompt            -> TV2A    sound effects for the picture, steered by text
  MOVIE strip + prompt that       -> TC-V2A  the text wins over the picture
        contradicts the picture
  MOVIE strip + Speaker Ref.      -> AC-V2A  picture-synced audio in the timbre of a
        (2-4 s of sound)                      reference clip (prompt optional)
  MOVIE strip alone               -> V2A     Foley from the picture only
  prompt alone                    -> T2A     text-to-audio

Output is a mono 44.1 kHz WAV (a SOUND strip) of the same length as the video
(at most the Duration setting), so the video itself is never re-encoded.
The model was trained on ~8 s clips. Everything runs in bf16 (CLAP in fp32): ~13 GB VRAM
resident, ~14 GB peak, ~9 s for 8 s of audio on an RTX 4090; spectra match the fp32 reference.

The code is a research checkout, not a pip package, so ``_controlfoley_runtime``
downloads the pinned source (controlfoley/ + lib/, Apache-2.0) into
%LOCALAPPDATA%/Pallaidium/controlfoley on first use; nothing is added to
Blender's site-packages. Weights come from the HF hub cache:
YJX-Xiaomi/ControlFoley (11.3 GB net + 5.3 GB encoders; **CC BY-NC 4.0**,
non-commercial), facebook/musicgen-style and apple/DFN5B-CLIP-ViT-H-14-384.
"""

import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename
from . import _controlfoley_runtime as rt


class ControlFoleyPlugin(ModelPlugin):
    MODEL_ID     = "YJX-Xiaomi/ControlFoley"
    DISPLAY_NAME = "Video to Audio: ControlFoley (Foley, 44.1 kHz)"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = ("Controllable Foley for video: sound effects synced to the picture, steered by text "
                    "and/or a reference sound's timbre (Speaker Ref.), or text-to-audio without a video "
                    "(weights CC BY-NC 4.0)")

    # Prompt and Speaker Ref. are both optional; the MOVIE strip is optional too (T2A).
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.VIDEO | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.NEG_PROMPT,
        UISection.VIDEO_STRIP,
        UISection.AUDIO_REF,
        UISection.AUDIO_DURATION,
        UISection.STEPS,
        UISection.GUIDANCE,
        UISection.SEED,
    ]
    PARAMS       = ParamSpec(steps=25, guidance=4.5, audio_length=8.0)
    # Heavy / version-pinned imports (laion_clap, open_clip, audiocraft ...) happen in load()
    # after the import shims; checked here by find_spec only.
    REQUIRED_PACKAGES = []

    supports_inpaint     = False
    supports_img2img     = False
    show_enhance         = False
    supports_batch       = True
    # Ask the queue to render a MOVIE input through the VSE first, so the model sees the
    # strip's trimmed/retimed range instead of the raw source file from frame 0.
    trim_movie_input     = True

    def is_available(self) -> bool:
        import importlib.util as ilu
        mods = ("torch", "torchaudio", "av", "soundfile", "open_clip", "laion_clap",
                "julius", "timm", "torchdiffeq", "omegaconf", "einops", "librosa")
        return all(ilu.find_spec(m) is not None for m in mods)

    def load(self, prefs, scene, **kw):
        import torch

        local = bool(getattr(prefs, "local_files_only", False))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("Loading ControlFoley (net + CLIP/CAV-MAE/Synchformer/CLAP/MusicGen-Style encoders)...")
        models = rt.load_models(device, torch.bfloat16, local_files_only=local)
        return {"pipe": None, "model": models["net"], "vocoder": None,
                "feature_extractor": models["feats"], "cf": models}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import numpy as np
        import soundfile as sf

        video = inputs.video_path if inputs.video_path and os.path.isfile(inputs.video_path) else None
        if video and os.path.splitext(video)[1].lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
            video = None        # a still image strip is not a video: fall back to text-to-audio
        ref = inputs.audio_ref if inputs.audio_ref and os.path.isfile(inputs.audio_ref) else None

        if video and ref:
            task = "AC-V2A (reference timbre)"
        elif video and inputs.prompt.strip():
            task = "TV2A (text + video)"
        elif video:
            task = "V2A (video only)"
        else:
            task = "T2A (text only)"
            ref = None          # a reference clip only steers timbre of a video-synced track
        self.set_phase(inputs, f"Generating audio: {task}")
        print(f"ControlFoley | {task} | prompt={inputs.prompt!r} | steps={inputs.steps} cfg={inputs.guidance} "
              f"| {inputs.audio_length:.1f}s | seed={inputs.seed}")

        wav, sr, used = rt.generate_audio(
            pipe_obj["cf"],
            video_path=video,
            ref_audio_path=ref,
            prompt=inputs.prompt,
            neg_prompt=inputs.neg_prompt,
            duration=float(inputs.audio_length),
            steps=int(inputs.steps),
            cfg_strength=float(inputs.guidance),
            seed=int(inputs.seed),
            progress_fn=inputs.progress_fn,
        )

        self.set_phase(inputs, "Saving...")
        label = inputs.prompt.strip() or ("foley" if video else "audio")
        out = solve_path(clean_filename(f"{inputs.seed}_{label}") + ".wav")
        data = np.clip(wav.numpy().T, -1.0, 1.0)
        sf.write(out, data, sr)
        print(f"ControlFoley saved {used:.1f}s @ {sr} Hz -> {out}")
        return out
