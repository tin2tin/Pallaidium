"""Zero-shot multilingual TTS via OmniVoice (k2-fsa/OmniVoice).

Supports 600+ languages. Voice Clone, Voice Design and Auto Voice can be
combined freely: provide a Speaker Ref., an Instruct string, both, or neither.

Non-verbal expressions can be embedded in the prompt: [laughter]  [sigh]
Chinese pronunciation: pinyin notation.  English: CMU dictionary notation.
Language is auto-detected from the prompt text unless overridden.
"""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename

_T_SHIFT_DEFAULT = 0.1   # hidden; use OmniVoice default


class OmniVoicePlugin(ModelPlugin):
    MODEL_ID     = "OmniVoice"
    DISPLAY_NAME = "TTS: OmniVoice"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Zero-shot TTS: 600+ languages, voice cloning & design, 40× real-time (k2-fsa/OmniVoice)"

    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF | InputSpec.TEXT_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.SPEED,
        UISection.STEPS,
        UISection.GUIDANCE,
        UISection.SEED,
    ]
    PARAMS = ParamSpec(steps=32, guidance=2.0)
    REQUIRED_PACKAGES = ["omnivoice"]

    def load(self, prefs, scene, **kw):
        import torch
        from omnivoice import OmniVoice

        use_cuda = torch.cuda.is_available()
        device_map = "cuda:0" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32

        print(f"Loading OmniVoice on {device_map}...")
        try:
            model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=device_map,
                dtype=dtype,
            )
        except OSError as e:
            if prefs.local_files_only:
                raise OSError(
                    "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
                ) from e
            print(f"OmniVoice preload failed ({e}), will load on first generate.")
            model = None
        except Exception as e:
            print(f"OmniVoice preload failed ({e}), will load on first generate.")
            model = None

        return {"pipe": None, "model": model, "vocoder": None, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import inspect
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice, OmniVoiceGenerationConfig

        model = pipe_obj["model"]
        use_cuda = torch.cuda.is_available()
        device_map = "cuda:0" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32

        if model is None:
            if prefs.local_files_only:
                raise OSError(
                    "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
                )
            model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=device_map,
                dtype=dtype,
            )
            pipe_obj["model"] = model

        instruct    = getattr(scene, "omnivoice_instruct",    "").strip() or None
        language    = getattr(scene, "omnivoice_language",    "AUTO")
        preprocess  = getattr(scene, "omnivoice_preprocess",  True)
        denoise     = getattr(scene, "omnivoice_denoise",     True)
        postprocess = getattr(scene, "omnivoice_postprocess", True)
        ref_audio   = inputs.audio_ref or None
        ref_text    = (inputs.text_ref or "").strip() or None
        # speed: None means 1.0 (model default); >1 = faster, <1 = slower
        speed       = inputs.speed if inputs.speed != 1.0 else None
        # language: "AUTO" means let OmniVoice detect from text
        lang_code   = None if language == "AUTO" else language

        # Build OmniVoiceGenerationConfig — only pass params the installed version accepts
        _cfg_params = set(inspect.signature(OmniVoiceGenerationConfig.__init__).parameters) - {"self"}
        _cfg_kwargs: dict = {
            "num_step":          max(4, inputs.steps),
            "guidance_scale":    inputs.guidance,
            "preprocess_prompt": (preprocess if ref_audio else False),
            "postprocess_output": postprocess,
        }
        for key, val in (("denoise", denoise), ("t_shift", _T_SHIFT_DEFAULT)):
            if key in _cfg_params:
                _cfg_kwargs[key] = val
        gen_cfg = OmniVoiceGenerationConfig(**_cfg_kwargs)

        output_path = solve_path(clean_filename(str(inputs.seed) + "_" + inputs.prompt) + ".wav")

        self.set_phase(inputs, "Generating")
        print(
            f"OmniVoice | ref={'yes' if ref_audio else 'no'} "
            f"| instruct={instruct!r} | lang={lang_code!r} "
            f"| speed={speed!r} | steps={inputs.steps} | guidance={inputs.guidance}"
        )

        # Assemble generate() kwargs — instruct and ref_audio are independent/combinable
        gen_kwargs: dict = {"text": inputs.prompt, "generation_config": gen_cfg}
        if ref_audio:
            gen_kwargs["ref_audio"] = ref_audio
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
        if instruct:
            gen_kwargs["instruct"] = instruct
        if lang_code:
            gen_kwargs["language"] = lang_code
        # speed is a direct generate() param, NOT a config field
        if speed is not None:
            gen_kwargs["speed"] = speed

        try:
            result = model.generate(**gen_kwargs)
        except Exception as e:
            print(f"OmniVoice generation failed: {e}")
            return output_path

        # generate() returns list[np.ndarray] — take the first result
        if result:
            arr = result[0]
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            sf.write(output_path, np.asarray(arr).flatten(), 24000)
            print(f"OmniVoice saved: {output_path}")
        else:
            print("OmniVoice: no audio generated.")

        return output_path

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene

        # Voice clone — optional
        col.prop(scene, "omnivoice_language")
        col.separator()
        row = col.row(align=True)
        row.prop(scene, "ref_audio_path", text="Speaker Ref.")
        row.operator("sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER")
        col.prop(scene, "ref_text", text="Ref. Text")
        col.prop(scene, "omnivoice_preprocess")
        col.separator()

        # Voice design — optional, combinable with ref audio
        col.prop(scene, "omnivoice_instruct")
        col.separator()

        col.prop(scene, "omnivoice_denoise")
        col.prop(scene, "omnivoice_postprocess")

        return False
