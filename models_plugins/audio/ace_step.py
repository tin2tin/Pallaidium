"""Text-to-music via ACE-Step (ModernAudio/ace-step-1b)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename


class AceStepPlugin(ModelPlugin):
    MODEL_ID     = "ACE-Step/acestep-v15-xl-turbo-diffusers"
    DISPLAY_NAME = "Music: ACE-Step"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "High-quality text-to-music via ACE-Step (Rectified Flow)"

    INPUTS       = InputSpec.PROMPT | InputSpec.MUSIC_PARAMS
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.AUDIO_DURATION,
        UISection.STEPS, UISection.GUIDANCE,
        UISection.MUSIC_PARAMS,
        UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=60, guidance=3.5, audio_length=30.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "accelerate"]
    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import AceStepPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {self.MODEL_ID}…")
        pipe = AceStepPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=_cache_dir
        )
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(gfx_device)
        return {"pipe": pipe, "model": None, "vocoder": None, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0
            else (torch.Generator(device=gfx_device).manual_seed(seed) if seed != 0 else None)
        )

        kwargs = dict(
            prompt=inputs.prompt,
            audio_duration=inputs.audio_length,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            generator=generator,
        )
        if inputs.bpm:
            kwargs["bpm"] = inputs.bpm
        if inputs.lyrics:
            kwargs["lyrics"] = inputs.lyrics
        if inputs.key_scale:
            kwargs["keyscale"] = inputs.key_scale
        if inputs.time_signature:
            kwargs["timesignature"] = inputs.time_signature

        self.set_phase(inputs, "Generating")
        cb = self.step_callback(inputs)
        if cb is not None:
            kwargs["callback_on_step_end"] = cb
        output = pipe(**kwargs)

        self.set_phase(inputs, "Saving")
        filename = solve_path(clean_filename(str(seed) + "_" + inputs.prompt[:30]) + ".wav")
        import soundfile as sf
        sf.write(filename, output.audios[0].T.cpu().float().numpy(), pipe.config.get("sample_rate", 44100))
        print(f"ACE-Step audio saved: {filename}")
        return filename
