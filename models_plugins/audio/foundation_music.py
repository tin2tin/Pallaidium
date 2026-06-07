"""Text-to-music via StableAudio (tintwotin/Foundation-1-Diffusers)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename


class FoundationMusicPlugin(ModelPlugin):
    MODEL_ID     = "tintwotin/Foundation-1-Diffusers"
    DISPLAY_NAME = "Music: Foundation-1"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Text to music via Stable Audio"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.NEG_PROMPT,
        UISection.AUDIO_DURATION,
        UISection.STEPS,
        UISection.SEED,
    ]
    PARAMS       = ParamSpec(steps=100, audio_length=10.0)
    REQUIRED_PACKAGES = ["torch", "scipy", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import StableAudioPipeline

        print("Loading Foundation-1 StableAudio…")
        _cache_dir = prefs.hf_cache_dir or None
        pipe = StableAudioPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16, cache_dir=_cache_dir,
            local_files_only=prefs.local_files_only,
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
        from scipy.io.wavfile import write as write_wav

        pipe = pipe_obj["pipe"]
        seed = inputs.seed

        if torch.cuda.is_available() and seed != 0:
            generator = torch.Generator("cuda").manual_seed(seed)
        elif seed != 0:
            generator = torch.Generator(device=gfx_device)
            generator.manual_seed(seed)
        else:
            generator = None

        self.set_phase(inputs, "Generating")
        import inspect
        call_params = inspect.signature(pipe.__call__).parameters
        extra = {}
        if "callback_on_step_end" in call_params:
            extra["callback_on_step_end"] = self.step_callback(inputs)
        audio = pipe(
            inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            audio_end_in_s=inputs.audio_length,
            num_waveforms_per_prompt=1,
            generator=generator,
            **extra,
        ).audios

        self.set_phase(inputs, "Saving")
        output = audio[0].T.float().cpu().numpy()
        filename = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")
        write_wav(filename, pipe.vae.sampling_rate, output)
        print("Foundation-1 audio saved:", filename)
        return filename
