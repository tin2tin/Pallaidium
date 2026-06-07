"""Text-to-music via stable-audio-tools (cocktailpeanut/stable-audio-3-medium-base)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename


class StableAudio3Plugin(ModelPlugin):
    MODEL_ID     = "cocktailpeanut/stable-audio-3-medium-base"
    DISPLAY_NAME = "Music: Stable Audio 3"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Text to music via Stable Audio 3 (44.1 kHz)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.NEG_PROMPT,
        UISection.AUDIO_DURATION,
        UISection.STEPS,
        UISection.GUIDANCE,
        UISection.SEED,
    ]
    PARAMS             = ParamSpec(steps=50, guidance=7.0, audio_length=30.0)
    REQUIRED_PACKAGES  = ["torch", "torchaudio", "einops", "stable_audio_tools"]

    def load(self, prefs, scene, **kw):
        import json
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from stable_audio_tools.models.factory import create_model_from_config

        print("Loading Stable Audio 3…")
        _cache_dir  = prefs.hf_cache_dir or None
        _local      = prefs.local_files_only

        config_path = hf_hub_download(
            repo_id=self.MODEL_ID, filename="model_config.json",
            cache_dir=_cache_dir, local_files_only=_local,
        )
        weights_path = hf_hub_download(
            repo_id=self.MODEL_ID, filename="model.safetensors",
            cache_dir=_cache_dir, local_files_only=_local,
        )

        with open(config_path) as f:
            model_config = json.load(f)

        model = create_model_from_config(model_config)
        state_dict = load_file(weights_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        if low_vram():
            model.to("cpu")
        else:
            model.to(gfx_device)

        sample_rate = model_config["sample_rate"]
        return {"pipe": model, "model": model_config, "vocoder": sample_rate, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        import torchaudio
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        model        = pipe_obj["pipe"]
        sample_rate  = pipe_obj["vocoder"]
        seed         = inputs.seed

        conditioning = [{
            "prompt": inputs.prompt,
            "seconds_start": 0,
            "seconds_total": inputs.audio_length,
        }]

        self.set_phase(inputs, "Generating")
        with torch.no_grad():
            output = generate_diffusion_cond(
                model,
                steps=inputs.steps,
                cfg_scale=inputs.guidance,
                conditioning=conditioning,
                sample_size=int(sample_rate * inputs.audio_length),
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=gfx_device,
                seed=seed if seed != 0 else None,
            )

        self.set_phase(inputs, "Saving")
        output = rearrange(output, "b d n -> d (b n)")
        output = (
            output.to(torch.float32)
            .div(torch.amax(torch.abs(output), dim=-1, keepdim=True).clamp(min=1e-8))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        filename = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")
        torchaudio.save(filename, output, sample_rate)
        print("Stable Audio 3 saved:", filename)
        return filename
