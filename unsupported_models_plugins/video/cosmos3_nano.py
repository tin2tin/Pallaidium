"""Text-to-video and image-to-video via Cosmos3-Nano (nvidia/Cosmos3-Nano)."""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename


_HF_REPO = "nvidia/Cosmos3-Nano"


class Cosmos3NanoVideoPlugin(ModelPlugin):
    MODEL_ID     = "nvidia/Cosmos3-Nano-video"   # unique registry key; HF repo is _HF_REPO
    DISPLAY_NAME = "Video: Cosmos3-Nano"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "NVIDIA Cosmos3-Nano multi-frame video generation (txt2video / img2video), runs under 24 GB VRAM"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS,
        UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS            = ParamSpec(width=1280, height=720, frames=49, steps=35, guidance=6.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = False
    supports_img2img  = True   # img2video counts as img2img in the framework

    def load(self, prefs, scene, **kw):
        import logging
        import torch
        from diffusers import Cosmos3OmniPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {_HF_REPO}…")

        # Cosmos3 checkpoints ship extra config keys (action_gen, vision_encoder, …)
        # for robotics features that the current diffusers classes don't use.
        # Raise the diffusers log level during load to suppress the harmless
        # "were passed … but are not expected and will be ignored" spam.
        _diffusers_log = logging.getLogger("diffusers")
        _prev_level = _diffusers_log.level
        _diffusers_log.setLevel(logging.ERROR)
        try:
            # Load weights to CPU; model_cpu_offload moves one submodule at a time
            # (transformer, VAE, …) to CUDA during inference.  Peak VRAM ≈ size of
            # the largest submodule (~10–14 GB for Nano) — fits comfortably in 24 GB
            # while being far faster than sequential (layer-by-layer) offload.
            pipe = Cosmos3OmniPipeline.from_pretrained(
                _HF_REPO,
                torch_dtype=torch.bfloat16,
                enable_safety_checker=False,
                cache_dir=_cache_dir,
            )
        finally:
            _diffusers_log.setLevel(_prev_level)

        pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from diffusers.utils import export_to_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator(device="cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        num_frames = max(9, inputs.frames)

        self.set_phase(inputs, "Generating")

        call_kwargs = dict(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt or "",
            num_frames=num_frames,
            height=inputs.height,
            width=inputs.width,
            fps=24.0,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        )

        if inputs.image is not None:
            call_kwargs["image"] = inputs.image

        result = pipe(**call_kwargs)

        self.set_phase(inputs, "Saving")
        src_path = export_to_video(result.video, fps=24, macro_block_size=1)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt[:40]) + ".mp4")
        shutil.move(src_path, dst_path)
        print(f"Cosmos3-Nano saved: {dst_path}")
        return dst_path
