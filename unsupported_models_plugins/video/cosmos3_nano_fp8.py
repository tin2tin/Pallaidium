"""Text-to-video and image-to-video via Cosmos3-Nano with low-VRAM offload.

Uses enable_sequential_cpu_offload (layer-by-layer) instead of model-level offload,
plus attention slicing, to keep peak VRAM well under 16 GB.  Slower than the standard
variant but fits on cards with 8–16 GB VRAM.
"""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename


class Cosmos3NanoFP8VideoPlugin(ModelPlugin):
    MODEL_ID     = "benjiaiplayground/Cosmos3-Nano_fp8-video"   # unique registry key
    DISPLAY_NAME = "Video: Cosmos3-Nano Low VRAM"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "NVIDIA Cosmos3-Nano multi-frame video generation with sequential CPU offload (~8–16 GB VRAM, slower)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS,
        UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS            = ParamSpec(width=1280, height=720, frames=49, steps=35, guidance=6.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = False
    supports_img2img  = True

    def load(self, prefs, scene, **kw):
        import logging
        import torch
        from diffusers import Cosmos3OmniPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print("Loading nvidia/Cosmos3-Nano with sequential CPU offload…")

        _diffusers_log = logging.getLogger("diffusers")
        _prev_level = _diffusers_log.level
        _diffusers_log.setLevel(logging.ERROR)
        try:
            pipe = Cosmos3OmniPipeline.from_pretrained(
                "nvidia/Cosmos3-Nano",
                torch_dtype=torch.bfloat16,
                enable_safety_checker=False,
                cache_dir=_cache_dir,
            )
        finally:
            _diffusers_log.setLevel(_prev_level)

        # Layer-by-layer offload: only one transformer layer on VRAM at a time.
        # Much lower peak VRAM than model_cpu_offload at the cost of speed.
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing()

        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from diffusers.utils import export_to_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed != 0 else None
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
        print(f"Cosmos3-Nano Low VRAM saved: {dst_path}")
        return dst_path
