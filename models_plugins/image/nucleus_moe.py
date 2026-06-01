"""Text-to-image via NucleusMoE-Image using FP8 quantised weights (fits 24 GB VRAM)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class NucleusMoEPlugin(ModelPlugin):
    MODEL_ID     = "NucleusAI/Nucleus-Image"
    FP8_REPO     = "D-Squarius-Green-Jr/Nucleus-Image-FP8"
    DISPLAY_NAME = "Image: NucleusMoE"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "High-quality text-to-image via Nucleus-Image with FP8 weights (24 GB VRAM friendly)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=20, guidance=8.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "huggingface_hub"]
    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        import importlib.util
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import hf_hub_download

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {self.MODEL_ID} with FP8 weights from {self.FP8_REPO}…")

        patch_py = hf_hub_download(self.FP8_REPO, "moe_fp8_patch.py", cache_dir=_cache_dir)
        weights  = hf_hub_download(self.FP8_REPO, "Nucleus-Image-FP8.safetensors", cache_dir=_cache_dir)
        hf_hub_download(self.FP8_REPO, "config.json", cache_dir=_cache_dir)

        spec = importlib.util.spec_from_file_location("moe_fp8_patch", patch_py)
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        patch.apply_patch()

        transformer = patch.load_fp8_safetensors_transformer(weights)

        pipe = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir,
        )

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        if seed != 0:
            device = "cuda" if torch.cuda.is_available() else gfx_device
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        self.set_phase(inputs, "Generating")
        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
