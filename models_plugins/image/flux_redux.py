"""FLUX Redux image-to-image (Runware/FLUX.1-Redux-dev)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class FluxReduxPlugin(ModelPlugin):
    MODEL_ID     = "Runware/FLUX.1-Redux-dev"
    DISPLAY_NAME = "Image: FLUX Redux (image restyle)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Image restyling via FLUX Redux — no text prompt needed"

    INPUTS       = InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS       = ParamSpec(steps=25, guidance=3.5)
    REQUIRED_PACKAGES          = ["torch", "diffusers", "transformers"]
    supports_inpaint           = False
    supports_img2img           = False
    uses_standard_input_strip  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import FluxPriorReduxPipeline, FluxPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print("Loading FLUX Redux…")
        pipe = FluxPipeline.from_pretrained(
            "ChuckMcSneed/FLUX.1-dev",
            text_encoder=None, text_encoder_2=None,
            torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir,
            local_files_only=prefs.local_files_only,
        )
        pipe_prior = FluxPriorReduxPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            local_files_only=prefs.local_files_only,
        ).to("cuda")

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        # Store pipe_prior in "refiner" slot
        return {"pipe": pipe, "converter": None, "refiner": pipe_prior, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe       = pipe_obj["pipe"]
        pipe_prior = pipe_obj["refiner"]
        image      = inputs.image
        if image is None:
            raise ValueError("FLUX Redux requires an input image.")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        self.set_phase(inputs, "Preprocessing")
        prior_output = pipe_prior(image)
        self.set_phase(inputs, "Generating")
        return pipe(
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            **prior_output,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
