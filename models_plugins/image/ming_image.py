"""Text-to-image (RGBA) via Ming-Image-0.1-Design — UI, poster and infographic designs with legible text."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


# Prompt openers the model was trained on for a transparent background (from the
# model card). Any other prompt yields an opaque image, so its alpha is dropped.
_TRANSPARENT_PHRASES = (
    "rgba, 4-channel, transparent background",
    "isolated subject, alpha matte, no background",
    "cutout png, alpha=0 outside the object",
    "transparent canvas, not white, not checkerboard",
    "production rgba layer for compositing",
)


class MingImagePlugin(ModelPlugin):
    MODEL_ID     = "OzzyGT/ming_image_0_1_design_diffusers"
    BLOCKS_ID    = "OzzyGT/ming_image_custom_blocks"
    DISPLAY_NAME = "Ming-Image 0.1 Design"
    DESCRIPTION  = "Text-to-image for UI, posters and infographics with legible text; transparent (RGBA) output on request"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED,
    ]
    # 12 steps, CFG-free (the original pipeline's cfg=1.0). Guidance > 1 turns on
    # classifier-free guidance against zero conditioning and doubles the cost.
    PARAMS            = ParamSpec(steps=12, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import ModularPipelineBlocks
        # Importing sdnq registers its quantizer with diffusers' from_pretrained.
        from sdnq import SDNQConfig

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        dtype = torch.bfloat16
        print(f"Loading {self.MODEL_ID}…")

        # Modular Diffusers port: the blocks repo holds the pipeline steps, the
        # components repo the weights plus the model code for the transformer,
        # connector and text encoder (hence both trust flags).
        blocks = ModularPipelineBlocks.from_pretrained(
            self.BLOCKS_ID,
            trust_remote_code=True,
            components_repo=self.MODEL_ID,
            trust_components_code=True,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        pipe = blocks.init_pipeline()

        # The text encoder is a 16B-parameter MoE (32 GB in bf16) — more than
        # the system RAM, so the model card's "load bf16, group-offload" recipe
        # can't work here. Quantize it on the fly instead, shard by shard on the
        # GPU: uint4 (~9 GB) plus the int8 transformer (~6 GB) keep the whole
        # pipeline resident on a 24 GB card with no offloading at all.
        on_gpu = gfx_device != "mps" and not low_vram()
        q_device = "cuda" if gfx_device == "cuda" else None
        return_device = "cuda" if on_gpu else "cpu"
        te_quant = SDNQConfig(
            weights_dtype="uint4",
            quantization_device=q_device,
            return_device=return_device,
            modules_to_not_convert=["word_embeddings"],
        )
        tr_quant = SDNQConfig(
            weights_dtype="int8",
            quantization_device=q_device,
            return_device=return_device,
        )
        hub = dict(cache_dir=_cache_dir, local_files_only=_lfo)
        pipe.load_components(names=["text_encoder"], dtype=dtype, quantization_config=te_quant, **hub)
        pipe.load_components(names=["transformer"], dtype=dtype, quantization_config=tr_quant, **hub)
        pipe.load_components(names=["connector", "vae", "tokenizer", "scheduler"], dtype=dtype, **hub)

        # 2048x2048 (the recommended size) decodes in one pass on 24 GB, but
        # tiling keeps larger sizes and smaller cards safe.
        pipe.vae.enable_tiling()

        if gfx_device == "mps":
            pipe.to("mps")
        elif on_gpu:
            pipe.connector.to("cuda")
            pipe.vae.to("cuda")
        else:
            # Whole-component moves only: group offload can strand SDNQ
            # scale/zero-point buffers on the CPU (see Qwen-Image 2.1 plugin).
            from diffusers import ComponentsManager
            manager = ComponentsManager()
            for name in ("text_encoder", "transformer", "connector", "vae"):
                manager.add(name, getattr(pipe, name))
            manager.enable_auto_cpu_offload(device="cuda")
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from diffusers import ClassifierFreeGuidance

        pipe = pipe_obj["pipe"]

        # Width and height must be multiples of 16.
        width = max(16, int(inputs.width) // 16 * 16)
        height = max(16, int(inputs.height) // 16 * 16)

        seed = inputs.seed
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        # The negative branch is zero conditioning, as in the original pipeline.
        guidance = float(inputs.guidance)
        pipe.update_components(guider=ClassifierFreeGuidance(
            guidance_scale=max(guidance, 1.0),
            use_original_formulation=True,
            enabled=guidance > 1.0,
        ))

        # Modular pipelines take no callback_on_step_end, so report progress
        # from the scheduler step, which runs once per denoising step.
        scheduler = pipe.scheduler
        original_step = scheduler.step
        progress_fn = inputs.progress_fn
        total = max(1, inputs.steps)
        counter = [0]

        def _step(*args, **kwargs):
            out = original_step(*args, **kwargs)
            counter[0] += 1
            if progress_fn is not None:
                progress_fn(counter[0], total)
            return out

        scheduler.step = _step
        self.set_phase(inputs, "Generating")
        try:
            image = pipe(
                prompt=inputs.prompt,
                height=height,
                width=width,
                num_inference_steps=inputs.steps,
                generator=generator,
                output="images",
            )[0]
        finally:
            scheduler.step = original_step

        # Output is always RGBA. Keep alpha only when the prompt asked for a
        # transparent background; snap its near-zero background to a clean
        # cutout. Otherwise drop the channel so stray low alpha can't make an
        # opaque design see-through in the VSE.
        prompt_head = inputs.prompt.lstrip().lower()
        if any(prompt_head.startswith(p) for p in _TRANSPARENT_PHRASES):
            r, g, b, a = image.split()
            a = a.point(lambda v: 0 if v < 8 else v)
            from PIL import Image as _PILImage
            return _PILImage.merge("RGBA", (r, g, b, a))
        return image.convert("RGB")
