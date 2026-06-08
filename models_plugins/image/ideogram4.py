"""Text-to-image via Ideogram 4 (uint4 SDNQ pre-quantized).

Uses vladmandic/Ideogram-4-sdnq-uint4-hadamard loaded through the standard
diffusers Ideogram4Pipeline.  Weights are pre-quantized (uint4 + Hadamard
rotation) at ~17.9 GB; no runtime quantization config required.

Supported  : prompt, height/width (up to 2048), steps, guidance_scale, seed,
             local prompt upsampling (requires 'outlines' package).
Unsupported: negative_prompt  — model uses a fixed unconditional_transformer.
             img2img / inpaint — not in Ideogram4Pipeline.
             reference images  — not supported.
             LoRA              — not documented / not supported.
"""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device


class Ideogram4Plugin(ModelPlugin):
    MODEL_ID     = "Disty0/Ideogram-4-SDNQ-4bit-dynamic-hadamard"
    DISPLAY_NAME = "Image: Ideogram 4"
    DESCRIPTION  = (
        "Text-to-image via Ideogram 4 (uint4 SDNQ, ~17.9 GB). "
        "Optional local prompt upsampling needs: pip install outlines."
    )
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.RESOLUTION, UISection.FRAMES,
        UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=40, guidance=7.0, width=1024, height=1024)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "outlines"]
    supports_inpaint      = False
    supports_img2img      = False

    _ENHANCER = "diffusers/qwen3-vl-8b-instruct-lm-head"

    def load(self, prefs, scene, **_kw):
        import torch
        import sdnq  # registers "sdnq" quantizer with diffusers/transformers before from_pretrained
        from diffusers import Ideogram4Pipeline

        _cache_dir = prefs.hf_cache_dir or None
        dtype      = torch.bfloat16
        print(f"Loading {self.MODEL_ID}…")

        _lfo = prefs.local_files_only
        load_kwargs = dict(torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo)

        prompt_upsampling = getattr(scene, "ideogram_prompt_upsampling", False)
        if prompt_upsampling:
            try:
                from diffusers import Ideogram4PromptEnhancerHead
                print("Ideogram4: loading prompt enhancer head…")
                enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
                    self._ENHANCER, torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo,
                )
                load_kwargs["prompt_enhancer_head"] = enhancer_head
                print("Ideogram4: prompt enhancer head attached")
            except Exception as e:
                print(f"Ideogram4: prompt enhancer head failed ({e}); skipping")

        pipe = Ideogram4Pipeline.from_pretrained(self.MODEL_ID, **load_kwargs)

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()
            # Ideogram4Pipeline calls text_encoder sub-modules directly inside
            # encode_prompt, bypassing the model_cpu_offload hook on text_encoder.
            # Patch encode_prompt to move text_encoder to CUDA for the duration of
            # the call, then return it to CPU so the normal offload sequence for
            # transformer / vae is unaffected.
            import types
            _orig_encode_prompt = pipe.__class__.encode_prompt
            def _encode_with_te_offload(self, *args, **kwargs):
                self.text_encoder.to("cuda")
                try:
                    return _orig_encode_prompt(self, *args, **kwargs)
                finally:
                    self.text_encoder.to("cpu")
                    torch.cuda.empty_cache()
            pipe.encode_prompt = types.MethodType(_encode_with_te_offload, pipe)

        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def draw_post_enhance_ui(self, col, context) -> None:
        col2 = col.column(heading="Prompt Upsampling", align=True)
        row = col2.row()
        row.prop(context.scene, "ideogram_prompt_upsampling", text="Enable")

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        seed = inputs.seed
        if seed != 0:
            exec_device = pipe_obj["pipe"]._execution_device
            generator = torch.Generator(exec_device).manual_seed(seed)
        else:
            generator = None

        prompt_upsampling = getattr(scene, "ideogram_prompt_upsampling", False)

        self.set_phase(inputs, "Generating")
        result = pipe_obj["pipe"](
            prompt=inputs.prompt,
            height=inputs.height,
            width=inputs.width,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            guidance_schedule=None,
            prompt_upsampling=prompt_upsampling,
            max_sequence_length=2048,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
        return result
