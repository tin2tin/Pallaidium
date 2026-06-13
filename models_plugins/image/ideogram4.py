"""Text-to-image via Ideogram 4.

Uses official ideogram-ai weights or pre-quantized SDNQ weights loaded 
through the standard diffusers Ideogram4Pipeline.

Supported  : prompt, height/width (up to 2048), steps, guidance_scale, seed,
             local prompt upsampling (requires 'outlines' package).
Unsupported: negative_prompt  - model uses a fixed unconditional_transformer.
             img2img / inpaint - not in Ideogram4Pipeline.
             reference images  - not supported.
             LoRA              - not documented / not supported.
"""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device


class Ideogram4Plugin(ModelPlugin):
    # Recommended default: Official NF4 model for higher speed, low VRAM, and clean, artifact-free output
    MODEL_ID     = "ideogram-ai/ideogram-4-nf4-diffusers"
    # Fallback to original FP8/SDNQ weights if preferred:
    # MODEL_ID     = "Disty0/Ideogram-4-SDNQ-FP8"
    # MODEL_ID     = "Disty0/Ideogram-4-SDNQ-4bit-dynamic-hadamard"
    
    DISPLAY_NAME = "Image: Ideogram 4"
    DESCRIPTION  = (
        "Text-to-image via Ideogram 4 (~10.5 GB NF4 / ~17.9 GB FP8). "
        "Optional local prompt upsampling needs: pip install outlines."
    )
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.HF_TOKEN
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=20, guidance=4.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "outlines"]
    supports_inpaint      = False
    supports_img2img      = False

    _ENHANCER = "diffusers/qwen3-vl-8b-instruct-lm-head"

    def load(self, prefs, scene, **_kw):
        import torch
        #from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
        #from sdnq.common import use_torch_compile as triton_is_available
        #from sdnq.loader import apply_sdnq_options_to_model
        from diffusers import Ideogram4Pipeline

        from huggingface_hub import login
        if prefs.hugginface_token:
            try:
                login(token=prefs.hugginface_token, add_to_git_credential=True)
            except Exception as e:
                raise RuntimeError(f"HuggingFace login failed: {e}")

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

        # Enable FP8 MatMul for AMD, Intel ARC and Nvidia GPUs (only executed for SDNQ models to avoid crashes on official NF4/FP8 models)
        # if "SDNQ" in self.MODEL_ID and triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        #     pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        #     pipe.unconditional_transformer = apply_sdnq_options_to_model(pipe.unconditional_transformer, use_quantized_matmul=True)
        #     pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()
            # Enable built-in VAE tiling and slicing to reduce memory spikes during decoding
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            
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

        # Optional torch.compile block to speed up inference (can be enabled via user preferences)
        use_compile = getattr(prefs, "enable_torch_compile", False)
        if use_compile and torch.cuda.is_available():
            print("Ideogram4: Compiling transformer blocks...")
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
            pipe.unconditional_transformer = torch.compile(pipe.unconditional_transformer, mode="reduce-overhead", fullgraph=True)

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