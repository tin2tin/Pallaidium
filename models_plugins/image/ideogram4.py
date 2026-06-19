"""Text-to-image via Ideogram 4.

Uses official ideogram-ai weights or pre-quantized SDNQ weights loaded
through the standard diffusers Ideogram4Pipeline.

Supported  : prompt, height/width (up to 2048), steps, guidance_scale, seed,
             local prompt upsampling (requires 'outlines' package),
             LoRA (requires diffusers >= PR#13921 with Ideogram4LoraLoaderMixin).
Unsupported: negative_prompt  - model uses a fixed unconditional_transformer.
             img2img / inpaint - not in Ideogram4Pipeline.
             reference images  - not supported.
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
    INPUTS       = InputSpec.PROMPT | InputSpec.HF_TOKEN | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=20, guidance=4.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "outlines"]
    supports_inpaint      = False
    supports_img2img      = False

    _ENHANCER = "diffusers/qwen3-vl-8b-instruct-lm-head"

    @staticmethod
    def _inject_lora_mixin(pipe):
        """Give *pipe* the standard load_lora_weights / set_adapters API.

        Uses the same LoraBaseMixin infrastructure that Klein/Flux2 rely on.
        Once diffusers ships Ideogram4LoraLoaderMixin this becomes a no-op.
        """
        from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
        from diffusers.loaders.lora_pipeline import TRANSFORMER_NAME
        from diffusers.utils import USE_PEFT_BACKEND
        _LOW_CPU = False
        try:
            from diffusers.loaders.lora_pipeline import _LOW_CPU_MEM_USAGE_DEFAULT_LORA
            _LOW_CPU = _LOW_CPU_MEM_USAGE_DEFAULT_LORA
        except ImportError:
            pass

        class _Ideogram4LoraLoader(LoraBaseMixin):
            _lora_loadable_modules = ["transformer"]
            transformer_name = TRANSFORMER_NAME

            @classmethod
            def lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
                weight_name = kwargs.pop("weight_name", None)
                use_safetensors = kwargs.pop("use_safetensors", None)
                local_files_only = kwargs.pop("local_files_only", None)
                cache_dir = kwargs.pop("cache_dir", None)
                force_download = kwargs.pop("force_download", False)
                proxies = kwargs.pop("proxies", None)
                token = kwargs.pop("token", None)
                revision = kwargs.pop("revision", None)
                subfolder = kwargs.pop("subfolder", None)
                return_lora_metadata = kwargs.pop("return_lora_metadata", False)
                allow_pickle = False
                if use_safetensors is None:
                    use_safetensors = True
                    allow_pickle = True
                state_dict, metadata = _fetch_state_dict(
                    pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
                    weight_name=weight_name,
                    use_safetensors=use_safetensors,
                    local_files_only=local_files_only,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent={"file_type": "attn_procs_weights", "framework": "pytorch"},
                    allow_pickle=allow_pickle,
                )
                state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}
                if not any(k.startswith("transformer.") for k in state_dict):
                    state_dict = {f"transformer.{k}": v for k, v in state_dict.items()}
                return (state_dict, metadata) if return_lora_metadata else state_dict

            def load_lora_weights(self, pretrained_model_name_or_path_or_dict,
                                  adapter_name=None, hotswap=False, **kwargs):
                if not USE_PEFT_BACKEND:
                    raise ValueError("PEFT backend is required for LoRA.")
                low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU)
                if isinstance(pretrained_model_name_or_path_or_dict, dict):
                    pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()
                kwargs["return_lora_metadata"] = True
                state_dict, metadata = self.lora_state_dict(
                    pretrained_model_name_or_path_or_dict, **kwargs)
                is_correct = all("lora" in k for k in state_dict)
                if not is_correct:
                    raise ValueError("Invalid LoRA checkpoint.")
                self.transformer.load_lora_adapter(
                    state_dict, adapter_name=adapter_name, metadata=metadata,
                    _pipeline=self, low_cpu_mem_usage=low_cpu_mem_usage,
                    hotswap=hotswap,
                )

        pipe.__class__ = type(
            pipe.__class__.__name__,
            (_Ideogram4LoraLoader, pipe.__class__),
            {},
        )

    def load(self, prefs, scene, **kw):
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
        mode = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")
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

        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            if not hasattr(pipe, "load_lora_weights"):
                self._inject_lora_mixin(pipe)
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            names, weights = [], []
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
            pipe.set_adapters(names, adapter_weights=weights)

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