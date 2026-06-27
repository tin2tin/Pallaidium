"""Multi-image editing via Qwen-Image-Edit-2511 (SDNQ uint4 weights)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, load_first_frame


class QwenImageEditPlugin(ModelPlugin):
    MODEL_ID     = "Qwen/Qwen-Image-Edit-2511"
    QUANT_ID     = "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32"
    DISPLAY_NAME = "Image: Qwen Image Edit (multi-image)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Multi-image instruction editing via Qwen-Image-Edit-2511 SDNQ uint4"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.MULTI_IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.MULTI_IMAGES,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=4, max_multi_images=3)
    REQUIRED_PACKAGES          = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint           = False
    supports_img2img           = True
    requires_input_strip       = True
    uses_standard_input_strip  = False

    def load(self, prefs, scene, **kw):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

        _cache_dir = prefs.hf_cache_dir or None
        dtype = torch.bfloat16
        print(f"Loading {self.QUANT_ID}…")

        # Import sdnq before from_pretrained so it can register its quantization
        # backend with transformers — without this transformers warns and skips it.
        _apply_sdnq = None
        _triton_ok = False 
        try:
            from sdnq import SDNQConfig  # noqa: F401 — registers quantization backend
            from sdnq.common import use_torch_compile as _triton_ok
            from sdnq.loader import apply_sdnq_options_to_model as _apply_sdnq
            _triton_ok = _triton_ok and (torch.cuda.is_available() or torch.xpu.is_available())
        except ImportError:
            pass

        _lfo = prefs.local_files_only
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.QUANT_ID,
            subfolder="text_encoder",
            dtype=dtype,
            device_map="cpu",
            cache_dir=_cache_dir,
            local_files_only=_lfo,
        )
        # Heavy SDNQ transformer: diffusers' from_pretrained(device_map="cpu")
        # faults with EXCEPTION_ACCESS_VIOLATION when materializing safetensors
        # shards on the queue worker thread (torch_cpu storage.__getitem__ is not
        # safe off the main thread in Blender's bundled build — diffusers does not
        # honor HF_DEACTIVATE_ASYNC_LOAD, unlike transformers, which is why the
        # text_encoder above loads fine off-thread but this did not). SDNQ's own
        # synchronous loader avoids that path — same fix the LTX-2.3 plugins use.
        import os as _os
        from huggingface_hub import snapshot_download as _snap
        from sdnq.loader import load_sdnq_model as _load_sdnq
        _sdnq_transformer_path = _os.path.join(
            _snap(self.QUANT_ID, cache_dir=_cache_dir, local_files_only=_lfo),
            "transformer",
        )
        transformer = _load_sdnq(
            model_path=_sdnq_transformer_path,
            model_cls=QwenImageTransformer2DModel,
            dtype=dtype, device="cpu",
        )

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.MODEL_ID,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=dtype,
            cache_dir=_cache_dir,
            local_files_only=_lfo,
        )

        if _apply_sdnq and _triton_ok:
            pipe.transformer  = _apply_sdnq(pipe.transformer,  use_quantized_matmul=True)
            pipe.text_encoder = _apply_sdnq(pipe.text_encoder, use_quantized_matmul=True)

        # Lightning distillation LoRA → 4-step inference (~10x fewer steps than
        # the stock 40-50). It is CFG-distilled, so generate() MUST run with true
        # CFG disabled (true_cfg_scale=1.0). Kept as an adapter — never fused,
        # since the base transformer is uint4-quantized. On download failure we
        # fall back to full-step inference so the plugin still works offline.
        _lora_names, _lora_weights = [], []
        _lightning_ok = False
        try:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Edit-2511-Lightning",
                weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                adapter_name="lightning",
                cache_dir=_cache_dir,
                local_files_only=_lfo,
            )
            _lora_names.append("lightning")
            _lora_weights.append(1.0)
            _lightning_ok = True
        except Exception as e:
            print(f"Qwen Image Edit: Lightning LoRA unavailable ({e}); "
                  "falling back to full-step inference.")

        # Apply user LoRAs on top of Lightning
        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
                _lora_names.append(name)
                _lora_weights.append(item.weight_value)

        if _lora_names:
            pipe.set_adapters(_lora_names, adapter_weights=_lora_weights)

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
            # Enable VAE tiling on the normal path too (not just low-VRAM): the
            # final full-size decode of a multi-image composition allocates a
            # large activation tensor that can fault in c10 at the end of the
            # run. Tiling caps that peak with negligible quality/speed cost.
            pipe.vae.enable_tiling()
        return {"pipe": pipe, "converter": None, "refiner": None,
                "preprocessor": None, "lightning": _lightning_ok}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        for attr, action in [
            ("qwen_strip_1", "qwen_select1"),
            ("qwen_strip_2", "qwen_select2"),
            ("qwen_strip_3", "qwen_select3"),
        ]:
            row = col.row(align=True)
            row.prop_search(
                vse_scene, attr, vse_scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = action
        return True

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import os
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        qwen_images = []
        if inputs.image is not None:
            qwen_images.append(inputs.image)
        for path_attr in ["qwen_strip_1_path", "qwen_strip_2_path", "qwen_strip_3_path"]:
            path = getattr(scene, path_attr, "")
            if path and os.path.isfile(path):
                img = load_first_frame(path)
                if img is not None:
                    qwen_images.append(img)

        if not qwen_images:
            raise ValueError(
                "Qwen Image Edit requires at least one image input. "
                "Select an image or movie strip and set Input to 'Strip', "
                "or pick strips using the image pickers below."
            )

        # Lightning LoRA is CFG-distilled → disable true CFG (this also skips the
        # negative-prompt pass entirely, ~2x). Fall back to 4.0 if Lightning is off.
        _true_cfg = 1.0 if pipe_obj.get("lightning") else 4.0
        self.set_phase(inputs, "Generating")
        with torch.inference_mode():
            return pipe(
                image=qwen_images,
                prompt=inputs.prompt,
                generator=generator,
                true_cfg_scale=_true_cfg,
                negative_prompt=inputs.neg_prompt + " ",
                num_inference_steps=inputs.steps,
                height=inputs.height,
                width=inputs.width,
                num_images_per_prompt=1,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
