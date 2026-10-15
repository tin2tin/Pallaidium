"""Text-to-image and multi-reference generation via Qwen-Image 2.1 Turbo (SDNQ 8-bit), with up to 9 reference image slots."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


# The Turbo checkpoint is distilled to a fixed 8-step schedule, stored as
# "sample_sigmas" in its model_index.json. Only diffusers with PR #14950 reads
# that key; older builds fall back to linspace(1, 1/n) and produce mush, so the
# schedule is also passed explicitly whenever the step count matches.
_TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]

class QwenImage21Plugin(ModelPlugin):
    MODEL_ID     = "OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit"
    DISPLAY_NAME = "Qwen-Image 2.1 Turbo"
    DESCRIPTION  = "Fast text-to-image via Qwen-Image 2.1 Turbo (8-step, SDNQ 8-bit) with up to 9 reference images"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED,
        UISection.LORA,
    ]
    # Turbo is distilled to 8 steps and sampled CFG-free (true_cfg_scale=1.0). The
    # Guidance slider maps straight onto true_cfg_scale and only takes effect
    # once > 1 and a negative prompt is also set (doubles the cost per step).
    PARAMS            = ParamSpec(steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint       = False
    # QwenImage21Pipeline has no strength param, no inpaint variant, and no
    # separate img2img class — image= is reference conditioning (VLM context +
    # VAE latent tokens), not a denoise blend, so there is nothing for the
    # Image Strength slider to control.

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import QwenImage21Pipeline
        # Registers SDNQ's quantizer so from_pretrained recognizes
        # quant_method="sdnq" in the transformer/text_encoder configs and wraps
        # the int8 (Hadamard-rotated) weights with the dequantizer. Without this
        # import the quantized tensors load raw → noise. Needs sdnq >= 0.2.2.
        from sdnq import SDNQConfig  # noqa: F401

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        dtype = torch.bfloat16
        print(f"Loading {self.MODEL_ID}…")

        # Every component (incl. the SDNQ transformer + Qwen3-VL text encoder)
        # ships in the one repo with its quantization config baked in, so a
        # single from_pretrained loads real tensors — no bnb on-the-fly quant.
        pipe = QwenImage21Pipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )

        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
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
                print(f"Qwen-Image 2.1 Turbo: user LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
            pipe.set_adapters(names, adapter_weights=weights)
            print(f"Qwen-Image 2.1 Turbo: active adapters={names} weights={weights}")

        # Large outputs (the card samples at 1696x2528) spike VRAM on decode.
        pipe.vae.enable_tiling()

        # NOT enable_group_offload: leaf-level streaming can leave SDNQ side
        # buffers (scale / zero_point) on CPU → cross-device dequant → noise.
        # enable_model_cpu_offload moves whole components together.
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene. Reuses the
        # klein_strip_N / klein_visible_strips scene props and add/remove
        # operators shared by every up-to-9-reference plugin in this codebase
        # (FLUX.2 Klein 9B/KV/Schematic) rather than duplicating that plumbing.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        for i in range(1, scene.klein_visible_strips + 1):
            row = col.row(align=True)
            row.prop_search(
                vse_scene, f"klein_strip_{i}", vse_scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"klein_select{i}"
            if i == scene.klein_visible_strips and scene.klein_visible_strips < 9:
                if scene.klein_visible_strips > 3:
                    row.operator("object.klein_hide_strip", text="", icon="REMOVE").strip_index = i
                row.operator("object.klein_add_strip", text="", icon="ADD")
        return True

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        seed = inputs.seed
        # CPU generator, as on the model card: with offloading the pipeline
        # device reports CPU.
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        from PIL import Image as _PILImage
        ref_images = []
        for attr in (f"klein_strip_{i}_path" for i in range(1, 10)):
            path = getattr(scene, attr, None)
            if path:
                try:
                    img = _PILImage.open(path).convert("RGB")
                    ref_images.append(img)
                    print(f"Qwen-Image 2.1 Turbo ref loaded: {attr} = '{path}' {img.size}")
                except Exception as e:
                    print(f"Qwen-Image 2.1 Turbo ref failed to open '{path}': {e}")
        # The active input (image/video/scene frame) is always the first
        # reference, ahead of the named ref slots, regardless of txt2img/img2img
        # mode — matching the Klein plugins.
        images = ([inputs.image.convert("RGB")] if inputs.image is not None else []) + ref_images
        print(f"Qwen-Image 2.1 Turbo: {len(images)} reference image(s) loaded, mode={inputs.mode}")

        # Ignored upstream unless paired with true_cfg_scale > 1 — passing "" would
        # still flip do_true_cfg on and log a wasted negative pass.
        neg_prompt = inputs.neg_prompt or None

        self.set_phase(inputs, "Generating")
        pipe_key = "converter" if (inputs.mode == "img2img" and inputs.image is not None) else "pipe"
        return pipe_obj[pipe_key](
            prompt=inputs.prompt,
            image=images if images else None,
            negative_prompt=neg_prompt,
            true_cfg_scale=inputs.guidance,
            num_inference_steps=inputs.steps,
            sigmas=_TURBO_SIGMAS if inputs.steps == len(_TURBO_SIGMAS) else None,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
