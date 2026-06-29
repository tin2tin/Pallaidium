"""Multi-image generation via OmniGen (Shitao/OmniGen-v1-diffusers)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, find_strip_by_name, get_strip_path, load_first_frame, load_strip_as_pil


class OmniGenPlugin(ModelPlugin):
    MODEL_ID     = "Shitao/OmniGen-v1-diffusers"
    DISPLAY_NAME = "OmniGen (multi-image)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Multi-image / instruction-based generation via OmniGen"

    INPUTS       = InputSpec.PROMPT | InputSpec.MULTI_IMAGE
    UI_SECTIONS  = [
        UISection.TRIPLE_PROMPT_IMG,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS       = ParamSpec(steps=50, guidance=3.0, max_multi_images=3)
    REQUIRED_PACKAGES          = ["torch", "diffusers"]
    supports_inpaint           = False
    supports_img2img           = False
    uses_standard_input_strip  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import OmniGenPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print("Loading OmniGen…")
        pipe = OmniGenPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=_cache_dir, local_files_only=prefs.local_files_only)
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        if vse_scene.sequence_editor is None:
            return True
        for idx in range(1, 4):
            col.prop(scene, f"omnigen_prompt_{idx}", text="", icon="ADD")
            row = col.row(align=True)
            row.prop_search(
                vse_scene, f"omnigen_strip_{idx}", vse_scene.sequence_editor, "strips",
                text="", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"omni_select{idx}"
        return True

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0
            else (torch.Generator(device=gfx_device).manual_seed(seed) if seed != 0 else None)
        )

        omnigen_images = []
        prompt = getattr(scene, "omnigen_prompt_1", inputs.prompt) or inputs.prompt
        for idx, strip_attr in enumerate(["omnigen_strip_1", "omnigen_strip_2", "omnigen_strip_3"], start=1):
            prompt_attr = f"omnigen_prompt_{idx}"
            if idx > 1:
                prompt += getattr(scene, prompt_attr, "") or ""
            strip_name = getattr(scene, strip_attr, None)
            if strip_name:
                strip = find_strip_by_name(scene, strip_name)
                if strip:
                    omnigen_images.append(load_strip_as_pil(strip))
                    prompt += f" <img><|image_{idx}|></img> "

        img_size = bool(omnigen_images)
        self.set_phase(inputs, "Generating")
        return pipe(
            prompt=prompt,
            input_images=omnigen_images or None,
            img_guidance_scale=getattr(scene, "img_guidance_scale", 1.6),
            use_input_image_size_as_output=img_size,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
