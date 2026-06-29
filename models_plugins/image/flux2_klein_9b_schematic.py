"""Image → schematic map via FLUX.2 Klein 9B + mode-specific LoRA (depth, normal, pose, seg)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram

_LORA_REPO = "nomadoor/flux-2-klein-9B-schematic-lora"
_LORA_FILES = {
    "DEPTH":      "loras/flux2-klein-schematic-relative-depth-lora.safetensors",
    "NORMAL":     "loras/flux2-klein-schematic-surface-normal-lora.safetensors",
    "BODY_POSE":  "loras/flux2-klein-schematic-body-pose-lora.safetensors",
    "FULL_POSE":  "loras/flux2-klein-schematic-full-pose-lora.safetensors",
    "BINARY_SEG": "loras/flux2-klein-schematic-binary-segmentation-lora.safetensors",
    "AMODAL_SEG": "loras/flux2-klein-schematic-amodal-segmentation-lora.safetensors",
}
_TRIGGER_PROMPTS = {
    "DEPTH":      "Generate a relative depth map of the input image.",
    "NORMAL":     "Generate a surface normal map of the input image.",
    "BODY_POSE":  "Generate a body pose map of all visible people in the input image.",
    "FULL_POSE":  "Generate a full pose map of all visible people in the input image.",
    "BINARY_SEG": "Generate a binary segmentation mask of {target} in the input image.",
    "AMODAL_SEG": "Generate an amodal segmentation mask of {target} in the input image.",
}


class Flux2Klein9BSchematicPlugin(ModelPlugin):
    MODEL_ID     = "nomadoor/flux-2-klein-9B-schematic-lora"
    DISPLAY_NAME = "FLUX.2 Klein 9B Schematic"
    DESCRIPTION  = "Transform images into schematic maps via Klein 9B (depth, normal, pose, segmentation)"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=20, guidance=5.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = False
    supports_img2img  = True
    requires_input_strip       = True
    requires_no_style          = True
    preserve_image_dimensions  = True

    _BASE_PIPELINE = "ModelsLab/FLUX.2-klein-9B"
    _TRANSFORMER   = "OzzyGT/flux2_klein_9B_bnb_4bit_transformer"
    _TEXT_ENCODER  = "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder"

    def on_model_selected(self, scene, context):
        mode   = getattr(scene, "klein_schematic_mode", "DEPTH")
        target = (getattr(scene, "klein_schematic_target", "person") or "person").strip()
        trigger = _TRIGGER_PROMPTS[mode].format(target=target)
        current = scene.generate_movie_prompt or ""
        has_trigger = any(
            current.startswith(t.split("{")[0]) for t in _TRIGGER_PROMPTS.values()
        )
        if not has_trigger:
            scene.generate_movie_prompt = trigger + (" " + current if current else "")

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM

        _cache_dir = prefs.hf_cache_dir or None
        mode = getattr(scene, "klein_schematic_mode", "DEPTH")
        print(f"Loading {self.MODEL_ID} (schematic mode: {mode})…")

        _lfo = prefs.local_files_only
        try:
            from transformers import BitsAndBytesConfig
            _bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception:
            _bnb4 = None
        _bnb_kw = {"quantization_config": _bnb4} if _bnb4 is not None else {}
        transformer = Flux2Transformer2DModel.from_pretrained(
            self._TRANSFORMER, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
            local_files_only=_lfo, **_bnb_kw,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self._TEXT_ENCODER, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
            local_files_only=_lfo, **_bnb_kw,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            self._BASE_PIPELINE,
            transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        from huggingface_hub import hf_hub_download
        lora_path = hf_hub_download(
            repo_id=_LORA_REPO,
            filename=_LORA_FILES[mode],
            cache_dir=_cache_dir,
            local_files_only=_lfo,
        )
        pipe.load_lora_weights(lora_path, adapter_name="schematic")
        pipe.set_adapters(["schematic"], adapter_weights=[1.0])

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None,
                "schematic_mode": mode}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        row = col.row()
        row.enabled = False
        try:
            row.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        col.prop(scene, "klein_schematic_mode", text="Mode")
        if getattr(scene, "klein_schematic_mode", "DEPTH") in ("BINARY_SEG", "AMODAL_SEG"):
            col.prop(scene, "klein_schematic_target", text="Target")
        return True

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from PIL import Image as _PIL

        if inputs.image is None:
            raise ValueError("FLUX.2 Klein Schematic requires an image strip as input.")

        cached_mode  = pipe_obj.get("schematic_mode", "DEPTH")
        current_mode = getattr(scene, "klein_schematic_mode", "DEPTH")
        if cached_mode != current_mode:
            print(
                f"WARNING: Schematic mode changed from '{cached_mode}' to '{current_mode}' "
                "since model was loaded — reload the model to apply the new LoRA."
            )

        img = inputs.image.convert("RGB")
        w, h = img.size
        src = img

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        self.set_phase(inputs, "Generating")
        result = pipe_obj["pipe"](
            prompt=inputs.prompt,
            image=src,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=h,
            width=w,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]

        if result.size != (w, h):
            result = result.resize((w, h), _PIL.LANCZOS)
        return result
