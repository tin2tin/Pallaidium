"""High-quality text-to-image via ERNIE-Image with SDNQ quantized weights."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device


class ErniePlugin(ModelPlugin):
    MODEL_ID     = "baidu/ERNIE-Image"
    SDNQ_4BIT_ID = "OzzyGT/ERNIE_Image_sdnq_dynamic_int4"
    SDNQ_8BIT_ID = "OzzyGT/ERNIE_Image_sdnq_dynamic_int8"
    DISPLAY_NAME = "Image: ERNIE-Image"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "High-quality text-to-image via ERNIE-Image (SDNQ quantized)"

    SDNQ_BITS           = 4     # 4 or 8
    USE_PROMPT_ENHANCER = True

    INPUTS      = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=50, guidance=4.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "sdnq"]
    supports_inpaint        = False
    supports_img2img        = False
    uses_standard_input_strip = False

    def load(self, prefs, scene, **kw):
        import torch
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("5.0.0"):
            raise RuntimeError(
                f"ERNIE-Image requires transformers>=5.0.0 "
                f"(installed: {transformers.__version__}). "
                "Upgrade with: pip install 'transformers>=5.0.0'"
            )
        from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel
        from transformers import Ministral3ForCausalLM, Mistral3Model
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        _cache_dir = prefs.hf_cache_dir or None
        sdnq_path = self.SDNQ_4BIT_ID if self.SDNQ_BITS == 4 else self.SDNQ_8BIT_ID
        dtype = torch.bfloat16

        print(f"Loading ERNIE-Image ({self.SDNQ_BITS}-bit SDNQ) from {sdnq_path}…")

        # Do NOT use device_map="cpu" — enable_model_cpu_offload() manages
        # devices automatically and device_map="cpu" causes a device mismatch
        # warning with newer Transformers/Accelerate versions.
        print("Loading transformer...")
        transformer = ErnieImageTransformer2DModel.from_pretrained(
            sdnq_path, subfolder="transformer", torch_dtype=dtype, cache_dir=_cache_dir,
        )
        print("Loading text encoder...")
        text_encoder = Mistral3Model.from_pretrained(
            sdnq_path, subfolder="text_encoder", torch_dtype=dtype, cache_dir=_cache_dir,
        )
        if self.USE_PROMPT_ENHANCER:
            print("Loading prompt enhancer...")
            prompt_enhancer = Ministral3ForCausalLM.from_pretrained(
                sdnq_path, subfolder="pe", torch_dtype=dtype, cache_dir=_cache_dir,
            )
        else:
            prompt_enhancer = None

        print("Building pipeline...")
        pipe = ErnieImagePipeline.from_pretrained(
            self.MODEL_ID,
            transformer=transformer,
            text_encoder=text_encoder,
            pe=prompt_enhancer,
            torch_dtype=dtype,
            cache_dir=_cache_dir,
        )

        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            print("Applying SDNQ quantized matmul optimizations...")
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            if self.USE_PROMPT_ENHANCER and pipe.pe is not None:
                pipe.pe = apply_sdnq_options_to_model(pipe.pe, use_quantized_matmul=True)

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()

        return {
            "pipe":   pipe,
            "use_pe": self.USE_PROMPT_ENHANCER,
            "converter": None, "refiner": None, "preprocessor": None,
        }

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import warnings

        pipe   = pipe_obj["pipe"]
        use_pe = pipe_obj["use_pe"]
        seed   = inputs.seed
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        # Suppress the spurious device-mismatch warning from transformers' generate()
        # pre-flight check: with enable_model_cpu_offload() the PE model's .device
        # reports CPU but accelerate moves it to CUDA at runtime — it works fine.
        self.set_phase(inputs, "Generating")
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*input_ids.*being on a device type different.*",
                category=UserWarning,
            )
            result = pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
                use_pe=use_pe,
                callback_on_step_end=self.step_callback(inputs),
            )
        return result.images[0]
