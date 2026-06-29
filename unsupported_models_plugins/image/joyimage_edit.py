"""Instruction-guided image editing via JoyAI-Image-Edit (spatial editing support)."""

import gc
import time

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, bench_print

# ── Scene props registered at import time ────────────────────────────────────
# draw_custom_ui is called on every panel redraw, long before load() runs,
# so the props must exist from the moment this module is imported.
try:
    import bpy as _bpy

    if not hasattr(_bpy.types.Scene, "joyimage_spatial_mode"):
        _bpy.types.Scene.joyimage_spatial_mode = _bpy.props.EnumProperty(
            name="Spatial Mode",
            items=[
                ("general", "General Edit",   "Instruction-guided image editing"),
                ("move",    "Object Move",     "Move object into a red-box region"),
                ("rotate",  "Object Rotation", "Rotate object to a canonical view"),
                ("camera",  "Camera Control",  "Shift camera viewpoint, keep 3D scene"),
            ],
            default="general",
        )
    if not hasattr(_bpy.types.Scene, "joyimage_object"):
        _bpy.types.Scene.joyimage_object = _bpy.props.StringProperty(
            name="Object",
            description="Name of the object to move/rotate",
            default="object",
        )
    if not hasattr(_bpy.types.Scene, "joyimage_rotate_view"):
        _bpy.types.Scene.joyimage_rotate_view = _bpy.props.EnumProperty(
            name="View",
            items=[
                ("front",       "Front",       ""),
                ("right",       "Right",       ""),
                ("left",        "Left",        ""),
                ("rear",        "Rear",        ""),
                ("front right", "Front Right", ""),
                ("front left",  "Front Left",  ""),
                ("rear right",  "Rear Right",  ""),
                ("rear left",   "Rear Left",   ""),
            ],
            default="front",
        )
    if not hasattr(_bpy.types.Scene, "joyimage_yaw"):
        _bpy.types.Scene.joyimage_yaw = _bpy.props.FloatProperty(
            name="Yaw (°)", default=0.0, min=-180.0, max=180.0, step=10,
        )
    if not hasattr(_bpy.types.Scene, "joyimage_pitch"):
        _bpy.types.Scene.joyimage_pitch = _bpy.props.FloatProperty(
            name="Pitch (°)", default=0.0, min=-90.0, max=90.0, step=10,
        )
    if not hasattr(_bpy.types.Scene, "joyimage_zoom"):
        _bpy.types.Scene.joyimage_zoom = _bpy.props.EnumProperty(
            name="Zoom",
            items=[
                ("unchanged", "Unchanged", ""),
                ("in",        "In",        ""),
                ("out",       "Out",       ""),
            ],
            default="unchanged",
        )
except Exception:
    pass


def _flush(cuda: bool = True) -> None:
    gc.collect()
    if cuda:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


class JoyImageEditPlugin(ModelPlugin):
    MODEL_ID     = "jdopensource/JoyAI-Image-Edit-Diffusers"
    DISPLAY_NAME = "JoyAI Image Edit (spatial)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = (
        "Instruction-guided image editing with object move, object rotation, "
        "and camera control via JoyAI-Image-Edit"
    )

    INPUTS      = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.IMAGE_STRIP,
        UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS = ParamSpec(steps=40, guidance=4.0)

    # diffusers must be installed from git until >0.38.0 is released:
    #   pip install git+https://github.com/huggingface/diffusers.git
    REQUIRED_PACKAGES         = ["torch", "diffusers", "transformers"]
    supports_inpaint          = False
    supports_img2img          = True
    requires_input_strip      = True
    uses_standard_input_strip = False   # prevents strip_power row from appearing
    show_enhance              = False   # hides Quality / Speed / Upscale 4x row
    requires_main_thread_for_load = True  # safetensors crashes in worker threads on Windows

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import JoyImageEditPipeline

        _cache_dir = prefs.hf_cache_dir or None
        t_total = time.perf_counter()
        print(f"[JoyAI] ── Loading {self.MODEL_ID} ──")
        _flush()
        bench_print("[JoyAI] before load")

        dtype = torch.bfloat16
        is_cuda = torch.cuda.is_available()

        # transformers.core_model_loading internally spawns a ThreadPoolExecutor
        # and crashes in torch_cpu.dll on Windows when worker threads call
        # item() on meta-device tensors.  Replace submit() with a synchronous
        # version so _job/_materialize_copy run in the calling thread instead.
        import concurrent.futures as _cf
        _orig_submit = _cf.ThreadPoolExecutor.submit

        def _sync_submit(ex_self, fn, /, *args, **kwargs):
            fut = _cf.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:
                fut.set_exception(exc)
            return fut

        _cf.ThreadPoolExecutor.submit = _sync_submit

        # safetensors uses mmap on Windows; the mmap can return a null/invalid
        # base pointer, making every tensor offset land in unaddressable memory
        # (e.g. 0x3D50) and crashing in torch_cpu.dll.  Patch load_file to read
        # the file into bytes first and use the non-mmap load(bytes) path.
        import safetensors.torch as _st_module
        _orig_st_load_file = _st_module.load_file

        def _no_mmap_load_file(filename, device="cpu", **_kw):
            with open(filename, "rb") as _fh:
                tensors = _st_module.load(_fh.read())
            if device and device != "cpu":
                tensors = {k: v.to(device) for k, v in tensors.items()}
            return tensors

        _st_module.load_file = _no_mmap_load_file

        t0 = bench_print("[JoyAI] from_pretrained start")
        pipe = None
        is_quantized = False
        try:
            if is_cuda:
                try:
                    from diffusers import BitsAndBytesConfig as _DiffusersBnB
                    # Pass a plain BitsAndBytesConfig directly — PipelineQuantizationConfig
                    # fails the isinstance(quantization_config, BitsAndBytesConfig) check
                    # inside transformers when loading the text_encoder component.
                    bnb = _DiffusersBnB(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                    pipe = JoyImageEditPipeline.from_pretrained(
                        self.MODEL_ID,
                        quantization_config=bnb,
                        torch_dtype=dtype,
                        cache_dir=_cache_dir,
                    )
                    is_quantized = True
                    bench_print("[JoyAI] NF4 weights loaded", t0)
                except Exception as e:
                    print(f"[JoyAI] NF4 unavailable ({e}), falling back to bf16.")
                    pipe = None

            if pipe is None:
                t0 = bench_print("[JoyAI] bf16 from_pretrained start")
                pipe = JoyImageEditPipeline.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=dtype,
                    cache_dir=_cache_dir,
                )
                bench_print("[JoyAI] bf16 weights loaded", t0)
        finally:
            _st_module.load_file          = _orig_st_load_file
            _cf.ThreadPoolExecutor.submit = _orig_submit

        t0 = bench_print("[JoyAI] offload setup start")
        if is_cuda:
            if is_quantized:
                pipe.enable_model_cpu_offload()
                bench_print("[JoyAI] NF4 + CPU-offload installed", t0)
            else:
                pipe.enable_model_cpu_offload()
                bench_print("[JoyAI] CPU-offload hooks installed", t0)
        elif gfx_device == "mps":
            pipe.to("mps")
            bench_print("[JoyAI] moved to MPS", t0)
        else:
            pipe.enable_model_cpu_offload()
            bench_print("[JoyAI] CPU-offload (CPU-only)", t0)

        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            print("[JoyAI] VAE slicing enabled.")

        try:
            import torch
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        try:
            pipe.transformer.fuse_qkv_projections()
        except Exception:
            pass

        _flush()
        print(f"[JoyAI] ── Load finished in {time.perf_counter()-t_total:.1f}s ──")
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_prompt(self, scene, user_prompt: str) -> str:
        mode = getattr(scene, "joyimage_spatial_mode", "general")
        obj  = getattr(scene, "joyimage_object", "object").strip() or "object"

        if mode == "move":
            return f"Move the {obj} into the red box and finally remove the red box."
        if mode == "rotate":
            view = getattr(scene, "joyimage_rotate_view", "front")
            return f"Rotate the {obj} to show the {view} side view."
        if mode == "camera":
            yaw   = getattr(scene, "joyimage_yaw",   0)
            pitch = getattr(scene, "joyimage_pitch",  0)
            zoom  = getattr(scene, "joyimage_zoom",   "unchanged")
            return (
                "Move the camera.\n"
                f"- Camera rotation: Yaw {yaw:.1f}°, Pitch {pitch:.1f}°.\n"
                f"- Camera zoom: {zoom}.\n"
                "- Keep the 3D scene static; only change the viewpoint."
            )
        return user_prompt

    # ── Custom UI ─────────────────────────────────────────────────────────────

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene

        # Show the input enum locked to Strips (model requires an image input).
        # The helpers.py input_strips_updated callback enforces the value on switch;
        # we only display it here as read-only so the user knows why it is fixed.
        row = col.row()
        row.enabled = False
        row.prop(scene, "input_strips", text="Input")

        col.prop(scene, "joyimage_spatial_mode", text="Mode")

        mode = getattr(scene, "joyimage_spatial_mode", "general")
        if mode == "move":
            col.label(text="Mark the target region with a red box in the input image.")
            col.prop(scene, "joyimage_object", text="Object")
        elif mode == "rotate":
            col.prop(scene, "joyimage_object", text="Object")
            col.prop(scene, "joyimage_rotate_view", text="View")
        elif mode == "camera":
            row = col.row(align=True)
            row.prop(scene, "joyimage_yaw",   text="Yaw°")
            row.prop(scene, "joyimage_pitch",  text="Pitch°")
            col.prop(scene, "joyimage_zoom",   text="Zoom")

        return True

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]

        if inputs.image is None:
            raise ValueError("JoyAI-Image-Edit requires an input image.")

        t_total = time.perf_counter()
        gc.collect()
        bench_print("[JoyAI] before inference")

        seed = inputs.seed
        if torch.cuda.is_available() and seed != 0:
            generator = torch.Generator("cuda").manual_seed(seed)
        elif seed != 0:
            generator = torch.Generator(device=gfx_device).manual_seed(seed)
        else:
            generator = None

        image = inputs.image
        orig_w, orig_h = image.size
        from PIL import Image as _PIL
        target_w = inputs.width  if (inputs.width  and inputs.width  > 0) else orig_w
        target_h = inputs.height if (inputs.height and inputs.height > 0) else orig_h
        max_px = 1024
        if max(target_w, target_h) > max_px:
            scale    = max_px / max(target_w, target_h)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
        out_w = (target_w // 8) * 8
        out_h = (target_h // 8) * 8
        if (orig_w, orig_h) != (out_w, out_h):
            image = image.resize((out_w, out_h), _PIL.LANCZOS)
            print(f"[JoyAI] input resized {orig_w}×{orig_h} → {out_w}×{out_h}")

        prompt = inputs.prompt  # pre-built at queue-add time via _build_prompt
        print(f"[JoyAI] prompt: {prompt!r}  |  output {out_w}×{out_h}")

        self.set_phase(inputs, "Generating")
        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=inputs.neg_prompt or None,
                height=out_h,
                width=out_w,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                generator=generator,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]

        _flush()
        print(f"[JoyAI] ── done in {time.perf_counter()-t_total:.1f}s ──")
        return result
