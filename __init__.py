from __future__ import annotations

if "bpy" in locals():
    import importlib
    utils = importlib.reload(utils)
    properties = importlib.reload(properties)
    ui = importlib.reload(ui)
    operators = importlib.reload(operators)
else:
    import bpy
    from . import utils
    from . import properties
    from . import ui
    from . import operators

from .operators.queue_ops import queue_classes as _queue_classes

import os

# transformers parallelizes weight materialization across a ThreadPoolExecutor.
# On Blender's bundled torch_cpu.dll (Win) this races and faults with an
# EXCEPTION_ACCESS_VIOLATION inside _materialize_copy worker threads. Force
# synchronous loading process-wide; setdefault keeps it overridable.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

# Reduce CUDA VRAM fragmentation across consecutive render-queue jobs. Without
# this, a second run of a heavy model can OOM even with GiB free ("12 GiB free
# but can't allocate 248 MiB"). PyTorch only reads this when its CUDA allocator
# first initializes, so it MUST be set before torch is imported — the plugins
# that set it inside generate() do so too late to have any effect. setdefault
# keeps it overridable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _find_msvc_cl():
    """Return the path to the highest-version MSVC cl.exe, or None.

    Locates it via vswhere (authoritative for VS / Build Tools) then falls back
    to globbing the default Program Files install layout. Never raises.
    """
    import glob as _glob
    import subprocess as _sp
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    pf64 = os.environ.get("ProgramW6432", r"C:\Program Files")
    candidates = []
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.exists(vswhere):
        try:
            inst = _sp.run(
                [vswhere, "-latest", "-prerelease", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
        except Exception:
            inst = ""
        if inst:
            candidates += _glob.glob(os.path.join(
                inst, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe"))
    if not candidates:
        for pf in (pf64, pf86):
            candidates += _glob.glob(os.path.join(
                pf, "Microsoft Visual Studio", "*", "*",
                "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe"))
    candidates = [c for c in candidates if os.path.exists(c)]
    return sorted(candidates)[-1] if candidates else None


def _configure_triton_build():
    """Either enable Triton's JIT via MSVC, or fully disable build-dependent
    fast paths so an incomplete toolchain degrades to eager instead of crashing.

    Installing `triton-windows` (via requirements.txt) makes `import triton`
    succeed on Windows. That alone is unsafe: code that probes `has_triton()`
    may then attempt a JIT build that fails at *runtime* when any of these is
    missing —
      - a usable compiler: Triton picks clang.exe off PATH, but clang treats the
        positional `.lib` args Triton emits (cuda.lib / python3xy.lib) as CWD
        input files (it only searches -L for -l flags) → link error. Only MSVC's
        cl.exe searches /LIBPATH: for positional .lib names and works unaided.
      - Python dev files (Python.h + python3xy.lib) in Blender's embedded Python,
        provisioned at install by operators/system.py::_ensure_python_headers_windows
        (absent if that download failed or deps were never installed).
      - CUDA dev lib (cuda.lib) from an installed CUDA Toolkit.

    Windows only; never raises. CUDA-centric (ROCm/CPU machines fall to the safe
    "disabled" branch, i.e. eager — never a crash). All env writes use setdefault
    so an advanced user can override every decision.
    """
    import sys
    if sys.platform != "win32":
        return  # Linux/macOS: gcc/clang build normally; don't touch their compile paths
    try:
        import sysconfig as _sc
        import glob as _glob

        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        pf64 = os.environ.get("ProgramW6432", r"C:\Program Files")

        # 1. Compiler: respect an explicit/dev-prompt CC, else find MSVC cl.exe.
        cc = os.environ.get("CC") or _find_msvc_cl()

        # 2. Python dev files (provisioned at install time).
        inc = _sc.get_path("include")
        ver = f"{sys.version_info.major}{sys.version_info.minor}"
        has_py_h   = bool(inc) and os.path.exists(os.path.join(inc, "Python.h"))
        has_py_lib = os.path.exists(os.path.join(sys.base_prefix, "libs", f"python{ver}.lib"))

        # 3. CUDA dev lib (cuda.lib) from a CUDA Toolkit install.
        def _has_cuda_lib():
            for var in ("CUDA_PATH", "CUDA_HOME"):
                p = os.environ.get(var)
                if p and os.path.exists(os.path.join(p, "lib", "x64", "cuda.lib")):
                    return True
            for pf in (pf64, pf86):
                if _glob.glob(os.path.join(pf, "NVIDIA GPU Computing Toolkit",
                                           "CUDA", "v*", "lib", "x64", "cuda.lib")):
                    return True
            return False
        has_cuda_lib = _has_cuda_lib()

        buildable = bool(cc) and has_py_h and has_py_lib and has_cuda_lib

        if buildable:
            # Steer Triton onto MSVC and make link.exe + toolchain DLLs resolvable.
            if not os.environ.get("CC"):
                os.environ["CC"] = cc
                os.environ["PATH"] = os.path.dirname(cc) + os.pathsep + os.environ.get("PATH", "")
            print(f"PALLAIDIUM: Triton JIT enabled — compiling with MSVC ({cc})", flush=True)
        else:
            # Incomplete toolchain → disable every build-dependent fast path so a
            # missing element can only cost speed, never crash a generation.
            os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # torch.compile → eager no-op
            os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")
            os.environ.setdefault("SDNQ_USE_TRITON_MM", "0")    # direct Triton matmul kernels off
            missing = []
            if not cc:           missing.append("MSVC cl.exe")
            if not has_py_h:     missing.append("Python.h")
            if not has_py_lib:   missing.append(f"python{ver}.lib")
            if not has_cuda_lib: missing.append("cuda.lib (CUDA Toolkit)")
            print("PALLAIDIUM: Triton JIT disabled, using eager fallback — missing: "
                  + ", ".join(missing) + " (reinstall deps and/or install MSVC + CUDA Toolkit to enable)",
                  flush=True)
    except Exception as exc:
        # Last-ditch safety net: if the probe itself failed, assume not buildable
        # and disable rather than risk a runtime compile crash.
        for _k, _v in (("TORCHDYNAMO_DISABLE", "1"),
                       ("SDNQ_USE_TORCH_COMPILE", "0"),
                       ("SDNQ_USE_TRITON_MM", "0")):
            try:
                os.environ.setdefault(_k, _v)
            except Exception:
                pass
        print(f"PALLAIDIUM: Triton build-env probe failed, JIT disabled — {exc}", flush=True)


_configure_triton_build()

from .utils.helpers import load_styles, filter_updated, input_strips_updated, get_enum_items, update_folder_callback
# from .utils.helpers import *
from .properties import *
from .ui import *
from .operators import *

bl_info = {
    "name": "Pallaidium - Generative AI",
    "author": "tintwotin",
    "version": (3, 0),
    "blender": (5, 2, 0),
    "location": "Video Sequence Editor > Sidebar > Generative AI",
    "description": "AI Generate media in the VSE",
    "category": "Sequencer",
}

classes = (
    *_queue_classes,
    GeneratorAddonPreferences,
    PALLAIDIUM_OT_refresh_remote_models,
    PALLAIDIUM_OT_start_backend,
    PALLAIDIUM_OT_stop_backend,
    PALLAIDIUM_OT_open_workflows_folder,
    PALLAIDIUM_OT_import_comfy_workflow,
    SEQUENCER_OT_generate_movie,
    SEQUENCER_OT_generate_audio,
    SEQUENCER_OT_generate_image,
    SEQUENCER_OT_generate_text,
    SEQUENCER_OT_ai_strip_picker,
    SEQUENCER_PT_pallaidium_panel,
    GENERATOR_OT_sound_notification,
    SEQUENCER_OT_strip_to_generatorAI,
    LORABrowserFileItem,
    LORA_OT_RefreshFiles,
    LORABROWSER_UL_files,
    GENERATOR_OT_install,
    GENERATOR_OT_uninstall,
    GENERATOR_OT_cancel_dep_op,
    GENERATOR_OT_copy_install_report,
    GENERATOR_OT_dismiss_install_errors,
    GENERATOR_OT_export_requirements,
    SequencerOpenAudioFile,
    IPAdapterFaceProperties,
    IPAdapterFaceFileBrowserOperator,
    IPAdapterStyleProperties,
    IPAdapterStyleFileBrowserOperator,
    SEQUENCER_OT_redo_from_metadata,
    AI_Metadata_PT_Panel,
    OBJECT_OT_FluxAddStrip,
    OBJECT_OT_FluxHideStrip,
    OBJECT_OT_KleinAddStrip,
    OBJECT_OT_KleinHideStrip,
    SEQUENCER_OT_stem_split,
)

_SCHEMATIC_TRIGGER_PROMPTS = {
    "DEPTH":      "Generate a relative depth map of the input image.",
    "NORMAL":     "Generate a surface normal map of the input image.",
    "BODY_POSE":  "Generate a body pose map of all visible people in the input image.",
    "FULL_POSE":  "Generate a full pose map of all visible people in the input image.",
    "BINARY_SEG": "Generate a binary segmentation mask of {target} in the input image.",
    "AMODAL_SEG": "Generate an amodal segmentation mask of {target} in the input image.",
}


def _schematic_mode_update(self, context):
    scene = context.scene
    mode   = scene.klein_schematic_mode
    target = (getattr(scene, "klein_schematic_target", "person") or "person").strip()
    new_trigger = _SCHEMATIC_TRIGGER_PROMPTS[mode].format(target=target)
    current = scene.generate_movie_prompt or ""
    for tmpl in _SCHEMATIC_TRIGGER_PROMPTS.values():
        prefix = tmpl.split("{")[0]
        if current.startswith(prefix):
            dot_idx = current.find(".", len(prefix))
            if dot_idx != -1:
                current = current[dot_idx + 1:].lstrip()
            else:
                current = ""
            break
    scene.generate_movie_prompt = new_trigger + (" " + current if current else "")
    if hasattr(scene, "generatorai_styles"):
        scene.generatorai_styles = "no_style"


def _schematic_target_update(self, context):
    scene = context.scene
    if getattr(scene, "klein_schematic_mode", "DEPTH") in ("BINARY_SEG", "AMODAL_SEG"):
        _schematic_mode_update(scene, context)


def register():
    bpy.types.Scene.generate_movie_prompt = bpy.props.StringProperty(
        name="generate_movie_prompt",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.generate_movie_negative_prompt = bpy.props.StringProperty(
        name="generate_movie_negative_prompt",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.parler_direction_prompt = bpy.props.StringProperty(
        name="parler_direction_prompt",
        default="Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x",
        default=1920,
        step=32,
        min=64,
        max=4096,
        description="Use the power of 64",
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=896,
        step=32,
        min=64,
        max=4096,
        description="Use the power of 64",
    )

    # The number of frames to be generated.
    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_frames",
        default=100,
        min=-1,
        max=500,
        description="Number of frames to generate. NB. some models have fixed values.",
    )

    # The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.
    bpy.types.Scene.movie_num_inference_steps = bpy.props.IntProperty(
        name="movie_num_inference_steps",
        default=23,
        min=1,
        max=200,
        description="Number of inference steps to improve the quality",
    )

    # The number of videos to generate.
    bpy.types.Scene.movie_num_batch = bpy.props.IntProperty(
        name="movie_num_batch",
        default=1,
        min=1,
        max=100,
        description="Number of generated media files",
        update=filter_updated,
    )

    # The seed number.
    bpy.types.Scene.movie_num_seed = bpy.props.IntProperty(
        name="movie_num_seed",
        default=1,
        min=-2147483647,
        max=2147483647,
        description="Seed value",
    )

    # The seed number.
    bpy.types.Scene.movie_use_random = bpy.props.BoolProperty(
        name="movie_use_random",
        default=1,
        description="Randomize seed value. Switched off will give more consistency.",
    )

    # The guidance number.
    bpy.types.Scene.movie_num_guidance = bpy.props.FloatProperty(
        name="movie_num_guidance",
        default=4.0,
        min=0,
        max=100,
    )

    # The frame audio duration.
    bpy.types.Scene.audio_length_in_f = bpy.props.IntProperty(
        name="audio_length_in_f",
        default=80,
        min=-1,
        max=10000,
        description="Audio duration: Maximum 47 sec.",
    )
    bpy.types.Scene.generatorai_typeselect = bpy.props.EnumProperty(
        name="Sound",
        items=[
            ("movie", "Video", "Generate Video"),
            ("image", "Image", "Generate Image"),
            ("audio", "Audio", "Generate Audio"),
            ("text", "Text", "Generate Text"),
            ("3d", "3D", "Generate 3D"),
        ],
        default="image",
        update=input_strips_updated,
    )
    bpy.types.Scene.speakers = bpy.props.EnumProperty(
        name="Speakers",
        items=[
            ("speaker_0", "Speaker 0", ""),
            ("speaker_1", "Speaker 1", ""),
            ("speaker_2", "Speaker 2", ""),
            ("speaker_3", "Speaker 3", ""),
            ("speaker_4", "Speaker 4", ""),
            ("speaker_5", "Speaker 5", ""),
            ("speaker_6", "Speaker 6", ""),
            ("speaker_7", "Speaker 7", ""),
            ("speaker_8", "Speaker 8", ""),
            ("speaker_9", "Speaker 9", ""),
        ],
        default="speaker_3",
    )
    bpy.types.Scene.languages = bpy.props.EnumProperty(
        name="Languages",
        items=[
            ("en", "English", ""),
            ("de", "German", ""),
            ("es", "Spanish", ""),
            ("fr", "French", ""),
            ("hi", "Hindi", ""),
            ("it", "Italian", ""),
            ("ja", "Japanese", ""),
            ("ko", "Korean", ""),
            ("pl", "Polish", ""),
            ("pt", "Portuguese", ""),
            ("ru", "Russian", ""),
            ("tr", "Turkish", ""),
            ("zh", "Chinese, simplified", ""),
        ],
        default="en",
    )

    # Inpaint
    bpy.types.Scene.inpaint_selected_strip = bpy.props.StringProperty(
        name="inpaint_selected_strip", default=""
    )

    # Inpaint
    bpy.types.Scene.out_frame = bpy.props.StringProperty(
        name="out_frame", default=""
    )

    # Upscale
    bpy.types.Scene.video_to_video = bpy.props.BoolProperty(
        name="video_to_video",
        default=0,
    )

    # Refine SD
    bpy.types.Scene.refine_sd = bpy.props.BoolProperty(
        name="refine_sd",
        default=0,
        description="Add a refinement step",
    )

    # ADetailer
    bpy.types.Scene.adetailer = bpy.props.BoolProperty(
        name="adetailer",
        default=0,
        description="Add Face Details",
        update=filter_updated,
    )

    # AuraSR
    bpy.types.Scene.aurasr = bpy.props.BoolProperty(
        name="aurasr",
        default=0,
        description="4x Upscale (Aura SR)",
        update=filter_updated,
    )

    # movie path
    bpy.types.Scene.movie_path = bpy.props.StringProperty(
        name="movie_path",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )

    # image path
    bpy.types.Scene.image_path = bpy.props.StringProperty(
        name="image_path",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )

    # sound path
    bpy.types.Scene.sound_path = bpy.props.StringProperty(
        name="sound_path",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )

    # LTX Multi N-anchor middle images (JSON list of [path, fraction] pairs)
    bpy.types.Scene.ltx_middle_images_json = bpy.props.StringProperty(
        name="ltx_middle_images_json",
        default="",
    )

    bpy.types.Scene.input_strips = bpy.props.EnumProperty(
        items=[
            ("input_prompt", "Prompts", "Prompts"),
            ("input_strips", "Strips", "Selected Strips"),
        ],
        default="input_prompt",
        update=input_strips_updated,
    )
    bpy.types.Scene.image_power = bpy.props.FloatProperty(
        name="image_power",
        default=0.50,
        min=0.00,
        max=1.00,
        description="Preserve the input image in vid/img to img/vid processes",
    )
    styles_array = load_styles(
        os.path.dirname(os.path.abspath(__file__)) + "/styles.json"
    )
    if styles_array:
        bpy.types.Scene.generatorai_styles = bpy.props.EnumProperty(
            name="Generator AI Styles",
            items=[("no_style", "No Style", "No Style")] + styles_array,
            default="no_style",
            description="Add style prompts",
        )
    bpy.types.Scene.use_lcm = bpy.props.BoolProperty(
        name="use_lcm",
        default=0,
        description="Higher Speed, lower quality. Try Quality Steps: 1-10",
        update=lcm_updated,
    )
    bpy.types.Scene.remove_silence = bpy.props.BoolProperty(
        name="remove_silence",
        default=1,
        description="Remove Silence",
    )

    # SVD decode chunck
    bpy.types.Scene.svd_decode_chunk_size = bpy.props.IntProperty(
        name="svd_decode_chunk_size",
        default=2,
        min=1,
        max=100,
        description="Number of frames to decode",
    )

    # SVD motion_bucket_id
    bpy.types.Scene.svd_motion_bucket_id = bpy.props.IntProperty(
        name="svd_motion_bucket_id",
        default=1,
        min=1,
        max=512,
        description="A higher number: more camera movement. A lower number: more character movement",
    )

    for cls in classes:
        bpy.utils.register_class(cls)

    # Render Queue
    from .operators.queue_ops import RenderQueueJob as _RQJ
    bpy.types.Scene.render_queue = bpy.props.CollectionProperty(type=_RQJ)

    # LoRA
    bpy.types.Scene.lora_files = bpy.props.CollectionProperty(type=LORABrowserFileItem)
    bpy.types.Scene.lora_files_index = bpy.props.IntProperty(name="Index", default=0)
    bpy.types.Scene.lora_folder = bpy.props.StringProperty(
        name="Folder",
        description="Select a folder",
        subtype="DIR_PATH",
        options={"TEXTEDIT_UPDATE"},
        default="",
        update=update_folder_callback,
    )
    bpy.types.Scene.ref_audio_path = bpy.props.StringProperty(
        name="ref_audio_path",
        default="",
        description="Path to speaker reference audio (TTS/VC models)",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.ref_text = bpy.props.StringProperty(
        name="ref_text",
        default="",
        description="Reference transcription text (TTS models)",
        options={"TEXTEDIT_UPDATE"},
    )
    # The frame audio duration.
    bpy.types.Scene.audio_speed = bpy.props.IntProperty(
        name="audio_speed",
        default=13,
        min=1,
        max=20,
        description="Speech speed.",
    )
    # The frame audio duration.
    bpy.types.Scene.audio_speed_tts = bpy.props.FloatProperty(
        name="Speed",
        default=1.0,
        min=0.1,
        max=3.0,
        description="Speech speed. 1.0 = normal. >1.0 = faster (shorter audio). <1.0 = slower (longer audio).",
    )
    bpy.types.Scene.ip_adapter_face_folder = bpy.props.StringProperty(
        name="File",
        description="Select a file or folder",
        default="",
        options={"TEXTEDIT_UPDATE"},
        # update=update_ip_adapter_face_callback,
    )
    bpy.types.Scene.ip_adapter_face_files_to_import = bpy.props.CollectionProperty(
        type=IPAdapterFaceProperties
    )
    bpy.types.Scene.ip_adapter_style_folder = bpy.props.StringProperty(
        name="File",
        description="Select a file or folder",
        default="",
        options={"TEXTEDIT_UPDATE"},
        # update=update_ip_adapter_style_callback,
    )
    bpy.types.Scene.ip_adapter_style_files_to_import = bpy.props.CollectionProperty(
        type=IPAdapterStyleProperties
    )

    bpy.types.Scene.genai_out_path = bpy.props.StringProperty(
        name="genai_out_path", default=""
    )
    bpy.types.Scene.genai_out_path = ""

    # Remote video models that support generated audio (e.g. Seedance 2.0).
    bpy.types.Scene.remote_generate_audio = bpy.props.BoolProperty(
        name="Generate Audio",
        description="Ask the backend to generate synchronized audio for the video "
                    "(supported by Seedance 2.0 models)",
        default=True,
    )

    bpy.types.Scene.minimax_subject = bpy.props.StringProperty(
        name="minimax_subject", default=""
    )

    bpy.types.Scene.omnigen_prompt_1 = bpy.props.StringProperty(
        name="omnigen_prompt_1",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.omnigen_prompt_2 = bpy.props.StringProperty(
        name="omnigen_prompt_2",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.omnigen_prompt_3 = bpy.props.StringProperty(
        name="omnigen_prompt_3",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.omnigen_strip_1 = bpy.props.StringProperty(
        name="omnigen_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.omnigen_strip_2 = bpy.props.StringProperty(
        name="omnigen_strip_2", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.omnigen_strip_3 = bpy.props.StringProperty(
        name="omnigen_strip_3", options={"TEXTEDIT_UPDATE"}, default=""
    )

    bpy.types.Scene.qwen_strip_1 = bpy.props.StringProperty(
        name="qwen_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.qwen_strip_2 = bpy.props.StringProperty(
        name="qwen_strip_2", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.qwen_strip_3 = bpy.props.StringProperty(
        name="qwen_strip_3", options={"TEXTEDIT_UPDATE"}, default=""
    )

    # Ideogram 4
    bpy.types.Scene.ideogram_prompt_upsampling = bpy.props.BoolProperty(
        name="ideogram_prompt_upsampling",
        description=(
            "Rewrite the prompt into Ideogram 4's native JSON caption on-device "
            "using the shared Qwen3-VL text encoder. Install 'outlines' for "
            "schema-constrained captions. Expect a quality decrease vs. remote "
            "upsampling. Requires a fresh model load when toggled."
        ),
        default=False,
    )

    bpy.types.Scene.klein_strip_1 = bpy.props.StringProperty(
        name="klein_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.klein_strip_2 = bpy.props.StringProperty(
        name="klein_strip_2", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.klein_strip_3 = bpy.props.StringProperty(
        name="klein_strip_3", options={"TEXTEDIT_UPDATE"}, default=""
    )
    # Klein optional overflow refs (4..9) — 1-3 are always shown, 4-9 are
    # revealed via the add/remove UI mirroring flux_visible_strips.
    for _i in range(4, 10):
        setattr(
            bpy.types.Scene,
            f"klein_strip_{_i}",
            bpy.props.StringProperty(
                name=f"klein_strip_{_i}", options={"TEXTEDIT_UPDATE"}, default=""
            )
        )
    bpy.types.Scene.klein_visible_strips = bpy.props.IntProperty(
        name="Visible Klein Strips",
        description="Number of Klein reference image strips visible in the UI",
        default=3,
        min=3,
        max=9,
    )

    # Klein Schematic LoRA plugin
    bpy.types.Scene.klein_schematic_mode = bpy.props.EnumProperty(
        name="Schematic Mode",
        items=[
            ("DEPTH",      "Relative Depth",
             "Generates a grayscale depth map — closer objects are brighter. Recommended: 20 steps, CFG 5.0"),
            ("NORMAL",     "Surface Normal",
             "Generates an RGB surface normal map (X=red, Y=green, Z=blue). Recommended: 20 steps, CFG 5.0"),
            ("BODY_POSE",  "Body Pose",
             "Generates a skeleton overlay for torso and limbs only. Recommended: 20 steps, CFG 5.0"),
            ("FULL_POSE",  "Full Pose",
             "Generates a full skeleton including hands and face. Use 1024px for best results. Recommended: 20 steps, CFG 5.0"),
            ("BINARY_SEG", "Binary Segmentation",
             "Generates a binary mask for the target class. Set the Target field to the object class, e.g. 'person' or 'car'"),
            ("AMODAL_SEG", "Amodal Segmentation",
             "Generates a mask including occluded/hidden regions of the target. Set the Target field."),
        ],
        default="DEPTH",
        update=_schematic_mode_update,
    )
    bpy.types.Scene.klein_schematic_target = bpy.props.StringProperty(
        name="Target",
        description="Object or class to segment, e.g. 'person' or 'car'. Used only in Binary/Amodal Segmentation modes.",
        default="person",
        options={"TEXTEDIT_UPDATE"},
        update=_schematic_target_update,
    )

    # JoyAI Image Edit
    bpy.types.Scene.joyimage_spatial_mode = bpy.props.EnumProperty(
        name="Spatial Mode",
        items=[
            ("general", "General Edit",    "Instruction-guided image editing"),
            ("move",    "Object Move",      "Move object into a red-box region"),
            ("rotate",  "Object Rotation",  "Rotate object to a canonical view"),
            ("camera",  "Camera Control",   "Shift camera viewpoint, keep 3D scene"),
        ],
        default="general",
    )
    bpy.types.Scene.joyimage_object = bpy.props.StringProperty(
        name="Object",
        description="Name of the object to move/rotate",
        default="object",
    )
    bpy.types.Scene.joyimage_rotate_view = bpy.props.EnumProperty(
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
    bpy.types.Scene.joyimage_yaw = bpy.props.FloatProperty(
        name="Yaw (°)", default=0.0, min=-180.0, max=180.0, step=10,
    )
    bpy.types.Scene.joyimage_pitch = bpy.props.FloatProperty(
        name="Pitch (°)", default=0.0, min=-90.0, max=90.0, step=10,
    )
    bpy.types.Scene.joyimage_zoom = bpy.props.EnumProperty(
        name="Zoom",
        items=[
            ("unchanged", "Unchanged", ""),
            ("in",        "In",        ""),
            ("out",       "Out",       ""),
        ],
        default="unchanged",
    )

    # FLUX 2
    for i in range(1, 10):
        setattr(
            bpy.types.Scene,
            f"flux_strip_{i}",
            bpy.props.StringProperty(
                name=f"flux_strip_{i}", options={"TEXTEDIT_UPDATE"}, default=""
            )
        )
    bpy.types.Scene.flux_visible_strips = bpy.props.IntProperty(
        name="Visible Flux Strips",
        description="Number of Flux image strips visible in the UI",
        default=1, # Start with one strip visible
        min=1,
        max=9
    )
    # ltx23_multi_v2 — independent audio / modality guidance
    bpy.types.Scene.ltx23m_enable_guidance      = bpy.props.BoolProperty( name="V2 Guidance",          default=False, description="Enable independent audio/modality guidance sliders")
    bpy.types.Scene.ltx23m_modality_scale       = bpy.props.FloatProperty(name="Modality Scale",       default=1.5, min=1.0, max=5.0,  description="Audio-to-video influence. >1.0 enables modality isolation guidance; 1.5 is a good default")
    bpy.types.Scene.ltx23m_audio_guidance       = bpy.props.FloatProperty(name="Audio CFG",            default=1.0, min=0.0, max=10.0, description="Independent audio guidance scale")
    bpy.types.Scene.ltx23m_audio_stg_scale      = bpy.props.FloatProperty(name="Audio STG",            default=0.0, min=0.0, max=5.0,  description="Audio spatio-temporal guidance scale")
    bpy.types.Scene.ltx23m_audio_modality_scale = bpy.props.FloatProperty(name="Audio Modality Scale", default=1.0, min=0.0, max=5.0,  description="Audio cross-modal scale")
    bpy.types.Scene.ltx23m_audio_noise_scale    = bpy.props.FloatProperty(name="Audio Noise Scale",    default=0.0, min=0.0, max=1.0,  description="Noise level for unmasked audio regions")
    bpy.types.Scene.ltx23m_audio_start_time     = bpy.props.FloatProperty(name="Audio Start Time",     default=0.0, min=0.0, max=3600.0, description="Audio condition start time in seconds (computed from strip offset)")
    bpy.types.Scene.ltx23m_image_strength       = bpy.props.FloatProperty(name="Image Strength",       default=1.0, min=0.0, max=1.0,  description="Conditioning strength for every image anchor (first frame, last frame, and any middle anchors). 1.0 hard-locks each anchor frame to its reference image; lower values loosen the lock, trading appearance fidelity for more motion freedom")

    # ltx23_multi_ic_lora — IC-LoRA control params
    bpy.types.Scene.ltx23ic_control_strip       = bpy.props.StringProperty(name="IC-LoRA Ref Strip",    default="",  description="Name of the IC-LoRA reference strip (META or MOVIE)")
    bpy.types.Scene.ltx23ic_control_strength    = bpy.props.FloatProperty( name="Control Strength",     default=1.0, min=0.0, max=1.0,  description="Strength of IC-LoRA reference token conditioning")
    bpy.types.Scene.ltx23ic_control_downscale   = bpy.props.IntProperty(   name="Control Downscale",    default=2,   min=1,   max=4,    description="Spatial downscale factor for IC-LoRA reference encoding (2 = quarter the control tokens, much lower Stage-1 VRAM)")
    bpy.types.Scene.ltx23ic_control_audio_str   = bpy.props.FloatProperty( name="Audio Ref Strength",   default=1.0, min=0.0, max=1.0,  description="Strength of IC-LoRA audio reference conditioning")
    bpy.types.Scene.ltx23ic_identity_guidance   = bpy.props.FloatProperty( name="Identity Guidance",    default=0.0, min=0.0, max=5.0,  description="Extra forward pass amplification for audio identity transfer")
    bpy.types.Scene.ltx23ic_input_downscale_pct = bpy.props.FloatProperty(
        name="Input Downscale %", default=100.0, min=25.0, max=100.0, subtype='PERCENTAGE',
        description="Downscale the rendered main input video/scene strip to this percentage of "
                    "its native resolution before it is fed to the model (rounded to the nearest "
                    "multiple of 64). Lower values render the intermediate clip faster and use "
                    "less VRAM at the cost of input detail")

    # ltx23_extend — clip extension params
    bpy.types.Scene.ltx23ext_extend_frames = bpy.props.IntProperty(
        name="Extend (frames)", default=96, min=8, max=1200,
        description="Number of new frames to generate after the source clip (8n+1 aligned)")
    bpy.types.Scene.ltx23ext_video_strength = bpy.props.FloatProperty(
        name="Source Lock", default=1.0, min=0.0, max=1.0,
        description="Conditioning strength of the carried-over source clip (1.0 = fully locked)")
    bpy.types.Scene.ltx23ext_audio_strip = bpy.props.StringProperty(
        name="Audio Strip", default="",
        description="Name of a SOUND strip whose audio drives the extended video (overrides the source clip's audio)")

    # Maxine VSR quality level
    bpy.types.Scene.maxine_quality = bpy.props.EnumProperty(
        name="Quality",
        items=[
            ("HIGH",    "High",    "AI upscaling, quality-favoring (default)"),
            ("ULTRA",   "Ultra",   "AI upscaling, maximum detail preservation"),
            ("MEDIUM",  "Medium",  "AI upscaling, balanced speed/quality"),
            ("LOW",     "Low",     "AI upscaling, speed-optimized"),
            ("DENOISE_MEDIUM", "Denoise", "Remove noise and compression artifacts (same resolution)"),
            ("DEBLUR_MEDIUM",  "Deblur",  "Sharpen blurry or out-of-focus footage (same resolution)"),
        ],
        default="HIGH",
        description="Maxine VSR processing mode",
    )

    # ltx23 staged — shared stage-mode enum for the *_staged plugin variants
    bpy.types.Scene.ltx23_stage_mode = bpy.props.EnumProperty(
        name="Stages",
        items=[
            ("FULL",  "Full (Both Steps)", "Run both stages — base generation + refinement"),
            ("STEP1", "Step 1",            "Only base generation — fast half-resolution preview"),
            ("STEP2", "Step 2 (input video)", "Skip generation; refine the input video via upsample + Stage 2"),
        ],
        default="FULL",
        description="Which stages of the two-stage LTX pipeline to run",
    )

    # ── Google Nano Banana (Gemini image) cloud plugin settings ────────────
    bpy.types.Scene.nano_banana_model = bpy.props.EnumProperty(
        name="Variant",
        items=[
            ("gemini-2.5-flash-image",     "Nano Banana (Flash)",  "Gemini 2.5 Flash Image — fast, low cost"),
            ("gemini-3.1-flash-lite-image","Nano Banana Lite",     "Gemini 3.1 Flash-Lite Image — ultra-low latency, cheapest"),
            ("gemini-3.1-flash-image",     "Nano Banana 2",        "Gemini 3.1 Flash Image — high-efficiency, optimized for speed"),
            ("gemini-3-pro-image-preview", "Nano Banana Pro",      "Gemini 3 Pro Image — highest quality, up to 4K"),
            ("imagen-4.0-generate-001",    "Imagen 4",             "Imagen 4 text-to-image"),
        ],
        default="gemini-2.5-flash-image",
        description="Which Google image model to use",
    )
    bpy.types.Scene.nano_banana_aspect = bpy.props.EnumProperty(
        name="Aspect",
        items=[
            ("1:1",  "1:1",  "Square"),
            ("16:9", "16:9", "Landscape"),
            ("9:16", "9:16", "Portrait"),
            ("4:3",  "4:3",  "Landscape"),
            ("3:4",  "3:4",  "Portrait"),
            ("21:9", "21:9", "Ultrawide"),
        ],
        default="1:1",
        description="Output aspect ratio",
    )
    bpy.types.Scene.nano_banana_resolution = bpy.props.EnumProperty(
        name="Resolution",
        items=[
            ("1K", "1K", "1024px"),
            ("2K", "2K", "2048px (Nano Banana Pro)"),
            ("4K", "4K", "4096px (Nano Banana Pro)"),
        ],
        default="1K",
        description="Output resolution tier (2K/4K require Nano Banana Pro)",
    )

    # ── Google Veo (video) cloud plugin settings ───────────────────────────
    bpy.types.Scene.veo_model = bpy.props.EnumProperty(
        name="Variant",
        items=[
            ("veo-3.1-generate-preview",      "Veo 3.1",       "Veo 3.1 — highest quality, native audio"),
            ("veo-3.1-fast-generate-preview", "Veo 3.1 Fast",  "Veo 3.1 Fast — faster, lower cost"),
            ("veo-3.0-generate-preview",      "Veo 3.0",       "Veo 3.0"),
        ],
        default="veo-3.1-fast-generate-preview",
        description="Which Google Veo model to use",
    )
    bpy.types.Scene.veo_aspect = bpy.props.EnumProperty(
        name="Aspect",
        items=[
            ("16:9", "16:9", "Landscape"),
            ("9:16", "9:16", "Portrait"),
        ],
        default="16:9",
        description="Output aspect ratio",
    )
    bpy.types.Scene.veo_resolution = bpy.props.EnumProperty(
        name="Resolution",
        items=[
            ("720p",  "720p",  "1280×720"),
            ("1080p", "1080p", "1920×1080 (16:9 only)"),
        ],
        default="720p",
        description="Output resolution",
    )
    bpy.types.Scene.veo_duration = bpy.props.EnumProperty(
        name="Duration",
        items=[
            ("4", "4s", "4 seconds"),
            ("6", "6s", "6 seconds"),
            ("8", "8s", "8 seconds"),
        ],
        default="8",
        description="Clip duration in seconds",
    )
    bpy.types.Scene.veo_person_generation = bpy.props.EnumProperty(
        name="People",
        items=[
            ("allow_adult", "Adults Only",  "Allow generation of adults only"),
            ("dont_allow",  "Don't Allow",  "Disallow generation of people"),
        ],
        default="allow_adult",
        description="Policy for generating people in the video "
                    "(the Developer API does not support allowing children)",
    )
    bpy.types.Scene.veo_image_mode = bpy.props.EnumProperty(
        name="Image Mode",
        items=[
            ("AUTO",        "Auto",               "Reference images if picked, else first+last frame, else first frame, else text-only"),
            ("FIRST",       "First Frame",        "Use the selected strip as the starting frame (image-to-video)"),
            ("INTERPOLATE", "First + Last Frame", "Interpolate between the first and last frame of a META strip (Veo 3.1)"),
            ("REFERENCE",   "Reference Images",   "Use the Ref. image pickers as subject/style ingredients (Veo 3.1)"),
        ],
        default="AUTO",
        description="How image inputs are interpreted by Veo",
    )

    # Reference-image strip pickers (Nano Banana composition / Veo 3.1 ingredients)
    # Nano Banana supports up to 9 reference images; how many picker rows are
    # shown is driven by nano_banana_ref_count (Nano Banana Pro handles the most).
    bpy.types.Scene.nano_banana_ref_count = bpy.props.IntProperty(
        name="References", default=3, min=1, max=9,
        description="Number of reference-image slots to use for Nano Banana",
    )
    for _i in range(1, 10):
        setattr(
            bpy.types.Scene, f"nano_banana_ref_strip_{_i}",
            bpy.props.StringProperty(
                name=f"nano_banana_ref_strip_{_i}",
                options={"TEXTEDIT_UPDATE"}, default="",
            ),
        )
    bpy.types.Scene.veo_ref_strip_1 = bpy.props.StringProperty(
        name="veo_ref_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.veo_ref_strip_2 = bpy.props.StringProperty(
        name="veo_ref_strip_2", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.veo_ref_strip_3 = bpy.props.StringProperty(
        name="veo_ref_strip_3", options={"TEXTEDIT_UPDATE"}, default=""
    )

    # The guidance number.
    bpy.types.Scene.img_guidance_scale = bpy.props.FloatProperty(
        name="img_guidance_scale",
        default=1.6,
        min=0,
        max=100,
    )
    bpy.types.Scene.chat_exaggeration = bpy.props.FloatProperty(
        name="Exaggeration",
        default=0.5,
        min=0,
        max=2,
        description="Chatterbox exaggeration",
    )
    bpy.types.Scene.chat_temperature = bpy.props.FloatProperty(
        name="Temperature",
        default=0.8,
        min=0,
        max=5,
        description="Chatterbox Temperature",
    )
    bpy.types.Scene.chat_pace = bpy.props.FloatProperty(
        name="Pace",
        default=0.5,
        min=0,
        max=1,
        description="Chatterbox Pace",
    )
    bpy.types.Scene.chatterbox_mtl_language = bpy.props.EnumProperty(
        name="Language",
        items=[
            ("ar", "Arabic",     ""),
            ("da", "Danish",     ""),
            ("de", "German",     ""),
            ("el", "Greek",      ""),
            ("en", "English",    ""),
            ("es", "Spanish",    ""),
            ("fi", "Finnish",    ""),
            ("fr", "French",     ""),
            ("he", "Hebrew",     ""),
            ("hi", "Hindi",      ""),
            ("it", "Italian",    ""),
            ("ja", "Japanese",   ""),
            ("ko", "Korean",     ""),
            ("ms", "Malay",      ""),
            ("nl", "Dutch",      ""),
            ("no", "Norwegian",  ""),
            ("pl", "Polish",     ""),
            ("pt", "Portuguese", ""),
            ("ru", "Russian",    ""),
            ("sv", "Swedish",    ""),
            ("sw", "Swahili",    ""),
            ("tr", "Turkish",    ""),
            ("zh", "Chinese",    ""),
        ],
        default="en",
        description="Chatterbox Multilingual output language",
    )
    bpy.types.Scene.kontext_strip_1 = bpy.props.StringProperty(
        name="kontext_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )

    bpy.types.Scene.illumination_style = bpy.props.EnumProperty(name="Lighting Style", items=get_enum_items(ILLUMINATION_OPTIONS), default="sunshine from window")
    bpy.types.Scene.light_direction = bpy.props.EnumProperty(name="Light Direction", items=get_enum_items(DIRECTION_OPTIONS), default="auto")
    bpy.types.Scene.music_bpm = bpy.props.IntProperty(
        name="BPM", default=0, min=0, max=300,
        description="Beats per minute (0 = model estimates)"
    )
    bpy.types.Scene.music_lyrics = bpy.props.StringProperty(
        name="Lyrics", default="",
        description="Lyrics text. Supports tags like [verse], [chorus]"
    )
    bpy.types.Scene.music_key_scale = bpy.props.StringProperty(
        name="Key", default="",
        description="Musical key, e.g. 'C major', 'A minor' (blank = model estimates)"
    )
    bpy.types.Scene.music_time_signature = bpy.props.StringProperty(
        name="Time Signature", default="",
        description="Time signature, e.g. '4' for 4/4, '3' for 3/4 (blank = model estimates)"
    )

    # Faster Whisper Transcription
    bpy.types.Scene.whisper_model_size = bpy.props.EnumProperty(
        name="Model Size",
        items=[
            ("tiny",           "Tiny",           "Fastest, least accurate (~39 MB)"),
            ("base",           "Base",           "Fast, decent accuracy (~74 MB)"),
            ("small",          "Small",          "Good balance (~244 MB)"),
            ("medium",         "Medium",         "High accuracy (~769 MB)"),
            ("large-v3-turbo", "Large-v3-turbo", "High accuracy, fast — recommended (~809 MB)"),
            ("large-v3",       "Large-v3",       "Best accuracy, slowest (~3.1 GB)"),
        ],
        default="large-v3-turbo",
    )
    from .utils.whisper_langs import WHISPER_LANG_ITEMS as _WL
    bpy.types.Scene.whisper_language = bpy.props.EnumProperty(
        name="Language",
        items=_WL,
        default="auto",
        description="Spoken language. 'Auto-detect' lets Whisper identify it automatically.",
    )

    # Mist / Depth Pass (3D)
    bpy.types.Scene.mist_range_mode = bpy.props.EnumProperty(
        name="Range",
        items=[
            ("AUTO",   "Auto",         "Compute mist start/depth from camera + scene geometry"),
            ("CUSTOM", "Custom",       "Use the Start/Depth values below"),
            ("SCENE",  "Scene World",  "Use the scene's existing World > Mist Settings"),
        ],
        default="AUTO",
        description="How the Mist start/depth range is determined",
    )
    bpy.types.Scene.mist_custom_start = bpy.props.FloatProperty(
        name="Start", default=0.0, min=0.0,
        description="Distance from the camera where mist begins",
    )
    bpy.types.Scene.mist_custom_depth = bpy.props.FloatProperty(
        name="Depth", default=100.0, min=0.001,
        description="Distance over which mist fades from 0 to 1",
    )
    bpy.types.Scene.mist_invert = bpy.props.BoolProperty(
        name="Invert (White = Near)", default=True,
        description="White = close to camera, black = far away",
    )

    # Stem Splitter
    bpy.types.Scene.stem_split_model = bpy.props.EnumProperty(
        name="Model",
        items=[
            ("htdemucs_ft", "HT-Demucs FT (Best)", "Fine-tuned, best quality — first-run download ~1.26 GB"),
            ("htdemucs",    "HT-Demucs (Fast)",     "Single model, fastest startup — first-run download 316 MB"),
            ("htdemucs_6s", "HT-Demucs 6-Stem",     "Adds guitar & piano — first-run download 258 MB"),
        ],
        default="htdemucs_ft",
    )
    bpy.types.Scene.stem_split_vocals = bpy.props.BoolProperty(name="Vocals", default=True)
    bpy.types.Scene.stem_split_drums  = bpy.props.BoolProperty(name="Drums",  default=True)
    bpy.types.Scene.stem_split_bass   = bpy.props.BoolProperty(name="Bass",   default=True)
    bpy.types.Scene.stem_split_other  = bpy.props.BoolProperty(name="Other",  default=True)
    bpy.types.Scene.stem_split_guitar = bpy.props.BoolProperty(name="Guitar", default=False)
    bpy.types.Scene.stem_split_piano  = bpy.props.BoolProperty(name="Piano",  default=False)

    # OmniVoice
    from .utils.omnivoice_langs import OMNIVOICE_LANG_ITEMS as _OV_LANGS
    bpy.types.Scene.omnivoice_language = bpy.props.EnumProperty(
        name="Language",
        items=_OV_LANGS,
        default="AUTO",
        description="Output language. Auto detects from the prompt text; select a language to force it.",
    )
    bpy.types.Scene.omnivoice_instruct = bpy.props.StringProperty(
        name="Instruct",
        default="",
        description=(
            "Describe the speaker voice. Can be combined with a Speaker Ref. "
            "Examples: 'female, adult, american'  |  'male, elderly, british'  |  'female, child, high pitch'. "
            "Accents (EN): american british australian canadian indian chinese korean japanese portuguese russian. "
            "Leave blank for auto-voice selection."
        ),
    )
    bpy.types.Scene.omnivoice_preprocess = bpy.props.BoolProperty(
        name="Preprocess Ref. Audio",
        default=True,
        description="Normalise reference-audio loudness before voice cloning",
    )
    bpy.types.Scene.omnivoice_denoise = bpy.props.BoolProperty(
        name="Denoise",
        default=True,
        description="Prepend a denoising token during diffusion (improves quality for noisy conditions)",
    )
    bpy.types.Scene.omnivoice_postprocess = bpy.props.BoolProperty(
        name="Remove Silence",
        default=True,
        description="Strip trailing silence from the generated audio output",
    )

    # MOSS-TTS
    from .utils.moss_tts_langs import MOSS_LANG_ITEMS as _MOSS_LANGS
    bpy.types.Scene.moss_model_variant = bpy.props.EnumProperty(
        name="Variant",
        items=[
            ("v1.5",     "MOSS-TTS-v1.5 (8B)",
             "Flagship. Best quality, voice cloning + 31 languages (incl. Danish) + "
             "duration control. Honours the Language setting and inline [pause Ns]. "
             "Needs ~16GB+ VRAM. Recommended for non-English text."),
            ("voicegen", "MOSS-VoiceGenerator (1.7B)",
             "Designs a NEW voice from a text description in the prompt — no reference "
             "audio (the Speaker Ref. field is hidden). Describe the voice, then the "
             "line to speak, e.g. 'A calm young woman. Hello, welcome!'"),
        ],
        default="v1.5",
        description="Which MOSS-TTS model variant to load (downloaded on first use)",
    )
    bpy.types.Scene.moss_language = bpy.props.EnumProperty(
        name="Language",
        items=_MOSS_LANGS,
        default="AUTO",
        description=(
            "Output language. Setting it explicitly improves quality on v1.5/TTSD — "
            "recommended whenever you know the language. AUTO lets the model infer it "
            "from the prompt. NOTE: MOSS-TTS-Nano ignores this and always auto-detects."
        ),
    )
    bpy.types.Scene.moss_duration_tokens = bpy.props.IntProperty(
        name="Duration Tokens",
        default=0,
        min=0,
        soft_max=2048,
        description=(
            "Token-level duration control. 0 = automatic length. A positive value "
            "forces a target length (v1.5: audio tokens; Nano: audio frames) — higher "
            "= longer. For precise pauses, prefer inline [pause 1.5s] markers in the "
            "prompt (v1.5)."
        ),
    )
    bpy.types.Scene.moss_max_new_tokens = bpy.props.IntProperty(
        name="Max New Tokens",
        default=4096,
        min=64,
        soft_max=8192,
        description=(
            "Upper bound on generated length, in audio tokens (v1.5/TTSD/VoiceGenerator). "
            "Raise it for long passages. Nano uses 'Duration Tokens' as a frame count "
            "instead and ignores this."
        ),
    )
    bpy.types.Scene.moss_temperature = bpy.props.FloatProperty(
        name="Temperature",
        default=1.7,
        min=0.0,
        soft_max=2.0,
        description=(
            "Audio sampling temperature. 1.7 is the MOSS default. Lower (~0.8-1.2) = "
            "steadier/cleaner; higher = more varied prosody but more artefacts. On v1.5, "
            "0 switches to greedy (deterministic)."
        ),
    )
    bpy.types.Scene.moss_top_p = bpy.props.FloatProperty(
        name="Top-p",
        default=0.8,
        min=0.0,
        max=1.0,
        description="Audio nucleus-sampling probability mass (MOSS default 0.8)",
    )
    bpy.types.Scene.moss_top_k = bpy.props.IntProperty(
        name="Top-k",
        default=25,
        min=0,
        soft_max=200,
        description="Audio top-k sampling cutoff (MOSS default 25; 0 disables top-k)",
    )
    bpy.types.Scene.moss_ref_audio_path = bpy.props.StringProperty(
        name="moss_ref_audio_path",
        default="",
        description=(
            "Path to a speaker reference audio for MOSS voice cloning (v1.5). Use a "
            "clean 24kHz+ mono clip. Ignored by VoiceGenerator, which designs a voice "
            "from the prompt text."
        ),
        options={"TEXTEDIT_UPDATE"},
    )

    # Fix read-only file attributes pip sometimes leaves on Windows.
    # Runs in a daemon thread so it doesn't slow down Blender startup.
    import threading as _threading
    def _startup_fix_permissions():
        try:
            from .operators.system import _fix_site_packages_permissions
            from .utils.helpers import site_packages_dir
            if site_packages_dir:
                _fix_site_packages_permissions(site_packages_dir)
        except Exception:
            pass
    _threading.Thread(target=_startup_fix_permissions, daemon=True).start()

    _restore_model_selections()
    if _restore_model_selections not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_restore_model_selections)

    _reset_queue_state()
    if _reset_queue_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_queue_state)

    # Pause the render-queue timer across .blend saves so a strip insert can't
    # land mid-serialisation (EXCEPTION_ACCESS_VIOLATION). Paired pre/post.
    if _queue_lock_on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_queue_lock_on_save_pre)
    if _queue_lock_on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_queue_lock_on_save_post)

    _reset_dep_state()
    if _reset_dep_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_dep_state)

    # Apply HuggingFace env vars (cache dir, Xet/CAS toggle) before any model
    # download can import huggingface_hub.  Re-applied on file load in case a
    # loaded .blend carried different preferences.
    _apply_hf_env_from_prefs()
    if _apply_hf_env_from_prefs not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_apply_hf_env_from_prefs)

    try:
        from .operators.mask_florence2 import register as _mf_register
        _mf_register()
    except Exception as _mfe:
        print(f"[Pallaidium] mask_florence2 register failed: {_mfe}")

    # Restore remote models discovered in a previous session (cineloom-style
    # persisted cache) so they appear in the dropdowns without re-querying a
    # backend. Best-effort: a missing/stale cache simply loads nothing.
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        from .utils.remote_backend import load_discovery_cache
        from .models import register_remote_models
        cached = load_discovery_cache(getattr(prefs, "remote_backend_url", ""))
        if cached:
            n = register_remote_models(cached, prefs)
            print(f"[Pallaidium] restored {n} cached remote model(s).")
    except Exception as _rce:
        print(f"[Pallaidium] remote model cache restore skipped: {_rce}")


def _apply_hf_env_from_prefs(_=None):
    """Set HF_HUB_CACHE / HF_HUB_DISABLE_XET from saved preferences.

    HF_HUB_DISABLE_XET is read by huggingface_hub at import time, so applying it
    here (at register and on file load) ensures it is in place before any plugin
    lazily imports huggingface_hub for a download.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
    except (AttributeError, KeyError):
        return
    try:
        from .utils.helpers import apply_hf_env
        apply_hf_env(prefs)
    except Exception as _e:
        print(f"[Pallaidium] apply_hf_env failed: {_e}")


def _reset_dep_state(_=None):
    """Clear stale dep-install runtime flags on every startup/file-load.

    SKIP_SAVE should prevent dep_needs_restart from persisting, but Blender
    sometimes writes AddonPreferences to userpref.blend regardless.  Resetting
    here guarantees the UI is never locked after a restart.

    If a _install_failures.json file exists from a previous failed install,
    the error state is restored so the user sees the error banner again.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
    except (AttributeError, KeyError):
        return
    prefs.dep_needs_restart = False
    prefs.dep_is_running    = False
    prefs.dep_progress      = 0.0
    prefs.dep_phase         = ""
    prefs.dep_status_line   = ""

    # Restore error state from disk if a previous install failed
    from .operators.system import load_install_errors_from_disk
    saved = load_install_errors_from_disk()
    if saved:
        prefs.dep_has_errors     = True
        prefs.dep_failure_report = saved.get("report", "")
    else:
        prefs.dep_has_errors     = False
        prefs.dep_failure_report = ""

def _restore_model_selections(_=None):
    """Sync each model-card EnumProperty from its string backing field.

    The EnumProperty uses SKIP_SAVE so its integer is never written to user
    preferences, eliminating stale-value RNA warnings.  The string backing
    (e.g. image_model_card_id) holds the MODEL_ID and is restored here after
    every Blender startup or .blend load.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
    except (AttributeError, KeyError):
        return

    from .models import get_enum_items as _gei

    for attr, backing, media_type in (
        ("image_model_card",  "image_model_card_id",  "image"),
        ("movie_model_card",  "movie_model_card_id",  "video"),
        ("audio_model_card",  "audio_model_card_id",  "audio"),
        ("text_model_card",   "text_model_card_id",   "text"),
    ):
        saved = getattr(prefs, backing, "")
        if not saved:
            continue
        valid = {item[0] for item in _gei(media_type)}
        if saved in valid:
            try:
                setattr(prefs, attr, saved)
            except Exception:
                pass

def _reset_queue_state(_=None):
    """On file load, clear stale runtime queue state left over from a previous
    session.  RUNNING/CANCELLING jobs are reset to PENDING so they can be
    re-submitted; queue_is_running is forced False because no background thread
    exists after restart.
    """
    # Clear any sequencer lock orphaned by a save whose save_post never fired,
    # so the queue timer isn't frozen after the next file load.
    try:
        from .utils.helpers import _sequencer_busy_event, sequencer_lock_release
        while _sequencer_busy_event.is_set():
            sequencer_lock_release()
    except Exception:
        pass
    try:
        scenes = bpy.data.scenes
    except AttributeError:
        return  # bpy.data not yet accessible (early startup restricted context)
    for scene in scenes:
        try:
            for job in scene.render_queue:
                if job.status in ("RUNNING", "CANCELLING"):
                    job.status   = "PENDING"
                    job.progress = 0.0
        except Exception:
            pass


def _queue_lock_on_save_pre(_=None):
    """Hold the sequencer lock for the duration of a .blend save.

    Saving serialises the whole scene; if the render-queue timer fires during
    the save and inserts a result strip (or starts a job that mutates the
    sequencer), the half-written data faults Blender. The lock makes _queue_tick
    idle until save_post clears it.
    """
    try:
        from .utils.helpers import sequencer_lock_acquire
        sequencer_lock_acquire()
    except Exception:
        pass


def _queue_lock_on_save_post(_=None):
    """Release the sequencer lock taken in _queue_lock_on_save_pre."""
    try:
        from .utils.helpers import sequencer_lock_release
        sequencer_lock_release()
    except Exception:
        pass


try:
    from bpy.app.handlers import persistent as _persistent
    _queue_lock_on_save_pre = _persistent(_queue_lock_on_save_pre)
    _queue_lock_on_save_post = _persistent(_queue_lock_on_save_post)
except Exception:
    pass


def unregister():
    # Stop any adapter subprocess Pallaidium launched so it never lingers after
    # the add-on is disabled or Blender quits.
    try:
        from .utils.adapter_launcher import stop_adapter
        stop_adapter()
    except Exception:
        pass
    try:
        from .operators.mask_florence2 import unregister as _mf_unregister
        _mf_unregister()
    except Exception:
        pass
    if _restore_model_selections in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_model_selections)
    if _reset_queue_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_queue_state)
    if _reset_dep_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_dep_state)
    if _apply_hf_env_from_prefs in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_apply_hf_env_from_prefs)
    if _queue_lock_on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_queue_lock_on_save_pre)
    if _queue_lock_on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_queue_lock_on_save_post)
    # Drop any lock left set by an interrupted save so a reload starts clean.
    try:
        from .utils.helpers import _sequencer_busy_event, sequencer_lock_release
        while _sequencer_busy_event.is_set():
            sequencer_lock_release()
    except Exception:
        pass
    from .operators.queue_ops import _queue_tick, _queue_stop
    try:
        if bpy.app.timers.is_registered(_queue_tick):
            bpy.app.timers.unregister(_queue_tick)
    except Exception:
        pass
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.generate_movie_prompt
    del bpy.types.Scene.generate_audio_prompt
    del bpy.types.Scene.generate_movie_x
    del bpy.types.Scene.generate_movie_y
    del bpy.types.Scene.movie_num_inference_steps
    del bpy.types.Scene.movie_num_batch
    del bpy.types.Scene.movie_num_seed
    del bpy.types.Scene.movie_use_random
    del bpy.types.Scene.movie_num_guidance
    del bpy.types.Scene.generatorai_typeselect
    del bpy.types.Scene.movie_path
    del bpy.types.Scene.image_path
    del bpy.types.Scene.sound_path
    del bpy.types.Scene.remote_generate_audio
    del bpy.types.Scene.ltx_middle_images_json
    del bpy.types.Scene.refine_sd
    del bpy.types.Scene.aurasr
    del bpy.types.Scene.adetailer
    del bpy.types.Scene.generatorai_styles
    del bpy.types.Scene.inpaint_selected_strip
    del bpy.types.Scene.out_frame
    del bpy.types.Scene.render_queue
    del bpy.types.Scene.lora_files
    del bpy.types.Scene.lora_files_index
    del bpy.types.Scene.ip_adapter_face_folder
    del bpy.types.Scene.ip_adapter_style_folder
    del bpy.types.Scene.ip_adapter_face_files_to_import
    del bpy.types.Scene.ip_adapter_style_files_to_import
    del bpy.types.Scene.music_bpm
    del bpy.types.Scene.music_lyrics
    del bpy.types.Scene.music_key_scale
    del bpy.types.Scene.music_time_signature
    del bpy.types.Scene.joyimage_spatial_mode
    del bpy.types.Scene.joyimage_object
    del bpy.types.Scene.joyimage_rotate_view
    del bpy.types.Scene.joyimage_yaw
    del bpy.types.Scene.joyimage_pitch
    del bpy.types.Scene.joyimage_zoom
    for _prop in ("ideogram_prompt_upsampling",):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("klein_schematic_mode", "klein_schematic_target"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("florence2_mode", "florence2_send_to_mask"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("marlin_mode", "marlin_find_query", "marlin_last_query"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("whisper_model_size", "whisper_language"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("mist_range_mode", "mist_custom_start", "mist_custom_depth", "mist_invert"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("stem_split_model", "stem_split_vocals", "stem_split_drums",
                  "stem_split_bass", "stem_split_other", "stem_split_guitar",
                  "stem_split_piano"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("omnivoice_language", "omnivoice_instruct", "omnivoice_preprocess",
                  "omnivoice_denoise", "omnivoice_postprocess"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("moss_model_variant", "moss_language", "moss_duration_tokens",
                  "moss_max_new_tokens", "moss_temperature", "moss_top_p", "moss_top_k",
                  "moss_ref_audio_path"):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in ("chatterbox_mtl_language",):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _prop in (
        "ltx23m_enable_guidance",
        "ltx23m_modality_scale", "ltx23m_audio_guidance", "ltx23m_audio_stg_scale",
        "ltx23m_audio_modality_scale", "ltx23m_audio_noise_scale", "ltx23m_audio_start_time",
        "ltx23m_image_strength",
        "ltx23ic_control_strip", "ltx23ic_control_strength", "ltx23ic_control_downscale",
        "ltx23ic_control_audio_str", "ltx23ic_identity_guidance",
        "ltx23ic_input_downscale_pct",
        "ltx23_stage_mode",
        "maxine_quality",
        "nano_banana_model", "nano_banana_aspect", "nano_banana_resolution",
        "veo_model", "veo_aspect", "veo_resolution", "veo_duration",
        "veo_person_generation",
        "veo_image_mode",
        "nano_banana_ref_count",
        *(f"nano_banana_ref_strip_{_n}" for _n in range(1, 10)),
        "veo_ref_strip_1", "veo_ref_strip_2", "veo_ref_strip_3",
    ):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)

if __name__ == "__main__":
    register()
