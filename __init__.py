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
    AI_Metadata_PT_Panel,
    OBJECT_OT_FluxAddStrip,
    OBJECT_OT_FluxHideStrip,
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
        default=1024,
        step=32,
        min=224,
        max=4096,
        description="Use the power of 64",
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=576,
        step=32,
        min=224,
        max=4096,
        description="Use the power of 64",
    )

    # The number of frames to be generated.
    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_frames",
        default=6,
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

    _reset_dep_state()
    if _reset_dep_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_dep_state)

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


def unregister():
    if _restore_model_selections in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_model_selections)
    if _reset_queue_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_queue_state)
    if _reset_dep_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_dep_state)
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
    for _prop in ("marlin_mode", "marlin_find_query", "marlin_last_query"):
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

if __name__ == "__main__":
    register()
