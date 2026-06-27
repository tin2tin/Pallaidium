"""Async render queue for Pallaidium Generative AI.

Jobs are enqueued from the VSE panel, executed one at a time in a background
thread, and strips are inserted back into the VSE timeline from the main
thread via bpy.app.timers (runs on the main thread, no context required).

Thread safety contract:
  - _worker_thread, _cancel_event, _result_queue, _progress_store are only
    mutated from the main thread EXCEPT:
      • _progress_store[job_id] is written by the worker (float assignment
        under the GIL is atomic enough for progress polling)
      • _result_queue.put() is called by the worker (thread-safe by design)
  - All bpy.types / bpy.context access happens in the main thread only.
  - The worker receives a plain dict snapshot; it never touches bpy.
"""
from __future__ import annotations

import gc
import json
import os
import random
import threading
import traceback
import types
import queue as _stdqueue
from datetime import date

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator, PropertyGroup

from ..utils.helpers import (
    ADDON_ID,
    apply_hf_env,
    clean_filename,
    clear_cuda_cache,
    closest_divisible_32,
    find_first_empty_channel,
    load_first_frame,
    release_model_cache,
    set_ai_metadata_from_dict,
    set_system_console_topmost,
    show_system_console,
    style_prompt,
)

# ---------------------------------------------------------------------------
# Reload / scene-input diagnostic — shared log file with main_ops.py so a single
# file shows what BOTH the immediate path and the queue path decided.
# ---------------------------------------------------------------------------
import time as _time
_PALLAIDIUM_DIAG_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "_diag_scene_input.log")

def _diag(msg):
    try:
        with open(_PALLAIDIUM_DIAG_LOG, "a", encoding="utf-8") as _fh:
            _fh.write(f"{_time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except Exception:
        pass

_diag("=== queue_ops.py IMPORTED (build: scene-strip-v3) ===")

# ---------------------------------------------------------------------------
# Module-level thread state  (main thread writes, worker reads via closure)
# ---------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_cancel_event = threading.Event()
_result_queue = _stdqueue.Queue()
_progress_store: dict = {}  # job_id → {"progress": float, "phase": str, "step": int, "total": int}
_queue_paused: bool = False  # True = timer stays alive but won't start new jobs
_active_scene = None  # direct bpy.types.Scene reference — avoids bpy.data in timer


# ---------------------------------------------------------------------------
# RenderQueueJob  —  one entry per queued job
# ---------------------------------------------------------------------------

class RenderQueueJob(PropertyGroup):
    """Stores a complete, self-contained snapshot of one generation job."""

    job_id:      StringProperty()
    output_type: StringProperty()   # "image" | "movie" | "audio" | "text"
    model_card:  StringProperty()

    status: EnumProperty(
        items=[
            ("PENDING",    "Pending",    "", "STRIP_COLOR_03", 0),
            ("RUNNING",    "Running",    "", "STRIP_COLOR_05", 1),
            ("COMPLETED",  "Done",       "", "STRIP_COLOR_04", 2),
            ("FAILED",     "Failed",     "", "STRIP_COLOR_01", 3),
            ("CANCELLED",  "Cancelled",  "", "STRIP_COLOR_09", 4),
            ("CANCELLING", "Stopping",   "", "STRIP_COLOR_02", 5),
        ],
        default="PENDING",
    )
    progress:             FloatProperty(min=0.0, max=1.0, default=0.0)
    phase:                StringProperty(default="")
    current_step:         IntProperty(default=0)
    total_steps:          IntProperty(default=0)
    sequencer_scene_name: StringProperty(default="")
    error_message:        StringProperty()
    error_traceback:      StringProperty()
    output_path:          StringProperty()
    tokens_info:          StringProperty()   # human-readable usage note (e.g. token cost)

    # Insertion point — calculated at add-time so user actions cannot
    # displace the intended location.
    insert_frame_start: IntProperty()
    insert_frame_end:   IntProperty()
    insert_channel:     IntProperty()
    insert_duration:    IntProperty()

    # Generation params snapshot
    prompt:       StringProperty()
    neg_prompt:   StringProperty()
    steps:        IntProperty()
    guidance:     FloatProperty()
    seed:         IntProperty()
    use_random:   BoolProperty()
    width:        IntProperty()
    height:       IntProperty()
    frames:       IntProperty()
    image_power:  FloatProperty()
    mode:         StringProperty()  # "txt2img" | "img2img" | "inpaint" | …

    use_lcm:           BoolProperty()
    refine_sd:         BoolProperty()
    adetailer:         BoolProperty()
    aurasr:            BoolProperty()
    remove_silence:    BoolProperty()
    audio_length:      FloatProperty()
    audio_speed_tts:   FloatProperty()
    chat_exaggeration: FloatProperty()
    chat_pace:         FloatProperty()
    chat_temperature:  FloatProperty()
    fps:               FloatProperty()
    music_bpm:         IntProperty()
    music_lyrics:      StringProperty()
    music_key_scale:   StringProperty()
    music_time_signature: StringProperty()

    # Resolved input file paths (resolved from strips at add-time)
    image_path:      StringProperty()
    movie_path:      StringProperty()
    sound_path:      StringProperty()
    last_image_path:   StringProperty()   # FLF/LFO last-frame image (LTX Multi)
    middle_images_json: StringProperty(default="")  # N-anchor middle images: JSON [[path, fraction], ...] (LTX Multi)
    ref_audio_path:  StringProperty()
    ref_text:        StringProperty()

    # Prefs snapshot
    hugginface_token: StringProperty()
    gemini_api_key:   StringProperty()
    remote_backend_url: StringProperty()
    remote_backend_key: StringProperty()
    local_files_only: BoolProperty()
    display_console:  BoolProperty(default=False)
    generator_ai:     StringProperty()
    hf_cache_dir:     StringProperty()

    # LoRA files — JSON list of {"name": path, "weight": float, "enabled": bool}
    lora_files_json: StringProperty()
    lora_folder:     StringProperty()   # absolute path to the LoRA folder

    # JoyAI spatial editing snapshot (captured at queue-add time)
    joyimage_spatial_mode: StringProperty(default="general")
    joyimage_object:       StringProperty(default="object")
    joyimage_rotate_view:  StringProperty(default="front")
    joyimage_yaw:          FloatProperty(default=0.0)
    joyimage_pitch:        FloatProperty(default=0.0)
    joyimage_zoom:         StringProperty(default="unchanged")

    # Resolved strip paths for model-specific inputs (captured at enqueue time)
    kontext_strip_1_path: StringProperty()   # FLUX Kontext reference / fallback image
    inpaint_mask_path:    StringProperty()   # inpaint mask image path
    qwen_strip_1_path:    StringProperty()   # Qwen Image Edit reference image 1
    qwen_strip_2_path:    StringProperty()   # Qwen Image Edit reference image 2
    qwen_strip_3_path:    StringProperty()   # Qwen Image Edit reference image 3
    klein_strip_1_path:   StringProperty()   # Klein reference image 1
    klein_strip_2_path:   StringProperty()   # Klein reference image 2
    klein_strip_3_path:   StringProperty()   # Klein reference image 3
    # FLUX.2 / remote multi-image reference strips (flux_strip_N pickers, up to 9)
    flux_strip_1_path:    StringProperty()
    flux_strip_2_path:    StringProperty()
    flux_strip_3_path:    StringProperty()
    flux_strip_4_path:    StringProperty()
    flux_strip_5_path:    StringProperty()
    flux_strip_6_path:    StringProperty()
    flux_strip_7_path:    StringProperty()
    flux_strip_8_path:    StringProperty()
    flux_strip_9_path:    StringProperty()
    minimax_subject_path: StringProperty()   # MiniMax subject2vid reference image
    # Nano Banana reference images (up to 9; nano_banana_ref_count = how many are active)
    nano_banana_ref_count: IntProperty(default=3)
    nano_banana_ref_strip_1_path: StringProperty()
    nano_banana_ref_strip_2_path: StringProperty()
    nano_banana_ref_strip_3_path: StringProperty()
    nano_banana_ref_strip_4_path: StringProperty()
    nano_banana_ref_strip_5_path: StringProperty()
    nano_banana_ref_strip_6_path: StringProperty()
    nano_banana_ref_strip_7_path: StringProperty()
    nano_banana_ref_strip_8_path: StringProperty()
    nano_banana_ref_strip_9_path: StringProperty()
    veo_ref_strip_1_path: StringProperty()   # Veo 3.1 reference image 1
    veo_ref_strip_2_path: StringProperty()   # Veo 3.1 reference image 2
    veo_ref_strip_3_path: StringProperty()   # Veo 3.1 reference image 3
    # Source strip names (persisted to metadata so Redo can re-render the refs)
    nano_banana_ref_strip_1: StringProperty()
    nano_banana_ref_strip_2: StringProperty()
    nano_banana_ref_strip_3: StringProperty()
    nano_banana_ref_strip_4: StringProperty()
    nano_banana_ref_strip_5: StringProperty()
    nano_banana_ref_strip_6: StringProperty()
    nano_banana_ref_strip_7: StringProperty()
    nano_banana_ref_strip_8: StringProperty()
    nano_banana_ref_strip_9: StringProperty()
    veo_ref_strip_1: StringProperty()
    veo_ref_strip_2: StringProperty()
    veo_ref_strip_3: StringProperty()

    # Faster Whisper Transcription
    whisper_model_size: StringProperty(default="large-v3-turbo")
    whisper_language:   StringProperty(default="auto")

    # OmniVoice
    omnivoice_instruct:    StringProperty(default="")
    omnivoice_language:    StringProperty(default="")
    omnivoice_preprocess:  BoolProperty(default=True)
    omnivoice_denoise:     BoolProperty(default=True)
    omnivoice_postprocess: BoolProperty(default=True)

    # Chatterbox Multilingual
    chatterbox_mtl_language: StringProperty(default="en")

    # MOSS-TTS
    moss_model_variant:   StringProperty(default="v1.5")
    moss_language:        StringProperty(default="AUTO")
    moss_duration_tokens: IntProperty(default=0)
    moss_max_new_tokens:  IntProperty(default=4096)
    moss_temperature:     FloatProperty(default=1.7)
    moss_top_p:           FloatProperty(default=0.8)
    moss_top_k:           IntProperty(default=25)
    moss_ref_audio_path:  StringProperty(default="")

    # Stem Splitter
    stem_split_model:  StringProperty(default="htdemucs_ft")
    stem_split_vocals: BoolProperty(default=True)
    stem_split_drums:  BoolProperty(default=True)
    stem_split_bass:   BoolProperty(default=True)
    stem_split_other:  BoolProperty(default=True)
    stem_split_guitar: BoolProperty(default=False)
    stem_split_piano:  BoolProperty(default=False)

    # Florence-2 plugin
    florence2_mode:         StringProperty(default="CAPTION")
    florence2_send_to_mask: BoolProperty(default=False)

    # Klein Schematic LoRA plugin
    klein_schematic_mode:   StringProperty(default="DEPTH")
    klein_schematic_target: StringProperty(default="person")

    # Extra UI properties
    img_guidance_scale:    FloatProperty(default=1.6)
    illumination_style:    StringProperty(default="")
    light_direction:       StringProperty(default="")
    ip_adapter_face_folder:  StringProperty(default="")
    ip_adapter_style_folder: StringProperty(default="")
    openpose_use_bones:    BoolProperty(default=False)
    use_scribble_image:    BoolProperty(default=False)
    ideogram_prompt_upsampling: BoolProperty(default=False)

    # ltx23_multi_v2 — independent audio/modality guidance
    ltx23m_modality_scale:       FloatProperty(default=1.0)
    ltx23m_audio_guidance:       FloatProperty(default=1.0)
    ltx23m_audio_stg_scale:      FloatProperty(default=0.0)
    ltx23m_audio_modality_scale: FloatProperty(default=1.0)
    ltx23m_audio_noise_scale:    FloatProperty(default=0.0)
    ltx23m_audio_start_time:     FloatProperty(default=0.0)

    # ltx23_multi_ic_lora — IC-LoRA control paths + params
    ltx23ic_control_video_path:  StringProperty(default="")
    ltx23ic_control_audio_path:  StringProperty(default="")
    ltx23ic_control_strength:    FloatProperty(default=1.0)
    ltx23ic_control_downscale:   IntProperty(default=1)
    ltx23ic_control_audio_str:   FloatProperty(default=1.0)
    ltx23ic_identity_guidance:   FloatProperty(default=0.0)
    # 3DREAL mode: dropdown holds an IMAGE → frame-0 appearance reference,
    # while the MAIN input video drives control_video.
    ltx23ic_ref_image_path:      StringProperty(default="")

    # ltx23_extend — clip extension params + resolved audio-strip path
    ltx23ext_extend_frames:      IntProperty(default=96)
    ltx23ext_video_strength:     FloatProperty(default=1.0)
    ltx23ext_audio_path:         StringProperty(default="")

    # ltx23 staged — stage-mode enum
    ltx23_stage_mode:            StringProperty(default="FULL")

    # Maxine VSR
    maxine_quality:              StringProperty(default="HIGH")

    # Google Nano Banana (Gemini image)
    nano_banana_model:           StringProperty(default="gemini-2.5-flash-image")
    nano_banana_aspect:          StringProperty(default="1:1")
    nano_banana_resolution:      StringProperty(default="1K")

    # Google Veo (video)
    veo_model:                   StringProperty(default="veo-3.1-fast-generate-preview")
    veo_aspect:                  StringProperty(default="16:9")
    veo_resolution:              StringProperty(default="720p")
    veo_duration:                StringProperty(default="8")
    veo_person_generation:       StringProperty(default="allow_adult")
    veo_image_mode:              StringProperty(default="AUTO")

    # Marlin Video Captions
    marlin_mode:        StringProperty(default="CAPTION")
    marlin_find_query:  StringProperty(default="")

    # VRAM management — set at run-time based on the next queued job
    should_unload: BoolProperty(default=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_job(scene, job_id: str):
    for job in scene.render_queue:
        if job.job_id == job_id:
            return job
    return None


def _find_next_pending(scene):
    for job in scene.render_queue:
        if job.status == "PENDING":
            return job
    return None


def _find_running_job(scene):
    for job in scene.render_queue:
        if job.status in ("RUNNING", "CANCELLING"):
            return job
    return None


def _find_free_channel(scene, frame_start: int, frame_end: int, preferred: int) -> int:
    """Scan upward from preferred_channel until a free slot is found.

    Called at insert-time so it reflects whatever has been added since the
    job was enqueued.
    """
    channel = max(1, preferred)
    for _ in range(128):
        occupied = any(
            s.channel == channel
            and s.frame_final_start < frame_end
            and (s.frame_final_start + s.frame_final_duration) > frame_start
            for s in scene.sequence_editor.strips_all
        )
        if not occupied:
            return channel
        channel += 1
    return max(1, preferred)


def _queue_solve_path(filename: str, generator_ai_dir: str) -> str:
    """Thread-safe output path resolver — does NOT use bpy.context."""
    out_dir = os.path.join(generator_ai_dir, str(date.today()))
    os.makedirs(out_dir, exist_ok=True)
    base = clean_filename(os.path.splitext(os.path.basename(filename))[0])
    ext = os.path.splitext(filename)[1] or ".png"
    out = os.path.join(out_dir, base + ext)
    i = 1
    while os.path.exists(out):
        out = os.path.join(out_dir, f"{base}({i}){ext}")
        i += 1
    return out


# ---------------------------------------------------------------------------
# Background worker — no bpy access allowed
# ---------------------------------------------------------------------------

def _run_job(snapshot: dict, result_queue, cancel_event, progress_store) -> None:
    """Runs in a background thread.

    Loads the model, runs inference, puts a result dict into result_queue,
    then exits.  Never touches bpy.context; uses SimpleNamespace proxies so
    plugins that read scene/prefs attributes still work.
    """
    job_id = snapshot["job_id"]
    progress_store[job_id] = {"progress": 0.0, "phase": "Queued", "step": 0, "total": 0}
    pipe_obj = None
    model_cache = None  # shared cache dict from main_ops; set once otype is known

    try:
        from ..models import get_plugin
        from ..models.base import ModelInputs

        plugin = get_plugin(snapshot["model_card"])
        if plugin is None:
            raise RuntimeError(f"Plugin not found: {snapshot['model_card']!r}")

        otype = snapshot["output_type"]
        model_card = snapshot["model_card"]

        # ---- Grab the shared model cache for this output type -----------
        from ..operators.main_ops import (
            _pallaidium_model_cache,
            _pallaidium_movie_model_cache,
            _pallaidium_audio_model_cache,
            _pallaidium_text_model_cache,
        )
        model_cache = {
            "image": _pallaidium_model_cache,
            "movie": _pallaidium_movie_model_cache,
            "audio": _pallaidium_audio_model_cache,
            "text":  _pallaidium_text_model_cache,
        }.get(otype)

        # ---- Reconstruct LoRA list first so scene_proxy can reference it ---
        try:
            lora_raw = json.loads(snapshot.get("lora_files_json", "[]"))
        except Exception:
            lora_raw = []
        enabled_items = [
            types.SimpleNamespace(
                name         = item["name"],
                weight_value = item.get("weight", 1.0),
                enabled      = True,
            )
            for item in lora_raw
            if item.get("enabled")
        ]

        # ---- Proxy objects — every field comes from the snapshot ----------
        prefs_proxy = types.SimpleNamespace(
            hugginface_token = snapshot.get("hugginface_token", ""),
            gemini_api_key   = snapshot.get("gemini_api_key", ""),
            remote_backend_url = snapshot.get("remote_backend_url", ""),
            remote_backend_key = snapshot.get("remote_backend_key", ""),
            local_files_only = snapshot.get("local_files_only", False),
            generator_ai     = snapshot.get("generator_ai", ""),
            hf_cache_dir     = snapshot.get("hf_cache_dir", ""),
            image_model_card = model_card if otype == "image" else "",
            movie_model_card = model_card if otype == "movie" else "",
            audio_model_card = model_card if otype == "audio" else "",
            text_model_card  = model_card if otype == "text"  else "",
        )
        scene_proxy = types.SimpleNamespace(
            generate_movie_prompt          = snapshot["prompt"],
            generate_movie_negative_prompt = snapshot["neg_prompt"],
            generate_movie_x               = snapshot["width"],
            generate_movie_y               = snapshot["height"],
            generate_movie_frames          = snapshot["frames"],
            movie_num_inference_steps      = snapshot["steps"],
            movie_num_guidance             = snapshot["guidance"],
            movie_num_seed                 = snapshot["seed"],
            movie_use_random               = snapshot["use_random"],
            image_power                    = snapshot["image_power"],
            use_lcm                        = snapshot["use_lcm"],
            refine_sd                      = snapshot["refine_sd"],
            adetailer                      = snapshot["adetailer"],
            aurasr                         = snapshot["aurasr"],
            remove_silence                 = snapshot["remove_silence"],
            audio_length_in_f              = int(snapshot["audio_length"]),
            audio_speed_tts                = snapshot["audio_speed_tts"],
            chat_exaggeration              = snapshot["chat_exaggeration"],
            chat_pace                      = snapshot["chat_pace"],
            chat_temperature               = snapshot["chat_temperature"],
            input_strips                   = (
                "input_strips"
                if (snapshot.get("image_path") or snapshot.get("movie_path"))
                else "input_prompt"
            ),
            image_path                     = snapshot.get("image_path", ""),
            movie_path                     = snapshot.get("movie_path", ""),
            sound_path                     = snapshot.get("sound_path", ""),
            ref_audio_path                 = snapshot.get("ref_audio_path", ""),
            ref_text                       = snapshot.get("ref_text", ""),
            inpaint_selected_strip         = "",
            kontext_strip_1                = "",
            kontext_strip_1_path           = snapshot.get("kontext_strip_1_path", ""),
            qwen_strip_1_path              = snapshot.get("qwen_strip_1_path", ""),
            qwen_strip_2_path              = snapshot.get("qwen_strip_2_path", ""),
            qwen_strip_3_path              = snapshot.get("qwen_strip_3_path", ""),
            music_bpm                      = snapshot["music_bpm"],
            music_lyrics                   = snapshot["music_lyrics"],
            music_key_scale                = snapshot["music_key_scale"],
            music_time_signature           = snapshot["music_time_signature"],
            ip_adapter_face_folder         = snapshot.get("ip_adapter_face_folder", ""),
            ip_adapter_style_folder        = snapshot.get("ip_adapter_style_folder", ""),
            svd_decode_chunk_size          = 2,
            svd_motion_bucket_id           = 1,
            img_guidance_scale             = snapshot.get("img_guidance_scale", 1.6),
            illumination_style             = snapshot.get("illumination_style", ""),
            light_direction                = snapshot.get("light_direction", ""),
            openpose_use_bones             = snapshot.get("openpose_use_bones", False),
            use_scribble_image             = snapshot.get("use_scribble_image", False),
            ideogram_prompt_upsampling     = snapshot.get("ideogram_prompt_upsampling", False),
            # ltx23_multi_v2 guidance params
            ltx23m_modality_scale          = snapshot.get("ltx23m_modality_scale",       1.0),
            ltx23m_audio_guidance          = snapshot.get("ltx23m_audio_guidance",       1.0),
            ltx23m_audio_stg_scale         = snapshot.get("ltx23m_audio_stg_scale",      0.0),
            ltx23m_audio_modality_scale    = snapshot.get("ltx23m_audio_modality_scale", 1.0),
            ltx23m_audio_noise_scale       = snapshot.get("ltx23m_audio_noise_scale",    0.0),
            ltx23m_audio_start_time        = snapshot.get("ltx23m_audio_start_time",     0.0),
            # ltx23_multi_ic_lora params
            ltx23ic_control_video_path     = snapshot.get("ltx23ic_control_video_path",  ""),
            ltx23ic_control_audio_path     = snapshot.get("ltx23ic_control_audio_path",  ""),
            ltx23ic_control_strength       = snapshot.get("ltx23ic_control_strength",    1.0),
            ltx23ic_control_downscale      = snapshot.get("ltx23ic_control_downscale",   1),
            ltx23ic_control_audio_str      = snapshot.get("ltx23ic_control_audio_str",   1.0),
            ltx23ic_identity_guidance      = snapshot.get("ltx23ic_identity_guidance",   0.0),
            ltx23ic_ref_image_path         = snapshot.get("ltx23ic_ref_image_path",      ""),
            # ltx23_extend params
            ltx23ext_extend_frames         = snapshot.get("ltx23ext_extend_frames",      96),
            ltx23ext_video_strength        = snapshot.get("ltx23ext_video_strength",     1.0),
            ltx23ext_audio_path            = snapshot.get("ltx23ext_audio_path",         ""),
            ltx23_stage_mode               = snapshot.get("ltx23_stage_mode",            "FULL"),
            maxine_quality                 = snapshot.get("maxine_quality",              "HIGH"),
            # Google Nano Banana / Veo cloud settings
            nano_banana_model              = snapshot.get("nano_banana_model",      "gemini-2.5-flash-image"),
            nano_banana_aspect             = snapshot.get("nano_banana_aspect",     "1:1"),
            nano_banana_resolution         = snapshot.get("nano_banana_resolution", "1K"),
            veo_model                      = snapshot.get("veo_model",              "veo-3.1-fast-generate-preview"),
            veo_aspect                     = snapshot.get("veo_aspect",             "16:9"),
            veo_resolution                 = snapshot.get("veo_resolution",         "720p"),
            veo_duration                   = snapshot.get("veo_duration",           "8"),
            veo_person_generation          = snapshot.get("veo_person_generation",  "allow_adult"),
            veo_image_mode                 = snapshot.get("veo_image_mode",         "AUTO"),
            marlin_mode                    = snapshot.get("marlin_mode",                 "CAPTION"),
            marlin_find_query              = snapshot.get("marlin_find_query",           ""),
            marlin_last_query              = "",
            lora_files                     = enabled_items,
            lora_folder                    = snapshot.get("lora_folder", ""),
            render                         = types.SimpleNamespace(
                fps=round(snapshot.get("fps", 24.0)), fps_base=1.0
            ),
            sequencer_scene_name = snapshot.get("sequencer_scene_name", ""),
            whisper_model_size = snapshot.get("whisper_model_size", "large-v3-turbo"),
            whisper_language   = snapshot.get("whisper_language",   ""),
            stem_split_model  = snapshot.get("stem_split_model",  "htdemucs_ft"),
            stem_split_vocals = snapshot.get("stem_split_vocals", True),
            stem_split_drums  = snapshot.get("stem_split_drums",  True),
            stem_split_bass   = snapshot.get("stem_split_bass",   True),
            stem_split_other  = snapshot.get("stem_split_other",  True),
            stem_split_guitar  = snapshot.get("stem_split_guitar",  False),
            stem_split_piano   = snapshot.get("stem_split_piano",   False),
            omnivoice_instruct    = snapshot.get("omnivoice_instruct",    ""),
            omnivoice_language    = snapshot.get("omnivoice_language",    ""),
            omnivoice_preprocess  = snapshot.get("omnivoice_preprocess",  True),
            omnivoice_denoise     = snapshot.get("omnivoice_denoise",     True),
            omnivoice_postprocess = snapshot.get("omnivoice_postprocess", True),
            chatterbox_mtl_language = snapshot.get("chatterbox_mtl_language", "en"),
            moss_model_variant    = snapshot.get("moss_model_variant",    "v1.5"),
            moss_language         = snapshot.get("moss_language",         "AUTO"),
            moss_duration_tokens  = snapshot.get("moss_duration_tokens",  0),
            moss_max_new_tokens   = snapshot.get("moss_max_new_tokens",   4096),
            moss_temperature      = snapshot.get("moss_temperature",      1.7),
            moss_top_p            = snapshot.get("moss_top_p",            0.8),
            moss_top_k            = snapshot.get("moss_top_k",            25),
            moss_ref_audio_path   = snapshot.get("moss_ref_audio_path",   ""),
            florence2_mode         = snapshot.get("florence2_mode",         "CAPTION"),
            florence2_send_to_mask = snapshot.get("florence2_send_to_mask", False),
            klein_schematic_mode   = snapshot.get("klein_schematic_mode",   "DEPTH"),
            klein_schematic_target = snapshot.get("klein_schematic_target", "person"),
            klein_strip_1_path     = snapshot.get("klein_strip_1_path",     ""),
            klein_strip_2_path     = snapshot.get("klein_strip_2_path",     ""),
            klein_strip_3_path     = snapshot.get("klein_strip_3_path",     ""),
            minimax_subject_path   = snapshot.get("minimax_subject_path",   ""),
            **{f"flux_strip_{_n}_path": snapshot.get(f"flux_strip_{_n}_path", "")
               for _n in range(1, 10)},
            nano_banana_ref_count  = snapshot.get("nano_banana_ref_count", 3),
            nano_banana_ref_strip_1_path = snapshot.get("nano_banana_ref_strip_1_path", ""),
            nano_banana_ref_strip_2_path = snapshot.get("nano_banana_ref_strip_2_path", ""),
            nano_banana_ref_strip_3_path = snapshot.get("nano_banana_ref_strip_3_path", ""),
            nano_banana_ref_strip_4_path = snapshot.get("nano_banana_ref_strip_4_path", ""),
            nano_banana_ref_strip_5_path = snapshot.get("nano_banana_ref_strip_5_path", ""),
            nano_banana_ref_strip_6_path = snapshot.get("nano_banana_ref_strip_6_path", ""),
            nano_banana_ref_strip_7_path = snapshot.get("nano_banana_ref_strip_7_path", ""),
            nano_banana_ref_strip_8_path = snapshot.get("nano_banana_ref_strip_8_path", ""),
            nano_banana_ref_strip_9_path = snapshot.get("nano_banana_ref_strip_9_path", ""),
            veo_ref_strip_1_path   = snapshot.get("veo_ref_strip_1_path",   ""),
            veo_ref_strip_2_path   = snapshot.get("veo_ref_strip_2_path",   ""),
            veo_ref_strip_3_path   = snapshot.get("veo_ref_strip_3_path",   ""),
        )

        mode = snapshot["mode"]

        if cancel_event.is_set():
            result_queue.put({"job_id": job_id, "status": "CANCELLED"})
            return

        # ---- Load model (or reuse from cache) ---------------------------
        progress_store[job_id] = {"progress": 0.02, "phase": "Loading model", "step": 0, "total": 0}
        if snapshot.get("display_console", True):
            show_system_console(True)
            set_system_console_topmost(True)

        # Ensure self-registering packages (e.g. sdnq) are imported before
        # generate() so their side-effects (registering quantization backends,
        # etc.) are visible to the loaded model weights.
        import importlib as _importlib
        for _pkg in getattr(plugin, "REQUIRED_PACKAGES", []):
            _pkg_name = _pkg if isinstance(_pkg, str) else _pkg[0]
            try:
                _importlib.import_module(_pkg_name)
            except (ImportError, PermissionError, OSError):
                pass

        # Many plugins load different pipeline objects per mode (e.g. ZImage
        # loads `pipe` for txt2img but `converter` for img2img — never both).
        # A cache built for one mode cannot serve another mode of the same
        # model, so we include `last_mode` in the hit check.
        _schematic_mode = snapshot.get("klein_schematic_mode", "")
        _lora_key = snapshot.get("lora_files_json", "[]")
        _cache_skip = {"last_model_card", "last_mode", "last_schematic_mode", "last_lora_key"}
        cache_hit = (
            model_cache is not None
            and model_cache.get("last_model_card") == model_card
            and model_cache.get("last_mode") == mode
            and model_cache.get("last_schematic_mode", "") == _schematic_mode
            and model_cache.get("last_lora_key", "[]") == _lora_key
            and any(v is not None for k, v in model_cache.items() if k not in _cache_skip)
        )

        if cache_hit:
            pipe_obj = model_cache
            print(f"[Queue] Reusing cached model: {model_card} ({mode}) schematic={_schematic_mode or '-'}")
        else:
            # Release a different model (or same model with incompatible mode)
            if (model_cache is not None
                    and (model_cache.get("last_model_card") != model_card
                         or model_cache.get("last_mode") != mode
                         or model_cache.get("last_schematic_mode", "") != _schematic_mode)):
                release_model_cache(model_cache)
            clear_cuda_cache()
            apply_hf_env(prefs_proxy)

            # Set initial state before load
            progress_store[job_id] = {
                "progress": 0.0,
                "phase":    "Loading model",
                "step":     0,
                "total":    0,
            }

            # Patch tqdm.std.tqdm.__init__ and .update directly on the base class.
            # This works regardless of when huggingface_hub imported tqdm, because
            # huggingface_hub.utils.tqdm.tqdm inherits from tqdm.std.tqdm and does
            # not override update() — so our patch is inherited by every instance.
            # Running in the download thread means KeyboardInterrupt here genuinely
            # aborts the HTTP transfer.
            try:
                import tqdm.std as _tqdm_std
            except Exception:
                _tqdm_std = None

            _active_bars: dict = {}   # id(bar) → [bytes_done, bytes_total]
            _dl_bars: set = set()     # ids of bars that are actual network downloads (unit='B')

            if _tqdm_std is not None:
                _orig_tqdm_init   = _tqdm_std.tqdm.__init__
                _orig_tqdm_update = _tqdm_std.tqdm.update

                def _patched_tqdm_init(tqdm_self, *a, **kw):
                    _orig_tqdm_init(tqdm_self, *a, **kw)
                    if not getattr(tqdm_self, "disable", False):
                        _active_bars[id(tqdm_self)] = [tqdm_self.n or 0, tqdm_self.total or 0]
                        if getattr(tqdm_self, "unit", "it") == "B":
                            _dl_bars.add(id(tqdm_self))

                def _patched_tqdm_update(tqdm_self, n=1):
                    if cancel_event.is_set():
                        raise KeyboardInterrupt("Queue job cancelled during download")
                    result = _orig_tqdm_update(tqdm_self, n)
                    entry = _active_bars.get(id(tqdm_self))
                    if entry is not None:
                        entry[0] = tqdm_self.n or 0
                        entry[1] = tqdm_self.total or 0
                    total_b = sum(v[1] for v in _active_bars.values() if v[1] > 0)
                    done_b  = sum(v[0] for v in _active_bars.values())
                    dl_frac = (done_b / total_b) if total_b > 0 else 0.0
                    phase = "Downloading model" if id(tqdm_self) in _dl_bars else "Loading model"
                    progress_store[job_id] = {
                        "progress": dl_frac,
                        "phase":    phase,
                        "step":     max(0, int(done_b  / 1_048_576)),
                        "total":    max(1, int(total_b / 1_048_576)),
                    }
                    return result

                _tqdm_std.tqdm.__init__ = _patched_tqdm_init
                _tqdm_std.tqdm.update   = _patched_tqdm_update

            if getattr(plugin, "requires_main_thread_for_load", False):
                # Plugin's load() crashes in worker threads on Windows
                # (e.g. PyTorch CPU ops via safetensors). Schedule it on
                # the main thread via a one-shot timer; worker blocks here
                # until done, then continues generate() in the thread.
                _load_done   = threading.Event()
                _load_result = {}

                def _main_thread_load():
                    try:
                        _load_result["loaded"] = plugin.load(
                            prefs_proxy, scene_proxy,
                            mode=mode, enabled_items=enabled_items,
                            use_lcm=snapshot["use_lcm"],
                            use_refine=snapshot["refine_sd"],
                            ip_adapter_face_folder=snapshot.get("ip_adapter_face_folder", ""),
                            ip_adapter_style_folder=snapshot.get("ip_adapter_style_folder", ""),
                            local_files_only=snapshot["local_files_only"],
                        )
                    except Exception as _exc:
                        _load_result["error"] = _exc
                    finally:
                        if _tqdm_std is not None:
                            _tqdm_std.tqdm.__init__ = _orig_tqdm_init
                            _tqdm_std.tqdm.update   = _orig_tqdm_update
                        _active_bars.clear()
                        _dl_bars.clear()
                        _load_done.set()
                    return None  # don't re-register the timer

                bpy.app.timers.register(_main_thread_load, first_interval=0)
                _load_done.wait()

                if "error" in _load_result:
                    raise _load_result["error"]
                loaded = _load_result["loaded"]
            else:
                try:
                    loaded = plugin.load(
                        prefs_proxy,
                        scene_proxy,
                        mode=mode,
                        enabled_items=enabled_items,
                        use_lcm=snapshot["use_lcm"],
                        use_refine=snapshot["refine_sd"],
                        ip_adapter_face_folder=snapshot.get("ip_adapter_face_folder", ""),
                        ip_adapter_style_folder=snapshot.get("ip_adapter_style_folder", ""),
                        local_files_only=snapshot["local_files_only"],
                    )
                finally:
                    if _tqdm_std is not None:
                        _tqdm_std.tqdm.__init__ = _orig_tqdm_init
                        _tqdm_std.tqdm.update   = _orig_tqdm_update
                    _active_bars.clear()
                    _dl_bars.clear()

            # If cancelled during load, stop here
            if cancel_event.is_set():
                result_queue.put({"job_id": job_id, "status": "CANCELLED"})
                return

            progress_store[job_id] = {
                "progress": 0.10, "phase": "Loaded", "step": 0, "total": 0,
            }

            if model_cache is not None and isinstance(loaded, dict):
                model_cache.update(loaded)
                model_cache["last_model_card"] = model_card
                model_cache["last_mode"] = mode
                model_cache["last_schematic_mode"] = _schematic_mode
                model_cache["last_lora_key"] = _lora_key
                pipe_obj = model_cache
            else:
                pipe_obj = loaded

        if cancel_event.is_set():
            result_queue.put({"job_id": job_id, "status": "CANCELLED"})
            return

        # ---- Build inputs -----------------------------------------------
        progress_store[job_id] = {"progress": 0.10, "phase": "Preparing", "step": 0, "total": 0}

        _preserve_dims = getattr(plugin, "preserve_image_dimensions", False)
        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".exr", ".webp", ".hdr"}

        init_image = None
        img_path = snapshot.get("image_path", "")
        vid_path = snapshot.get("movie_path", "")
        if img_path and os.path.isfile(img_path):
            init_image = load_first_frame(img_path)
            if init_image and not _preserve_dims:
                init_image = init_image.resize((snapshot["width"], snapshot["height"]))
        if init_image is None and vid_path and os.path.isfile(vid_path):
            init_image = load_first_frame(vid_path)
            if init_image and not _preserve_dims:
                init_image = init_image.resize((snapshot["width"], snapshot["height"]))

        # FLF/LFO last-frame image (LTX Multi)
        _last_image_path = snapshot.get("last_image_path", "")
        _flf_last_image = None
        if _last_image_path and os.path.isfile(_last_image_path):
            _flf_last_image = load_first_frame(_last_image_path)
            if _flf_last_image and not _preserve_dims:
                _flf_last_image = _flf_last_image.resize((snapshot["width"], snapshot["height"]))
            # FLF: movie_path holds the first frame as an image file, not a real video;
            # suppress video_path so the plugin doesn't try to open it with av.
            if os.path.splitext(vid_path)[1].lower() in _IMAGE_EXTS:
                vid_path = ""

        # N-anchor middle images (LTX Multi)
        _middle_images_paths = []
        _middle_json = snapshot.get("middle_images_json", "") or ""
        if _middle_json:
            try:
                _middle_raw = json.loads(_middle_json)
                _middle_images_paths = [
                    (str(p), float(f)) for p, f in _middle_raw
                    if p and os.path.isfile(str(p))
                ]
            except Exception as _e:
                print(f"[Queue] middle_images_json parse error: {_e}")

        inpaint_mask = None
        mask_path = snapshot.get("inpaint_mask_path", "")
        if mask_path and os.path.isfile(mask_path):
            inpaint_mask = load_first_frame(mask_path)
            if inpaint_mask:
                inpaint_mask = inpaint_mask.resize((snapshot["width"], snapshot["height"]))

        def _phase_fn(label: str) -> None:
            store = progress_store.get(job_id)
            if store is not None:
                store["phase"] = label

        def _progress_fn(step: int, total: int) -> None:
            # Maps denoising steps to the 10 %–99 % band; 0–10 % = load,
            # 100 % = done.
            cur_phase = progress_store.get(job_id, {}).get("phase", "Generating")
            progress_store[job_id] = {
                "progress": 0.10 + 0.89 * (step / max(1, total)),
                "phase":    cur_phase,
                "step":     step,
                "total":    total,
            }

        # audio_length is stored in frames; models expect seconds
        _fps = snapshot.get("fps", 24.0) or 24.0
        _audio_length_in_s = snapshot["audio_length"] / _fps

        _infer_w, _infer_h = snapshot["width"], snapshot["height"]
        if _preserve_dims and init_image is not None:
            _infer_w, _infer_h = init_image.size

        inputs = ModelInputs(
            prompt               = snapshot["prompt"],
            neg_prompt           = snapshot["neg_prompt"],
            mode                 = mode,
            image                = init_image,
            last_image           = _flf_last_image,
            middle_images_paths  = _middle_images_paths,
            inpaint_mask         = inpaint_mask,
            width          = _infer_w,
            height         = _infer_h,
            frames         = snapshot["frames"],
            fps            = snapshot.get("fps", 24.0),
            steps          = snapshot["steps"],
            guidance       = snapshot["guidance"],
            strength       = snapshot["image_power"],
            seed           = snapshot["seed"],
            audio_ref      = snapshot.get("ref_audio_path") or snapshot.get("sound_path") or None,
            text_ref       = snapshot.get("ref_text", ""),
            video_path     = vid_path or None,
            audio_length   = _audio_length_in_s,
            speed          = snapshot["audio_speed_tts"],
            exaggeration   = snapshot["chat_exaggeration"],
            pace           = snapshot["chat_pace"],
            temperature    = snapshot["chat_temperature"],
            remove_silence = snapshot["remove_silence"],
            use_lcm        = snapshot["use_lcm"],
            use_adetailer  = snapshot["adetailer"],
            use_upscale    = snapshot["aurasr"],
            bpm            = snapshot["music_bpm"],
            lyrics         = snapshot["music_lyrics"],
            key_scale      = snapshot["music_key_scale"],
            time_signature = snapshot["music_time_signature"],
            lora_files     = [(i.name, i.weight_value) for i in enabled_items],
            progress_fn    = _progress_fn,
            phase_fn       = _phase_fn,
            should_cancel  = cancel_event.is_set,
        )

        # ---- Generate ---------------------------------------------------
        # Some plugins (e.g. SDNQ-quantized LTX-2.3) load their model weights
        # inside generate() rather than load(). On Blender 5.2's bundled
        # torch_cpu, materializing those tensors on a background worker thread
        # faults with EXCEPTION_ACCESS_VIOLATION (the same non-main-thread torch
        # instability that HF_DEACTIVATE_ASYNC_LOAD only fixes for transformers).
        # Such plugins opt into running generate() on the main thread — identical
        # to non-queue generation, which never crashes — while the worker blocks
        # here until it finishes. The UI is frozen for the duration, exactly as
        # it is during a direct (non-queue) generation of the same model.
        if getattr(plugin, "requires_main_thread_for_generate", False):
            _gen_done   = threading.Event()
            _gen_result = {}

            def _main_thread_generate():
                try:
                    _gen_result["value"] = plugin.generate(
                        pipe_obj, inputs, scene_proxy, prefs_proxy
                    )
                except BaseException as _exc:  # propagate every failure to the worker
                    _gen_result["error"] = _exc
                finally:
                    _gen_done.set()
                return None  # one-shot timer

            bpy.app.timers.register(_main_thread_generate, first_interval=0)
            _gen_done.wait()
            if "error" in _gen_result:
                raise _gen_result["error"]
            result = _gen_result["value"]
        else:
            result = plugin.generate(pipe_obj, inputs, scene_proxy, prefs_proxy)

        if cancel_event.is_set():
            result_queue.put({"job_id": job_id, "status": "CANCELLED"})
            return

        # ---- Save output (image plugins → PIL; video/audio → file path) --
        generator_ai = snapshot.get("generator_ai", "") or os.path.join(
            os.path.expanduser("~"), "Pallaidium Media"
        )

        out_path = None
        text_content = None
        if isinstance(result, str) and result.startswith("MULTI_STEM:"):
            out_path = result  # pass through; _queue_insert_strip handles multi-stem
        elif isinstance(result, str) and os.path.isfile(result):
            out_path = result
        elif isinstance(result, str) and result.strip():
            # Plain text result (caption, rewritten prompt) — save to .txt and
            # pass content directly so _queue_insert_strip can create a TEXT strip.
            text_content = result
            fname = clean_filename(f"{snapshot['seed']}_{snapshot['prompt'][:40]}")
            if snapshot.get("florence2_mode") == "IDEOGRAM4":
                # Derive filename from scene description (shot type + unique content).
                try:
                    _desc = json.loads(result).get("high_level_description", "")
                    if _desc:
                        fname = clean_filename(_desc[:70])
                except Exception:
                    pass
                # When sending to box editor: rename the rendered PNG to match the
                # scene description so the image file and the JSON share the same name.
                if snapshot.get("florence2_send_to_mask"):
                    _src = snapshot.get("image_path") or snapshot.get("movie_path") or ""
                    if _src and os.path.isfile(_src) and _src.lower().endswith(".png"):
                        _new_src = os.path.join(os.path.dirname(_src), fname + ".png")
                        if _new_src != _src:
                            try:
                                # os.replace overwrites existing destination (Windows-safe)
                                os.replace(_src, _new_src)
                                _src = _new_src
                                print(f"[Queue] Florence2 PNG renamed → {_new_src!r}")
                            except Exception as _rename_err:
                                print(f"[Queue] Florence2 PNG rename failed: {_rename_err}")
                    if _src:
                        text_content = f"// image: {_src}\n" + text_content
            out_path = _queue_solve_path(fname + ".txt", generator_ai)
            with open(out_path, "w", encoding="utf-8") as _fh:
                _fh.write(text_content)
        elif result is not None and not isinstance(result, str):
            fname = clean_filename(f"{snapshot['seed']}_{snapshot['prompt']}")
            out_path = _queue_solve_path(fname + ".png", generator_ai)
            result.save(out_path)

        progress_store[job_id] = {"progress": 1.0, "phase": "Done", "step": 0, "total": 0}
        result_queue.put({
            "job_id":               job_id,
            "status":               "COMPLETED",
            "output_path":          out_path or "",
            "text_content":         text_content or "",
            "result_note":          getattr(inputs, "usage_note", "") or "",
            "output_type":          otype,
            "frame_start":          snapshot["insert_frame_start"],
            "frame_end":            snapshot["insert_frame_end"],
            "channel":              snapshot["insert_channel"],
            "duration":             snapshot["insert_duration"],
            "sequencer_scene_name": snapshot.get("sequencer_scene_name", ""),
            "prompt":          snapshot["prompt"],
            "neg_prompt":      snapshot["neg_prompt"],
            "seed":            snapshot["seed"],
            "model_card":      snapshot["model_card"],
            "mode":            snapshot["mode"],
            "steps":           snapshot["steps"],
            "guidance":        snapshot["guidance"],
            "width":           snapshot["width"],
            "height":          snapshot["height"],
            "frames":          snapshot["frames"],
            "lora_files_json":        snapshot.get("lora_files_json", "[]"),
            "lora_folder":            snapshot.get("lora_folder", ""),
            "img_guidance_scale":     snapshot.get("img_guidance_scale", 1.6),
            "illumination_style":     snapshot.get("illumination_style", ""),
            "light_direction":        snapshot.get("light_direction", ""),
            "ip_adapter_face_folder":  snapshot.get("ip_adapter_face_folder", ""),
            "ip_adapter_style_folder": snapshot.get("ip_adapter_style_folder", ""),
            "openpose_use_bones":     snapshot.get("openpose_use_bones", False),
            "use_scribble_image":     snapshot.get("use_scribble_image", False),
            "ideogram_prompt_upsampling": snapshot.get("ideogram_prompt_upsampling", False),
            # ltx23_multi_v2
            "ltx23m_modality_scale":       snapshot.get("ltx23m_modality_scale",       1.0),
            "ltx23m_audio_guidance":       snapshot.get("ltx23m_audio_guidance",       1.0),
            "ltx23m_audio_stg_scale":      snapshot.get("ltx23m_audio_stg_scale",      0.0),
            "ltx23m_audio_modality_scale": snapshot.get("ltx23m_audio_modality_scale", 1.0),
            "ltx23m_audio_noise_scale":    snapshot.get("ltx23m_audio_noise_scale",    0.0),
            "ltx23m_audio_start_time":     snapshot.get("ltx23m_audio_start_time",     0.0),
            # ltx23_multi_ic_lora
            "ltx23ic_control_video_path":  snapshot.get("ltx23ic_control_video_path",  ""),
            "ltx23ic_control_audio_path":  snapshot.get("ltx23ic_control_audio_path",  ""),
            "ltx23ic_control_strength":    snapshot.get("ltx23ic_control_strength",    1.0),
            "ltx23ic_control_downscale":   snapshot.get("ltx23ic_control_downscale",   1),
            "ltx23ic_control_audio_str":   snapshot.get("ltx23ic_control_audio_str",   1.0),
            "ltx23ic_identity_guidance":   snapshot.get("ltx23ic_identity_guidance",   0.0),
            "ltx23ic_ref_image_path":      snapshot.get("ltx23ic_ref_image_path",      ""),
            # ltx23_extend
            "ltx23ext_extend_frames":      snapshot.get("ltx23ext_extend_frames",      96),
            "ltx23ext_video_strength":     snapshot.get("ltx23ext_video_strength",     1.0),
            "ltx23ext_audio_path":         snapshot.get("ltx23ext_audio_path",         ""),
            "ltx23_stage_mode":            snapshot.get("ltx23_stage_mode",            "FULL"),
            "maxine_quality":              snapshot.get("maxine_quality",              "HIGH"),
            # Google Nano Banana / Veo cloud settings
            "nano_banana_model":           snapshot.get("nano_banana_model",      "gemini-2.5-flash-image"),
            "nano_banana_aspect":          snapshot.get("nano_banana_aspect",     "1:1"),
            "nano_banana_resolution":      snapshot.get("nano_banana_resolution", "1K"),
            "veo_model":                   snapshot.get("veo_model",              "veo-3.1-fast-generate-preview"),
            "veo_aspect":                  snapshot.get("veo_aspect",             "16:9"),
            "veo_resolution":              snapshot.get("veo_resolution",         "720p"),
            "veo_duration":                snapshot.get("veo_duration",           "8"),
            "veo_person_generation":       snapshot.get("veo_person_generation",  "allow_adult"),
            "veo_image_mode":              snapshot.get("veo_image_mode",         "AUTO"),
            # Reference strips — names (drive Redo) + resolved paths (record)
            "nano_banana_ref_count":        snapshot.get("nano_banana_ref_count", 3),
            **{f"nano_banana_ref_strip_{_n}":
                   snapshot.get(f"nano_banana_ref_strip_{_n}", "") for _n in range(1, 10)},
            "veo_ref_strip_1":              snapshot.get("veo_ref_strip_1",              ""),
            "veo_ref_strip_2":              snapshot.get("veo_ref_strip_2",              ""),
            "veo_ref_strip_3":              snapshot.get("veo_ref_strip_3",              ""),
            **{f"nano_banana_ref_strip_{_n}_path":
                   snapshot.get(f"nano_banana_ref_strip_{_n}_path", "") for _n in range(1, 10)},
            "veo_ref_strip_1_path":         snapshot.get("veo_ref_strip_1_path",         ""),
            "veo_ref_strip_2_path":         snapshot.get("veo_ref_strip_2_path",         ""),
            "veo_ref_strip_3_path":         snapshot.get("veo_ref_strip_3_path",         ""),
            "florence2_send_to_mask":      snapshot.get("florence2_send_to_mask",      False),
            "florence2_source_image_path": snapshot.get("image_path") or snapshot.get("movie_path") or "",
            # Chatterbox Multilingual
            "chatterbox_mtl_language": snapshot.get("chatterbox_mtl_language", "en"),
            # MOSS-TTS
            "moss_model_variant":   snapshot.get("moss_model_variant",   "v1.5"),
            "moss_language":        snapshot.get("moss_language",        "AUTO"),
            "moss_duration_tokens": snapshot.get("moss_duration_tokens", 0),
            "moss_max_new_tokens":  snapshot.get("moss_max_new_tokens",  4096),
            "moss_temperature":     snapshot.get("moss_temperature",     1.7),
            "moss_top_p":           snapshot.get("moss_top_p",           0.8),
            "moss_top_k":           snapshot.get("moss_top_k",           25),
            "moss_ref_audio_path":  snapshot.get("moss_ref_audio_path",  ""),
        })

    except KeyboardInterrupt:
        # User cancelled during download — report as CANCELLED, not FAILED
        result_queue.put({
            "job_id":    job_id,
            "status":    "CANCELLED",
        })
    except Exception as exc:
        _err_str = str(exc)
        if isinstance(exc, ModuleNotFoundError):
            _err_str = (
                f"Missing dependency: {exc.name}. "
                "Install Dependencies in the add-on Preferences."
            )
        elif snapshot.get("local_files_only") and isinstance(exc, OSError):
            _err_str = (
                "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
            )
        result_queue.put({
            "job_id":    job_id,
            "status":    "FAILED",
            "error":     _err_str,
            "traceback": traceback.format_exc(),
        })
    finally:
        pipe_obj = None
        gc.collect()
        if snapshot.get("should_unload", True) and model_cache is not None:
            release_model_cache(model_cache)
        # Always trim the CUDA cache between jobs, even when the model is kept
        # loaded for the next same-model job: empty_cache() only returns unused
        # reserved blocks to the driver (it never frees still-referenced model
        # tensors), so it reduces fragmentation that otherwise OOMs a 2nd run.
        # Plugins that load inside generate() (stub load()) cache nothing, so
        # this is their only between-job cleanup on the queue side.
        clear_cuda_cache()
        _progress_store.pop(job_id, None)


# ---------------------------------------------------------------------------
# Operator: Add to Queue
# ---------------------------------------------------------------------------

class SEQUENCER_OT_add_to_queue(Operator):
    """Add current settings as a job in the render queue"""

    bl_idname = "sequencer.add_to_queue"
    bl_label = "Add to Queue"
    bl_description = "Enqueue the current generation settings without blocking the UI"
    bl_options = {"REGISTER"}

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_named_strip_path(scene, strip_name: str) -> str:
        """Resolve a named strip to its absolute file path for queue snapshotting."""
        if not strip_name:
            return ""
        from ..utils.helpers import find_strip_by_name, get_strip_path
        strip = find_strip_by_name(scene, strip_name)
        if strip is None:
            return ""
        path = get_strip_path(strip)
        return bpy.path.abspath(path) if path else ""

    @staticmethod
    def _render_named_strip_image(context, scene, strip_name: str) -> str:
        """Resolve a named strip to an image file path suitable for model input.

        Mirrors load_strip_as_pil()'s decision tree but runs at queue-add time
        (main thread, bpy available) and returns a file path instead of a PIL image:
          - IMAGE without transforms → raw source file (no letterboxing)
          - MOVIE                    → raw source file (load_first_frame will seek)
          - IMAGE with transforms / SCENE / META / MASK / COLOR → VSE render to PNG
        """
        if not strip_name:
            return ""
        from ..utils.helpers import find_strip_by_name, get_strip_path, render_strip_to_path
        strip = find_strip_by_name(scene, strip_name)
        if strip is None:
            # Fallback: the slot may hold a direct file path (e.g. a Screenwriter
            # script-to-screen reference) rather than a timeline strip name. This
            # makes reference delivery scene-independent — no source strip needs to
            # exist in the enqueue scene.
            try:
                if strip_name and os.path.isfile(bpy.path.abspath(strip_name)):
                    return bpy.path.abspath(strip_name)
            except Exception:
                pass
            return ""

        if strip.type == "IMAGE":
            try:
                tx = strip.transform
                has_transform = (
                    tx.scale_x != 1.0 or tx.scale_y != 1.0
                    or tx.offset_x != 0.0 or tx.offset_y != 0.0
                    or getattr(strip, "use_crop", False)
                )
            except Exception:
                has_transform = False
            if not has_transform:
                path = get_strip_path(strip)
                return bpy.path.abspath(path) if path else ""

        if strip.type == "MOVIE":
            path = get_strip_path(strip)
            if path:
                return bpy.path.abspath(path)

        # Complex strips (META, SCENE, IMAGE with transforms, etc.) — render through VSE
        path = render_strip_to_path(context, strip, image_output=True)
        return path or ""

    @staticmethod
    def _paths_from_strip(strip):
        """Extract (image_path, movie_path, sound_path, last_image_path) from a VSE strip.

        Returns empty strings for strip types that carry no media (TEXT, COLOR,
        ADJUSTMENT, META, effect strips, …).  The caller uses these to drive
        mode detection: empty paths → txt2* mode, but frame alignment still
        follows the strip's timing.

        last_image_path is non-empty only for LTX Multi FLF/LFO patterns inside
        a META strip:
          FLF  (2 images, different frame_starts, no MOVIE):
              movie_path      = first image (lower frame_start)
              last_image_path = second image (higher frame_start)
          LFO  (1 image whose frame_start > all other children, no MOVIE):
              movie_path      = ""
              last_image_path = that image
        """
        image_path = movie_path = sound_path = last_image_path = middle_images_json = ""
        control_video_path = control_audio_path = ""
        if strip is None:
            return image_path, movie_path, sound_path, last_image_path, middle_images_json, control_video_path, control_audio_path
        if strip.type == "IMAGE":
            dirname = os.path.dirname(bpy.path.abspath(strip.directory))
            try:
                fname = strip.elements[0].filename
            except (IndexError, AttributeError):
                fname = ""
            image_path = os.path.join(dirname, fname) if fname else ""
        elif strip.type == "MOVIE":
            movie_path = bpy.path.abspath(strip.filepath)
        elif strip.type == "TEXT":
            # Florence-2 output strips embed "// image: /path/to/file.png" as the
            # first line of their text so they can be re-queued on the same image.
            strip_text = getattr(strip, "text", "")
            for _ln in strip_text.splitlines():
                _ln = _ln.strip()
                if _ln.startswith("// image:"):
                    _candidate = _ln[len("// image:"):].strip()
                    if _candidate and os.path.isfile(_candidate):
                        image_path = _candidate
                    break
        elif strip.type == "SOUND":
            # Assign to sound_path, NOT movie_path — otherwise _detect_mode
            # sees has_vid=True and incorrectly routes to vid2vid / img2vid.
            try:
                sound_path = bpy.path.abspath(strip.sound.filepath)
            except AttributeError:
                sound_path = ""
        elif strip.type == "META":
            # Decompose META children into typed paths so the job gets real
            # media references rather than empty strings.
            _meta_images = []   # (path, frame_start, child_strip) for IMAGE children
            for child in strip.strips:
                if child.type == "IMAGE":
                    try:
                        dirname = os.path.dirname(bpy.path.abspath(child.directory))
                        fname = child.elements[0].filename
                        candidate = os.path.join(dirname, fname)
                        if fname and os.path.isfile(candidate):
                            _meta_images.append((candidate, child.frame_start, child))
                    except (IndexError, AttributeError):
                        pass
                elif child.type == "MOVIE" and not movie_path:
                    try:
                        mp = bpy.path.abspath(child.filepath)
                        if os.path.isfile(mp):
                            movie_path = mp
                    except Exception:
                        pass
                elif child.type == "MOVIE" and movie_path and not control_video_path:
                    # Second MOVIE in META → IC-LoRA control reference video
                    try:
                        cv = bpy.path.abspath(child.filepath)
                        if os.path.isfile(cv):
                            control_video_path = cv
                    except Exception:
                        pass
                elif child.type == "SOUND" and not sound_path:
                    try:
                        sp = bpy.path.abspath(child.sound.filepath)
                        if os.path.isfile(sp):
                            sound_path = sp
                    except AttributeError:
                        pass
                elif child.type == "SOUND" and sound_path and not control_audio_path:
                    # Second SOUND in META → IC-LoRA control reference audio
                    try:
                        ca = bpy.path.abspath(child.sound.filepath)
                        if os.path.isfile(ca):
                            control_audio_path = ca
                    except AttributeError:
                        pass

            if not movie_path and len(_meta_images) >= 3:
                # Multi-anchor: 3+ images → first=start anchor, last=end anchor, rest=middle
                _sorted = sorted(_meta_images, key=lambda x: x[1])
                movie_path      = _sorted[0][0]
                last_image_path = _sorted[-1][0]
                _meta_fs  = strip.frame_final_start
                _meta_dur = max(1, strip.frame_final_duration)
                _middle = []
                for _mp, _, _mchild in _sorted[1:-1]:
                    _frac = (_mchild.frame_final_start - _meta_fs) / _meta_dur
                    _frac = max(0.001, min(0.999, _frac))
                    _middle.append([_mp, _frac])
                import json as _json_q
                middle_images_json = _json_q.dumps(_middle)
            elif not movie_path and len(_meta_images) == 2 and _meta_images[0][1] != _meta_images[1][1]:
                # FLF: 2 images with different frame_starts → first → movie_path, second → last_image_path
                _sorted = sorted(_meta_images, key=lambda x: x[1])
                movie_path      = _sorted[0][0]
                last_image_path = _sorted[1][0]
            elif not movie_path and len(_meta_images) == 1:
                # Check LFO: image's frame_start > all other media children's frame_start.
                # Exclude TEXT strips — they carry prompts, not temporal anchors, and a
                # late-positioned TEXT strip would otherwise block LFO detection.
                _other_starts = [c.frame_start for c in strip.strips if c.type not in ("IMAGE", "TEXT")]
                if _other_starts and _meta_images[0][1] > max(_other_starts):
                    # LFO: image is the last strip → last-frame-only
                    last_image_path = _meta_images[0][0]
                else:
                    image_path = _meta_images[0][0]
            elif _meta_images:
                image_path = _meta_images[0][0]

        # TEXT, COLOR, ADJUSTMENT, SCENE, effect strips, etc.:
        # return empty strings → mode detection falls through to txt2* path.
        return image_path, movie_path, sound_path, last_image_path, middle_images_json, control_video_path, control_audio_path

    @staticmethod
    def _detect_mode(otype, image_path, movie_path, inpaint_strip):
        has_img = bool(image_path)
        has_vid = bool(movie_path)
        if otype == "image":
            do_inpaint = bool(inpaint_strip) and has_img
            return "inpaint" if do_inpaint else ("img2img" if (has_img or has_vid) else "txt2img")
        elif otype == "movie":
            return "vid2vid" if has_vid else ("img2vid" if has_img else "txt2vid")
        elif otype == "audio":
            return "txt2audio"
        return "txt2text"

    def execute(self, context):
        scene = context.scene
        prefs = context.preferences.addons[ADDON_ID].preferences
        otype = scene.generatorai_typeselect

        # In Blender 5.x the VSE edits context.sequencer_scene (a workspace
        # property) which can differ from the active scene.  All picked-strip
        # references (input strips and the per-plugin ref pickers) live in that
        # scene, so resolve them from it rather than context.scene.  Record its
        # name too so generated strips land back in the right place.
        seq_scene = (getattr(context, "sequencer_scene", None)
                     or getattr(context.workspace, "sequencer_scene", None)
                     or scene)
        sequencer_scene_name = seq_scene.name

        if not scene.sequence_editor:
            scene.sequence_editor_create()

        model_card = {
            "image": prefs.image_model_card,
            "movie": prefs.movie_model_card,
            "audio": prefs.audio_model_card,
            "text":  prefs.text_model_card,
        }.get(otype, "")

        if not model_card:
            self.report({"ERROR"}, "No model selected.")
            return {"CANCELLED"}

        # ---- Shared settings (same for every job in this batch) ----------
        x          = closest_divisible_32(scene.generate_movie_x)
        y          = closest_divisible_32(scene.generate_movie_y)
        # raw_frames: -1 means "match input strip duration"; 0/positive = explicit
        raw_frames = scene.generate_movie_frames
        audio_dur  = float(getattr(scene, "audio_length_in_f", 80))

        styled     = style_prompt(scene.generate_movie_prompt)
        prompt     = styled[0]

        # Let the plugin bake any scene-specific state into the prompt now so
        # the job snapshot is self-contained (e.g. JoyAI spatial mode).
        from ..models import get_plugin as _get_plugin_for_prompt
        _pi = _get_plugin_for_prompt(model_card)
        if _pi is not None and hasattr(_pi, "_build_prompt"):
            prompt = _pi._build_prompt(scene, prompt)

        neg_prompt = (
            scene.generate_movie_negative_prompt
            + ", " + styled[1]
        )
        lora_json  = json.dumps([
            {"name": f.name, "weight": getattr(f, "weight_value", 1.0), "enabled": f.enabled}
            for f in getattr(scene, "lora_files", [])
        ])
        # 'frames' and 'audio_length' are per-strip — omitted from common
        common = dict(
            output_type  = otype,
            model_card   = model_card,
            prompt       = prompt,
            neg_prompt   = neg_prompt,
            steps        = scene.movie_num_inference_steps,
            guidance     = scene.movie_num_guidance,
            use_random   = scene.movie_use_random,
            width        = x,
            height       = y,
            image_power  = scene.image_power,
            use_lcm           = getattr(scene, "use_lcm", False),
            refine_sd         = getattr(scene, "refine_sd", False),
            adetailer         = getattr(scene, "adetailer", False),
            aurasr            = getattr(scene, "aurasr", False),
            remove_silence    = getattr(scene, "remove_silence", True),
            audio_speed_tts   = getattr(scene, "audio_speed_tts", 1.0),
            chat_exaggeration = getattr(scene, "chat_exaggeration", 0.5),
            chat_pace         = getattr(scene, "chat_pace", 0.5),
            chat_temperature  = getattr(scene, "chat_temperature", 0.8),
            fps               = round(scene.render.fps / scene.render.fps_base, 3),
            music_bpm         = getattr(scene, "music_bpm", 0),
            music_lyrics      = getattr(scene, "music_lyrics", ""),
            music_key_scale   = getattr(scene, "music_key_scale", ""),
            music_time_signature = getattr(scene, "music_time_signature", ""),
            ref_audio_path    = bpy.path.abspath(getattr(scene, "ref_audio_path", "") or ""),
            ref_text          = getattr(scene, "ref_text", ""),
            hugginface_token  = getattr(prefs, "hugginface_token", ""),
            gemini_api_key    = getattr(prefs, "gemini_api_key", ""),
            remote_backend_url = getattr(prefs, "remote_backend_url", ""),
            remote_backend_key = getattr(prefs, "remote_backend_key", ""),
            local_files_only  = getattr(prefs, "local_files_only", False),
            display_console   = getattr(prefs, "display_console", True),
            generator_ai      = getattr(prefs, "generator_ai", "") or os.path.join(
                bpy.utils.user_resource("DATAFILES"), "Pallaidium Media"
            ),
            hf_cache_dir      = getattr(prefs, "hf_cache_dir", ""),
            lora_files_json      = lora_json,
            lora_folder          = bpy.path.abspath(getattr(scene, "lora_folder", "") or ""),
            sequencer_scene_name = sequencer_scene_name,
            joyimage_spatial_mode = getattr(scene, "joyimage_spatial_mode", "general"),
            joyimage_object       = getattr(scene, "joyimage_object", "object"),
            joyimage_rotate_view  = getattr(scene, "joyimage_rotate_view", "front"),
            joyimage_yaw          = getattr(scene, "joyimage_yaw", 0.0),
            joyimage_pitch        = getattr(scene, "joyimage_pitch", 0.0),
            joyimage_zoom         = getattr(scene, "joyimage_zoom", "unchanged"),
            kontext_strip_1_path  = self._resolve_named_strip_path(seq_scene, getattr(seq_scene, "kontext_strip_1", "")),
            inpaint_mask_path     = self._resolve_named_strip_path(seq_scene, getattr(seq_scene, "inpaint_selected_strip", "")),
            qwen_strip_1_path     = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "qwen_strip_1", "")),
            qwen_strip_2_path     = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "qwen_strip_2", "")),
            qwen_strip_3_path     = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "qwen_strip_3", "")),
            klein_strip_1_path    = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "klein_strip_1", "")),
            klein_strip_2_path    = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "klein_strip_2", "")),
            klein_strip_3_path    = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "klein_strip_3", "")),
            minimax_subject_path  = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "minimax_subject", "")),
            **{f"flux_strip_{_n}_path":
                   self._render_named_strip_image(context, seq_scene, getattr(seq_scene, f"flux_strip_{_n}", ""))
               for _n in range(1, 10)},
            nano_banana_ref_count = getattr(scene, "nano_banana_ref_count", 3),
            nano_banana_ref_strip_1_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_1", "")),
            nano_banana_ref_strip_2_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_2", "")),
            nano_banana_ref_strip_3_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_3", "")),
            nano_banana_ref_strip_4_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_4", "")),
            nano_banana_ref_strip_5_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_5", "")),
            nano_banana_ref_strip_6_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_6", "")),
            nano_banana_ref_strip_7_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_7", "")),
            nano_banana_ref_strip_8_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_8", "")),
            nano_banana_ref_strip_9_path = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "nano_banana_ref_strip_9", "")),
            veo_ref_strip_1_path  = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "veo_ref_strip_1", "")),
            veo_ref_strip_2_path  = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "veo_ref_strip_2", "")),
            veo_ref_strip_3_path  = self._render_named_strip_image(context, seq_scene, getattr(seq_scene, "veo_ref_strip_3", "")),
            # Source strip names (carried through to metadata for Redo)
            nano_banana_ref_strip_1 = getattr(seq_scene, "nano_banana_ref_strip_1", ""),
            nano_banana_ref_strip_2 = getattr(seq_scene, "nano_banana_ref_strip_2", ""),
            nano_banana_ref_strip_3 = getattr(seq_scene, "nano_banana_ref_strip_3", ""),
            nano_banana_ref_strip_4 = getattr(seq_scene, "nano_banana_ref_strip_4", ""),
            nano_banana_ref_strip_5 = getattr(seq_scene, "nano_banana_ref_strip_5", ""),
            nano_banana_ref_strip_6 = getattr(seq_scene, "nano_banana_ref_strip_6", ""),
            nano_banana_ref_strip_7 = getattr(seq_scene, "nano_banana_ref_strip_7", ""),
            nano_banana_ref_strip_8 = getattr(seq_scene, "nano_banana_ref_strip_8", ""),
            nano_banana_ref_strip_9 = getattr(seq_scene, "nano_banana_ref_strip_9", ""),
            veo_ref_strip_1       = getattr(seq_scene, "veo_ref_strip_1", ""),
            veo_ref_strip_2       = getattr(seq_scene, "veo_ref_strip_2", ""),
            veo_ref_strip_3       = getattr(seq_scene, "veo_ref_strip_3", ""),
            whisper_model_size = getattr(scene, "whisper_model_size", "large-v3-turbo"),
            whisper_language   = getattr(scene, "whisper_language",   ""),
            stem_split_model  = getattr(scene, "stem_split_model",  "htdemucs_ft"),
            stem_split_vocals = getattr(scene, "stem_split_vocals", True),
            stem_split_drums  = getattr(scene, "stem_split_drums",  True),
            stem_split_bass   = getattr(scene, "stem_split_bass",   True),
            stem_split_other  = getattr(scene, "stem_split_other",  True),
            stem_split_guitar  = getattr(scene, "stem_split_guitar",  False),
            stem_split_piano   = getattr(scene, "stem_split_piano",   False),
            omnivoice_instruct    = getattr(scene, "omnivoice_instruct",    ""),
            omnivoice_language    = getattr(scene, "omnivoice_language",    ""),
            omnivoice_preprocess  = getattr(scene, "omnivoice_preprocess",  True),
            omnivoice_denoise     = getattr(scene, "omnivoice_denoise",     True),
            omnivoice_postprocess = getattr(scene, "omnivoice_postprocess", True),
            chatterbox_mtl_language = getattr(scene, "chatterbox_mtl_language", "en"),
            moss_model_variant    = getattr(scene, "moss_model_variant",    "v1.5"),
            moss_language         = getattr(scene, "moss_language",         "AUTO"),
            moss_duration_tokens  = getattr(scene, "moss_duration_tokens",  0),
            moss_max_new_tokens   = getattr(scene, "moss_max_new_tokens",   4096),
            moss_temperature      = getattr(scene, "moss_temperature",      1.7),
            moss_top_p            = getattr(scene, "moss_top_p",            0.8),
            moss_top_k            = getattr(scene, "moss_top_k",            25),
            moss_ref_audio_path   = getattr(scene, "moss_ref_audio_path",   ""),
            florence2_mode         = getattr(scene, "florence2_mode",         "CAPTION"),
            florence2_send_to_mask = getattr(scene, "florence2_send_to_mask", False),
            klein_schematic_mode   = getattr(scene, "klein_schematic_mode",   "DEPTH"),
            klein_schematic_target = getattr(scene, "klein_schematic_target", "person"),
            img_guidance_scale     = getattr(scene, "img_guidance_scale",     1.6),
            illumination_style     = getattr(scene, "illumination_style",     ""),
            light_direction        = getattr(scene, "light_direction",        ""),
            ip_adapter_face_folder  = bpy.path.abspath(getattr(scene, "ip_adapter_face_folder",  "") or ""),
            ip_adapter_style_folder = bpy.path.abspath(getattr(scene, "ip_adapter_style_folder", "") or ""),
            openpose_use_bones     = getattr(scene, "openpose_use_bones",     False),
            use_scribble_image     = getattr(scene, "use_scribble_image",     False),
            ideogram_prompt_upsampling = getattr(scene, "ideogram_prompt_upsampling", False),
            # ltx23_multi_v2 guidance params
            ltx23m_modality_scale       = getattr(scene, "ltx23m_modality_scale",       1.0),
            ltx23m_audio_guidance       = getattr(scene, "ltx23m_audio_guidance",       1.0),
            ltx23m_audio_stg_scale      = getattr(scene, "ltx23m_audio_stg_scale",      0.0),
            ltx23m_audio_modality_scale = getattr(scene, "ltx23m_audio_modality_scale", 1.0),
            ltx23m_audio_noise_scale    = getattr(scene, "ltx23m_audio_noise_scale",    0.0),
            # ltx23_multi_ic_lora params
            ltx23ic_control_strength    = getattr(scene, "ltx23ic_control_strength",    1.0),
            ltx23ic_control_downscale   = getattr(scene, "ltx23ic_control_downscale",   1),
            ltx23ic_control_audio_str   = getattr(scene, "ltx23ic_control_audio_str",   1.0),
            ltx23ic_identity_guidance   = getattr(scene, "ltx23ic_identity_guidance",   0.0),
            # ltx23_extend params (ltx23ext_audio_path is resolved per-job below)
            ltx23ext_extend_frames      = getattr(scene, "ltx23ext_extend_frames",      96),
            ltx23ext_video_strength     = getattr(scene, "ltx23ext_video_strength",     1.0),
            ltx23_stage_mode            = getattr(scene, "ltx23_stage_mode",            "FULL"),
            maxine_quality              = getattr(scene, "maxine_quality",              "HIGH"),
            # Google Nano Banana / Veo cloud settings
            nano_banana_model           = getattr(scene, "nano_banana_model",      "gemini-2.5-flash-image"),
            nano_banana_aspect          = getattr(scene, "nano_banana_aspect",     "1:1"),
            nano_banana_resolution      = getattr(scene, "nano_banana_resolution", "1K"),
            veo_model                   = getattr(scene, "veo_model",              "veo-3.1-fast-generate-preview"),
            veo_aspect                  = getattr(scene, "veo_aspect",             "16:9"),
            veo_resolution              = getattr(scene, "veo_resolution",         "720p"),
            veo_duration                = getattr(scene, "veo_duration",           "8"),
            veo_person_generation       = getattr(scene, "veo_person_generation",  "allow_adult"),
            veo_image_mode              = getattr(scene, "veo_image_mode",         "AUTO"),
            marlin_mode                 = getattr(scene, "marlin_mode",                 "CAPTION"),
            marlin_find_query           = getattr(scene, "marlin_find_query",           ""),
        )

        # ---- Decide which strips to iterate over -------------------------
        # Only content-bearing strip types are meaningful as queue inputs.
        # Effect strips (SPEED, GLOW, GAUSSIAN_BLUR, TRANSFORM, WIPE, …) are
        # automatically selected alongside their parent strip in Blender, which
        # would create spurious extra jobs — exclude them with a whitelist.
        _CONTENT_TYPES = {
            "IMAGE", "MOVIE", "SOUND", "TEXT",
            "COLOR", "META", "SCENE", "MOVIECLIP", "MASK", "ADJUSTMENT",
        }
        input_mode = scene.input_strips
        # context.selected_strips is the authoritative source in Blender 5.x —
        # it works correctly from the N-panel (same as strip_to_generatorAI).
        # Fall back to se.strips + s.select for non-VSE invocations (keybinds,
        # Python calls) where context may not carry the sequence editor state.
        # Resolve input/reference strips from the sequencer scene (see top of
        # execute) so picked refs and the select fallback see the strips
        # actually shown in the editor.
        se = seq_scene.sequence_editor
        _ctx_sel = list(context.selected_strips) if context.selected_strips else []
        if _ctx_sel:
            selected = sorted(
                (s for s in _ctx_sel if s.type in _CONTENT_TYPES),
                key=lambda s: s.frame_final_start,
            )
        else:
            # Fallback: iterate top-level strips and check s.select directly.
            top_level = list(se.strips) if se else []
            selected = sorted(
                (s for s in top_level if s.select and s.type in _CONTENT_TYPES),
                key=lambda s: s.frame_final_start,
            )
        if input_mode == "input_strips" and not selected:
            self.report({"WARNING"}, "No strips selected. Select one or more strips before adding to the queue.")
            return {"CANCELLED"}
        strip_list = selected if (input_mode == "input_strips" and selected) else [None]

        inpaint_strip = getattr(seq_scene, "inpaint_selected_strip", "")
        # Deterministic single-output plugins (supports_batch=False) gain nothing
        # from batching — the UI hides Batch Count for them, so clamp to 1 here
        # to avoid enqueuing identical duplicate jobs from a stale movie_num_batch.
        if _pi is not None and not getattr(_pi, "supports_batch", True):
            batch_count = 1
        else:
            batch_count = max(1, getattr(scene, "movie_num_batch", 1))
        added = 0

        for strip in strip_list:
            if strip is not None:
                image_path, movie_path, sound_path, last_image_path, middle_images_json, control_video_path, control_audio_path = self._paths_from_strip(strip)
                print(f"[Queue][dbg] loop strip='{strip.name}' type={strip.type} otype={otype} "
                      f"input_mode={input_mode} → image_path={image_path!r} movie_path={movie_path!r}")

                # Render audio to a trimmed WAV so the job never holds a pointer to
                # the full source file (avoids CUDA OOM when frame count is derived
                # from audio duration).
                # • META strip  → use render_meta_child_to_path (META-range export)
                # • SOUND strip → use render_strip_to_wav (strip's own trimmed range)
                if strip.type == "META" and sound_path:
                    from ..utils.helpers import render_meta_child_to_path
                    for _c in strip.strips:
                        if _c.type == "SOUND":
                            _trimmed = render_meta_child_to_path(context, strip, _c, image_output=False)
                            if _trimmed:
                                sound_path = _trimmed
                                print(f"[Queue] META SOUND trimmed → {_trimmed!r}")
                            else:
                                print(f"[Queue] META SOUND trim failed, keeping raw: {sound_path!r}")
                            break
                elif strip.type == "SOUND" and sound_path:
                    from ..utils.helpers import render_strip_to_wav
                    _trimmed = render_strip_to_wav(context, strip)
                    if _trimmed:
                        sound_path = _trimmed
                        print(f"[Queue] SOUND trimmed → {_trimmed!r}")
                    else:
                        print(f"[Queue] SOUND trim failed, keeping raw: {sound_path!r}")

                # Florence-2 (text) jobs: render the source strip to a single-frame
                # PNG so the model and box editor both receive the correct trimmed
                # first frame rather than the raw unclipped source file.
                if otype == "text" and strip.type in ("IMAGE", "MOVIE") and (image_path or movie_path):
                    from ..utils.helpers import render_strip_to_path
                    _f2_rendered = render_strip_to_path(context, strip, image_output=True)
                    if _f2_rendered:
                        image_path = _f2_rendered
                        movie_path = ""
                        print(f"[Queue] Florence2 input rendered → {_f2_rendered!r}")
                    else:
                        print(f"[Queue] Florence2 input render failed, keeping raw")

                # SCENE strips carry no source file, so _paths_from_strip returns
                # empty paths and the job would silently fall through to a txt2*
                # mode that ignores the strip. Render the scene through the VSE and
                # use the rendered file as the model input — a single-frame PNG for
                # image/text plugins, an animation MP4 for video plugins. Mirrors
                # load_strip_as_pil()/_render_named_strip_image()'s decision tree.
                # A failed render is surfaced to the UI so the job never silently
                # degrades to txt2* (which is what "input is a scene strip but mode
                # came out txt2img" looks like to the user).
                if strip.type == "SCENE" and not (image_path or movie_path):
                    from ..utils.helpers import render_strip_to_path
                    print(f"[Queue][dbg] SCENE block ENTERED for '{strip.name}', "
                          f"rendering (otype={otype})…")
                    _diag(f"SCENE block ENTERED for '{strip.name}' otype={otype}")
                    _want_video = (otype == "movie")
                    _scene_rendered = None
                    try:
                        _scene_rendered = render_strip_to_path(
                            context, strip, image_output=not _want_video
                        )
                    except Exception as _scene_err:
                        print(f"[Queue] SCENE render raised: {_scene_err!r}")
                        _diag(f"SCENE render RAISED: {_scene_err!r}")
                    _diag(f"SCENE render returned: {_scene_rendered!r}")
                    if _scene_rendered:
                        if _want_video:
                            movie_path = _scene_rendered
                        else:
                            image_path = _scene_rendered
                        print(f"[Queue] SCENE input rendered ({'video' if _want_video else 'image'}) → {_scene_rendered!r}")
                    elif otype in ("image", "movie", "text"):
                        # Empty path → _detect_mode would pick txt2*, dropping the
                        # scene the user explicitly selected. Make that loud.
                        self.report(
                            {"WARNING"},
                            f"Could not render scene strip '{strip.name}' to an input "
                            f"image — job will run without it. Check the scene has a camera/output.",
                        )
                        print(f"[Queue] SCENE render returned no file for '{strip.name}'")

                # MOVIE main inputs (Output: Video): re-render through the VSE so the
                # model receives only the strip's visible (trimmed) duration — and any
                # speed/transform baked in — instead of the full untrimmed source file.
                # Mirrors the SCENE branch; a failed render keeps the raw path so the
                # job still runs.
                if strip.type == "MOVIE" and otype == "movie" and movie_path:
                    from ..utils.helpers import render_strip_to_path
                    _trim_mov = None
                    try:
                        _trim_mov = render_strip_to_path(context, strip, image_output=False)
                    except Exception as _mov_err:
                        print(f"[Queue] MOVIE trim render raised: {_mov_err!r}")
                    if _trim_mov:
                        movie_path = _trim_mov
                        print(f"[Queue] MOVIE input rendered (trimmed) → {_trim_mov!r}")
                    else:
                        print(f"[Queue] MOVIE trim render failed, keeping raw: {movie_path!r}")

                # Compute audio start offset (seconds) from SOUND strip position in META
                _audio_start_time = 0.0
                if strip.type == "META" and sound_path:
                    _fps_scene = scene.render.fps / max(1.0, getattr(scene.render, "fps_base", 1.0))
                    for _c in strip.strips:
                        if _c.type == "SOUND":
                            _off_f = _c.frame_final_start - strip.frame_final_start
                            _audio_start_time = max(0.0, _off_f / _fps_scene)
                            break

                # Pre-trim IC-LoRA control MOVIE strips to honour frame_offset_start
                # and avoid loading the full source file (OOM risk).
                if strip.type == "META" and control_video_path:
                    from ..utils.helpers import render_meta_child_to_path
                    for _c in strip.strips:
                        try:
                            _cv = bpy.path.abspath(_c.filepath) if _c.type == "MOVIE" else ""
                        except Exception:
                            _cv = ""
                        if _c.type == "MOVIE" and _cv == control_video_path:
                            _trimmed_cv = render_meta_child_to_path(context, strip, _c, image_output=False)
                            if _trimmed_cv:
                                control_video_path = _trimmed_cv
                                print(f"[Queue] IC-LoRA MOVIE trimmed → {_trimmed_cv!r}")
                            else:
                                print(f"[Queue] IC-LoRA MOVIE trim failed, keeping raw: {control_video_path!r}")
                            break
                elif strip.type == "MOVIE" and control_video_path:
                    # Single MOVIE control strip: render trimmed version
                    _ctrl_strip_name = getattr(seq_scene, "ltx23ic_control_strip", "")
                    _ctrl_strip_obj  = se.strips.get(_ctrl_strip_name) if _ctrl_strip_name and se else None
                    if _ctrl_strip_obj and _ctrl_strip_obj.type == "MOVIE":
                        from ..utils.helpers import render_strip_to_path
                        _trimmed_cv = render_strip_to_path(context, _ctrl_strip_obj, image_output=False)
                        if _trimmed_cv:
                            control_video_path = _trimmed_cv
                            print(f"[Queue] IC-LoRA single-MOVIE trimmed → {_trimmed_cv!r}")

                # Pre-trim IC-LoRA control audio strip
                if strip.type == "META" and control_audio_path:
                    from ..utils.helpers import render_meta_child_to_path
                    for _c in strip.strips:
                        if _c.type == "SOUND" and sound_path:
                            try:
                                _ca = bpy.path.abspath(_c.sound.filepath)
                            except Exception:
                                _ca = ""
                            if _ca == control_audio_path:
                                _trimmed_ca = render_meta_child_to_path(context, strip, _c, image_output=False)
                                if _trimmed_ca:
                                    control_audio_path = _trimmed_ca
                                    print(f"[Queue] IC-LoRA SOUND trimmed → {_trimmed_ca!r}")
                                break

                # Align output to the input strip's in-point
                strip_frame_start = strip.frame_final_start
                # -1 (or any negative) = match the input strip's length
                strip_dur = strip.frame_final_duration
                gen_frames = strip_dur if raw_frames < 0 else max(1, abs(raw_frames))
                if otype == "audio":
                    # audio_dur < 0 is the -1 sentinel: match the input strip.
                    # Also match when the plugin requires a strip (e.g. Stem Splitter).
                    _pi_req = _pi is not None and getattr(_pi, "requires_input_strip", False)
                    insert_dur = strip_dur if (audio_dur < 0 or _pi_req) else max(1, int(audio_dur))
                elif otype == "text":
                    # Text output (captions, transcriptions) always spans the full
                    # input strip so it can be read/used for the strip's entire range.
                    insert_dur = strip_dur
                else:
                    insert_dur = gen_frames
                # For SOUND strips generating audio or text (e.g. transcription):
                # store the strip's file as the audio reference so it overrides
                # the scene-level ref_audio_path and each queued job uses its
                # own strip rather than the same (first) strip for all jobs.
                _pi_strip_input = _pi is not None and getattr(_pi, "requires_input_strip", False)
                strip_audio_path = sound_path if (
                    sound_path and (otype == "audio" or _pi_strip_input)
                ) else ""
            else:
                image_path          = bpy.path.abspath(getattr(scene, "image_path", "") or "")
                movie_path          = bpy.path.abspath(getattr(scene, "movie_path", "") or "")
                sound_path          = bpy.path.abspath(getattr(scene, "sound_path", "") or "")
                last_image_path     = ""
                middle_images_json  = ""
                control_video_path  = ""
                control_audio_path  = ""
                _audio_start_time   = 0.0
                strip_audio_path    = ""

                # Negative raw_frames = "auto" sentinel. In prompt mode there is no
                # input strip to derive duration from, so fall back to 100 frames.
                if raw_frames < 0:
                    gen_frames = 100
                elif raw_frames > 0:
                    gen_frames = raw_frames
                else:
                    gen_frames = 1
                if otype == "audio":
                    # audio_dur < 0 is the -1 sentinel: no strip to match, use default
                    insert_dur = 80 if audio_dur < 0 else max(1, int(audio_dur))
                else:
                    insert_dur = gen_frames

            mode = self._detect_mode(otype, image_path, movie_path,
                                     inpaint_strip if strip is None else "")
            print(f"[Queue][dbg] mode-detect otype={otype} image_path={image_path!r} "
                  f"movie_path={movie_path!r} inpaint={inpaint_strip!r} → mode={mode}")
            _diag(f"queue enqueue: strip={(strip.name, strip.type) if strip is not None else None} "
                  f"otype={otype} input_mode={input_mode} image_path={image_path!r} "
                  f"movie_path={movie_path!r} → mode={mode}")

            # For strip mode: find the output channel once, then advance the
            # frame cursor after each batch copy so they form a sequence.
            if strip is not None:
                _batch_frame = strip_frame_start
                _batch_ch    = _find_free_channel(
                    scene,
                    strip_frame_start,
                    strip_frame_start + max(1, insert_dur),
                    strip.channel + 1,
                )

            for _batch_i in range(batch_count):
                seed = (
                    random.randint(-2147483647, 2147483647)
                    if scene.movie_use_random else scene.movie_num_seed
                )
                # Reflect the latest random seed in the UI field so the user
                # can see / reuse the value that was actually generated.
                if scene.movie_use_random:
                    scene.movie_num_seed = seed

                if strip is not None:
                    # Strip mode: batch copies placed consecutively in time,
                    # all in the same channel directly above the input strip.
                    frame_start = _batch_frame
                    frame_end   = frame_start + max(1, insert_dur)
                    channel     = _batch_ch
                    _batch_frame = frame_end  # advance cursor for next copy
                else:
                    # Prompt mode: each batch copy is placed consecutively in time.
                    frame_start = scene.frame_current
                    frame_end   = frame_start + max(1, insert_dur)
                    channel     = find_first_empty_channel(frame_start, frame_end)

                job = scene.render_queue.add()
                job.job_id   = f"{id(job)}_{random.randint(0, 999999)}"
                job.status   = "PENDING"
                job.progress = 0.0
                job.mode     = mode
                job.seed     = seed
                job.frames   = gen_frames
                job.audio_length = insert_dur

                job.insert_frame_start = frame_start
                job.insert_frame_end   = frame_end
                job.insert_channel     = channel
                job.insert_duration    = insert_dur

                job.image_path         = image_path
                job.movie_path         = movie_path
                job.sound_path         = sound_path
                job.last_image_path    = last_image_path
                job.middle_images_json = middle_images_json

                # IC-LoRA / V2 — resolved at queue time so the worker is self-contained
                job.ltx23ic_control_video_path = control_video_path
                job.ltx23ic_control_audio_path = control_audio_path
                job.ltx23ic_ref_image_path     = ""
                job.ltx23m_audio_start_time    = _audio_start_time

                def _image_strip_path(_s):
                    """Absolute path to an IMAGE strip's first element, or ''."""
                    try:
                        _dir = os.path.dirname(bpy.path.abspath(_s.directory))
                        _fn  = _s.elements[0].filename
                        _p   = os.path.join(_dir, _fn) if _fn else ""
                        return _p if _p and os.path.isfile(_p) else ""
                    except Exception:
                        return ""

                # Single-file control strip: resolve ltx23ic_control_strip prop
                # from the sequencer scene. Supports a MOVIE strip (control video),
                # an IMAGE strip (3DREAL frame-0 appearance reference → the MAIN
                # input video drives control_video), or a META whose first
                # MOVIE/SOUND/IMAGE children supply the control references.
                if not control_video_path:
                    _ctrl_name = getattr(seq_scene, "ltx23ic_control_strip", "")
                    if _ctrl_name and se:
                        _ctrl_s = se.strips.get(_ctrl_name)
                        if _ctrl_s and _ctrl_s.type == "MOVIE":
                            try:
                                _cp = bpy.path.abspath(_ctrl_s.filepath)
                                if os.path.isfile(_cp):
                                    job.ltx23ic_control_video_path = _cp
                            except Exception:
                                pass
                        elif _ctrl_s and _ctrl_s.type == "IMAGE":
                            job.ltx23ic_ref_image_path = _image_strip_path(_ctrl_s)
                        elif _ctrl_s and _ctrl_s.type == "META":
                            for _c in _ctrl_s.strips:
                                try:
                                    if _c.type == "MOVIE" and not job.ltx23ic_control_video_path:
                                        _cp = bpy.path.abspath(_c.filepath)
                                        if os.path.isfile(_cp):
                                            job.ltx23ic_control_video_path = _cp
                                    elif _c.type == "IMAGE" and not job.ltx23ic_ref_image_path:
                                        job.ltx23ic_ref_image_path = _image_strip_path(_c)
                                    elif (_c.type == "SOUND"
                                          and not job.ltx23ic_control_audio_path
                                          and getattr(_c, "sound", None)):
                                        _ap = bpy.path.abspath(_c.sound.filepath)
                                        if os.path.isfile(_ap):
                                            job.ltx23ic_control_audio_path = _ap
                                except Exception:
                                    pass

                # ltx23_extend: resolve the picked SOUND strip → file path for the worker.
                job.ltx23ext_audio_path = ""
                _ext_name = getattr(seq_scene, "ltx23ext_audio_strip", "")
                if _ext_name and se:
                    _ext_s = se.strips.get(_ext_name)
                    if _ext_s and _ext_s.type == "SOUND" and getattr(_ext_s, "sound", None):
                        try:
                            _ap = bpy.path.abspath(_ext_s.sound.filepath)
                            if os.path.isfile(_ap):
                                job.ltx23ext_audio_path = _ap
                        except Exception:
                            pass

                for attr, val in common.items():
                    setattr(job, attr, val)

                # For TEXT strips: prepend the strip's text to the prompt,
                # matching the non-queue behaviour: strip.text + ", " + base_prompt
                # Strip // comment lines (Florence-2 metadata) before using as prompt.
                # Skip when the strip carries its own ai_meta_prompt (a metadata
                # carrier, e.g. a Screenwriter script-to-screen TEXT strip): Redo
                # already loaded that authoritative prompt into job.prompt, so
                # prepending the visible text would duplicate it (dialogue spoken
                # twice / shot annotation repeated).
                if (strip is not None and strip.type == "TEXT"
                        and not strip.get("ai_meta_prompt")):
                    strip_text = getattr(strip, "text", "").strip()
                    strip_text = "\n".join(
                        ln for ln in strip_text.splitlines()
                        if not ln.strip().startswith("//")
                    ).strip()
                    if strip_text:
                        job.prompt = (strip_text + ", " + job.prompt) if job.prompt else strip_text

                # For META strips: prepend TEXT children to the prompt
                if strip is not None and strip.type == "META":
                    meta_texts = [
                        c.text for c in strip.strips
                        if c.type == "TEXT" and getattr(c, "text", "").strip()
                    ]
                    if meta_texts:
                        meta_text = ", ".join(meta_texts)
                        job.prompt = (meta_text + ", " + job.prompt) if job.prompt else meta_text

                # Override ref_audio_path from a SOUND strip (wins over scene-level value)
                if strip_audio_path:
                    job.ref_audio_path = strip_audio_path

                # For strip-input plugins (e.g. Stem Splitter) use the input
                # filename as the queue job name instead of the text prompt.
                if strip is not None and _pi is not None and getattr(_pi, "requires_input_strip", False):
                    input_path = sound_path or movie_path or ""
                    input_name = os.path.splitext(os.path.basename(input_path))[0]
                    if input_name:
                        job.prompt = input_name

                added += 1

        if added == 0:
            self.report({"WARNING"}, "Nothing to queue.")
            return {"CANCELLED"}

        label = f"{added} job(s)" if added > 1 else (prompt[:40] + ("…" if len(prompt) > 40 else ""))
        self.report({"INFO"}, f"Queued: {label}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Timer-based queue processing  (bpy.app.timers — main thread, no context needed)
# ---------------------------------------------------------------------------

_TIMER_INTERVAL = 0.2


def _queue_get_scene():
    """Return the scene being processed, or None if the RNA pointer has been freed."""
    try:
        if _active_scene is None:
            return None
        _active_scene.name  # validate the RNA pointer is still live
        return _active_scene
    except ReferenceError:
        return None


def _queue_start_job(scene, job) -> None:
    global _worker_thread
    print(f"[Pallaidium Queue] Starting job: {job.prompt[:60]!r}")
    job.status   = "RUNNING"
    job.progress = 0.0

    # Keep VRAM warm if the next pending job uses the same model
    next_job = next((j for j in scene.render_queue if j.status == "PENDING"), None)
    job.should_unload = (next_job is None or next_job.model_card != job.model_card)

    snapshot = {k: getattr(job, k) for k in (
        "job_id", "output_type", "model_card", "prompt", "neg_prompt",
        "steps", "guidance", "seed", "use_random", "width", "height",
        "frames", "image_power", "mode", "use_lcm", "refine_sd",
        "adetailer", "aurasr", "remove_silence", "audio_length",
        "audio_speed_tts", "chat_exaggeration", "chat_pace",
        "chat_temperature", "fps", "music_bpm", "music_lyrics",
        "music_key_scale", "music_time_signature",
        "image_path", "movie_path", "sound_path", "last_image_path", "middle_images_json", "ref_audio_path",
        "ref_text", "hugginface_token", "gemini_api_key",
        "remote_backend_url", "remote_backend_key", "local_files_only", "display_console",
        "generator_ai", "hf_cache_dir", "lora_files_json", "lora_folder",
        "insert_frame_start", "insert_frame_end",
        "insert_channel", "insert_duration",
        "sequencer_scene_name",
        "should_unload",
        "whisper_model_size", "whisper_language",
        "omnivoice_instruct", "omnivoice_language",
        "omnivoice_preprocess", "omnivoice_denoise", "omnivoice_postprocess",
        "chatterbox_mtl_language",
        "moss_model_variant", "moss_language", "moss_duration_tokens",
        "moss_max_new_tokens", "moss_temperature", "moss_top_p", "moss_top_k",
        "moss_ref_audio_path",
        "stem_split_model", "stem_split_vocals", "stem_split_drums",
        "stem_split_bass", "stem_split_other", "stem_split_guitar", "stem_split_piano",
        "qwen_strip_1_path", "qwen_strip_2_path", "qwen_strip_3_path",
        "klein_strip_1_path", "klein_strip_2_path", "klein_strip_3_path",
        *(f"flux_strip_{_n}_path" for _n in range(1, 10)),
        "minimax_subject_path",
        "nano_banana_ref_count",
        *(f"nano_banana_ref_strip_{_n}_path" for _n in range(1, 10)),
        "veo_ref_strip_1_path", "veo_ref_strip_2_path", "veo_ref_strip_3_path",
        *(f"nano_banana_ref_strip_{_n}" for _n in range(1, 10)),
        "veo_ref_strip_1", "veo_ref_strip_2", "veo_ref_strip_3",
        "florence2_mode", "florence2_send_to_mask",
        "klein_schematic_mode", "klein_schematic_target",
        "img_guidance_scale", "illumination_style", "light_direction",
        "ip_adapter_face_folder", "ip_adapter_style_folder",
        "openpose_use_bones", "use_scribble_image",
        "ideogram_prompt_upsampling",
        # ltx23_multi_v2 guidance params
        "ltx23m_modality_scale", "ltx23m_audio_guidance", "ltx23m_audio_stg_scale",
        "ltx23m_audio_modality_scale", "ltx23m_audio_noise_scale", "ltx23m_audio_start_time",
        # ltx23_multi_ic_lora control paths + params
        "ltx23ic_control_video_path", "ltx23ic_control_audio_path",
        "ltx23ic_control_strength", "ltx23ic_control_downscale",
        "ltx23ic_control_audio_str", "ltx23ic_identity_guidance",
        "ltx23ic_ref_image_path",
        # ltx23_extend params + resolved audio-strip path
        "ltx23ext_extend_frames", "ltx23ext_video_strength", "ltx23ext_audio_path",
        "ltx23_stage_mode",
        # Google Nano Banana / Veo cloud settings
        "nano_banana_model", "nano_banana_aspect", "nano_banana_resolution",
        "veo_model", "veo_aspect", "veo_resolution", "veo_duration",
        "veo_person_generation", "veo_image_mode",
        "marlin_mode", "marlin_find_query",
    )}
    _cancel_event.clear()
    _worker_thread = threading.Thread(
        target=_run_job,
        args=(snapshot, _result_queue, _cancel_event, _progress_store),
        daemon=True,
    )
    _worker_thread.start()


def _run_job_main_thread(scene, job) -> None:
    """Execute a requires_main_thread plugin synchronously on the main thread.

    Used by plugins (hviske_subtitles, marlin_video_captions) that must call
    bpy directly inside generate().  The UI will be unresponsive for the
    duration — same behaviour as interactive generation.
    """
    from ..models import get_plugin
    from ..models.base import ModelInputs
    from ..operators.main_ops import (
        _pallaidium_model_cache,
        _pallaidium_movie_model_cache,
        _pallaidium_audio_model_cache,
        _pallaidium_text_model_cache,
    )

    job_id = job.job_id
    print(f"[Pallaidium Queue] Starting main-thread job: {job.prompt[:60]!r}")
    job.status   = "RUNNING"
    job.progress = 0.0
    job.phase    = "Loading model"

    _progress_store[job_id] = {"progress": 0.0, "phase": "Loading model", "step": 0, "total": 0}

    # Determine whether to unload the model after this job
    following = next(
        (j for j in scene.render_queue
         if j.status == "PENDING" and j.job_id != job_id),
        None,
    )
    should_unload = following is None or following.model_card != job.model_card

    model_cache = None
    try:
        plugin = get_plugin(job.model_card)
        if plugin is None:
            raise RuntimeError(f"Plugin not found: {job.model_card!r}")

        otype = job.output_type
        model_cache = {
            "image": _pallaidium_model_cache,
            "movie": _pallaidium_movie_model_cache,
            "audio": _pallaidium_audio_model_cache,
            "text":  _pallaidium_text_model_cache,
        }.get(otype)

        mode = job.mode
        prefs = bpy.context.preferences.addons[ADDON_ID].preferences

        apply_hf_env(prefs)

        # Load model (or reuse cache)
        _schematic_mode_mt = getattr(job, "klein_schematic_mode", "")
        _cache_skip = {"last_model_card", "last_mode", "last_schematic_mode"}
        cache_hit = (
            model_cache is not None
            and model_cache.get("last_model_card") == job.model_card
            and model_cache.get("last_mode") == mode
            and model_cache.get("last_schematic_mode", "") == _schematic_mode_mt
            and any(v is not None for k, v in model_cache.items() if k not in _cache_skip)
        )
        if cache_hit:
            pipe_obj = model_cache
            print(f"[Queue] Reusing cached model: {job.model_card} ({mode}) schematic={_schematic_mode_mt or '-'}")
        else:
            if (model_cache is not None
                    and (model_cache.get("last_model_card") != job.model_card
                         or model_cache.get("last_mode") != mode
                         or model_cache.get("last_schematic_mode", "") != _schematic_mode_mt)):
                release_model_cache(model_cache)
            clear_cuda_cache()
            if job.display_console:
                show_system_console(True)
                set_system_console_topmost(True)

            _progress_store[job_id] = {
                "progress": 0.0,
                "phase":    "Loading model",
                "step":     0,
                "total":    0,
            }

            import tqdm.std as _tqdm_std_mt

            _active_bars_mt: dict = {}
            _dl_bars_mt: set = set()     # ids of bars that are actual network downloads (unit='B')
            _orig_tqdm_init_mt   = _tqdm_std_mt.tqdm.__init__
            _orig_tqdm_update_mt = _tqdm_std_mt.tqdm.update

            def _patched_tqdm_init_mt(tqdm_self, *a, **kw):
                _orig_tqdm_init_mt(tqdm_self, *a, **kw)
                if not getattr(tqdm_self, "disable", False):
                    _active_bars_mt[id(tqdm_self)] = [tqdm_self.n or 0, tqdm_self.total or 0]
                    if getattr(tqdm_self, "unit", "it") == "B":
                        _dl_bars_mt.add(id(tqdm_self))

            def _patched_tqdm_update_mt(tqdm_self, n=1):
                result = _orig_tqdm_update_mt(tqdm_self, n)
                entry = _active_bars_mt.get(id(tqdm_self))
                if entry is not None:
                    entry[0] = tqdm_self.n or 0
                    entry[1] = tqdm_self.total or 0
                total_b = sum(v[1] for v in _active_bars_mt.values() if v[1] > 0)
                done_b  = sum(v[0] for v in _active_bars_mt.values())
                dl_frac = (done_b / total_b) if total_b > 0 else 0.0
                phase = "Downloading model" if id(tqdm_self) in _dl_bars_mt else "Loading model"
                _progress_store[job_id] = {
                    "progress": dl_frac,
                    "phase":    phase,
                    "step":     max(0, int(done_b  / 1_048_576)),
                    "total":    max(1, int(total_b / 1_048_576)),
                }
                job.progress = dl_frac
                job.phase    = phase
                return result

            _tqdm_std_mt.tqdm.__init__ = _patched_tqdm_init_mt
            _tqdm_std_mt.tqdm.update   = _patched_tqdm_update_mt

            try:
                loaded = plugin.load(
                    prefs, scene,
                    mode=mode,
                    enabled_items=[],
                    use_lcm=job.use_lcm,
                    use_refine=job.refine_sd,
                    ip_adapter_face_folder="",
                    ip_adapter_style_folder="",
                    local_files_only=job.local_files_only,
                )
            finally:
                _tqdm_std_mt.tqdm.__init__ = _orig_tqdm_init_mt
                _tqdm_std_mt.tqdm.update   = _orig_tqdm_update_mt
                _active_bars_mt.clear()
                _dl_bars_mt.clear()

            if model_cache is not None and isinstance(loaded, dict):
                model_cache.update(loaded)
                model_cache["last_model_card"] = job.model_card
                model_cache["last_mode"] = mode
                model_cache["last_schematic_mode"] = _schematic_mode_mt
                pipe_obj = model_cache
            else:
                pipe_obj = loaded

        _progress_store[job_id] = {"progress": 0.10, "phase": "Preparing", "step": 0, "total": 0}

        def _phase_fn(label: str) -> None:
            store = _progress_store.get(job_id)
            if store is not None:
                store["phase"] = label
            job.phase = label  # direct update — we are on the main thread

        def _progress_fn(step: int, total: int) -> None:
            cur_phase = _progress_store.get(job_id, {}).get("phase", "Generating")
            _progress_store[job_id] = {
                "progress": 0.10 + 0.89 * (step / max(1, total)),
                "phase":    cur_phase,
                "step":     step,
                "total":    total,
            }

        _mt_last_image = None
        _mt_last_path  = getattr(job, "last_image_path", "")
        if _mt_last_path and os.path.isfile(_mt_last_path):
            _mt_last_image = load_first_frame(_mt_last_path)

        _mt_middle_paths = []
        _mt_middle_json = getattr(job, "middle_images_json", "") or ""
        if _mt_middle_json:
            try:
                _mt_middle_raw = json.loads(_mt_middle_json)
                _mt_middle_paths = [
                    (str(p), float(f)) for p, f in _mt_middle_raw
                    if p and os.path.isfile(str(p))
                ]
            except Exception as _e:
                print(f"[Queue] middle_images_json parse error (main thread): {_e}")

        inputs = ModelInputs(
            prompt               = job.prompt,
            neg_prompt           = job.neg_prompt,
            mode                 = mode,
            steps                = job.steps,
            guidance             = job.guidance,
            strength             = job.image_power,
            seed                 = job.seed,
            audio_ref            = job.ref_audio_path or job.sound_path or None,
            video_path           = job.movie_path or None,
            last_image           = _mt_last_image,
            middle_images_paths  = _mt_middle_paths,
            insert_channel      = job.insert_channel,
            insert_frame_start  = job.insert_frame_start,
            progress_fn         = _progress_fn,
            phase_fn            = _phase_fn,
        )

        _result = plugin.generate(pipe_obj, inputs, scene, prefs)
        del _result  # None for bpy-inserting plugins; discarded intentionally

        # plugin handled its own VSE strip creation
        job.status   = "COMPLETED"
        job.progress = 1.0
        job.phase    = "Done"
        _progress_store.pop(job_id, None)

    except Exception as exc:
        _err_str = str(exc)
        if job.local_files_only and isinstance(exc, OSError):
            _err_str = (
                "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
            )
        job.status          = "FAILED"
        job.progress        = 0.0
        job.error_message   = _err_str[:200]
        job.error_traceback = traceback.format_exc()[:4000]
        print("=== Pallaidium Queue Error (main-thread job) ===")
        print(traceback.format_exc())
        print("================================================")
        _progress_store.pop(job_id, None)
    finally:
        if should_unload and model_cache is not None:
            release_model_cache(model_cache)
            clear_cuda_cache()


def _find_vse_override():
    """Return (window, area, region) for the first SEQUENCE_EDITOR area, or (None, None, None).

    Including the WINDOW region is required for temp_override in timer callbacks
    on Blender 5.x — without it, strip-creation calls can silently fail.
    """
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "SEQUENCE_EDITOR":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            return window, area, region
    except Exception:
        pass
    return None, None, None


def _queue_insert_strip(scene, result: dict) -> None:
    """Insert the generated strip into the VSE timeline (main thread).

    Uses the sequencer_scene stored at job-add time so the strip lands in the
    correct scene even when workspace.sequencer_scene differs from the active scene.
    Tries a direct call first; falls back to temp_override with the WINDOW
    region for timer-callback invocations where Blender 5.x requires it.
    """
    out_path     = result.get("output_path", "")
    otype        = result.get("output_type", "")
    text_content = result.get("text_content", "")

    # Multi-stem output from StemSplitterPlugin: insert one SOUND strip per stem.
    _MULTI_PREFIX = "MULTI_STEM:"
    if out_path.startswith(_MULTI_PREFIX):
        import json as _json
        stem_paths = _json.loads(out_path[len(_MULTI_PREFIX):])
        seq_scene_name = result.get("sequencer_scene_name", "")
        seq_scene = (bpy.data.scenes.get(seq_scene_name) if seq_scene_name else None) or scene
        if not seq_scene.sequence_editor:
            seq_scene.sequence_editor_create()
        ed          = seq_scene.sequence_editor
        frame_start = result["frame_start"]
        frame_end   = result["frame_end"]
        base_ch     = result["channel"]
        next_ch     = base_ch
        for stem_name, stem_path in stem_paths.items():
            if not os.path.isfile(stem_path):
                print(f"[Queue] Stem file not found: {stem_path!r}")
                continue
            ch = _find_free_channel(seq_scene, frame_start, frame_end, next_ch)
            ch = max(ch, next_ch)
            next_ch = ch + 1
            # Derive input filename from the stem file: "{stem}_{orig_base}.wav"
            file_stem = os.path.splitext(os.path.basename(stem_path))[0]
            orig_base = file_stem[len(stem_name) + 1:] if file_stem.startswith(stem_name + "_") else file_stem
            strip_name = f"{stem_name} | {orig_base}"
            snd = ed.strips.new_sound(
                name=strip_name,
                filepath=stem_path,
                channel=ch,
                frame_start=frame_start,
            )
            snd.frame_final_duration = (result["frame_end"] - result["frame_start"]) - 1
            print(f"[Queue] Inserted stem '{stem_name}' on channel {ch}")
        return

    # For text-type jobs the content may be carried directly; no file required.
    if otype == "text" and text_content:
        pass  # proceed to strip creation
    elif otype == "text" and not text_content and not out_path:
        print("[Queue] Text model produced no output — skipping strip insertion.")
        return
    elif not out_path or not os.path.isfile(out_path):
        print(f"[Queue] Output file not found: {out_path!r}")
        return

    # Resolve the target scene — use the sequencer_scene recorded at add-time.
    seq_scene_name = result.get("sequencer_scene_name", "")
    seq_scene = (bpy.data.scenes.get(seq_scene_name) if seq_scene_name else None) or scene

    if not seq_scene.sequence_editor:
        seq_scene.sequence_editor_create()

    frame_start = result["frame_start"]
    frame_end   = result["frame_end"]
    channel     = _find_free_channel(seq_scene, frame_start, frame_end, result["channel"])
    duration    = result["duration"]
    name        = f"{result['seed']}_{result['prompt'][:30]}"
    ed          = seq_scene.sequence_editor

    def _do_insert():
        nonlocal new_strip
        if otype == "image":
            new_strip = ed.strips.new_image(
                name=name,
                filepath=out_path,
                frame_start=frame_start,
                channel=channel,
                fit_method="FIT",
            )
            new_strip.frame_final_duration = duration
            new_strip.use_proxy = True

        elif otype == "movie":
            new_strip = ed.strips.new_movie(
                name=name,
                filepath=out_path,
                frame_start=frame_start,
                channel=channel,
                fit_method="FIT",
            )
            snd_ch = _find_free_channel(seq_scene, frame_start, frame_end, max(1, channel - 1))
            snd_strip = ed.strips.new_sound(
                name=name,
                filepath=out_path,
                channel=snd_ch,
                frame_start=frame_start,
            )
            snd_strip.select = False

        elif otype == "audio":
            new_strip = ed.strips.new_sound(
                name=result["prompt"][:50],
                filepath=out_path,
                channel=channel,
                frame_start=frame_start,
            )

        elif otype == "text":
            # Build the text body: prefer the in-memory content; fall back to
            # reading the .txt file saved by the worker.
            text_body = text_content or ""
            if not text_body and out_path and out_path.endswith(".txt"):
                try:
                    with open(out_path, encoding="utf-8") as _tf:
                        text_body = _tf.read()
                except Exception:
                    pass
            if text_body:
                # Use // comment-free text for the strip name so UIList looks clean,
                # but keep // image: in strip.text so Florence-2 can recover the
                # source image path when this strip is re-queued.
                clean_name = "\n".join(
                    ln for ln in text_body.splitlines()
                    if not ln.strip().startswith("//")
                ).strip()
                new_strip = ed.strips.new_effect(
                    name=(clean_name or text_body)[:63],
                    type="TEXT",
                    frame_start=frame_start,
                    length=max(1, frame_end - frame_start),
                    channel=channel,
                )
                new_strip.text = text_body

    new_strip = None
    try:
        _do_insert()
    except Exception as exc1:
        # Direct call failed (common from timer callbacks) — retry with override.
        new_strip = None
        window, area, region = _find_vse_override()
        if window is None:
            print(f"[Queue] No SEQUENCE_EDITOR area found ({exc1}) — cannot insert strip.")
            return
        try:
            with bpy.context.temp_override(window=window, area=area, region=region):
                _do_insert()
        except Exception as exc2:
            print(f"[Queue] Strip insertion failed: {exc2}")
            traceback.print_exc()
            return

    # --- Florence-2 → Mask Editor (text/Ideogram4 with send_to_mask enabled) ---
    if (
        result.get("output_type") == "text"
        and result.get("florence2_send_to_mask")
        and result.get("text_content")
    ):
        try:
            from .mask_florence2 import apply_florence_json_to_mask

            # The worker already rendered the frame, ran Florence-2, built the
            # name, renamed the PNG, and embedded "// image: <path>" in the JSON.
            # Pass source_image_path="" so apply_florence_json_to_mask reads the
            # path from the embedded comment instead of re-rendering here.
            apply_florence_json_to_mask(result["text_content"], "")
        except Exception as _mex:
            print(f"[Queue] Florence2 mask creation failed: {_mex}")

    if new_strip is not None:
        try:
            # Use the actual output dimensions from the strip element rather than
            # the snapshot values, which may differ due to model-specific rounding.
            actual_w = result.get("width", 0)
            actual_h = result.get("height", 0)
            if new_strip.type in ('IMAGE', 'MOVIE'):
                try:
                    elem = new_strip.elements[0]
                    if elem.orig_width and elem.orig_height:
                        actual_w = elem.orig_width
                        actual_h = elem.orig_height
                except Exception:
                    pass

            # Write actual resolution back into the job so Redo restores it correctly.
            job = _find_job(scene, result.get("job_id", ""))
            if job is not None:
                job.width  = actual_w
                job.height = actual_h

            # Build enabled-only LoRA list for metadata display.
            try:
                lora_all = json.loads(result.get("lora_files_json", "[]") or "[]")
            except (json.JSONDecodeError, ValueError):
                lora_all = []
            lora_enabled = [item for item in lora_all if item.get("enabled", True)]
            lora_enabled_json = json.dumps(lora_enabled)

            # Individual per-LoRA props for string-box display in the panel.
            lora_meta = {}
            if lora_enabled:
                lora_meta["lora_folder"] = result.get("lora_folder", "")
                for i, item in enumerate(lora_enabled, 1):
                    lora_meta[f"lora_{i}"]        = os.path.basename(item.get("name", ""))
                    lora_meta[f"lora_{i}_weight"]  = f"{item.get('weight', 1.0):.2f}"

            extra_meta = {}
            if result.get("img_guidance_scale", 1.6) != 1.6:
                extra_meta["img_guidance_scale"] = result.get("img_guidance_scale", 1.6)
            if result.get("illumination_style"):
                extra_meta["illumination_style"] = result.get("illumination_style", "")
            if result.get("light_direction"):
                extra_meta["light_direction"] = result.get("light_direction", "")
            if result.get("ip_adapter_face_folder"):
                extra_meta["ip_adapter_face_folder"] = result.get("ip_adapter_face_folder", "")
            if result.get("ip_adapter_style_folder"):
                extra_meta["ip_adapter_style_folder"] = result.get("ip_adapter_style_folder", "")
            if result.get("openpose_use_bones"):
                extra_meta["openpose_use_bones"] = result.get("openpose_use_bones", False)
            if result.get("use_scribble_image"):
                extra_meta["use_scribble_image"] = result.get("use_scribble_image", False)
            if result.get("ideogram_prompt_upsampling"):
                extra_meta["ideogram_prompt_upsampling"] = result.get("ideogram_prompt_upsampling", False)
            # ltx23_multi_v2 — only write non-default values to keep metadata compact
            for _k, _def in [
                ("ltx23m_modality_scale",       1.0),
                ("ltx23m_audio_guidance",       1.0),
                ("ltx23m_audio_stg_scale",      0.0),
                ("ltx23m_audio_modality_scale", 1.0),
                ("ltx23m_audio_noise_scale",    0.0),
                ("ltx23m_audio_start_time",     0.0),
                # ltx23_extend params
                ("ltx23ext_extend_frames",      96),
                ("ltx23ext_video_strength",     1.0),
                # ltx23 staged
                ("ltx23_stage_mode",            "FULL"),
                # Maxine VSR
                ("maxine_quality",              "HIGH"),
                # Google Nano Banana (Gemini image)
                ("nano_banana_model",      "gemini-2.5-flash-image"),
                ("nano_banana_aspect",     "1:1"),
                ("nano_banana_resolution", "1K"),
                # Google Veo (video)
                ("veo_model",              "veo-3.1-fast-generate-preview"),
                ("veo_aspect",             "16:9"),
                ("veo_resolution",         "720p"),
                ("veo_duration",           "8"),
                ("veo_person_generation",  "allow_adult"),
                ("veo_image_mode",         "AUTO"),
            ]:
                _v = result.get(_k, _def)
                if _v != _def:
                    extra_meta[_k] = _v
            # Google Nano Banana / Veo reference strips — write names (for Redo)
            # and resolved paths (record) only when a slot was actually used.
            # Only the first nano_banana_ref_count slots are active (the rest are
            # hidden in the UI), so ignore stale names/paths beyond that count.
            _nb_count = max(1, min(int(result.get("nano_banana_ref_count", 3) or 3), 9))
            if any(result.get(f"nano_banana_ref_strip_{_n}") for _n in range(1, _nb_count + 1)):
                extra_meta["nano_banana_ref_count"] = _nb_count
            for _k in (
                *(f"nano_banana_ref_strip_{_n}" for _n in range(1, _nb_count + 1)),
                "veo_ref_strip_1", "veo_ref_strip_2", "veo_ref_strip_3",
                *(f"nano_banana_ref_strip_{_n}_path" for _n in range(1, _nb_count + 1)),
                "veo_ref_strip_1_path", "veo_ref_strip_2_path", "veo_ref_strip_3_path",
            ):
                if result.get(_k):
                    extra_meta[_k] = result.get(_k)
            # ltx23_extend — write the resolved audio-strip path when present
            if result.get("ltx23ext_audio_path"):
                extra_meta["ltx23ext_audio_path"] = result.get("ltx23ext_audio_path", "")
            # ltx23_multi_ic_lora — always write so redo-from-metadata can reconstruct paths
            for _k in (
                "ltx23ic_control_video_path", "ltx23ic_control_audio_path",
                "ltx23ic_control_strength", "ltx23ic_control_downscale",
                "ltx23ic_control_audio_str", "ltx23ic_identity_guidance",
                "ltx23ic_ref_image_path",
            ):
                _v = result.get(_k)
                if _v is not None:
                    extra_meta[_k] = _v
            # Chatterbox Multilingual
            if result.get("model_card", "") == "ChatterboxMultilingual":
                _v = result.get("chatterbox_mtl_language", "en")
                extra_meta["chatterbox_mtl_language"] = _v
            # MOSS-TTS — write only when it was the model used, so other models'
            # strips don't carry irrelevant TTS metadata.
            if result.get("model_card", "") == "MOSS-TTS":
                for _k, _def in [
                    ("moss_model_variant",   "v1.5"),
                    ("moss_language",        "AUTO"),
                    ("moss_duration_tokens", 0),
                    ("moss_max_new_tokens",  4096),
                    ("moss_temperature",     1.7),
                    ("moss_top_p",           0.8),
                    ("moss_top_k",           25),
                    ("moss_ref_audio_path",  ""),
                ]:
                    _v = result.get(_k, _def)
                    # variant always written (identifies the run); others only if non-default
                    if _k == "moss_model_variant" or _v != _def:
                        extra_meta[_k] = _v

            # Generic input paths — write so redo-from-metadata can reconstruct the
            # inputs even when the metadata carrier is a TEXT strip (Screenwriter
            # script-to-screen jobs) rather than the rendered media itself. Only
            # non-empty values are written to keep metadata compact.
            for _k in (
                "image_path", "last_image_path", "middle_images_json",
                "movie_path", "sound_path", "ref_audio_path", "ref_text",
                "kontext_strip_1_path",
                "qwen_strip_1_path", "qwen_strip_2_path", "qwen_strip_3_path",
                "klein_strip_1_path", "klein_strip_2_path", "klein_strip_3_path",
            ):
                _v = result.get(_k)
                if _v:
                    extra_meta[_k] = _v

            set_ai_metadata_from_dict(new_strip, {
                "output_type":     result.get("output_type", ""),
                "model":           result.get("model_card", ""),
                "mode":            result.get("mode", ""),
                "prompt":          result.get("prompt", ""),
                "negative_prompt": result.get("neg_prompt", ""),
                "seed":            result.get("seed", 0),
                "width":           actual_w,
                "height":          actual_h,
                "frames":          result.get("frames", 0),
                "steps":           result.get("steps", 0),
                "guidance":        result.get("guidance", 0.0),
                "lora_files_json": lora_enabled_json,
                **lora_meta,
                **extra_meta,
            })
            new_strip.select = False
            ed.active_strip = new_strip
        except Exception:
            pass


def _queue_stop() -> None:
    """Unregister the processing timer."""
    if bpy.app.timers.is_registered(_queue_tick):
        bpy.app.timers.unregister(_queue_tick)


def _queue_tick() -> float | None:
    """Called by bpy.app.timers every 0.2 s on the main thread.

    Returns the next interval (float) to keep running, or None to stop.
    """
    global _worker_thread

    try:
        scene = _queue_get_scene()
        if scene is None:
            _queue_stop()
            return None

        # ---- Drain one result from the worker ---------------------------
        try:
            result = _result_queue.get_nowait()
        except _stdqueue.Empty:
            result = None

        if result:
            job = _find_job(scene, result["job_id"])
            status = result["status"]

            if job:
                if job.status in ("CANCELLING", "CANCELLED"):
                    job.status   = "CANCELLED"
                    job.progress = 0.0
                elif status == "COMPLETED":
                    job.output_path = result.get("output_path", "")
                    job.tokens_info = result.get("result_note", "")
                    job.status      = "COMPLETED"
                    job.progress    = 1.0
                    _queue_insert_strip(scene, result)
                elif status == "FAILED":
                    job.status          = "FAILED"
                    job.progress        = 0.0
                    job.error_message   = result.get("error", "Unknown error")[:200]
                    job.error_traceback = result.get("traceback", "")[:4000]
                    print("=== Pallaidium Queue Error ===")
                    print(result.get("traceback", ""))
                    print("==============================")
                else:
                    job.status = "CANCELLED"

            # Do NOT set _worker_thread = None here. The thread may still be
            # executing its finally block (release_model_cache, clear_cuda_cache).
            # Setting it to None too early would let the next job start while the
            # previous thread is still tearing down the model cache, causing a race
            # where the new job loads the model and the old thread then clears it.
            # The is_alive() check below is the correct gate for starting the next job.
            _cancel_event.clear()

        # ---- Sync progress for running job ------------------------------
        running = _find_running_job(scene)
        if running and running.job_id in _progress_store:
            try:
                store = _progress_store[running.job_id]
                running.progress     = store["progress"]
                running.phase        = store["phase"]
                running.current_step = store["step"]
                running.total_steps  = store["total"]
            except Exception:
                pass  # stale or unexpected store format; skip this tick

        # ---- Force panel redraw so slider animates without mouse movement ---
        try:
            for _win in bpy.context.window_manager.windows:
                for _area in _win.screen.areas:
                    if _area.type == "SEQUENCE_EDITOR":
                        _area.tag_redraw()
        except Exception:
            pass

        # ---- Start next job if worker is idle ---------------------------
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = None
            if running and running.status == "RUNNING":
                running.status        = "FAILED"
                running.error_message = "Worker thread died unexpectedly."
                running = None
            if not _queue_paused:
                next_job = _find_next_pending(scene)
                if next_job:
                    from ..models import get_plugin as _get_plugin
                    _next_plugin = _get_plugin(next_job.model_card)
                    if _next_plugin and getattr(_next_plugin, "requires_main_thread", False):
                        # Run synchronously on the main thread (UI will freeze).
                        _run_job_main_thread(scene, next_job)
                    else:
                        _queue_start_job(scene, next_job)

        # ---- Stop conditions --------------------------------------------
        still_active = any(
            j.status in ("PENDING", "RUNNING", "CANCELLING")
            for j in scene.render_queue
        )
        worker_idle = _worker_thread is None or not _worker_thread.is_alive()

        if (not still_active and worker_idle) or (_queue_paused and worker_idle):
            _queue_stop()
            return None

        return _TIMER_INTERVAL

    except ReferenceError:
        # Scene was freed mid-tick (file reload, scene deletion, etc.) — stop cleanly.
        _queue_stop()
        return None
    except BaseException:
        traceback.print_exc()
        _queue_stop()
        return None


# ---------------------------------------------------------------------------
# Operator: Queue runner  (simple execute — starts the app timer)
# ---------------------------------------------------------------------------

class SEQUENCER_OT_queue_runner(Operator):
    """Start processing the render queue"""

    bl_idname = "sequencer.queue_runner"
    bl_label  = "Start Queue"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _queue_paused, _active_scene
        _queue_paused = False
        _active_scene = context.scene   # direct reference — no bpy.data needed in timer
        # Always (re-)register to clear any stale timer from a previous crash
        if bpy.app.timers.is_registered(_queue_tick):
            bpy.app.timers.unregister(_queue_tick)
        bpy.app.timers.register(_queue_tick, first_interval=_TIMER_INTERVAL)
        print("[Pallaidium Queue] Started — scene:", context.scene.name)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Cancel a job
# ---------------------------------------------------------------------------

class SEQUENCER_OT_cancel_queue_job(Operator):
    """Cancel a pending or running job"""

    bl_idname = "sequencer.cancel_queue_job"
    bl_label  = "Cancel Job"

    job_id: StringProperty()

    def execute(self, context):
        job = _find_job(context.scene, self.job_id)
        if job is None:
            return {"CANCELLED"}
        if job.status == "PENDING":
            job.status = "CANCELLED"
        elif job.status == "RUNNING":
            # Signal the worker; it will stop at the next cancellation checkpoint
            _cancel_event.set()
            job.status = "CANCELLING"
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Redo — reload a job's settings back into the Pallaidium UI
# ---------------------------------------------------------------------------

class SEQUENCER_OT_redo_from_job(Operator):
    """Reload this job's generation settings into the Pallaidium panel"""

    bl_idname = "sequencer.redo_from_job"
    bl_label  = "Redo Job Settings"
    bl_description = "Reload this job's generation settings into the Pallaidium panel"

    job_id: StringProperty()

    def execute(self, context):
        job = _find_job(context.scene, self.job_id)
        if job is None:
            self.report({'WARNING'}, "Job not found")
            return {"CANCELLED"}

        scene = context.scene
        prefs = context.preferences.addons[ADDON_ID].preferences

        # Set typeselect + model first — input_strips_updated fires here and
        # overwrites x/y/frames with model defaults.  All explicit values
        # are written afterwards so they win over those defaults.
        scene.generatorai_typeselect = job.output_type

        model_attr = {
            "image": "image_model_card",
            "movie": "movie_model_card",
            "audio": "audio_model_card",
            "text":  "text_model_card",
        }.get(job.output_type)
        if model_attr and job.model_card:
            try:
                setattr(prefs, model_attr, job.model_card)
            except TypeError:
                pass

        # Now set all generation params — these override any callback defaults.
        scene.generate_movie_prompt          = job.prompt
        scene.generate_movie_negative_prompt = job.neg_prompt
        scene.movie_num_inference_steps      = job.steps
        scene.movie_num_guidance             = job.guidance
        scene.movie_num_seed                 = job.seed
        scene.movie_use_random               = job.use_random
        scene.generate_movie_x               = job.width
        scene.generate_movie_y               = job.height
        scene.generate_movie_frames          = job.frames
        scene.image_power                    = job.image_power
        scene.use_lcm                        = job.use_lcm
        scene.refine_sd                      = job.refine_sd
        scene.adetailer                      = job.adetailer
        scene.aurasr                         = job.aurasr
        scene.remove_silence                 = job.remove_silence
        scene.audio_length_in_f              = int(job.audio_length)
        scene.audio_speed_tts                = job.audio_speed_tts
        scene.chat_exaggeration              = job.chat_exaggeration
        scene.chat_pace                      = job.chat_pace
        scene.chat_temperature               = job.chat_temperature
        scene.music_bpm                      = job.music_bpm
        scene.music_lyrics                   = job.music_lyrics
        scene.music_key_scale                = job.music_key_scale
        scene.music_time_signature           = job.music_time_signature
        if hasattr(scene, "img_guidance_scale"):
            scene.img_guidance_scale         = job.img_guidance_scale
        if hasattr(scene, "illumination_style") and job.illumination_style:
            scene.illumination_style         = job.illumination_style
        if hasattr(scene, "light_direction") and job.light_direction:
            scene.light_direction            = job.light_direction
        if hasattr(scene, "ip_adapter_face_folder") and job.ip_adapter_face_folder:
            scene.ip_adapter_face_folder     = job.ip_adapter_face_folder
        if hasattr(scene, "ip_adapter_style_folder") and job.ip_adapter_style_folder:
            scene.ip_adapter_style_folder    = job.ip_adapter_style_folder
        if hasattr(scene, "openpose_use_bones"):
            scene.openpose_use_bones         = job.openpose_use_bones
        if hasattr(scene, "use_scribble_image"):
            scene.use_scribble_image         = job.use_scribble_image
        if hasattr(scene, "whisper_model_size") and job.whisper_model_size:
            scene.whisper_model_size         = job.whisper_model_size
        if hasattr(scene, "whisper_language") and job.whisper_language:
            scene.whisper_language           = job.whisper_language
        # ltx23 staged
        if hasattr(scene, "ltx23_stage_mode"):
            scene.ltx23_stage_mode           = job.ltx23_stage_mode
        # Google Nano Banana (Gemini image)
        if hasattr(scene, "nano_banana_model"):
            scene.nano_banana_model          = job.nano_banana_model
            scene.nano_banana_aspect         = job.nano_banana_aspect
            scene.nano_banana_resolution     = job.nano_banana_resolution
        # Google Veo (video)
        if hasattr(scene, "veo_model"):
            scene.veo_model                  = job.veo_model
            scene.veo_aspect                 = job.veo_aspect
            scene.veo_resolution             = job.veo_resolution
            scene.veo_duration               = job.veo_duration
            scene.veo_person_generation      = job.veo_person_generation
            scene.veo_image_mode             = job.veo_image_mode
        # Reference strips (re-rendered from the restored names at queue time)
        if hasattr(scene, "nano_banana_ref_count"):
            scene.nano_banana_ref_count = getattr(job, "nano_banana_ref_count", 3) or 3
        for _attr in (
            *(f"nano_banana_ref_strip_{_n}" for _n in range(1, 10)),
            "veo_ref_strip_1", "veo_ref_strip_2", "veo_ref_strip_3",
        ):
            if hasattr(scene, _attr):
                setattr(scene, _attr, getattr(job, _attr, ""))
        # Chatterbox Multilingual
        if hasattr(scene, "chatterbox_mtl_language"):
            scene.chatterbox_mtl_language    = job.chatterbox_mtl_language
        # MOSS-TTS
        if hasattr(scene, "moss_model_variant"):
            scene.moss_model_variant         = job.moss_model_variant
            scene.moss_language              = job.moss_language
            scene.moss_duration_tokens       = job.moss_duration_tokens
            scene.moss_max_new_tokens        = job.moss_max_new_tokens
            scene.moss_temperature           = job.moss_temperature
            scene.moss_top_p                 = job.moss_top_p
            scene.moss_top_k                 = job.moss_top_k
            scene.moss_ref_audio_path        = job.moss_ref_audio_path

        # Restore LoRA: scan full folder so all files appear in the UIList,
        # then apply saved enabled state and weights from the job snapshot.
        if job.lora_folder:
            scene.lora_folder = job.lora_folder
        try:
            lora_raw = json.loads(job.lora_files_json) if job.lora_files_json else []
        except (json.JSONDecodeError, ValueError):
            lora_raw = []
        saved_loras = {
            item.get("name", ""): {"weight": item.get("weight", 1.0), "enabled": item.get("enabled", True)}
            for item in lora_raw
        }
        scene.lora_files.clear()
        directory = bpy.path.abspath(scene.lora_folder) if scene.lora_folder else ""
        if directory and os.path.isdir(directory):
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".safetensors"):
                    stem  = filename.replace(".safetensors", "")
                    entry = scene.lora_files.add()
                    entry.name = stem
                    if stem in saved_loras:
                        entry.enabled      = saved_loras[stem]["enabled"]
                        entry.weight_value = saved_loras[stem]["weight"]
                    else:
                        entry.enabled      = False
                        entry.weight_value = 1.0
        elif lora_raw:
            # Folder not accessible — fall back to just the saved entries.
            for item in lora_raw:
                entry = scene.lora_files.add()
                entry.name         = item.get("name", "")
                entry.weight_value = item.get("weight", 1.0)
                entry.enabled      = item.get("enabled", True)

        self.report({'INFO'}, "Settings loaded from job")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Remove a single job from the list
# ---------------------------------------------------------------------------

class SEQUENCER_OT_remove_queue_job(Operator):
    """Remove a finished or cancelled job from the list"""

    bl_idname = "sequencer.remove_queue_job"
    bl_label  = "Remove Job"

    job_id: StringProperty()

    def execute(self, context):
        queue = context.scene.render_queue
        for i, job in enumerate(queue):
            if job.job_id == self.job_id:
                if job.status == "RUNNING":
                    self.report({"WARNING"}, "Cannot remove a running job — cancel it first.")
                    return {"CANCELLED"}
                queue.remove(i)
                return {"FINISHED"}
        return {"CANCELLED"}


# ---------------------------------------------------------------------------
# Operator: Clear all finished jobs
# ---------------------------------------------------------------------------

class SEQUENCER_OT_clear_queue(Operator):
    """Remove all jobs that are not actively running"""

    bl_idname = "sequencer.clear_queue"
    bl_label  = "Clear Queue"

    def execute(self, context):
        queue = context.scene.render_queue
        indices = [
            i for i, j in enumerate(queue)
            if j.status != "RUNNING"
        ]
        for i in reversed(indices):
            queue.remove(i)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Show full error traceback in a popup
# ---------------------------------------------------------------------------

class SEQUENCER_OT_show_queue_error(Operator):
    """Show the full error traceback for a failed job"""

    bl_idname = "sequencer.show_queue_error"
    bl_label  = "Job Error Details"

    job_id: StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=620)

    def draw(self, context):
        job = _find_job(context.scene, self.job_id)
        layout = self.layout
        if job is None:
            layout.label(text="Job not found.")
            return

        col = layout.column(align=True)
        col.label(text=f"Failed job: {job.prompt[:60]}", icon="ERROR")
        col.label(text=f"Model: {job.model_card}")
        col.separator()

        box = col.box()
        box.label(text=f"Error: {job.error_message}", icon="CANCEL")
        col.separator()

        col.label(text="Traceback (for bug reports):")
        tb_box = col.box()
        for line in (job.error_traceback or "(no traceback)").splitlines():
            tb_box.label(text=line[:95])

        col.separator()
        col.label(text="Full traceback also printed to the System Console.", icon="INFO")

    def execute(self, context):
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Stop queue after current job (keeps pending jobs for later restart)
# ---------------------------------------------------------------------------

class SEQUENCER_OT_stop_queue(Operator):
    """Stop the queue after the current job finishes (pending jobs are kept)"""

    bl_idname = "sequencer.stop_queue"
    bl_label  = "Stop Queue"

    def execute(self, context):
        global _queue_paused
        _queue_paused = True
        # If no job is running right now, stop the timer immediately;
        # otherwise let the current job finish and the timer will stop itself.
        worker_idle = _worker_thread is None or not _worker_thread.is_alive()
        if worker_idle:
            _queue_stop()
            self.report({"INFO"}, "Queue stopped.")
        else:
            self.report({"INFO"}, "Queue will stop after the current job.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Exported list (used in __init__.py's classes tuple)
# ---------------------------------------------------------------------------

queue_classes = (
    RenderQueueJob,
    SEQUENCER_OT_add_to_queue,
    SEQUENCER_OT_queue_runner,
    SEQUENCER_OT_stop_queue,
    SEQUENCER_OT_cancel_queue_job,
    SEQUENCER_OT_redo_from_job,
    SEQUENCER_OT_remove_queue_job,
    SEQUENCER_OT_clear_queue,
    SEQUENCER_OT_show_queue_error,
)
