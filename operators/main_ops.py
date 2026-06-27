import bpy
from bpy_extras.io_utils import ExportHelper
import ctypes
import random
import site
import platform
import json
import subprocess
import sys
import os
import aud
import re
import glob
import string
from os.path import dirname, realpath, isdir, join, basename
import shutil
from datetime import date
import pathlib
import gc
import time
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, Panel, AddonPreferences, UIList, PropertyGroup
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
)
import sys
import base64
from io import BytesIO
import asyncio
import inspect
from fractions import Fraction
import importlib
import importlib.metadata
import warnings
import logging
import bpy

# ── Reload diagnostic ─────────────────────────────────────────────────────────
# Writes a line to a file every time THIS module is imported by Blender, and a
# helper generate_image() uses to log the img2img decision.  Because Blender
# caches imported modules, editing this file has NO effect until the addon is
# reloaded (full restart, or disable→enable the extension).  If the timestamp in
# this file does not update after a restart, the edit is not being loaded.
_PALLAIDIUM_DIAG_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "_diag_scene_input.log")

def _diag(msg):
    try:
        with open(_PALLAIDIUM_DIAG_LOG, "a", encoding="utf-8") as _fh:
            _fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except Exception:
        pass

_diag("=== main_ops.py IMPORTED (build: scene-strip-v3) ===")
import os
import re
from datetime import date

from ..utils.helpers import *
from ..properties.scene_props import *
from ..properties.preferences import *
from ..ui.panels import *


# Set True to re-enable [Florence2Mask] debug logging.
_DEBUG = False


def _dbg(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


_pallaidium_movie_model_cache = {
    "pipe": None,
    "refiner": None,
    "last_model_card": None,
}

_pallaidium_model_cache = {
    "pipe": None,
    "converter": None,
    "refiner": None,
    "preprocessor": None,
    "last_model_card": None,
}

_pallaidium_audio_model_cache = {
    "pipe": None,
    "vocoder": None,
    "model": None,
    "feature_extractor": None,
    "last_model_card": None,
}

_pallaidium_text_model_cache = {
    "model": None,
    "processor": None,
    "tokenizer": None,
    "last_model_card": None,
}


class SEQUENCER_OT_generate_movie(Operator):
    """Generate Video"""

    bl_idname = "sequencer.generate_movie"
    bl_label = "Prompt"
    bl_description = "Convert text to video"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):  # noqa: C901
        global _pallaidium_movie_model_cache
        import os
        import random

        scene = context.scene
        if not scene.sequence_editor:
            scene.sequence_editor_create()

        try:
            import torch
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
        except ModuleNotFoundError as e:
            print(f"Dependencies not installed: {e.name}")
            self.report({"INFO"}, "In the add-on preferences, install dependencies.")
            return {"CANCELLED"}

        preferences   = context.preferences
        addon_prefs   = preferences.addons[ADDON_ID].preferences
        if addon_prefs.display_console:
            show_system_console(True)
            set_system_console_topmost(True)
        movie_model_card = addon_prefs.movie_model_card

        from ..models import get_plugin
        from ..models.base import ModelInputs

        plugin = get_plugin(movie_model_card)
        if plugin is None:
            self.report({"ERROR"}, f"Video plugin not found: {movie_model_card!r}")
            return {"CANCELLED"}
        if not plugin.is_available():
            self.report({"INFO"}, "Dependencies need to be installed in the add-on preferences.")
            return {"CANCELLED"}

        current_frame    = scene.frame_current
        prompt           = style_prompt(scene.generate_movie_prompt)[0]
        negative_prompt  = (
            scene.generate_movie_negative_prompt
            + ", "
            + style_prompt(scene.generate_movie_prompt)[1]
        )
        x = scene.generate_movie_x = closest_divisible_32(scene.generate_movie_x)
        y = scene.generate_movie_y = closest_divisible_32(scene.generate_movie_y)
        old_duration = duration = scene.generate_movie_frames
        input_mode = scene.input_strips
        # Use active TEXT strip content as prompt when in input_strip mode.
        if input_mode == "input_strips":
            _active = scene.sequence_editor.active_strip
            if _active and _active.type == "TEXT" and _active.text:
                _text = _active.text
                _raw = scene.generate_movie_prompt
                if not (_raw == _text or _raw.startswith(_text + ", ")):
                    _raw = _text + (", " + _raw if _raw else "")
                    prompt = style_prompt(_raw)[0]
                    negative_prompt = (
                        scene.generate_movie_negative_prompt
                        + ", "
                        + style_prompt(_raw)[1]
                    )

        should_load   = context.scene.get("ai_load_state",   True)
        should_unload = context.scene.get("ai_unload_state", True)

        cache     = _pallaidium_movie_model_cache
        has_model = cache["pipe"] is not None
        if not has_model and not should_load:
            print("Video model cache empty. Forcing load.")
            should_load = True
        if cache["last_model_card"] != movie_model_card:
            print("Video model changed. Releasing old model.")
            release_model_cache(cache)
            should_load = True

        # Detect mode
        # scene.movie_path is set by both the UI eyedropper (input_strips mode) and by the
        # strip processor (strip_to_generatorAI) which works regardless of input_strips mode.
        # The input_strips_updated callback clears movie_path when switching to input_prompt,
        # so it is safe to read movie_path unconditionally here.
        has_video = bool(getattr(scene, "movie_path", ""))
        has_image = bool(getattr(scene, "image_path", "")) and input_mode == "input_strips"
        if has_video or has_image:
            mode = "img2vid"   # img2vid covers both; vid2vid handled by plugin if video_path is set
            if has_video:
                mode = "vid2vid"
        else:
            mode = "txt2vid"

        # Collect LoRA enabled items
        lora_files    = getattr(scene, "lora_files", [])
        enabled_items = [item for item in lora_files if item.enabled]

        if should_load:
            clear_cuda_cache()
            apply_hf_env(addon_prefs)
            t_load = bench_print(f"[{plugin.MODEL_ID}] load start")
            pipe_obj = plugin.load(
                addon_prefs, scene,
                mode=mode,
                enabled_items=enabled_items,
            )
            bench_print(f"[{plugin.MODEL_ID}] load done", t_load)
            cache["pipe"]            = pipe_obj.get("pipe")
            cache["refiner"]         = pipe_obj.get("refiner")
            cache["last_model_card"] = movie_model_card
        else:
            pipe_obj = {
                "pipe":    cache["pipe"],
                "refiner": cache["refiner"],
            }

        # Batch loop
        for i in range(scene.movie_num_batch):
            if duration == -1 and input_mode == "input_strips":
                strip = scene.sequence_editor.active_strip
                if strip:
                    duration = scene.generate_movie_frames = strip.frame_final_duration + 1

            start_time = timer()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = (
                    scene.sequence_editor.active_strip.frame_final_start
                    + scene.sequence_editor.active_strip.frame_final_duration
                )
                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * abs(duration)) + scene.frame_current,
                )
                start_frame = scene.frame_current

            seed = context.scene.movie_num_seed
            seed = (
                seed if not context.scene.movie_use_random
                else random.randint(-2147483647, 2147483647)
            )
            print("Seed: " + str(seed))
            context.scene.movie_num_seed = seed

            # Load input image / video path
            init_image  = None
            video_path  = None
            audio_ref   = None

            if has_image and input_mode == "input_strips":
                active = scene.sequence_editor.active_strip
                if active and active.type == "IMAGE":
                    img_path = bpy.path.abspath(
                        os.path.join(active.directory, active.elements[0].filename)
                    )
                    if os.path.isfile(img_path):
                        from PIL import Image as _PIL_Image
                        init_image = _PIL_Image.open(img_path).convert("RGB")

            # Detect FLF / last-frame-only modes for LTX Multi plugin
            _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".exr", ".webp", ".hdr"}
            _is_ltx_multi = getattr(plugin, "MODEL_ID", "") in {
                "LTX-2 Multi-Input File", "LTX-2.3 Multi-Input File"
            }
            _last_img_raw = bpy.path.abspath(getattr(scene, "image_path", "") or "")
            _last_img_is_image = (
                bool(_last_img_raw)
                and os.path.splitext(_last_img_raw)[1].lower() in _IMAGE_EXTS
                and os.path.isfile(_last_img_raw)
            )
            _movie_ext = os.path.splitext(getattr(scene, "movie_path", ""))[1].lower()
            _flf_mode = _is_ltx_multi and _movie_ext in _IMAGE_EXTS and _last_img_is_image
            _lfo_mode = _is_ltx_multi and not getattr(scene, "movie_path", "") and _last_img_is_image
            _flf_last_image = None

            if has_video:
                vp = bpy.path.abspath(scene.movie_path)
                if os.path.isfile(vp):
                    if _flf_mode:
                        init_image      = load_first_frame(vp)
                        video_path      = None
                        _flf_last_image = load_first_frame(_last_img_raw)
                    else:
                        video_path = vp
                        if init_image is None:
                            init_image = load_first_frame(vp)

            if _lfo_mode:
                _flf_last_image = load_first_frame(_last_img_raw)

            if getattr(scene, "sound_path", ""):
                sp = bpy.path.abspath(scene.sound_path)
                if os.path.isfile(sp):
                    audio_ref = sp

            # LTX Multi N-anchor: parse middle image paths+fractions from scene property
            _middle_images_paths = []
            if _is_ltx_multi:
                _middle_json = getattr(scene, "ltx_middle_images_json", "") or ""
                if _middle_json:
                    try:
                        import json as _json_mi
                        _middle_images_paths = [
                            (str(p), float(f)) for p, f in _json_mi.loads(_middle_json)
                            if p and os.path.isfile(str(p))
                        ]
                    except Exception as _e:
                        print(f"[LTX Multi] middle_images_json parse error: {_e}")

            inputs = ModelInputs(
                prompt               = prompt,
                neg_prompt           = negative_prompt,
                mode                 = mode,
                image                = init_image,
                last_image           = _flf_last_image,
                middle_images_paths  = _middle_images_paths,
                video_path           = video_path,
                audio_ref            = audio_ref,
                width       = x,
                height      = y,
                frames      = abs(duration),
                fps         = round(scene.render.fps / scene.render.fps_base, 3),
                steps       = scene.movie_num_inference_steps,
                guidance    = scene.movie_num_guidance,
                strength    = scene.image_power,
                seed        = seed,
            )

            print("=" * 60)
            print("[VIDEO GENERATE] ModelInputs summary:")
            print(f"  prompt      = {inputs.prompt!r}")
            print(f"  mode        = {inputs.mode!r}")
            print(f"  image       = {'<PIL Image>' if inputs.image is not None else None}")
            print(f"  video_path  = {inputs.video_path!r}")
            print(f"  audio_ref   = {inputs.audio_ref!r}")
            print(f"  width/height= {inputs.width} x {inputs.height}")
            print(f"  frames      = {inputs.frames}")
            print(f"  seed        = {inputs.seed}")
            print("=" * 60)

            t_gen = bench_print(f"[{plugin.MODEL_ID}] generate start")
            try:
                dst_path = plugin.generate(pipe_obj, inputs, scene, addon_prefs)
                bench_print(f"[{plugin.MODEL_ID}] generate done", t_gen)
            except Exception as e:
                print(f"Video generation error: {e}")
                import traceback
                traceback.print_exc()
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}

            if not dst_path or not os.path.isfile(dst_path):
                print("No resulting video file found.")
                return {"CANCELLED"}

            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == "SEQUENCE_EDITOR":
                        from bpy import context as _ctx
                        with _ctx.temp_override(window=window, area=area):
                            _strip_name = str(seed) + "_" + prompt
                            new_movie_strip = scene.sequence_editor.strips.new_movie(
                                name=_strip_name,
                                filepath=dst_path,
                                frame_start=start_frame,
                                channel=empty_channel,
                                fit_method="FIT",
                            )
                            set_ai_metadata_from_dict(new_movie_strip, {
                                "model":    movie_model_card,
                                "prompt":   inputs.prompt,
                                "steps":    inputs.steps,
                                "guidance": inputs.guidance,
                                "seed":     seed,
                                "width":    inputs.width,
                                "height":   inputs.height,
                                "frames":   inputs.frames,
                                "mode":     inputs.mode,
                            })
                            scene.sequence_editor.active_strip = new_movie_strip
                            if i > 0:
                                scene.frame_current = new_movie_strip.frame_final_start
                            if empty_channel > 1:
                                snd_ch = max(1, empty_channel - 1)
                                snd = scene.sequence_editor.strips.new_sound(
                                    name=_strip_name,
                                    filepath=dst_path,
                                    channel=snd_ch,
                                    frame_start=start_frame,
                                )
                                snd.select = False
                        break
                else:
                    continue
                break

            print_elapsed_time(start_time)

        if old_duration == -1 and input_mode == "input_strips":
            scene.generate_movie_frames = -1

        if should_unload:
            print("Unloading video models from memory...")
            release_model_cache(cache)

        scene.movie_path = ""
        scene.frame_current = current_frame
        return {"FINISHED"}

class SEQUENCER_OT_generate_audio(Operator):
    """Generate Audio"""

    bl_idname = "sequencer.generate_audio"
    bl_label = "Prompt"
    bl_description = "Convert text to audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):  # noqa: C901
        global _pallaidium_audio_model_cache
        import os

        scene = context.scene
        if not scene.sequence_editor:
            scene.sequence_editor_create()
        preferences = context.preferences
        addon_prefs = preferences.addons[ADDON_ID].preferences

        from ..models import get_plugin
        from ..models.base import ModelInputs

        plugin = get_plugin(addon_prefs.audio_model_card)
        if plugin is None:
            self.report({"ERROR"}, f"Audio plugin not found: {addon_prefs.audio_model_card!r}")
            return {"CANCELLED"}
        if not plugin.is_available():
            self.report({"INFO"}, "Dependencies need to be installed in the add-on preferences.")
            return {"CANCELLED"}

        prompt        = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        strip         = scene.sequence_editor.active_strip
        input_mode    = scene.input_strips
        # Use active TEXT strip content as the text-to-speak prompt when in input_strip mode.
        if input_mode == "input_strips" and strip and strip.type == "TEXT" and strip.text:
            _text = strip.text
            _raw = scene.generate_movie_prompt
            if not (_raw == _text or _raw.startswith(_text + ", ")):
                prompt = _text + (", " + _raw if _raw else "")

        # --- load / cache ---
        should_load   = context.scene.get("ai_load_state", True)
        should_unload = context.scene.get("ai_unload_state", True)

        cache = _pallaidium_audio_model_cache
        pipe_obj = {
            "pipe":              cache["pipe"],
            "model":             cache["model"],
            "vocoder":           cache["vocoder"],
            "feature_extractor": cache["feature_extractor"],
        }

        has_model = cache["pipe"] is not None or cache["model"] is not None
        if not has_model and not should_load:
            print("Audio model cache empty. Forcing load.")
            should_load = True
        if cache["last_model_card"] != addon_prefs.audio_model_card:
            print("Audio model changed. Forcing load.")
            should_load = True

        if addon_prefs.display_console:
            show_system_console(True)
            set_system_console_topmost(True)

        if should_load:
            clear_cuda_cache()
            apply_hf_env(addon_prefs)
            t_load = bench_print(f"[{plugin.MODEL_ID}] load start")
            pipe_obj = plugin.load(addon_prefs, scene)
            bench_print(f"[{plugin.MODEL_ID}] load done", t_load)
            cache["pipe"]              = pipe_obj.get("pipe")
            cache["model"]             = pipe_obj.get("model")
            cache["vocoder"]           = pipe_obj.get("vocoder")
            cache["feature_extractor"] = pipe_obj.get("feature_extractor")
            cache["last_model_card"]   = addon_prefs.audio_model_card

        # --- duration (before batch loop) ---
        strips = context.selected_strips
        if strip in strips:
            duration = scene.audio_length_in_f = strip.frame_final_duration + 1
        else:
            duration = scene.audio_length_in_f
        audio_length_in_s = duration / (scene.render.fps / scene.render.fps_base)
        old_duration = duration

        # --- batch loop ---
        for i in range(scene.movie_num_batch):
            import random
            start_time = timer()
            strip = scene.sequence_editor.active_strip

            if strip and input_mode == "input_strips" and duration == -1:
                duration = scene.audio_length_in_f = strip.frame_final_duration + 1
                audio_length_in_s = duration / (scene.render.fps / scene.render.fps_base)
            else:
                duration = scene.audio_length_in_f
                audio_length_in_s = duration / (scene.render.fps / scene.render.fps_base)

            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame   = (
                    scene.sequence_editor.active_strip.frame_final_start
                    + scene.sequence_editor.active_strip.frame_final_duration
                )
                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
            else:
                if input_mode != "input_strips":
                    empty_channel = find_first_empty_channel(
                        scene.frame_current,
                        (scene.movie_num_batch * (len(prompt) * 4)) + scene.frame_current,
                    )
                else:
                    empty_channel = find_first_empty_channel(
                        strip.frame_final_start if strip else scene.frame_current,
                        duration + scene.frame_current,
                    )
                start_frame = scene.frame_current

            # seed
            seed = context.scene.movie_num_seed
            seed = seed if not context.scene.movie_use_random else random.randint(-2147483647, 2147483647)
            context.scene.movie_num_seed = seed

            # VC mode: input strip is SOUND -> voice clone
            is_vc = (
                input_mode == "input_strips"
                and strip is not None
                and strip.type == "SOUND"
            )
            if is_vc:
                _trimmed_ref = render_strip_to_wav(context, strip)
                audio_ref = _trimmed_ref if _trimmed_ref else bpy.path.abspath(strip.sound.filepath)
            elif (
                input_mode == "input_strips"
                and strip is not None
                and strip.type == "META"
                and any(c.type == "SOUND" for c in strip.strips)
            ):
                _trimmed_ref = render_meta_audio_to_path(context, strip)
                audio_ref = _trimmed_ref or None
            elif getattr(scene, "ref_audio_path", None):
                audio_ref = bpy.path.abspath(scene.ref_audio_path)
            else:
                audio_ref = None

            text_ref   = bpy.path.abspath(scene.ref_text) if getattr(scene, "ref_text", None) else ""
            if input_mode == "input_strips" and strip is not None and strip.type == "MOVIE":
                video_path = bpy.path.abspath(strip.filepath)
            else:
                video_path = scene.movie_path if getattr(scene, "movie_path", None) else None

            inputs = ModelInputs(
                prompt         = prompt,
                neg_prompt     = negative_prompt,
                audio_ref      = audio_ref,
                text_ref       = text_ref,
                video_path     = video_path,
                steps          = scene.movie_num_inference_steps,
                guidance       = scene.movie_num_guidance,
                seed           = seed,
                audio_length   = audio_length_in_s,
                speed          = getattr(scene, "audio_speed_tts",   1.0),
                exaggeration   = getattr(scene, "chat_exaggeration",  0.5),
                pace           = getattr(scene, "chat_pace",          0.5),
                temperature    = getattr(scene, "chat_temperature",   0.8),
                remove_silence = getattr(scene, "remove_silence",     False),
                is_voice_clone = is_vc,
                bpm            = getattr(scene, "music_bpm",          0),
                lyrics         = getattr(scene, "music_lyrics",       ""),
                key_scale      = getattr(scene, "music_key_scale",    ""),
                time_signature = getattr(scene, "music_time_signature", ""),
            )

            t_gen = bench_print(f"[{plugin.MODEL_ID}] generate start")
            try:
                filename = plugin.generate(pipe_obj, inputs, scene, addon_prefs)
            except Exception as e:
                print(f"Audio generation error: {e}")
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            bench_print(f"[{plugin.MODEL_ID}] generate done", t_gen)

            if filename and os.path.isfile(filename):
                new_strip = scene.sequence_editor.strips.new_sound(
                    name=prompt,
                    filepath=filename,
                    channel=empty_channel,
                    frame_start=start_frame,
                )
                scene.sequence_editor.active_strip = new_strip
                if i > 0:
                    scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            else:
                print("No resulting audio file found!")

            print_elapsed_time(start_time)
            if old_duration == -1 and input_mode == "input_strips":
                scene.audio_length_in_f = scene.generate_movie_frames = -1

        # --- unload ---
        if should_unload:
            print("Unloading audio model...")
            release_model_cache(cache)

        return {"FINISHED"}

class SEQUENCER_OT_generate_image(Operator):
    """Generate Image"""

    # NOTE: Legacy immediate "Generate" path. Normal inference is now ALWAYS routed
    # through the render queue (operators/queue_ops.py), which runs plugins on a
    # background worker and never calls this operator. Input handling for queued jobs
    # lives in queue_ops.py — edit there, not here.
    bl_idname = "sequencer.generate_image"
    bl_label = "Prompt"
    bl_description = "Convert text to image"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):  # noqa: C901
        global _pallaidium_model_cache
        import os, random

        scene = context.scene
        seq_editor = scene.sequence_editor
        preferences = context.preferences
        addon_prefs = preferences.addons[ADDON_ID].preferences
        local_files_only = addon_prefs.local_files_only
        image_model_card = addon_prefs.image_model_card
        strips = context.selected_strips

        if addon_prefs.display_console:
            show_system_console(True)
            set_system_console_topmost(True)

        if not seq_editor:
            scene.sequence_editor_create()

        # Dependency check
        try:
            import torch
            from PIL import Image
        except ModuleNotFoundError as e:
            self.report({"INFO"}, "Dependencies needs to be installed in the add-on preferences: " + str(e.name))
            return {"CANCELLED"}

        # Plugin lookup
        from ..models import get_plugin
        from ..models.base import ModelInputs

        plugin = get_plugin(image_model_card)
        if plugin is None:
            self.report({"ERROR"}, "No image plugin for model: " + str(image_model_card))
            return {"CANCELLED"}
        if not plugin.is_available():
            self.report({"ERROR"}, "Missing dependencies for: " + str(image_model_card))
            return {"CANCELLED"}

        # Scene / prompt setup
        current_frame = scene.frame_current
        input_mode = scene.input_strips
        type_select = scene.generatorai_typeselect
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        negative_prompt = (
            scene.generate_movie_negative_prompt
            + ", "
            + style_prompt(scene.generate_movie_prompt)[1]
        )
        # When running in input_strip mode and the active strip is a TEXT strip, use its
        # content as the prompt. The startswith guard prevents doubling when
        # strip_to_generatorAI already incorporated the text before calling this operator.
        if input_mode == "input_strips":
            _active = scene.sequence_editor.active_strip
            if _active and _active.type == "TEXT" and _active.text:
                _text = _active.text
                _raw = scene.generate_movie_prompt
                if not (_raw == _text or _raw.startswith(_text + ", ")):
                    _raw = _text + (", " + _raw if _raw else "")
                    prompt = style_prompt(_raw)[0]
                    negative_prompt = (
                        scene.generate_movie_negative_prompt
                        + ", "
                        + style_prompt(_raw)[1]
                    )
        x = scene.generate_movie_x = closest_divisible_32(scene.generate_movie_x)
        y = scene.generate_movie_y = closest_divisible_32(scene.generate_movie_y)
        duration = scene.generate_movie_frames
        steps = scene.movie_num_inference_steps
        guidance = scene.movie_num_guidance
        active_strip = scene.sequence_editor.active_strip

        lora_files = scene.lora_files
        enabled_items = [item for item in lora_files if item.enabled]

        do_inpaint = (
            bool(find_strip_by_name(scene, scene.inpaint_selected_strip))
            and type_select == "image"
            and plugin.supports_inpaint
        )
        do_convert = (
            bool(scene.image_path or scene.movie_path)
            and plugin.supports_img2img
            and not do_inpaint
        )
        do_refine = getattr(scene, "refine_sd", False) and not do_convert
        mode = "inpaint" if do_inpaint else ("img2img" if do_convert else "txt2img")

        _sel_dbg = [(s.name, s.type) for s in (strips or [])]
        print("[generate_image][dbg] ===== SCENE-INPUT DIAG (build: scene-strip-v3) =====")
        print(f"[generate_image][dbg] input_mode={input_mode!r} type_select={type_select!r} "
              f"selected_strips={_sel_dbg}")
        print(f"[generate_image][dbg] scene.image_path={getattr(scene, 'image_path', '')!r}")
        print(f"[generate_image][dbg] scene.movie_path={getattr(scene, 'movie_path', '')!r}")
        print(f"[generate_image][dbg] supports_img2img={plugin.supports_img2img} "
              f"requires_input_strip={getattr(plugin, 'requires_input_strip', False)}")
        print("do_inpaint: " + str(do_inpaint))
        print("do_convert: " + str(do_convert))
        print("do_refine:  " + str(do_refine))
        print(f"[generate_image][dbg] → mode={mode}")
        _diag(f"generate_image: model={image_model_card} input_mode={input_mode} "
              f"type_select={type_select} selected={_sel_dbg} "
              f"image_path={getattr(scene, 'image_path', '')!r} "
              f"movie_path={getattr(scene, 'movie_path', '')!r} "
              f"supports_img2img={plugin.supports_img2img} → mode={mode}")

        # Validate strip selection when input strip is required
        if do_inpaint or do_convert or plugin.requires_input_strip:
            if not strips:
                self.report({"INFO"}, "Select strip(s) for processing.")
                return {"CANCELLED"}
            for strip in strips:
                if strip.type in {"MOVIE", "IMAGE", "TEXT", "SCENE"}:
                    break
            else:
                self.report({"INFO"}, "None of the selected strips are movie, image, text or scene types.")
                return {"CANCELLED"}

        # --- Cache / load state ---
        should_load = context.scene.get("ai_load_state", True)
        should_unload = context.scene.get("ai_unload_state", True)

        if (_pallaidium_model_cache["pipe"] is None and
                _pallaidium_model_cache["converter"] is None and not should_load):
            print("Model cache missing. Forcing load.")
            should_load = True

        if (_pallaidium_model_cache["last_model_card"] != image_model_card and
                _pallaidium_model_cache["last_model_card"] is not None):
            print("Model card changed. Releasing old model.")
            release_model_cache(_pallaidium_model_cache)
            should_load = True

        if should_load:
            clear_cuda_cache()
            apply_hf_env(addon_prefs)
            print("Loading: " + image_model_card + " (" + mode + ")")
            t_load = bench_print(f"[{plugin.MODEL_ID}] load start")
            try:
                pipe_obj = plugin.load(
                    addon_prefs, scene,
                    mode=mode,
                    enabled_items=enabled_items,
                    use_lcm=getattr(scene, "use_lcm", False),
                    use_refine=do_refine,
                    ip_adapter_face_folder=getattr(scene, "ip_adapter_face_folder", ""),
                    ip_adapter_style_folder=getattr(scene, "ip_adapter_style_folder", ""),
                    local_files_only=local_files_only,
                )
            except OSError as _load_err:
                if local_files_only:
                    self.report({"ERROR"}, (
                        "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
                    ))
                else:
                    self.report({"ERROR"}, f"Failed to load model: {_load_err}")
                return {"CANCELLED"}
            bench_print(f"[{plugin.MODEL_ID}] load done", t_load)
            _pallaidium_model_cache.update(pipe_obj)
            _pallaidium_model_cache["last_model_card"] = image_model_card

        # --- Pre-load init image and mask ---
        init_image = None
        mask_image = None

        if do_inpaint:
            mask_strip = find_strip_by_name(scene, scene.inpaint_selected_strip)
            if not mask_strip:
                self.report({"INFO"}, "Inpaint mask strip not found.")
                return {"CANCELLED"}
            mask_image = load_strip_as_pil(mask_strip, context)
            if not mask_image:
                self.report({"INFO"}, "Failed to load mask image.")
                return {"CANCELLED"}
            mask_image = mask_image.resize((x, y))
            _pipe = _pallaidium_model_cache.get("pipe")
            if _pipe and hasattr(_pipe, "mask_processor"):
                mask_image = _pipe.mask_processor.blur(mask_image, blur_factor=33)

            if scene.image_path:
                init_image = load_first_frame(scene.image_path)
            elif scene.movie_path:
                init_image = load_first_frame(scene.movie_path)
            if not init_image and active_strip and active_strip.type in ("IMAGE", "MOVIE"):
                if active_strip.name != scene.inpaint_selected_strip:
                    strip_path = get_strip_path(active_strip)
                    if strip_path:
                        init_image = load_first_frame(strip_path)
            if not init_image:
                self.report({"INFO"}, "Failed to load init image for inpaint. Select a source image strip.")
                return {"CANCELLED"}
            init_image = init_image.resize((x, y))

        elif do_convert:
            if scene.movie_path:
                init_image = load_first_frame(scene.movie_path)
            elif scene.image_path:
                init_image = load_first_frame(scene.image_path)
            if init_image:
                from PIL import ImageOps as _ImageOps
                init_image = _ImageOps.fit(init_image, (x, y), Image.LANCZOS)
            else:
                self.report({"ERROR"}, "img2img: could not load source image from strip. Check that the strip has a valid file.")
                return {"CANCELLED"}

        # ===================== BATCH GENERATION LOOP =====================
        for i in range(scene.movie_num_batch):
            start_time = timer()

            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = (
                    scene.sequence_editor.active_strip.frame_final_start
                    + scene.sequence_editor.active_strip.frame_final_duration
                )
                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * duration) + scene.frame_current,
                )
                start_frame = scene.frame_current

            # Seed
            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(-2147483647, 2147483647)
            )
            context.scene.movie_num_seed = seed
            print("Seed: " + str(seed))

            # Build plugin inputs
            inputs = ModelInputs(
                prompt=prompt,
                neg_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                strength=scene.image_power,
                height=y,
                width=x,
                seed=seed,
                mode=mode,
                image=init_image,
                inpaint_mask=mask_image,
            )

            # Generate via plugin
            t_gen = bench_print(f"[{plugin.MODEL_ID}] generate start")
            try:
                image = plugin.generate(_pallaidium_model_cache, inputs, scene, addon_prefs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            bench_print(f"[{plugin.MODEL_ID}] generate done", t_gen)

            # Refiner post-processing
            refiner = _pallaidium_model_cache.get("refiner")
            if refiner is not None and do_refine:
                print("Refine: Image")
                gen = (
                    torch.Generator("cuda").manual_seed(seed)
                    if torch.cuda.is_available() and seed != 0 else None
                )
                image = refiner(
                    prompt=prompt,
                    image=image,
                    strength=max(1.0 - scene.image_power, 0.1),
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=max(guidance, 1.1),
                    generator=gen,
                ).images[0]

            # ADetailer
            if scene.adetailer:
                try:
                    from asdff.base import AdPipelineBase
                    from huggingface_hub import hf_hub_download
                    from diffusers import StableDiffusionXLPipeline, AutoencoderKL

                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=torch.float16, local_files_only=local_files_only,
                    )
                    ad_base = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        vae=vae, variant="fp16", torch_dtype=torch.float16,
                        local_files_only=local_files_only,
                    )
                    if gfx_device == "mps":
                        ad_base.to("mps")
                    elif low_vram():
                        ad_base.enable_model_cpu_offload()
                    else:
                        ad_base.to(gfx_device)
                    model_path = hf_hub_download(
                        "Bingsu/adetailer", "face_yolov8n.pt",
                        local_dir="asdff/yolo_models", local_dir_use_symlinks=False,
                    )
                    ad_pipe = AdPipelineBase(**ad_base.components)
                    result = ad_pipe(
                        common={
                            "prompt": prompt + ", face, (8k, RAW photo, best quality, masterpiece:1.2)",
                            "n_prompt": "nsfw, blurry, disfigured",
                            "num_inference_steps": int(steps),
                            "target_size": (x, y),
                        },
                        inpaint_only={"strength": 0.4},
                        images=image,
                        mask_dilation=4,
                        mask_blur=4,
                        mask_padding=32,
                        model_path=model_path,
                    )
                    image = result.images[0]
                except Exception as e:
                    print("ADetailer skipped: " + str(e))

            # AuraSR
            if scene.aurasr:
                if do_convert and init_image is not None:
                    image = init_image
                if image:
                    from aura_sr import AuraSR
                    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
                    image = aura_sr.upscale_4x_overlapped(image)

            # Save and create strip
            filename = clean_filename(str(seed) + "_" + context.scene.generate_movie_prompt)
            out_path = solve_path(filename + ".png")
            image.save(out_path)
            bpy.types.Scene.genai_out_path = out_path

            if input_mode == "input_strips":
                old_strip = active_strip

            if os.path.isfile(out_path):
                new_strip = scene.sequence_editor.strips.new_image(
                    name=str(seed) + "_" + context.scene.generate_movie_prompt,
                    frame_start=start_frame,
                    filepath=out_path,
                    channel=empty_channel,
                    fit_method="FIT",
                )
                if scene.generate_movie_frames == -1 and input_mode == "input_strips":
                    new_strip.frame_final_duration = old_strip.frame_final_duration
                else:
                    new_strip.frame_final_duration = abs(scene.generate_movie_frames)

                set_ai_metadata_from_dict(new_strip, {
                    "model":           image_model_card,
                    "prompt":          inputs.prompt,
                    "negative_prompt": inputs.neg_prompt,
                    "steps":           inputs.steps,
                    "guidance":        inputs.guidance,
                    "seed":            seed,
                    "width":           inputs.width,
                    "height":          inputs.height,
                    "mode":            inputs.mode,
                })

                scene.sequence_editor.active_strip = new_strip
                if i > 0:
                    scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
                new_strip.use_proxy = True
            else:
                print("No resulting file found.")

            import gc
            gc.collect()

            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == "SEQUENCE_EDITOR":
                        from bpy import context as bpy_ctx
                        with bpy_ctx.temp_override(window=window, area=area):
                            if i > 0:
                                scene.frame_current = (
                                    scene.sequence_editor.active_strip.frame_final_start
                                )
                            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                            break

            print_elapsed_time(start_time)

        # Unload
        if should_unload:
            print("Unloading image model from memory...")
            release_model_cache(_pallaidium_model_cache)

        scene.movie_num_guidance = guidance
        scene.frame_current = current_frame
        return {"FINISHED"}

class SEQUENCER_OT_generate_text(Operator):
    """Generate Text"""

    bl_idname = "sequencer.generate_text"
    bl_label = "Prompt"
    bl_description = "Generate texts from strips"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        global _pallaidium_text_model_cache
        import os

        scene = context.scene
        input_mode = scene.input_strips
        seq_editor = scene.sequence_editor
        preferences = context.preferences
        addon_prefs = preferences.addons[ADDON_ID].preferences
        guidance = scene.movie_num_guidance
        current_frame = scene.frame_current
        prompt = scene.generate_movie_prompt
        x = scene.generate_movie_x = closest_divisible_32(scene.generate_movie_x)
        y = scene.generate_movie_y = closest_divisible_32(scene.generate_movie_y)
        active_strip = context.scene.sequence_editor.active_strip
        # Use active TEXT strip content as prompt when in input_strip mode.
        if input_mode == "input_strips" and active_strip and active_strip.type == "TEXT" and active_strip.text:
            _text = active_strip.text
            _raw = scene.generate_movie_prompt
            if not (_raw == _text or _raw.startswith(_text + ", ")):
                prompt = _text + (", " + _raw if _raw else "")

        if active_strip:
            old_duration = duration = active_strip.frame_final_duration
        else:
            old_duration = duration = 100

        render = bpy.context.scene.render
        fps = render.fps / render.fps_base
        if addon_prefs.display_console:
            show_system_console(True)
            set_system_console_topmost(True)

        if not seq_editor:
            scene.sequence_editor_create()

        # --- Get plugin ---
        from ..models import get_plugin
        from ..models.base import InputSpec, ModelInputs
        plugin = get_plugin(addon_prefs.text_model_card)
        if not plugin:
            self.report({"ERROR"}, f"No plugin registered for {addon_prefs.text_model_card}")
            return {"CANCELLED"}

        if not plugin.is_available():
            self.report({"INFO"}, "Dependencies need to be installed in the add-on preferences.")
            return {"CANCELLED"}

        # --- Cache management ---
        should_load   = context.scene.get("ai_load_state", True)
        should_unload = context.scene.get("ai_unload_state", True)

        if _pallaidium_text_model_cache["model"] is None and not should_load:
            should_load = True
        if _pallaidium_text_model_cache["last_model_card"] != addon_prefs.text_model_card:
            print("Text model card changed. Forcing load.")
            should_load = True

        if should_load:
            clear_cuda_cache()
            apply_hf_env(addon_prefs)
            t_load = bench_print(f"[{plugin.MODEL_ID}] load start")
            pipe_data = plugin.load(addon_prefs, scene, gfx_device=gfx_device)
            bench_print(f"[{plugin.MODEL_ID}] load done", t_load)
            _pallaidium_text_model_cache["model"]      = pipe_data.get("model")
            _pallaidium_text_model_cache["processor"]  = pipe_data.get("processor")
            _pallaidium_text_model_cache["tokenizer"]  = pipe_data.get("tokenizer")
            _pallaidium_text_model_cache["last_model_card"] = addon_prefs.text_model_card

        # --- Collect inputs ---
        inputs = ModelInputs()
        inputs.prompt = prompt
        inputs.width  = x
        inputs.height = y

        if InputSpec.IMAGE & plugin.INPUTS:
            init_image = None
            if scene.movie_path:
                init_image = load_first_frame(bpy.path.abspath(scene.movie_path))
            elif scene.image_path:
                init_image = load_first_frame(bpy.path.abspath(scene.image_path))

            if init_image:
                inputs.image = init_image.resize((x, y))
            else:
                print("No input image found. Cancelling.")
                return {"CANCELLED"}

        # --- Generate ---
        pipe_obj = {
            "model":     _pallaidium_text_model_cache["model"],
            "processor": _pallaidium_text_model_cache["processor"],
            "tokenizer": _pallaidium_text_model_cache["tokenizer"],
        }

        # Output strip placement (base position; batch copies are placed consecutively).
        if input_mode == "input_strips" and active_strip:
            start_frame  = int(getattr(active_strip, "left_handle", getattr(active_strip, "frame_final_start", 1)))
            strip_length = int(getattr(active_strip, "duration",    getattr(active_strip, "frame_final_duration", 100)))
        else:
            start_frame  = int(scene.frame_current)
            strip_length = 100

        # Batch loop — only stochastic plugins (supports_batch) gain from >1.
        batch = max(1, scene.movie_num_batch) if getattr(plugin, "supports_batch", True) else 1

        text = None
        for i in range(batch):
            t_gen = bench_print(f"[{plugin.MODEL_ID}] generate start")
            text = plugin.generate(pipe_obj, inputs, scene, addon_prefs)
            bench_print(f"[{plugin.MODEL_ID}] generate done", t_gen)

            # --- Create text strip ---
            frame_start   = start_frame + i * strip_length
            empty_channel = find_first_empty_channel(frame_start, frame_start + strip_length)

            if text:
                new_strip = scene.sequence_editor.strips.new_effect(
                    name=str(text),
                    type="TEXT",
                    frame_start=frame_start,
                    length=strip_length,
                    channel=empty_channel,
                )
                if hasattr(new_strip, "right_handle"):
                    new_strip.right_handle = frame_start + strip_length
                else:
                    new_strip.frame_final_end = frame_start + strip_length

                new_strip.text        = text
                new_strip.wrap_width  = 0.68
                new_strip.font_size   = 16
                new_strip.location[0] = 0.5
                new_strip.location[1] = 0.2
                new_strip.anchor_x    = "CENTER"
                new_strip.anchor_y    = "TOP"
                new_strip.alignment_x = "LEFT"
                new_strip.use_shadow  = True
                new_strip.use_box     = True
                new_strip.box_color   = (0, 0, 0, 0.7)
                scene.sequence_editor.active_strip = new_strip

        # --- Florence-2 → Mask Editor ---
        if (
            text
            and getattr(scene, "florence2_mode",         "CAPTION") == "IDEOGRAM4"
            and getattr(scene, "florence2_send_to_mask", False)
        ):
            source_path = bpy.path.abspath(scene.movie_path or scene.image_path or "")
            try:
                from .mask_florence2 import apply_florence_json_to_mask
                apply_florence_json_to_mask(text, source_path)
            except Exception as _mex:
                _dbg(f"[Florence2Mask] mask creation failed: {_mex}")

        # --- UI redraw ---
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "SEQUENCE_EDITOR":
                    from bpy import context as _ctx
                    with _ctx.temp_override(window=window, area=area):
                        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                    break

        scene.movie_num_guidance = guidance
        scene.frame_current = current_frame

        # --- Unload ---
        if should_unload:
            print("Unloading text model from memory...")
            _pallaidium_text_model_cache["model"]      = None
            _pallaidium_text_model_cache["processor"]  = None
            _pallaidium_text_model_cache["tokenizer"]  = None
            clear_cuda_cache()

        return {"FINISHED"}

class SEQUENCER_OT_strip_to_generatorAI(Operator):
    """Convert selected text strips to Generative AI with Smart Memory Management"""

    # NOTE: The immediate "Generate" path (this dispatcher + sequencer.generate_image/
    # movie/audio/text) is NO LONGER the entry point for normal use. All inference is
    # now initiated through the render QUEUE (OBJECT_OT_QueueAddOperator in
    # operators/queue_ops.py), which builds its own job snapshot and runs the plugin on
    # a background worker WITHOUT calling these operators. When changing input handling
    # (scene-strip rendering, img2img mode detection, reference images, …) edit the
    # queue path; changes here will not affect queued generations.
    bl_idname = "sequencer.text_to_generator"
    bl_label = "Pallaidium"
    bl_options = {"INTERNAL"}
    bl_description = "Adds selected strips as inputs to the Generative AI process"

    @classmethod
    def poll(cls, context):
        return context.sequencer_scene and context.scene.sequence_editor

    def execute(self, context):
        import os
        # --- Initialization ---
        scene = context.scene
        scene.movie_path = ""
        scene.image_path = ""
        scene.sound_path = ""
        preferences = context.preferences
        
        try:
            addon_prefs = preferences.addons[ADDON_ID].preferences
            play_sound = addon_prefs.playsound
            addon_prefs.playsound = False
        except:
            play_sound = False

        scene = context.scene
        sequencer = bpy.ops.sequencer
        strips = context.selected_strips
        active_strip = context.scene.sequence_editor.active_strip
        
        if not strips == context.selected_strips:
            active_strip.select = True
            
        # STORE BASE PROMPTS HERE - These are our "Clean" copies
        base_prompt = scene.generate_movie_prompt
        base_negative_prompt = scene.generate_movie_negative_prompt
        current_prompt_text = base_prompt
        current_negative_text = base_negative_prompt
        current_frame = scene.frame_current
        target_type = scene.generatorai_typeselect 
        seed = scene.movie_num_seed
        use_random = scene.movie_use_random
        temp_strip = None
        temp_strips = []
        current_temp_strip = None
        run_generation = False
        
        # --- Input Validation ---
        if not strips:
            self.report({"INFO"}, "Select strip(s) for processing.")
            return {"CANCELLED"}
        else:
            print("\nStrip input processing started...")
        
        valid_types = {"MOVIE", "IMAGE", "TEXT", "SCENE", "META", "SOUND"}
        for strip in strips:
            if strip.type in valid_types:
                break
        else:
            self.report({"INFO"}, "None of the selected strips are valid types.")
            return {"CANCELLED"}

        if target_type == "text":
            for strip in strips:
                if strip.type in {"MOVIE", "IMAGE", "TEXT", "SCENE", "META", "SOUND"}:
                    break
            else:
                self.report({"INFO"}, "None of the selected strips are possible to process to text.")
                return {"CANCELLED"}

        # --- Hardware Info (Optional) ---
        try:
            if gfx_device == "cuda":
                print(f"CUDA version: {torch.version.cuda}")
        except:
            pass

        # --- Main Processing Loop ---
        total_strips = len(strips)
        
        for count, strip in enumerate(strips):
            # 1. Selection Logic
            for dsel_strip in bpy.context.scene.sequence_editor.strips:
                dsel_strip.select = False
            strip.select = True
            context.scene.sequence_editor.active_strip = strip

            # 2. Smart Memory Management Logic
            is_first_strip = (count == 0)
            is_last_strip = (count == total_strips - 1)
            current_strip_type = strip.type
            
            prev_strip_type = strips[count-1].type if count > 0 else None
            type_has_changed = (prev_strip_type is not None and current_strip_type != prev_strip_type)

            next_strip_type = strips[count+1].type if count < total_strips - 1 else None
            next_type_is_different = (next_strip_type is not None and next_strip_type != current_strip_type)

            should_load_model = is_first_strip or type_has_changed
            should_unload_model = is_last_strip or next_type_is_different

            context.sequencer_scene["ai_load_state"] = should_load_model
            context.sequencer_scene["ai_unload_state"] = should_unload_model
            
            print(f"Processing {count+1}/{total_strips} [{strip.type}]. Load: {should_load_model}, Unload: {should_unload_model}")


            # 3A. Intermediate META Strip Handling
            _multi_input_video = target_type == "movie" and addon_prefs.movie_model_card in {
                "LTX-2 Multi-Input File", "LTX-2.3 Multi-Input File"
            }

            # Fast path: META strip for image plugin → composite PNG + TEXT as prompt
            if target_type == "image" and strip.type == "META" and not _multi_input_video:
                _decomp = decompose_meta(context, strip, target_type="image")
                current_prompt_text  = (
                    (_decomp["text"] + ", " + base_prompt) if _decomp["text"] and base_prompt
                    else _decomp["text"] or base_prompt
                )
                current_negative_text = base_negative_prompt
                if _decomp["image"]:
                    scene.image_path = _decomp["image"]
                    print(f"META image composite → {_decomp['image']!r}")
                run_generation = True

            elif _multi_input_video or (target_type == "image"): # and addon_prefs.image_model_card == "Tongyi-MAI/Z-Image-Turbo"
                if strip.type == "META":
                    meta_strip = strip
                    strips_array = strip.strips
                else:
                    meta_strip = None
                    strips_array = [strip]
                
                # For multi-input video plugins: pre-read any Text strip from the meta so
                # the prompt is already set before IMAGE/SOUND iterations fire generation.
                if _multi_input_video and meta_strip:
                    for _c in meta_strip.strips:
                        if _c.type == "TEXT" and _c.text:
                            _stripped = _c.text.strip()
                            current_prompt_text = (_stripped + ", " + base_prompt) if base_prompt else _stripped
                            print(f"LTX Multi: text strip prompt: {current_prompt_text!r}")
                            break

                current_temp_strip = None
                _multi_text_only = False  # tracks standalone TEXT strip in non-META context
                # Decompose trigger: first non-TEXT child.  If TEXT is strips[0] the
                # "child_strip == strips[0]" check inside the elif never fires because TEXT
                # takes the if-TEXT branch first → decompose silently skipped → audio lost.
                if _multi_input_video and meta_strip:
                    _meta_trigger_child = next(
                        (s for s in meta_strip.strips if s.type != "TEXT"),
                        meta_strip.strips[0] if meta_strip.strips else None,
                    )
                else:
                    _meta_trigger_child = None
                for child_strip in strips_array:
                    for dsel_strip in bpy.context.scene.sequence_editor.strips:
                        dsel_strip.select = False
                    child_strip.select = True
                    context.scene.sequence_editor.active_strip = child_strip

                    # 4. Processing Variables Setup
                    run_generation = False

                    # --- TEXT STRIP ---
                    if child_strip.type == "TEXT":
                        if _multi_input_video:
                            if meta_strip:
                                pass  # prompt already collected in pre-scan above
                            else:
                                # Standalone TEXT strip: use as prompt for text-to-video
                                if child_strip.text:
                                    _stripped = child_strip.text.strip()
                                    current_prompt_text = (_stripped + ", " + base_prompt) if base_prompt else _stripped
                                    print(f"LTX Multi: standalone text strip prompt: {current_prompt_text!r}")
                                _multi_text_only = True
                        elif meta_strip and meta_strip.type == 'META':
                            for child in meta_strip.strips:
                                if child.type == 'TEXT':
                                    print("Found text:", child.text)
                                    current_prompt_text = child.text + ", " + base_prompt
                            run_generation = True
                        else:
                            current_prompt_text = child_strip.text + ", " + base_prompt
                            run_generation = True

                    elif _multi_input_video and meta_strip:
                        # LTX multi-input with META: render all children once (triggered by
                        # first non-TEXT child), then skip subsequent children.
                        if child_strip is _meta_trigger_child:
                            # Clear stale paths from prior runs / other plugins
                            scene.movie_path = ""
                            scene.image_path = ""
                            scene.sound_path = ""
                            scene.ltx_middle_images_json = ""

                            print(f"[LTX Multi META] ── BEGIN META decompose ──────────────────────")
                            print(f"[LTX Multi META] meta='{meta_strip.name}'  children={len(list(meta_strip.strips))}")

                            # Render every child through the META compositor so trims/
                            # transforms/color corrections are respected.
                            _images_wf  = []   # (rendered_path, frame_start, child_strip)
                            _video_path = None
                            _audio_path = None
                            _text_parts = []

                            for _c in meta_strip.strips:
                                print(f"[LTX Multi META]   child type={_c.type!r} name={_c.name!r}"
                                      f"  frame_start={_c.frame_start}"
                                      f"  frame_final_start={_c.frame_final_start}"
                                      f"  frame_final_duration={_c.frame_final_duration}")

                                if _c.type == "TEXT":
                                    if _c.text and _c.text.strip():
                                        _text_parts.append(_c.text.strip())
                                        print(f"[LTX Multi META]     TEXT accepted: {_c.text.strip()!r}")
                                    else:
                                        print(f"[LTX Multi META]     TEXT empty, skipped")

                                elif _c.type == "IMAGE":
                                    print(f"[LTX Multi META]     IMAGE: rendering through META compositor...")
                                    _rpath = render_meta_child_to_path(context, meta_strip, _c, image_output=True)
                                    if _rpath:
                                        _images_wf.append((_rpath, _c.frame_start, _c))
                                        print(f"[LTX Multi META]     IMAGE OK → {_rpath!r}")
                                    else:
                                        _raw = get_strip_path(_c)
                                        if _raw and os.path.isfile(_raw):
                                            _images_wf.append((_raw, _c.frame_start, _c))
                                            print(f"[LTX Multi META]     IMAGE fallback (raw) → {_raw!r}")
                                        else:
                                            print(f"[LTX Multi META]     IMAGE WARN: no path found, skipped")

                                elif _c.type == "MOVIE":
                                    print(f"[LTX Multi META]     MOVIE: rendering through META compositor...")
                                    _rpath = render_meta_child_to_path(context, meta_strip, _c, image_output=False)
                                    if _rpath:
                                        _video_path = _rpath
                                        print(f"[LTX Multi META]     MOVIE OK → {_rpath!r}")
                                    else:
                                        _raw = get_strip_path(_c)
                                        if _raw and os.path.isfile(_raw):
                                            _video_path = _raw
                                            print(f"[LTX Multi META]     MOVIE fallback (raw) → {_raw!r}")
                                        else:
                                            print(f"[LTX Multi META]     MOVIE WARN: no path found, skipped")

                                elif _c.type == "SOUND":
                                    if _audio_path is None:
                                        # Mix ALL SOUND children in one pass.
                                        print(f"[LTX Multi META]     SOUND: mixing all META audio children...")
                                        _rpath = render_meta_audio_to_path(context, meta_strip)
                                        if _rpath:
                                            _audio_path = _rpath
                                            print(f"[LTX Multi META]     SOUND MIX OK → {_rpath!r}")
                                        else:
                                            try:
                                                _raw = bpy.path.abspath(_c.sound.filepath)
                                                if _raw and os.path.isfile(_raw):
                                                    _audio_path = _raw
                                                    print(f"[LTX Multi META]     SOUND fallback (raw) → {_raw!r}")
                                                else:
                                                    print(f"[LTX Multi META]     SOUND WARN: no path found, skipped")
                                            except Exception as _se:
                                                print(f"[LTX Multi META]     SOUND WARN: error getting path: {_se}")
                                    else:
                                        print(f"[LTX Multi META]     SOUND {_c.name!r}: already mixed, skipped")

                                else:
                                    print(f"[LTX Multi META]     type {_c.type!r} not handled, skipped")

                            # Apply text prompt
                            if _text_parts:
                                _decomp_text = ", ".join(_text_parts)
                                if not scene.generate_movie_prompt.startswith(_decomp_text):
                                    current_prompt_text = (
                                        (_decomp_text + ", " + base_prompt) if base_prompt
                                        else _decomp_text
                                    )
                                    print(f"[LTX Multi META] prompt from TEXT: {current_prompt_text!r}")
                            else:
                                print(f"[LTX Multi META] no TEXT strip, using base_prompt: {base_prompt!r}")

                            print(f"[LTX Multi META] collected images_wf={[(p, fs) for p, fs, _ in _images_wf]}")
                            print(f"[LTX Multi META] collected video_path={_video_path!r}")
                            print(f"[LTX Multi META] collected audio_path={_audio_path!r}")

                            # ── Mode detection ────────────────────────────────────────────
                            # Mode MA: 3+ images with no video → multi-anchor
                            _multi_anchor_mode = not _video_path and len(_images_wf) >= 3
                            # Mode A: exactly 2 images with different frame_start, no video
                            _flf_mode = (
                                not _video_path
                                and len(_images_wf) == 2
                                and _images_wf[0][1] != _images_wf[1][1]
                            )
                            # Mode B: 1 image whose frame_start > every other media child's frame_start.
                            # Exclude TEXT strips — they are prompt anchors, not temporal anchors.
                            _other_starts = (
                                [_oc.frame_start for _oc in meta_strip.strips
                                 if _oc is not _images_wf[0][2] and _oc.type != "TEXT"]
                                if len(_images_wf) == 1 else []
                            )
                            _lfo_mode = (
                                not _video_path
                                and len(_images_wf) == 1
                                and bool(_other_starts)
                                and _images_wf[0][1] > max(_other_starts)
                            )

                            print(f"[LTX Multi META] mode: MULTI_ANCHOR={_multi_anchor_mode}  FLF={_flf_mode}  LFO={_lfo_mode}")

                            if _multi_anchor_mode:
                                import json as _json
                                _sorted_wf = sorted(_images_wf, key=lambda x: x[1])
                                scene.movie_path = _sorted_wf[0][0]   # first image → start anchor
                                scene.image_path = _sorted_wf[-1][0]  # last image → end anchor
                                _meta_fs  = meta_strip.frame_final_start
                                _meta_dur = max(1, meta_strip.frame_final_duration)
                                _middle = []
                                for _path, _fstart, _mstrip in _sorted_wf[1:-1]:
                                    _frac = (_mstrip.frame_final_start - _meta_fs) / _meta_dur
                                    _frac = max(0.001, min(0.999, _frac))
                                    _middle.append([_path, _frac])
                                scene.ltx_middle_images_json = _json.dumps(_middle)
                                print(f"[LTX Multi META] MULTI-ANCHOR → first={scene.movie_path!r}")
                                print(f"[LTX Multi META] MULTI-ANCHOR → last ={scene.image_path!r}")
                                print(f"[LTX Multi META] MULTI-ANCHOR → middle={_middle}")

                            elif _flf_mode:
                                _sorted_wf = sorted(_images_wf, key=lambda x: x[1])
                                scene.movie_path = _sorted_wf[0][0]   # lower frame_start → first frame
                                scene.image_path = _sorted_wf[1][0]   # higher frame_start → last frame
                                print(f"[LTX Multi META] FLF → first={scene.movie_path!r}")
                                print(f"[LTX Multi META] FLF → last ={scene.image_path!r}")

                            elif _lfo_mode:
                                scene.movie_path = ""
                                scene.image_path = _images_wf[0][0]
                                print(f"[LTX Multi META] LFO → last ={scene.image_path!r}")

                            elif _video_path:
                                scene.movie_path = _video_path
                                if _images_wf:
                                    scene.image_path = _images_wf[0][0]
                                    print(f"[LTX Multi META] video+supplement → movie={scene.movie_path!r}  image={scene.image_path!r}")
                                else:
                                    print(f"[LTX Multi META] video only → movie={scene.movie_path!r}")

                            elif _images_wf:
                                scene.movie_path = _images_wf[0][0]
                                print(f"[LTX Multi META] single image → movie={scene.movie_path!r}")

                            else:
                                print(f"[LTX Multi META] WARN: no video or image found in meta strip")

                            # Audio — always assign to clear stale voice-plugin paths
                            scene.sound_path = _audio_path or ""
                            if scene.sound_path:
                                print(f"[LTX Multi META] audio → {scene.sound_path!r}")
                            else:
                                print(f"[LTX Multi META] no audio (sound_path cleared)")

                            print(f"[LTX Multi META] ── END META decompose ────────────────────────")
                            print(f"[LTX Multi META] scene.movie_path={scene.movie_path!r}")
                            print(f"[LTX Multi META] scene.image_path={scene.image_path!r}")
                            print(f"[LTX Multi META] scene.sound_path={scene.sound_path!r}")

                    else:
                        # Standard render pipeline for image generation or non-meta strips
                        print(f"[dispatcher][dbg] standard-render branch: child='{child_strip.name}' "
                              f"type={child_strip.type} target_type={target_type}")
                        current_temp_strip = get_render_strip(self, context, child_strip, meta_strip=meta_strip)
                        print("Adding: " + str(current_temp_strip))
                        print(f"[dispatcher][dbg] get_render_strip → "
                              f"{(current_temp_strip.name, current_temp_strip.type) if current_temp_strip else None}")
                        if current_temp_strip:
                            temp_strips.append(current_temp_strip)

                        # --- IMAGE / MOVIE / SOUND rendered strip ---
                        if current_temp_strip and current_temp_strip.type in {"IMAGE", "MOVIE", "SOUND"}:
                            if current_temp_strip.type == "IMAGE":
                                strip_dirname = os.path.dirname(current_temp_strip.directory)
                                file_path = bpy.path.abspath(os.path.join(strip_dirname, current_temp_strip.elements[0].filename))
                                scene.movie_path = file_path
                            elif current_temp_strip.type == "MOVIE":
                                file_path = bpy.path.abspath(current_temp_strip.filepath)
                                scene.movie_path = file_path
                            elif current_temp_strip.type == "SOUND" and target_type == "movie":
                                file_path = bpy.path.abspath(current_temp_strip.sound.filepath)
                                scene.sound_path = file_path
                            current_temp_strip = None
                            run_generation = True

                    print(scene.movie_path)
                    print(scene.image_path)
                    print(scene.sound_path)

                print(f"Prompt: {current_prompt_text}")

                # For multi-input video: guarantee generation fires if any media path was
                # collected (even if TEXT was the last child) OR if a standalone TEXT strip
                # was the sole input (prompt-only / text-to-video mode).
                # Use scene.movie_path (instance) not bpy.types.Scene.movie_path (always-truthy class descriptor).
                if _multi_input_video and (scene.movie_path or scene.sound_path or scene.image_path or _multi_text_only):
                    run_generation = True

            # 3B. Intermediate Strip Handling
            elif strip.type in {"SCENE", "MOVIE", "META", "SOUND", "TEXT", "IMAGE"}:
                _orig_strip_3b = strip  # preserved before get_render_strip reassigns it

                if (target_type == "image" or target_type == "text") and strip.type not in {"TEXT", "IMAGE"}:
                    trim_frame = find_overlapping_frame(strip, current_frame)
                    if trim_frame and len(strips) == 1:
                        bpy.ops.sequencer.duplicate_move(
                            SEQUENCER_OT_duplicate={},
                            TRANSFORM_OT_seq_slide={"value": (0, 1), "use_restore_handle_selection": False, "snap": False}
                        )
                        intermediate_strip = bpy.context.selected_strips[0]
                        intermediate_strip.frame_start = strip.frame_start
                        intermediate_strip.frame_offset_start = int(trim_frame)
                        intermediate_strip.frame_final_duration = 1
                        temp_strip = strip = get_render_strip(self, context, intermediate_strip)
                        if intermediate_strip: delete_strip(intermediate_strip)

                    elif target_type == "text":
                        bpy.ops.sequencer.copy()
                        bpy.ops.sequencer.paste(keep_offset=True)
                        intermediate_strip = bpy.context.selected_strips[0]
                        intermediate_strip.frame_start = strip.frame_start
                        intermediate_strip.frame_final_duration = strip.frame_final_duration
                        temp_strip = strip = get_render_strip(self, context, intermediate_strip)
                        if intermediate_strip: delete_strip(intermediate_strip)
                    else:
                        temp_strip = strip = get_render_strip(self, context, strip)
                elif strip.type == "SOUND" and target_type == "movie":
                    # Render a properly-trimmed WAV before get_render_strip can fall
                    # back to the potentially-full-length mixdown output.
                    _trimmed_wav = render_strip_to_wav(context, strip)
                    if _trimmed_wav:
                        scene.sound_path = _trimmed_wav
                        print(f"[3B SOUND] trimmed WAV → {_trimmed_wav!r}")
                    temp_strip = strip = get_render_strip(self, context, strip)
                elif strip.type not in {"TEXT", "IMAGE"}:
                    temp_strip = strip = get_render_strip(self, context, strip)

                # META with SOUND children feeding a video model: render the full
                # mixed audio of the META duration once, regardless of child count.
                if (target_type == "movie"
                        and _orig_strip_3b.type == "META"
                        and not scene.sound_path
                        and any(c.type == "SOUND" for c in _orig_strip_3b.strips)):
                    _meta_wav = render_meta_audio_to_path(context, _orig_strip_3b)
                    if _meta_wav:
                        scene.sound_path = _meta_wav
                        print(f"[3B META] audio mix → {_meta_wav!r}")

                if strip is None:
                    continue

                # 4. Processing Variables Setup
                # We calculate specific prompts into these variables, then apply them
                run_generation = False
                current_prompt_text = base_prompt
                current_negative_text = base_negative_prompt

                # --- TEXT STRIP ---
                if strip.type == "TEXT":
                    if strip.text:
                        # Combine Strip Text + Base Prompt
                        current_prompt_text = strip.text + ", " + base_prompt
                        run_generation = True

                # --- SOUND STRIP ---
                if strip.type == "SOUND":
                    if strip.sound:
                        # Sound usually uses just the base prompt, or you can add logic here
                        current_prompt_text = base_prompt 
                        run_generation = True

                # --- IMAGE / MOVIE STRIP ---
                if strip.type == "IMAGE" or strip.type == "MOVIE" or strip.type == "SOUND":
                    # Set path on the scene instance so SEQUENCER_OT_generate_image reads it.
                    # For image/text target: render IMAGE and MOVIE through VSE so trims,
                    # crops, and transforms are respected before passing to the plugin.
                    if strip.type == "IMAGE" and target_type == "image":
                        rendered_path = render_strip_to_path(context, strip, image_output=True)
                        if rendered_path:
                            scene.image_path = rendered_path
                            file_path = rendered_path
                        else:
                            strip_dirname = os.path.dirname(strip.directory)
                            file_path = bpy.path.abspath(os.path.join(strip_dirname, strip.elements[0].filename))
                            scene.image_path = file_path
                    elif strip.type == "IMAGE":
                        strip_dirname = os.path.dirname(strip.directory)
                        file_path = bpy.path.abspath(os.path.join(strip_dirname, strip.elements[0].filename))
                        scene.image_path = file_path
                    elif strip.type == "MOVIE" and target_type == "image":
                        rendered_path = render_strip_to_path(context, strip, image_output=True)
                        if rendered_path:
                            scene.image_path = rendered_path
                            file_path = rendered_path
                        else:
                            file_path = bpy.path.abspath(strip.filepath)
                            scene.movie_path = file_path
                    elif strip.type == "MOVIE":
                        file_path = bpy.path.abspath(strip.filepath)
                        scene.movie_path = file_path
                    elif strip.type == "SOUND":
                        # scene.sound_path is already a trimmed WAV when the original
                        # strip went through render_strip_to_wav above; only fall back
                        # to the rendered strip's filepath if nothing was set yet.
                        if scene.sound_path:
                            file_path = scene.sound_path
                        else:
                            file_path = bpy.path.abspath(strip.sound.filepath)
                            scene.sound_path = file_path
                    run_generation = True

                    if strip.name:
                        strip_prompt = os.path.splitext(strip.name)[0]
                        seed_nr = extract_numbers(str(strip_prompt))

                        if seed_nr:
                            file_seed = int(seed_nr)
                            strip_prompt = strip_prompt.replace(str(file_seed) + "_", "")
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed

                        # Style Prompts using BASE prompt
                        styled = style_prompt(strip_prompt + ", " + base_prompt)
                        
                        current_prompt_text = styled[0]
                        current_negative_text = styled[1]
                    
                    if current_prompt_text == "":
                        current_prompt_text = base_prompt
                        
#                    if target_type != "text":
#                        print(f"Prompt: {current_prompt_text}")
                    print(f"Prompt: {current_prompt_text}")

            # 5. EXECUTE GENERATION
            if run_generation:
                # Apply the calculated prompt to the scene property
                if current_prompt_text == None: current_prompt_text = ""
                if current_negative_text == None: current_negative_text = ""
                scene.generate_movie_prompt = current_prompt_text
                scene.generate_movie_negative_prompt = current_negative_text
                scene.frame_current = strip.frame_final_start
                context.scene.sequence_editor.active_strip = strip
                
                # Seed was already set from strip name above

                # Call the actual operator
                if target_type == "movie": sequencer.generate_movie()
                elif target_type == "audio": sequencer.generate_audio()
                elif target_type == "image": sequencer.generate_image()
                elif target_type == "text": sequencer.generate_text()

                # --- IMMEDIATE RESTORE ---
                # Restore the clean base prompt immediately after the call
                scene.generate_movie_prompt = base_prompt
                scene.generate_movie_negative_prompt = base_negative_prompt
                context.scene.movie_use_random = use_random
                context.scene.movie_num_seed = seed
                
                # Clean up paths
                scene.image_path = ""
                scene.movie_path = ""
                scene.sound_path = ""
                # --- Single temp strip cleanup ---
                if temp_strip is not None:
                    if temp_strip.type == 'MOVIE':
                        delete_linked_audio(context, temp_strip)

                    delete_strip(temp_strip)
                    temp_strip = None


                # --- Batch Cleanup: Delete all temporary strips collected ---
                for s in temp_strips:
                    if s:
                        try:
                            seq_editor = context.scene.sequence_editor
                            if seq_editor and s.name in seq_editor.strips_all:

                                if s.type == 'MOVIE':
                                    delete_linked_audio(context, s)

                                delete_strip(s)

                        except Exception as e:
                            print(f"Warning: Could not delete temp strip {s.name}: {e}")
                
                # Clear the list for the next iteration
                temp_strips.clear()

                # Delete any temp files rendered by render_strip_to_path() / decompose_meta()
                from ..utils.helpers import _rendered_temp_paths
                for _p in list(_rendered_temp_paths):
                    try:
                        os.remove(_p)
                    except OSError:
                        pass
                _rendered_temp_paths.clear()

        # --- Final Cleanup ---
        scene.frame_current = current_frame
        # Final safety restore
        scene.generate_movie_prompt = base_prompt
        scene.generate_movie_negative_prompt = base_negative_prompt
        context.scene.movie_use_random = use_random
        context.scene.movie_num_seed = seed
        # Note: active_strip is intentionally NOT restored here so that the
        # last generated output strip remains active, allowing the Metadata
        # panel to display its stored AI metadata immediately after generation.

#        try:
#            addon_prefs.playsound = play_sound
#            bpy.ops.renderreminder.pallaidium_play_notification()
#        except:
#            pass

        print("Processing finished.")

        return {"FINISHED"}

