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

# Throttle key for the one-shot Strip-Power gate diagnostic (see draw()).
_STRIP_POWER_DBG_LAST = None
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
import os
import re
from datetime import date

from ..utils.helpers import *
from ..properties.scene_props import *
from ..properties.preferences import *

def _draw_queue(layout, context):
    """Draw the render queue section at the bottom of the Pallaidium panel."""
    from ..operators.queue_ops import _queue_tick

    scene = context.scene
    queue = getattr(scene, "render_queue", None)
    if queue is None:
        return

    #layout.separator()
    outer = layout.column(align=True)
    outer = outer.box()
    outer.use_property_split = False
    outer.use_property_decorate = False

    # Derive running state from the timer — never from a stored property
    is_running = bpy.app.timers.is_registered(_queue_tick)

    # Determine queue state safely
    has_jobs = len(queue) > 0
    has_pending = any(j.status == "PENDING" for j in queue) if has_jobs else False

    # Create a single row for both controls
    ctrl_row = outer.row(align=True)

    # Sub-row for Start / Stop
    start_stop_sub = ctrl_row.row(align=True)
    if is_running:
        start_stop_sub.operator("sequencer.stop_queue", text="Stop", icon="PAUSE")
    else:
        start_stop_sub.enabled = has_pending
        start_stop_sub.operator("sequencer.queue_runner", text="Start", icon="PLAY")
    
    # Sub-row for Clear
    clear_sub = ctrl_row.row(align=True)
    clear_sub.enabled = has_jobs
    clear_sub.operator("sequencer.clear_queue", text="", icon="TRASH")

    _STATUS_ICONS = {
        "PENDING":    "STRIP_COLOR_03",
        "RUNNING":    "STRIP_COLOR_05",
        "COMPLETED":  "STRIP_COLOR_04",
        "FAILED":     "STRIP_COLOR_01",
        "CANCELLED":  "STRIP_COLOR_09",
        "CANCELLING": "STRIP_COLOR_02",
    }

    for job in queue:
        box = outer.box()

        # ---- Top row: icon + prompt label + action button ----
        top = box.row(align=True)
        top.label(text="", icon=_STATUS_ICONS.get(job.status, "STRIP_COLOR_09"))
        label = (job.prompt[:36] + "…") if len(job.prompt) > 38 else job.prompt
        top.label(text=label)

        op_redo = top.operator("sequencer.redo_from_job", text="", icon="LOOP_BACK")
        op_redo.job_id = job.job_id

        if job.status == "PENDING" or (job.status == "RUNNING" and job.phase != "Downloading model"):
            op = top.operator("sequencer.cancel_queue_job", text="", icon="X")
            op.job_id = job.job_id
        elif job.status == "RUNNING" and job.phase == "Downloading model":
            top.label(text="", icon="IMPORT")
        elif job.status == "FAILED":
            op = top.operator("sequencer.show_queue_error", text="", icon="INFO")
            op.job_id = job.job_id
            op2 = top.operator("sequencer.remove_queue_job", text="", icon="X")
            op2.job_id = job.job_id
        elif job.status in ("COMPLETED", "CANCELLED", "CANCELLING"):
            op = top.operator("sequencer.remove_queue_job", text="", icon="X")
            op.job_id = job.job_id

        # ---- Detail row depending on status ----
        if job.status == "RUNNING":
            prog_row = box.row(align=True)
            if job.phase == "Downloading model" and job.total_steps > 0:
                pct_text = f"Downloading  {job.current_step} / {job.total_steps} MB"
            elif job.total_steps > 0:
                pct_text = f"{job.phase}  {job.current_step}/{job.total_steps}"
            else:
                pct_text = f"{job.phase}  {int(job.progress * 100)}%"
            prog_row.prop(job, "progress", text=pct_text, slider=True)
        elif job.status == "CANCELLING":
            box.label(text="Stopping…", icon="TIME")
        elif job.status == "PENDING":
            info = f"{job.output_type.title()} · {job.model_card.split('/')[-1][:28]}"
            box.label(text=info, icon="TIME")
        elif job.status == "COMPLETED" and getattr(job, "tokens_info", ""):
            box.label(text=job.tokens_info, icon="RNA")
        elif job.status == "FAILED":
            box.label(text=f"Error: {job.error_message[:58]}", icon="ERROR")


class SEQUENCER_PT_pallaidium_panel(Panel):  # UI
    """Generate Media using AI"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_movie_panel"
    bl_label = "Pallaidium"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generative AI"

    @classmethod
    def poll(cls, context):
        return context.area.type == "SEQUENCE_EDITOR"

    def draw(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[ADDON_ID].preferences
        audio_model_card = addon_prefs.audio_model_card
        movie_model_card = addon_prefs.movie_model_card
        image_model_card = addon_prefs.image_model_card
        text_model_card = addon_prefs.text_model_card
        scene = context.scene
        type = scene.generatorai_typeselect
        input = scene.input_strips
        layout = self.layout

        if addon_prefs.dep_is_running:
            op_label = "Installing" if addon_prefs.dep_op_type == "install" else "Uninstalling"
            pct = int(addon_prefs.dep_progress * 100)
            box = layout.box()
            header = box.row()
            header.label(text=f"Dependencies {op_label}...", icon="IMPORT")
            header.operator("sequencer.cancel_dep_op", text="", icon="X")
            prog_row = box.row()
            prog_row.prop(addon_prefs, "dep_progress",
                          text=f"{addon_prefs.dep_phase or op_label}  {pct}%",
                          slider=True)
            if addon_prefs.dep_status_line:
                box.label(text=addon_prefs.dep_status_line, icon="INFO")
            layout.enabled = False
            return

        if addon_prefs.dep_needs_restart:
            box = layout.box()
            box.label(text="Restart Blender to use newly installed dependencies.", icon="ERROR")
            layout.enabled = False
            return

        # --- Plugin-driven rendering ---
        from ..models import get_plugin as _reg_get_plugin
        from ..models.base import UISection
        _card = {"movie": movie_model_card, "image": image_model_card,
                 "audio": audio_model_card, "text": text_model_card}.get(type, "")
        plugin = _reg_get_plugin(_card)
        def _has(sec): return plugin is None or sec in (plugin.UI_SECTIONS or [])
        col = layout.column(align=False)
        col.use_property_split = True
        col.use_property_decorate = False
        col = col.box()
        col = col.column()

        if scene.sequence_editor is None:
            scene.sequence_editor_create()

        drew_custom = (plugin.draw_custom_ui(col, context) is True) if (plugin and type == "image") else False
        if not drew_custom:
            try:
                _strip_required = getattr(plugin, "requires_input_strip", False)
                row = col.row()
                row.enabled = not _strip_required
                row.prop(context.scene, "input_strips", text="Input")
            except:
                pass


        if type != "text":
            if type != "audio":
                if type == "movie" and plugin is not None and not plugin.uses_standard_input_strip:
                    plugin.draw_custom_ui(col, context)

                elif (type == "movie") or (type == "image" and (plugin is None or plugin.uses_standard_input_strip)):
                    _show_strength = (
                        not scene.inpaint_selected_strip
                        or image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                        or (plugin is not None and getattr(plugin, "inpaint_uses_strength", False))
                    )
                    _usp = getattr(plugin, "uses_strip_power", True)
                    # ── Strip-Power gate diagnostic (throttled: prints only when the
                    #    decision changes, so it never spams the redraw loop) ──────────
                    if type == "image":
                        global _STRIP_POWER_DBG_LAST
                        _sig = (image_model_card, input, bool(_show_strength), bool(_usp),
                                getattr(scene, "inpaint_selected_strip", ""))
                        if _sig != _STRIP_POWER_DBG_LAST:
                            _STRIP_POWER_DBG_LAST = _sig
                            _shown = (input == "input_strips" and _show_strength and _usp)
                            print(f"[Strip Power gate] model={image_model_card!r} "
                                  f"plugin={plugin.__class__.__name__ if plugin else None} "
                                  f"input={input!r} (need 'input_strips') "
                                  f"show_strength={bool(_show_strength)} "
                                  f"uses_strip_power={bool(_usp)} "
                                  f"inpaint_mask={getattr(scene, 'inpaint_selected_strip', '')!r} "
                                  f"=> Strip Power {'SHOWN' if _shown else 'HIDDEN'}")
                    if input == "input_strips" and _show_strength and _usp:
                        col = col.column(heading="Use", align=True)
                        col.prop(context.scene, "image_power", text="Strip Power")

                    if (
                        bpy.context.scene.sequence_editor is not None
                        and (plugin is None or plugin.supports_inpaint)
                    ):
                        if type == "image":
                            row = col.row(align=True)
                            row.prop_search(
                                scene,
                                "inpaint_selected_strip",
                                scene.sequence_editor,
                                "strips",
                                text="Inpaint Mask",
                                icon="SEQ_STRIP_DUPLICATE",
                            )
                            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "inpaint_select"

            if image_model_card == "yuvraj108c/FLUX.1-Kontext-dev" and type == "image":
                row = col.row(align=True)
                row.prop_search(
                    scene,
                    "kontext_strip_1",
                    scene.sequence_editor,
                    "strips",
                    text="Ref.",
                    icon="FILE_IMAGE",
                )
                row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "kontext_select1"

            if _has(UISection.POSE_TOGGLE):
                col = col.column(heading="Read as", align=True)
                col.prop(context.scene, "openpose_use_bones", text="OpenPose Rig Image")
            if _has(UISection.SCRIBBLE_TOGGLE):
                col = col.column(heading="Read as", align=True)
                col.prop(context.scene, "use_scribble_image", text="Scribble Image")

            # IPAdapter.
            if _has(UISection.IP_ADAPTER) and type == "image":
                row = col.row(align=True)
                row.prop(scene, "ip_adapter_face_folder", text="Adapter Face")
                row.operator(
                    "ip_adapter_face.file_browser", text="", icon="FILE_FOLDER"
                )

                row = col.row(align=True)
                row.prop(scene, "ip_adapter_style_folder", text="Adapter Style")
                row.operator(
                    "ip_adapter_style.file_browser", text="", icon="FILE_FOLDER"
                )

            # Prompts
            if plugin is None or plugin.UI_SECTIONS:
                col = layout.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
            if _has(UISection.PROMPT):
                col = col.box()
                col = col.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
                if type != "audio":
                    col.prop(context.scene, "generatorai_styles", text="")
                col.textbox(context.scene, "generate_movie_prompt", placeholder='Positive prompt...')#, text="")#prop, icon="ADD")
                if _has(UISection.NEG_PROMPT):
                    col.textbox(
                        context.scene,
                        "generate_movie_negative_prompt",
                        placeholder='Negative prompt...',
                        #text="",
                        #icon="REMOVE", 
                    )
                    
            layout = self.layout
            layout = layout.column(align=True)
            layout = layout.box()
            layout.use_property_split = True
            layout.use_property_decorate = False
            if _has(UISection.RESOLUTION):
                col = layout.column(align=True)
                col.prop(context.scene, "generate_movie_x", text="X")
                col.prop(context.scene, "generate_movie_y", text="Y")
            col = layout.column(align=True)
            if _has(UISection.FRAMES):
                col.prop(context.scene, "generate_movie_frames", text="Frames")
            if type == "movie" and _has(UISection.AUDIO_OUTPUT):
                col.prop(context.scene, "remote_generate_audio", text="Generate Audio")
            if _has(UISection.AUDIO_DURATION):
                col.prop(context.scene, "audio_length_in_f", text="Frames")
            if type == "audio" and _has(UISection.SPEED):
                col.prop(context.scene, "audio_speed_tts", text="Speed")
            if type == "audio" and _has(UISection.AUDIO_REF):
                row = col.row(align=True)
                row.prop(context.scene, "ref_audio_path", text="Speaker Ref.")
                row.operator(
                    "sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER"
                )
            if type == "audio" and _has(UISection.TEXT_REF):
                row = col.row(align=True)
                row.prop(context.scene, "ref_text", text="Text Ref.")
            if type == "audio" and _has(UISection.CHAT_PARAMS):
                col.prop(context.scene, "chat_exaggeration")
                col.prop(context.scene, "chat_pace")
                col.prop(context.scene, "chat_temperature")

            if _has(UISection.STEPS) and not scene.use_lcm:
                col.prop(
                    context.scene,
                    "movie_num_inference_steps",
                    text="Quality Steps",
                )

            if _has(UISection.GUIDANCE) and not scene.use_lcm:
                if image_model_card == "Shitao/OmniGen-v1-diffusers" and type == "image":
                    col.prop(
                        context.scene, "img_guidance_scale", text="Image Power"
                    )
                col.prop(context.scene, "movie_num_guidance", text="Word Power")

            if _has(UISection.ILLUMINATION):
                col.prop(context.scene, "illumination_style", text="Relight Style")
                col.prop(context.scene, "light_direction", text="Direction")
            if type == "audio" and _has(UISection.MUSIC_PARAMS):
                col.prop(context.scene, "music_bpm", text="BPM")
                col.prop(context.scene, "music_key_scale", text="Key")
                col.prop(context.scene, "music_time_signature", text="Time Sig.")
                col.textbox(context.scene, "music_lyrics", placeholder="Lyrics")

            if type == "audio" and plugin is not None:
                plugin.draw_custom_ui(col, context)

            if type == "movie" and plugin is not None and plugin.uses_standard_input_strip:
                plugin.draw_custom_ui(col, context)

            if _has(UISection.SEED):
                row = col.row(align=True)
                row.use_property_split = False
                
                split = row.split(factor=0.4, align=True)
                
                left = split.row(align=True)
                left.alignment = 'RIGHT'
                left.label(text="Seed")
                
                right = split.row(align=True)
                
                sub = right.row(align=True)
                is_rand = context.scene.movie_use_random
                sub.active = is_rand
                sub.prop(context.scene, "movie_num_seed", text="")
                
                right.prop(
                    context.scene, "movie_use_random",
                    text="",
                    icon="UNLOCKED" if is_rand else "LOCKED",
                    icon_only=True
                )

            _col_pre_enhance = col
            if type == "image" and (plugin is None or plugin.UI_SECTIONS) and getattr(plugin, "show_enhance", True):
                col = col.column(heading="Enhance", align=True)
                row = col.row()

                if _has(UISection.ENHANCE):
                    row.prop(context.scene, "refine_sd", text="Quality")
                    sub_col = col.row()
                    sub_col.active = context.scene.refine_sd
                    row.prop(context.scene, "use_lcm", text="Speed")

                if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    col = col.column(heading="Details", align=True)

                row = col.row()
                if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    row.prop(context.scene, "adetailer", text="Faces")

                row.prop(context.scene, "aurasr", text="Upscale 4x")

            if type == "image" and plugin is not None:
                plugin.draw_post_enhance_ui(_col_pre_enhance, context)

            if type == "movie" and movie_model_card == "stable-diffusion-xl/frame2frame":
                col = layout.column(heading="Upscale", align=True)
                col.prop(context.scene, "aurasr", text="4x")

            # LoRA.
            if _has(UISection.LORA):
                layout = self.layout
                #layout.use_property_split = True
                #layout.use_property_decorate = False
                col = layout.column(align=True)
                col = col.box()
                col = col.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False

                # Folder selection and refresh button
                row = col.row(align=True)
                row.prop(scene, "lora_folder", text="LoRA", placeholder='LoRA path...')
                row.operator("lora.refresh_files", text="", icon="FILE_REFRESH")

                # Custom UIList
                lora_files = scene.lora_files
                list_len = len(lora_files)
                if list_len > 0:
                    col.template_list(
                        "LORABROWSER_UL_files",
                        "The_List",
                        scene,
                        "lora_files",
                        scene,
                        "lora_files_index",
                        rows=2,
                    )

        elif text_model_card == "ZuluVision/MoviiGen1.1_Prompt_Rewriter":
                col = layout.column(align=True)
                col = col.box()
                col = col.column(align=True)
                col.use_property_split = False
                col.use_property_decorate = False
                col.prop(context.scene, "generate_movie_prompt", text="", icon="ADD")

        # Output.
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.box()
        col = col.column(align=True)
        try:
            col.prop(context.scene, "generatorai_typeselect", text="Output")
        except:
            pass


        if type == "movie":
            col.prop(addon_prefs, "movie_model_card", text=" ")
        if type == "audio":
            col.prop(addon_prefs, "audio_model_card", text=" ")
        if type == "text":
            col.prop(addon_prefs, "text_model_card", text=" ")
            from ..models.base import InputSpec as _InputSpec
            if plugin is not None and _InputSpec.HF_TOKEN in plugin.INPUTS:
                row = col.row(align=True)
                row.prop(addon_prefs, "hugginface_token")
                row.operator(
                    "wm.url_open", text="", icon="URL"
                ).url = "https://huggingface.co/settings/tokens"
            if plugin is not None:
                plugin.draw_custom_ui(col, context)
        if type == "image":
            col.prop(addon_prefs, "image_model_card", text=" ")
            from ..models.base import InputSpec as _InputSpec
            if plugin is not None and _InputSpec.HF_TOKEN in plugin.INPUTS:
                row = col.row(align=True)
                row.prop(addon_prefs, "hugginface_token")
                row.operator(
                    "wm.url_open", text="", icon="URL"
                ).url = "https://huggingface.co/settings/tokens"
        # Batch Count: shown only when the active plugin actually produces
        # multiple distinct outputs per run. Deterministic single-output models
        # (captioning, transcription, stem split, external single-shot APIs)
        # set supports_batch=False so the control is hidden where it has no effect.
        if plugin is None or getattr(plugin, "supports_batch", True):
            col = col.column()
            col.prop(context.scene, "movie_num_batch", text="Batch Count")

        if plugin is not None and hasattr(plugin, "draw_post_seed_ui"):
            plugin.draw_post_seed_ui(col, context)
        if type == "image":
            if plugin.__class__.__name__ == "Ideogram4Plugin":
                col.operator("florence2.open_box_editor", text="Open Box Editor", icon="MOD_MASK")

        # Generate.
        col = layout.column()
        col = col.box()
        if input == "input_strips":
            ed = scene.sequence_editor
            row = col.row(align=True)
            #row.scale_y = 1.2
            #row.operator("sequencer.text_to_generator", text="Generate from Strips")
            row.operator("sequencer.add_to_queue", text="Add to Queue", icon="ADD")
        else:
            row = col.row(align=True)
            #row.scale_y = 1.2
            if type == "movie":
                # Frame by Frame
#                if movie_model_card == "stable-diffusion-xl/frame2frame":
#                    row.operator(
#                        "sequencer.text_to_generator", text="Generate from Strips"
#                    )
#                else:
#                    #row.operator("sequencer.generate_movie", text="Generate")
                row.operator("sequencer.add_to_queue", text="Add to Queue", icon="ADD")
            if type == "image":
                #row.operator("sequencer.generate_image", text="Generate")
                row.operator("sequencer.add_to_queue", text="Add to Queue", icon="ADD")
            if type == "audio":
                #row.operator("sequencer.generate_audio", text="Generate")
                row.operator("sequencer.add_to_queue", text="Add to Queue", icon="ADD")
            if type == "text":
                #row.operator("sequencer.generate_text", text="Generate")
                row.operator("sequencer.add_to_queue", text="Add to Queue", icon="ADD")

        # Render Queue panel
        _draw_queue(col, context)#self.layout, context)