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
import os
import re
from datetime import date

from ..utils.helpers import *


# ---------------------------------------------------------------------------
# Dynamic EnumProperty item callbacks — delegates to the plugin registry
# ---------------------------------------------------------------------------

def _video_enum_items(self, context):
    from ..models import get_enum_items
    return get_enum_items("video")


def _image_enum_items(self, context):
    from ..models import get_enum_items
    return get_enum_items("image")


def _audio_enum_items(self, context):
    from ..models import get_enum_items
    return get_enum_items("audio")


def _text_enum_items(self, context):
    from ..models import get_enum_items
    return get_enum_items("text")


# Update wrappers: persist the selected MODEL_ID string alongside the enum
# so we can restore the correct selection across Blender restarts without
# relying on the enum's internal integer (which changes as plugins are added).

def _movie_model_update(self, context):
    self.movie_model_card_id = self.movie_model_card
    input_strips_updated(self, context)

def _image_model_update(self, context):
    self.image_model_card_id = self.image_model_card
    output_strips_updated(self, context)
    try:
        from ..models import get_plugin
        plugin = get_plugin(self.image_model_card)
        if plugin is not None and getattr(plugin, "requires_input_strip", False):
            if context and context.scene:
                context.scene.input_strips = "input_strips"
        if plugin is not None and getattr(plugin, "requires_no_style", False):
            if context and context.scene:
                context.scene.generatorai_styles = "no_style"
        if plugin is not None and hasattr(plugin, "on_model_selected"):
            if context and context.scene:
                plugin.on_model_selected(context.scene, context)
    except Exception:
        pass

def _audio_model_update(self, context):
    self.audio_model_card_id = self.audio_model_card
    output_strips_updated(self, context)
    try:
        from ..models import get_plugin
        plugin = get_plugin(self.audio_model_card)
        if plugin is not None and getattr(plugin, "requires_input_strip", False):
            if context and context.scene:
                context.scene.input_strips = "input_strips"
    except Exception:
        pass

def _text_model_update(self, context):
    self.text_model_card_id = self.text_model_card
    output_strips_updated(self, context)


def _hf_cache_dir_update(self, context):
    cache_dir = self.hf_cache_dir
    if cache_dir:
        os.environ["HF_HUB_CACHE"] = cache_dir


class GeneratorAddonPreferences(AddonPreferences):
    bl_idname = __package__.rsplit(".", 1)[0]
    soundselect: EnumProperty(
        name="Sound",
        items={
            ("ding", "Ding", "A simple bell sound"),
            ("coin", "Coin", "A Mario-like coin sound"),
            ("user", "User", "Load a custom sound file"),
        },
        default="ding",
    )
    default_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sounds", "*.wav")

    usersound: StringProperty(
        name="User",
        description="Load a custom sound from your computer",
        subtype="FILE_PATH",
        default=default_folder,
        maxlen=1024,
    )

    playsound: BoolProperty(
        name="Audio Notification",
        default=True,
    )

    # String backings: these are what get saved to user preferences.
    # The EnumProperty integer is NOT saved (SKIP_SAVE) to avoid stale-value
    # warnings when plugins are added, removed, or renamed.
    movie_model_card_id: StringProperty(default="")
    image_model_card_id: StringProperty(default="")
    audio_model_card_id: StringProperty(default="")
    text_model_card_id: StringProperty(default="")

    movie_model_card: bpy.props.EnumProperty(
        name="Video Model",
        items=_video_enum_items,
        options={'SKIP_SAVE'},
        update=_movie_model_update,
    )
    image_model_card: bpy.props.EnumProperty(
        name="Image Model",
        items=_image_enum_items,
        options={'SKIP_SAVE'},
        update=_image_model_update,
    )
    audio_model_card: bpy.props.EnumProperty(
        name="Audio Model",
        items=_audio_enum_items,
        options={'SKIP_SAVE'},
        update=_audio_model_update,
    )
    hugginface_token: bpy.props.StringProperty(
        name="Hugginface Token",
        default="hugginface_token",
        subtype="PASSWORD",
    )
    text_model_card: EnumProperty(
        name="Text Model",
        items=_text_enum_items,
        options={'SKIP_SAVE'},
        update=_text_model_update,
    )
    generator_ai: StringProperty(
        name="Filepath",
        description="Path to the folder where the generated files are stored",
        subtype="DIR_PATH",
        default=join(bpy.utils.user_resource("DATAFILES"), "Pallaidium_Media"),
    )
    hf_cache_dir: StringProperty(
        name="HuggingFace Cache",
        description="Path where HuggingFace models are stored and loaded from (sets HF_HUB_CACHE). Must point to the 'hub' subdirectory.",
        subtype="DIR_PATH",
        default=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        maxlen=1024,
        update=_hf_cache_dir_update,
    )
    local_files_only: BoolProperty(
        name="Use Local Files Only",
        default=False,
    )

    # --- Async dependency operation state (SKIP_SAVE — reset on restart) ---
    dep_is_running:     BoolProperty(default=False,  options={'SKIP_SAVE'})
    dep_op_type:        StringProperty(default="",   options={'SKIP_SAVE'})
    dep_progress:       FloatProperty(min=0.0, max=1.0, default=0.0, options={'SKIP_SAVE'})
    dep_phase:          StringProperty(default="",   options={'SKIP_SAVE'})
    dep_status_line:    StringProperty(default="",   options={'SKIP_SAVE'})
    dep_has_errors:     BoolProperty(default=False,  options={'SKIP_SAVE'})
    dep_failure_report: StringProperty(default="",   options={'SKIP_SAVE'})
    dep_needs_restart:  BoolProperty(default=False,  options={'SKIP_SAVE'})

    def draw(self, context):
        layout = self.layout
        box = layout.box()

        if self.dep_is_running:
            op_label = "Installing" if self.dep_op_type == "install" else "Uninstalling"
            pct = int(self.dep_progress * 100)
            header_row = box.row()
            header_row.label(text=f"Dependencies {op_label}...", icon="IMPORT")
            header_row.operator("sequencer.cancel_dep_op", text="Cancel", icon="X")
            prog_row = box.row()
            prog_row.prop(self, "dep_progress",
                          text=f"{self.dep_phase or op_label}  {pct}%",
                          slider=True)
            if self.dep_status_line:
                box.label(text=self.dep_status_line, icon="INFO")
            layout.enabled = False
            return

        if self.dep_needs_restart:
            restart_box = layout.box()
            restart_box.label(text="Restart Blender to use newly installed dependencies.", icon="ERROR")

        if self.dep_has_errors:
            err_box = layout.box()
            # Header row: icon + title + dismiss button
            hdr = err_box.row()
            hdr.label(text="Dependency installation incomplete — some packages failed.", icon="ERROR")
            hdr.operator("sequencer.dismiss_install_errors", text="", icon="X")
            # Action row: copy button + text-editor hint
            act = err_box.row()
            act.operator("sequencer.copy_install_report", icon="COPYDOWN", text="Copy Error Report")
            act.label(text='Full log: Text Editor > "Pallaidium Install Errors"')
            # Show the actual failure content — skip the markdown header boilerplate
            lines = self.dep_failure_report.splitlines()
            start = 0
            for i, ln in enumerate(lines):
                if ln.startswith("### Failed batches"):
                    start = i + 1
                    break
            shown = 0
            for ln in lines[start:]:
                if shown >= 10:
                    break
                text = ln.strip()
                if not text or text.startswith("---") or text.startswith("_Please"):
                    continue
                if text == "```":
                    continue
                err_box.label(text=text[:110])
                shown += 1

        row = box.row()
        row.operator("sequencer.install_generator")
        row.operator("sequencer.uninstall_generator")
        row.operator("sequencer.export_requirements")
        try:
            box.prop(self, "movie_model_card")
            box.prop(self, "image_model_card")
        except:
            pass
        from ..models import get_plugin as _gp
        from ..models.base import InputSpec as _IS
        _img_plugin = _gp(self.image_model_card)
        if (
            (_img_plugin is not None and _IS.HF_TOKEN in _img_plugin.INPUTS)
            or (self.image_model_card == "ChuckMcSneed/FLUX.1-dev" and os_platform == "Darwin")
            or (self.image_model_card == "lzyvegetable/FLUX.1-schnell" and os_platform == "Darwin")
        ):
            row = box.row(align=True)
            row.prop(self, "hugginface_token")
            row.operator(
                "wm.url_open", text="", icon="URL"
            ).url = "https://huggingface.co/settings/tokens"
        try:
            box.prop(self, "audio_model_card")
        except:
            pass
        box.prop(self, "generator_ai")
        box.prop(self, "hf_cache_dir")
        row = box.row(align=True)
        row.label(text="Notification:")
        row.prop(self, "playsound", text="")
        sub_row = row.row()
        sub_row.prop(self, "soundselect", text="")
        if self.soundselect == "user":
            sub_row.prop(self, "usersound", text="")
        sub_row.operator(
            "renderreminder.pallaidium_play_notification", text="", icon="PLAY"
        )
        sub_row.active = self.playsound

        row_row = box.row(align=True)
        row_row.label(text="Use Local Files Only:")
        row_row.prop(self, "local_files_only", text="")
        row_row.label(text="")
        row_row.label(text="")
        row_row.label(text="")