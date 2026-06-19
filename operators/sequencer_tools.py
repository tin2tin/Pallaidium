"""
Sequencer utility operators: audio file browser, VSE strip eyedropper picker.
"""

import bpy
import os

from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator
from bpy.props import StringProperty

from ..utils.helpers import find_strip_by_name


class SequencerOpenAudioFile(Operator, ImportHelper):
    bl_idname = "sequencer.open_audio_filebrowser"
    bl_label = "Open Audio File Browser"
    filter_glob: StringProperty(
        default="*.wav;",
        options={"HIDDEN"},
    )
    # Which scene StringProperty to write the chosen path into. Defaults to the
    # shared ref_audio_path; plugins with their own field (e.g. MOSS-TTS) pass
    # their property name here.
    target_prop: StringProperty(default="ref_audio_path", options={"HIDDEN"})

    def execute(self, context):
        scene = context.scene
        if self.filepath and os.path.exists(self.filepath):
            valid_extensions = {".wav"}
            filename, extension = os.path.splitext(self.filepath)
            if extension.lower() in valid_extensions:
                print("Selected audio file:", self.filepath)
                target = self.target_prop or "ref_audio_path"
                if hasattr(scene, target):
                    setattr(scene, target, bpy.path.abspath(self.filepath))
                else:
                    scene.ref_audio_path = bpy.path.abspath(self.filepath)
            else:
                print("Info: Only wav is allowed.")
        else:
            self.report({"ERROR"}, "Selected file does not exist.")
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class SEQUENCER_OT_ai_strip_picker(Operator):
    """Pick a strip"""
    bl_idname = "sequencer.strip_picker"
    bl_label = "Pick Strip"
    bl_description = "Pick a strip in the VSE"
    bl_options = {"REGISTER", "UNDO"}
    action: StringProperty(
        name="Action",
        description="Action to perform on the picked strip",
        default="select"
    )

    def modal(self, context, event):
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            print("Picking...")
            area = context.area
            region = context.region
            mouse_region_coord = (event.mouse_region_x, event.mouse_region_y)
            if area.type != "SEQUENCE_EDITOR" or not region:
                self.report({"WARNING"}, "Invalid region or area for VSE")
                context.window.cursor_modal_restore()
                return {"CANCELLED"}

            v2d = region.view2d
            mouse_x_view, mouse_y_view = v2d.region_to_view(*mouse_region_coord)

            for strip in context.scene.sequence_editor.strips_all:
                if hasattr(strip, 'transform'):
                    scale_y = strip.transform.scale_y
                else:
                    scale_y = 1.0

                strip_y_min_view = strip.channel - 0.5 * scale_y
                strip_y_max_view = strip.channel + 0.5 * scale_y

                if (
                    strip.frame_start <= mouse_x_view < strip.frame_final_end and
                    strip_y_min_view <= mouse_y_view < strip_y_max_view
                ):
                    self.perform_action(context, strip)
                    context.window.cursor_modal_restore()
                    return {"FINISHED"}

            return {"RUNNING_MODAL"}

        elif event.type in {"RIGHTMOUSE", "ESC"}:
            context.window.cursor_modal_restore()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def perform_action(self, context, strip):
        scene = context.scene

        if self.action == "omni_select1":
            self.report({"INFO"}, f"Picked: {strip.name}")
            if find_strip_by_name(scene, strip.name):
                scene.omnigen_strip_1 = strip.name
        elif self.action == "omni_select2":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.omnigen_strip_2 = strip.name
        elif self.action == "omni_select3":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.omnigen_strip_3 = strip.name

        if self.action == "qwen_select1":
            self.report({"INFO"}, f"Picked: {strip.name}")
            if find_strip_by_name(scene, strip.name):
                scene.qwen_strip_1 = strip.name
        elif self.action == "qwen_select2":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.qwen_strip_2 = strip.name
        elif self.action == "qwen_select3":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.qwen_strip_3 = strip.name

        if self.action == "klein_select1":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.klein_strip_1 = strip.name
        elif self.action == "klein_select2":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.klein_strip_2 = strip.name
        elif self.action == "klein_select3":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.klein_strip_3 = strip.name

        elif self.action == "ltx23ic_control_select":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.ltx23ic_control_strip = strip.name

        elif self.action == "ltx23ext_audio_select":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.ltx23ext_audio_strip = strip.name

        elif self.action == "minimax_select":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.minimax_subject = strip.name

        elif self.action == "inpaint_select":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.inpaint_selected_strip = strip.name

        elif self.action == "out_frame_select":
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.out_frame = strip.name

        for i in range(1, 10):
            if self.action == f"flux_select{i}":
                self.report({"INFO"}, f"Picked: {strip.name}")
                if find_strip_by_name(scene, strip.name):
                    setattr(scene, f"flux_strip_{i}", strip.name)
                break

        if self.action == "kontext_select1":
            self.report({"INFO"}, f"Picked: {strip.name}")
            if find_strip_by_name(scene, strip.name):
                scene.kontext_strip_1 = strip.name

    def invoke(self, context, event):
        if context.area.type == 'SEQUENCE_EDITOR':
            context.window_manager.modal_handler_add(self)
            context.window.cursor_modal_set("EYEDROPPER")
            return {"RUNNING_MODAL"}
        else:
            self.report({'WARNING'}, "This operator only works in the Video Sequence Editor")
            return {"CANCELLED"}
