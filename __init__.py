# https://modelscope.cn/models/damo/text-to-video-synthesis/summary

bl_info = {
    "name": "Text to Video",
    "author": "tintwotin",
    "version": (1, 0),
    "blender": (3, 4, 0),
    "location": "Video Sequence Editor > Sidebar > Generate",
    "description": "Convert text to video",
    "category": "Sequencer",
}

import bpy, ctypes
from bpy.types import Operator, Panel
import site
import subprocess
import sys, os
import string


def show_system_console(show):
    # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
    SW_HIDE = 0
    SW_SHOW = 5

    ctypes.windll.user32.ShowWindow(
        ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW if show else SW_HIDE
    )


def set_system_console_topmost(top):
    # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowpos
    HWND_NOTOPMOST = -2
    HWND_TOPMOST = -1
    HWND_TOP = 0
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOZORDER = 0x0004

    ctypes.windll.user32.SetWindowPos(
        ctypes.windll.kernel32.GetConsoleWindow(),
        HWND_TOP if top else HWND_NOTOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
    )


def clean_path(string_path):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    clean_path = "".join(c if c in valid_chars else "_" for c in string_path)
    return clean_path


def import_module(self, module, install_module):
    show_system_console(True)
    set_system_console_topmost(True)

    module = str(module)
    try:
        exec("import " + module)
    except ModuleNotFoundError:
        app_path = site.USER_SITE
        if app_path not in sys.path:
            sys.path.append(app_path)
        pybin = sys.executable
        self.report({"INFO"}, "Installing: " + module + " module.")
        print("Installing: " + module + " module")
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                install_module,
                "--no-warn-script-location",
                "--user",
            ]
        )
        try:
            exec("import " + module)
        except ModuleNotFoundError:
            return False
    return True


def install_modules(self):
    app_path = site.USER_SITE
    if app_path not in sys.path:
        sys.path.append(app_path)
    pybin = sys.executable

    print("Ensuring: pip")
    try:
        subprocess.call([pybin, "-m", "ensurepip"])
        subprocess.call([pybin, "-m", "pip", "install", "--upgrade", "pip"])
    except ImportError:
        pass
    try:
        exec("import torch")
    except ModuleNotFoundError:
        app_path = site.USER_SITE
        if app_path not in sys.path:
            sys.path.append(app_path)
        pybin = sys.executable
        self.report({"INFO"}, "Installing: torch module.")
        print("Installing: torch module")
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
                "--no-warn-script-location",
                "--user",
            ]
        )
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
                "--no-warn-script-location",
                "--user",
            ]
        )
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
                "--no-warn-script-location",
                "--user",
            ]
        )
    import_module(self, "PySoundFile", "PySoundFile")  # Sox for Linux pip install sox
    import_module(self, "diffusers", "diffusers")
    import_module(self, "accelerate", "accelerate")
    import_module(self, "transformers", "transformers")
    import_module(self, "opencv_python", "opencv_python")


class SEQUENCER_OT_generate_movie(Operator):
    """Text to Video"""

    bl_idname = "sequencer.generate_movie"
    bl_label = "Prompt"
    bl_description = "Convert text to video"
    bl_options = {"REGISTER", "UNDO"}

    #    generate_movie_prompt: bpy.props.StringProperty(
    #        name="generate_movie_prompt", default=""
    #    )

    def execute(self, context):
        if not bpy.types.Scene.generate_movie_prompt:
            return {"CANCELLED"}
        scene = context.scene

        install_modules(self)
        import torch
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import export_to_video

        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16",
        )  # "damo-vilab/text-to-video-ms-1.7b"
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        prompt = context.scene.generate_movie_prompt
        video_frames = pipe(prompt, num_inference_steps=25).frames
        video_path = export_to_video(video_frames)

        filepath = bpy.path.abspath(video_path)
        if os.path.isfile(filepath):
            strip = scene.sequence_editor.sequences.new_movie(
                name=context.scene.generate_movie_prompt,
                filepath=filepath,
                channel=1,
                frame_start=scene.frame_current,
            )
        else:
            print("Modelscope did not produce a file!")
        return {"FINISHED"}


class SEQUENCER_OT_generate_audio(Operator):
    """Text to Audio"""

    bl_idname = "sequencer.generate_audio"
    bl_label = "Prompt"
    bl_description = "Convert text to audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not bpy.types.Scene.generate_audio_prompt:
            return {"CANCELLED"}
        scene = context.scene

        install_modules(self)

        from diffusers import AudioLDMPipeline
        import torch

        repo_id = "cvssp/audioldm"
        pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        prompt = context.scene.generate_audio_prompt
        audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

        import scipy

        filename = clean_path(prompt + ".wav")
        scipy.io.wavfile.write(filename, rate=16000, data=audio)  ###

        filepath = bpy.path.abspath(filename)  ###
        if os.path.isfile(filepath):
            strip = scene.sequence_editor.sequences.new_sound(
                name=prompt,
                filepath=filepath,
                channel=1,
                frame_start=scene.frame_current,
            )
        else:
            print("No file was saved!")
        return {"FINISHED"}


class SEQEUNCER_PT_generate_movie(Panel):
    """Text to Video using ModelScope"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_movie_panel"
    bl_label = "Text to Video"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generate"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        row.prop(context.scene, "generate_movie_prompt", text="")
        row = layout.row()
        row.operator("sequencer.generate_movie", text="Generate Movie")


class SEQEUNCER_PT_generate_audio(Panel):
    """Text to Audio"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_audio_panel"
    bl_label = "Text to Audio"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generate"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        row.prop(context.scene, "generate_audio_prompt", text="")
        row = layout.row()
        row.operator("sequencer.generate_audio", text="Generate Audio")


classes = (
    SEQUENCER_OT_generate_movie,
    SEQUENCER_OT_generate_audio,
    SEQEUNCER_PT_generate_movie,
    SEQEUNCER_PT_generate_audio,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.generate_movie_prompt = bpy.props.StringProperty(
        name="generate_movie_prompt", default=""
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.generate_movie_prompt
    del bpy.types.Scene.generate_audio_prompt


if __name__ == "__main__":
    register()
