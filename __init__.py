# https://modelscope.cn/models/damo/text-to-video-synthesis/summary

bl_info = {
    "name": "Generative AI",
    "author": "tintwotin",
    "version": (1, 0),
    "blender": (3, 4, 0),
    "location": "Video Sequence Editor > Sidebar > Generative AI",
    "description": "Generate media in the VSE",
    "category": "Sequencer",
}

import bpy, ctypes
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
import site
import subprocess
import sys, os, aud
import string
from os.path import dirname, realpath, isfile
import shutil


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


def closest_divisible_64(num):
    # Determine the remainder when num is divided by 64
    remainder = num % 64

    # If the remainder is less than or equal to 32, return num - remainder,
    # but ensure the result is not less than 64
    if remainder <= 32:
        result = num - remainder
        return max(result, 192)
    # Otherwise, return num + (64 - remainder)
    else:
        return num + (64 - remainder)


def find_first_empty_channel(start_frame, end_frame):
    for ch in range(1, len(bpy.context.scene.sequence_editor.sequences_all) + 1):
        for seq in bpy.context.scene.sequence_editor.sequences_all:
            if (
                seq.channel == ch
                and seq.frame_final_start < end_frame 
                and (seq.frame_final_start + seq.frame_final_duration) > start_frame
            ):
                break
        else:
            return ch
    return 1


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
    import_module(self, "soundfile", "PySoundFile")  # Sox for Linux pip install sox
    import_module(self, "diffusers", "diffusers")
    import_module(self, "accelerate", "accelerate")
    import_module(self, "transformers", "transformers")
    import_module(self, "cv2", "opencv_python")


class GeneratorAddonPreferences(AddonPreferences):
    bl_idname = __name__

    soundselect: EnumProperty(
        name="Sound",
        items={
            ("ding", "Ding", "A simple bell sound"),
            ("coin", "Coin", "A Mario-like coin sound"),
            ("user", "User", "Load a custom sound file"),
        },
        default="ding",
    )

    default_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sounds','*.wav')
    if default_folder not in sys.path:
        sys.path.append(default_folder)

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

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.prop(self, "playsound")
        row = box.row()
        row.prop(self, "soundselect")
        if self.soundselect == "user":
            row.prop(self, "usersound", text="")
        row.operator("renderreminder.play_notification", text="", icon="PLAY")
        row.active = self.playsound


class GENERATOR_OT_sound_notification(Operator):
    """Test your notification settings"""

    bl_idname = "renderreminder.play_notification"
    bl_label = "Test Notification"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        if addon_prefs.playsound:
            device = aud.Device()

            def coinSound():
                sound = aud.Sound("")
                handle = device.play(
                    sound.triangle(1000)
                    .highpass(20)
                    .lowpass(2000)
                    .ADSR(0, 0.5, 1, 0)
                    .fadeout(0.1, 0.1)
                    .limit(0, 1)
                )

                handle = device.play(
                    sound.triangle(1500)
                    .highpass(20)
                    .lowpass(2000)
                    .ADSR(0, 0.5, 1, 0)
                    .fadeout(0.2, 0.2)
                    .delay(0.1)
                    .limit(0, 1)
                )

            def ding():
                sound = aud.Sound("")
                handle = device.play(
                    sound.triangle(3000)
                    .highpass(20)
                    .lowpass(1000)
                    .ADSR(0, 0.5, 1, 0)
                    .fadeout(0, 1)
                    .limit(0, 1)
                )

            if addon_prefs.soundselect == "ding":
                ding()
            if addon_prefs.soundselect == "coin":
                coinSound()
            if addon_prefs.soundselect == "user":
                file = str(addon_prefs.usersound)
                sound = aud.Sound(file)
                handle = device.play(sound)
        return {"FINISHED"}


class SEQUENCER_OT_generate_movie(Operator):
    """Generate Video"""

    bl_idname = "sequencer.generate_movie"
    bl_label = "Prompt"
    bl_description = "Convert text to video"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not bpy.types.Scene.generate_movie_prompt:
            return {"CANCELLED"}
        scene = context.scene
        seq_editor = scene.sequence_editor
        if not seq_editor:
            scene.sequence_editor_create()
        install_modules(self)

        import torch
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import export_to_video

        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        movie_x = scene.generate_movie_x
        movie_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_64(movie_x)
        y = scene.generate_movie_y = closest_divisible_64(movie_y)
        duration = scene.generate_movie_frames
        movie_num_inference_steps = scene.movie_num_inference_steps  

        wm = bpy.context.window_manager
        tot = scene.movie_num_batch
        wm.progress_begin(0, tot)

        for i in range(scene.movie_num_batch):

            wm.progress_update(i)
            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = scene.sequence_editor.active_strip.frame_final_start + scene.sequence_editor.active_strip.frame_final_duration
            else:
                empty_channel = find_first_empty_channel(scene.frame_current, (scene.movie_num_batch*duration)+scene.frame_current)
                start_frame = scene.frame_current

            # Options: https://huggingface.co/docs/diffusers/api/pipelines/text_to_video
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                variant="fp16",
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()

            # memory optimization
            pipe.enable_vae_slicing()

            video_frames = pipe(
                prompt, negative_prompt=negative_prompt, num_inference_steps=movie_num_inference_steps, height=y, width=x, num_frames=duration,
            ).frames
            src_path = export_to_video(video_frames)

            dst_path = dirname(realpath(__file__)) + "/" + os.path.basename(src_path)
            shutil.move(src_path, dst_path)
            if os.path.isfile(dst_path):
                strip = scene.sequence_editor.sequences.new_movie(
                    name=context.scene.generate_movie_prompt,
                    frame_start=start_frame,
                    filepath=dst_path,
                    channel=empty_channel,
                    fit_method="FIT",
                )
                scene.sequence_editor.active_strip = strip
            else:
                print("No resulting file found.")
        bpy.ops.renderreminder.play_notification()
        wm.progress_end()
        return {"FINISHED"}


class SEQEUNCER_PT_generate_movie(Panel):
    """Generate Video using AI"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_movie_panel"
    bl_label = "Generate Video"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generator"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        col = layout.column(align=True)
        row = col.row()
        row.scale_y = 1.2
        row.prop(context.scene, "generate_movie_prompt", text="", icon="ADD")
        row = col.row()
        row.scale_y = 1.2
        row.prop(context.scene, "generate_movie_negative_prompt", text="", icon="REMOVE")
        col = layout.column(align=True)
        row = col.row()
        row.prop(context.scene, "generate_movie_x", text="X")
        row.prop(context.scene, "generate_movie_frames", text="Frames")
        row = col.row()
        row.prop(context.scene, "generate_movie_y", text="Y")
        row.prop(context.scene, "movie_num_inference_steps", text="Inference")
        
        row = layout.row(align=True)
        row.scale_y = 1.1
        row.operator("sequencer.generate_movie", text="Generate")
        row.prop(context.scene, "movie_num_batch", text="")


class SEQUENCER_OT_generate_audio(Operator):
    """Generate Audio"""

    bl_idname = "sequencer.generate_audio"
    bl_label = "Prompt"
    bl_description = "Convert text to audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not bpy.types.Scene.generate_audio_prompt:
            return {"CANCELLED"}
        scene = context.scene
        seq_editor = scene.sequence_editor
        if not seq_editor:
            scene.sequence_editor_create()
        install_modules(self)

        from diffusers import AudioLDMPipeline
        import torch
        import scipy

        repo_id = "cvssp/audioldm"
        pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        prompt = context.scene.generate_audio_prompt
        # Options: https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm
        audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
        print(audio.tostring())
        filename = dirname(realpath(__file__)) + "/" + clean_path(prompt + ".wav")
        scipy.io.wavfile.write(filename, 48000, audio.transpose())

        filepath = filename
        if os.path.isfile(filepath):
            empty_channel = find_first_empty_channel(0, 10000000000)
            strip = scene.sequence_editor.sequences.new_sound(
                name=prompt,
                filepath=filepath,
                channel=empty_channel,
                frame_start=scene.frame_current,
            )
            scene.sequence_editor.active_strip = strip
        else:
            print("No resulting file found!")
        return {"FINISHED"}


class SEQEUNCER_PT_generate_audio(Panel):
    """Generate Audio with AI"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_audio_panel"
    bl_label = "Generate Audio"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generator"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        row.scale_y = 1.2
        row.prop(context.scene, "generate_audio_prompt", text="")
        row = layout.row()
        row.scale_y = 1.2
        row.operator("sequencer.generate_audio", text="Generate Audio")


classes = (
    SEQUENCER_OT_generate_movie,
    #SEQUENCER_OT_generate_audio,
    SEQEUNCER_PT_generate_movie,
    #SEQEUNCER_PT_generate_audio,
    GeneratorAddonPreferences,
    GENERATOR_OT_sound_notification,
    
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.generate_movie_prompt = bpy.props.StringProperty(
        name="generate_movie_prompt", default=""
    )
    bpy.types.Scene.generate_movie_negative_prompt = bpy.props.StringProperty(
        name="generate_movie_negative_prompt", default="text, watermark, copyright, blurry"
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x", default=512, step=64, min=192
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=256,
        step=64,
        min=192,
    )
    # The number of frames to be generated.
    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_y",
        default=16,
        min=1,
    )
    # The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.
    bpy.types.Scene.movie_num_inference_steps = bpy.props.IntProperty(
        name="movie_num_inference_steps",
        default=25,
        min=1,
    )
    # The number of videos to generate.
    bpy.types.Scene.movie_num_batch = bpy.props.IntProperty(
        name="movie_num_batch",
        default=1,
        min=1,
    )


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.generate_movie_prompt
    del bpy.types.Scene.generate_audio_prompt
    del bpy.types.Scene.generate_movie_x
    del bpy.types.Scene.generate_movie_y
    del bpy.types.Scene.movie_num_inference_steps
    del bpy.types.Scene.movie_num_batch


if __name__ == "__main__":
    register()
