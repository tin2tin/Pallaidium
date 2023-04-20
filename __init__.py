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

import bpy, ctypes, random
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
import site, platform
import subprocess
import sys, os, aud
import string
from os.path import dirname, realpath, isfile
import shutil
os_platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'

def show_system_console(show):
    if os_platform == "Windows":
        # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
        SW_HIDE = 0
        SW_SHOW = 5

        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW #if show else SW_HIDE
        )


def set_system_console_topmost(top):
    if os_platform == "Windows":
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


def clean_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    clean_filename = "".join(c if c in valid_chars else "_" for c in filename)
    return clean_filename


def clean_path(full_path):
    max_chars = 250
    full_path = full_path[:max_chars]
    dir_path, filename = os.path.split(full_path)
    cleaned_filename = clean_filename(filename)
    new_filename = cleaned_filename
    i = 1
    while os.path.exists(os.path.join(dir_path, new_filename)):
        name, ext = os.path.splitext(cleaned_filename)
        new_filename = f"{name}({i}){ext}"
        i += 1
    return os.path.join(dir_path, new_filename)


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
        if os_platform == "Windows":
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
        else:
            import_module(self, "torch", "torch")
    if os_platform == 'Darwin':
        import_module(self, "sox", "sex")
    else:
        import_module(self, "soundfile", "PySoundFile")
    import_module(self, "diffusers", "diffusers")
    import_module(self, "accelerate", "accelerate")
    import_module(self, "transformers", "transformers")
    import_module(self, "cv2", "opencv_python")
    import_module(self, "scipy", "scipy")
    import_module(self, "xformers", "xformers")


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

    default_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sounds", "*.wav"
    )
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
        box.operator("sequencer.install_generator")
        row = box.row(align=True)
        row.prop(self, "playsound", text="Notification")
        row.prop(self, "soundselect", text="")
        if self.soundselect == "user":
            row.prop(self, "usersound", text="")
        row.operator("renderreminder.play_notification", text="", icon="PLAY")
        row.active = self.playsound


class GENERATOR_OT_install(Operator):
    """Install all dependencies"""

    bl_idname = "sequencer.install_generator"
    bl_label = "Install Dependencies"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        install_modules(self)
        return {"FINISHED"}


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
                if os.path.isfile(file):
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

        show_system_console(True)
        set_system_console_topmost(True)

        scene = context.scene
        seq_editor = scene.sequence_editor
        if not seq_editor:
            scene.sequence_editor_create()
        try:
            import torch
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import export_to_video
        except ModuleNotFoundError:
            print("Dependencies needs to be installed in the add-on preferences.")
            self.report(
                {"INFO"},
                "Dependencies needs to be installed in the add-on preferences.",
            )
            return {"CANCELLED"}

        current_frame = scene.frame_current
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        movie_x = scene.generate_movie_x
        movie_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_64(movie_x)
        y = scene.generate_movie_y = closest_divisible_64(movie_y)
        duration = scene.generate_movie_frames
        movie_num_inference_steps = scene.movie_num_inference_steps
        movie_num_guidance = scene.movie_num_guidance

        #wm = bpy.context.window_manager
        #tot = scene.movie_num_batch
        #wm.progress_begin(0, tot)

        # Options: https://huggingface.co/docs/diffusers/api/pipelines/text_to_video
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            #"strangeman3107/animov-0.1.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # memory optimization
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        for i in range(scene.movie_num_batch):
            #wm.progress_update(i)
            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = (
                    scene.sequence_editor.active_strip.frame_final_start
                    + scene.sequence_editor.active_strip.frame_final_duration
                )
                scene.frame_current = (
                    scene.sequence_editor.active_strip.frame_final_start
                )
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * duration) + scene.frame_current,
                )
                start_frame = scene.frame_current

            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(0, 2147483647)
            )
            context.scene.movie_num_seed = seed

            # Use cuda if possible
            if torch.cuda.is_available():
                generator = (
                    torch.Generator("cuda").manual_seed(seed) if seed != 0 else None
                )
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None

            video_frames = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=movie_num_inference_steps,
                guidance_scale=movie_num_guidance,
                height=y,
                width=x,
                num_frames=duration,
                generator=generator,
            ).frames

            # Move to folder
            src_path = export_to_video(video_frames)
            dst_path = clean_path(dirname(realpath(__file__)) + "/" + os.path.basename(src_path))
            shutil.move(src_path, dst_path)

            # Add strip
            if os.path.isfile(dst_path):
                strip = scene.sequence_editor.sequences.new_movie(
                    name=context.scene.generate_movie_prompt + " " + str(seed),
                    frame_start=start_frame,
                    filepath=dst_path,
                    channel=empty_channel,
                    fit_method="FILL",
                )
                strip.transform.filter = 'SUBSAMPLING_3x3'
                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = (
                        scene.sequence_editor.active_strip.frame_final_start
                    )
            else:
                print("No resulting file found.")

            # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
            #bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        bpy.ops.renderreminder.play_notification()
        #wm.progress_end()
        scene.frame_current = current_frame

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"FINISHED"}


class SEQEUNCER_PT_generate_movie(Panel):
    """Generate Video using AI"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_movie_panel"
    bl_label = "Generative AI"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generative AI"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False
        scene = context.scene
        type = scene.generatorai_typeselect
        col = layout.column()
        col.prop(context.scene, "generatorai_typeselect", text="")

        layout = self.layout
        col = layout.column(align=True)
        col.use_property_split = True
        col.use_property_decorate = False
        col.scale_y = 1.2
        col.prop(context.scene, "generate_movie_prompt", text="", icon="ADD")
        col.prop(context.scene, "generate_movie_negative_prompt", text="", icon="REMOVE")
 
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False 
        if type == "movie" or type == "image":
            col = layout.column(align=True)
            col.prop(context.scene, "generate_movie_x", text="X")
            col.prop(context.scene, "generate_movie_y", text="Y")
        col = layout.column(align=True)
        if type == "movie" or type == "image":
            col.prop(context.scene, "generate_movie_frames", text="Frames")
        if type == "audio":
            col.prop(context.scene, "audio_length_in_f", text="Frames")
        col.prop(context.scene, "movie_num_inference_steps", text="Quality Steps")
        col.prop(context.scene, "movie_num_guidance", text="Word Power")
        if type == "movie" or type == "image":
            col.prop(context.scene, "movie_num_batch", text="Batch Count")

        if type == "movie" or type == "image":
            col = layout.column(align=True)
            row = col.row(align=True)
            sub_row = row.row(align=True)
            sub_row.prop(context.scene, "movie_num_seed", text="Seed")
            row.prop(context.scene, "movie_use_random", text="", icon="QUESTION")
            sub_row.active = not context.scene.movie_use_random

        row = layout.row(align=True)
        row.scale_y = 1.1
        if type == "movie":
            row.operator("sequencer.generate_movie", text="Generate")
        if type == "image":
            row.operator("sequencer.generate_image", text="Generate")
        if type == "audio":
            row.operator("sequencer.generate_audio", text="Generate")


class SEQUENCER_OT_generate_audio(Operator):
    """Generate Audio"""

    bl_idname = "sequencer.generate_audio"
    bl_label = "Prompt"
    bl_description = "Convert text to audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not bpy.types.Scene.generate_movie_prompt:
            self.report({"INFO"}, "Text prompt in the GeneratorAI tab is empty!")
            return {"CANCELLED"}
        scene = context.scene

        if not scene.sequence_editor:
            scene.sequence_editor_create()
        current_frame = scene.frame_current
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        movie_num_inference_steps = scene.movie_num_inference_steps
        movie_num_guidance = scene.movie_num_guidance
        audio_length_in_s = scene.audio_length_in_f/(scene.render.fps / scene.render.fps_base)

        try:
            from diffusers import AudioLDMPipeline
            import torch
            import scipy
        except ModuleNotFoundError:
            print("Dependencies needs to be installed in the add-on preferences.")
            self.report(
                {"INFO"},
                "Dependencies needs to be installed in the add-on preferences.",
            )
            return {"CANCELLED"}
        repo_id = "cvssp/audioldm"
        pipe = AudioLDMPipeline.from_pretrained(repo_id)  # , torch_dtype=torch.float16z

        # Use cuda if possible
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            
        for i in range(1):#scene.movie_num_batch): seed do not work for audio
            #wm.progress_update(i)
            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = (
                    scene.sequence_editor.active_strip.frame_final_start
                    + scene.sequence_editor.active_strip.frame_final_duration
                )
                scene.frame_current = (
                    scene.sequence_editor.active_strip.frame_final_start
                )
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * scene.audio_length_in_f) + scene.frame_current,
                )
                start_frame = scene.frame_current            

            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(0, 2147483647)
            )
            context.scene.movie_num_seed = seed

            # Use cuda if possible
            if torch.cuda.is_available():
                generator = (
                    torch.Generator("cuda").manual_seed(seed) if seed != 0 else None
                )
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None
            
            prompt = context.scene.generate_movie_prompt
            # Options: https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm
            audio = pipe(
                prompt,
                num_inference_steps=movie_num_inference_steps,
                audio_length_in_s=audio_length_in_s,
                guidance_scale=movie_num_guidance,
                generator=generator,
            ).audios[0]
            filename = clean_path(dirname(realpath(__file__)) + "/" + prompt + ".wav")
            scipy.io.wavfile.write(filename, 16000, audio.transpose())

            filepath = filename
            if os.path.isfile(filepath):
                empty_channel = empty_channel
                strip = scene.sequence_editor.sequences.new_sound(
                    name=prompt,
                    filepath=filepath,
                    channel=empty_channel,
                    frame_start=start_frame,
                )
                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = (
                        scene.sequence_editor.active_strip.frame_final_start
                    )
            else:
                print("No resulting file found!")
        
        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bpy.ops.renderreminder.play_notification()

        return {"FINISHED"}


#class SEQEUNCER_PT_generate_audio(Panel):
#    """Generate Audio with AI"""

#    bl_idname = "SEQUENCER_PT_sequencer_generate_audio_panel"
#    bl_label = "Generate Audio"
#    bl_space_type = "SEQUENCE_EDITOR"
#    bl_region_type = "UI"
#    bl_category = "Generative AI"

#    def draw(self, context):
#        layout = self.layout
#        scene = context.scene
#        row = layout.row()
#        row.scale_y = 1.2
#        row.prop(context.scene, "generate_audio_prompt", text="")
#        row = layout.row()
#        row.scale_y = 1.2
#        row.operator("sequencer.generate_audio", text="Generate Audio")


class SEQUENCER_OT_generate_image(Operator):
    """Generate Image"""

    bl_idname = "sequencer.generate_image"
    bl_label = "Prompt"
    bl_description = "Convert text to image"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not bpy.types.Scene.generate_movie_prompt:
            return {"CANCELLED"}

        show_system_console(True)
        set_system_console_topmost(True)

        scene = context.scene
        seq_editor = scene.sequence_editor
        if not seq_editor:
            scene.sequence_editor_create()
        try:
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            import torch
        except ModuleNotFoundError:
            print("Dependencies needs to be installed in the add-on preferences.")
            self.report(
                {"INFO"},
                "Dependencies needs to be installed in the add-on preferences.",
            )
            return {"CANCELLED"}

        current_frame = scene.frame_current
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        image_x = scene.generate_movie_x
        image_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_64(image_x)
        y = scene.generate_movie_y = closest_divisible_64(image_y)
        duration = scene.generate_movie_frames
        image_num_inference_steps = scene.movie_num_inference_steps
        image_num_guidance = scene.movie_num_guidance

        #wm = bpy.context.window_manager
        #tot = scene.movie_num_batch
        #wm.progress_begin(0, tot)

        # Options: https://huggingface.co/docs/diffusers/api/pipelines/text_to_video
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # memory optimization
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        for i in range(scene.movie_num_batch):
            #wm.progress_update(i)
            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = (
                    scene.sequence_editor.active_strip.frame_final_start
                    + scene.sequence_editor.active_strip.frame_final_duration
                )
                scene.frame_current = (
                    scene.sequence_editor.active_strip.frame_final_start
                )
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * duration) + scene.frame_current,
                )
                start_frame = scene.frame_current

            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(0, 2147483647)
            )
            context.scene.movie_num_seed = seed

            # Use cuda if possible
            if torch.cuda.is_available():
                generator = (
                    torch.Generator("cuda").manual_seed(seed) if seed != 0 else None
                )
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=image_num_inference_steps,
                guidance_scale=image_num_guidance,
                height=y,
                width=x,
                generator=generator,
            ).images[0]

            # Move to folder
            image.save("temp.png")
            #print(src_path)
            dst_path = clean_path(dirname(realpath(__file__)) + "/" + context.scene.generate_movie_prompt + ".png")
            shutil.move("temp.png", dst_path)

            # Add strip
            if os.path.isfile(dst_path):
                strip = scene.sequence_editor.sequences.new_image(
                    name=context.scene.generate_movie_prompt + " " + str(seed),
                    frame_start=start_frame,
                    filepath=dst_path,
                    channel=empty_channel,
                    fit_method="FILL",
                )
                strip.frame_final_duration = scene.generate_movie_frames
                strip.transform.filter = 'SUBSAMPLING_3x3'

                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = (
                        scene.sequence_editor.active_strip.frame_final_start
                    )
            else:
                print("No resulting file found.")

            # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
            #bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        bpy.ops.renderreminder.play_notification()
        #wm.progress_end()
        scene.frame_current = current_frame

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"FINISHED"}


class SEQUENCER_OT_strip_to_generatorAI(Operator):
    """Convert selected text strips to GeneratorAI"""

    bl_idname = "sequencer.text_to_generator"
    bl_label = "Convert Text Strips to GeneratorAI"
    bl_options = {"INTERNAL"}
    bl_description = "Adds selected text strips as GeneratorAI strips"

    @classmethod
    def poll(cls, context):
        return context.scene and context.scene.sequence_editor

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        play_sound = addon_prefs.playsound
        addon_prefs.playsound = False
        scene = context.scene
        sequencer = bpy.ops.sequencer
        sequences = bpy.context.sequences
        strips = context.selected_sequences
        prompt = scene.generate_movie_prompt
        current_frame = scene.frame_current
        type = scene.generatorai_typeselect
        for strip in strips:
            if strip.type == "TEXT":
                if strip.text:
                    print("Processing: " + strip.text)
                    scene.generate_movie_prompt = strip.text
                    scene.frame_current = strip.frame_final_start
                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
        scene.frame_current = current_frame
        context.scene.generate_movie_prompt = prompt
        addon_prefs.playsound = play_sound
        bpy.ops.renderreminder.play_notification()

        return {"FINISHED"}


def panel_text_to_generatorAI(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(
        "sequencer.text_to_generator", text="Text to GeneratorAI", icon="SHADERFX"
    )


classes = (
    SEQUENCER_OT_generate_movie,
    SEQUENCER_OT_generate_audio,
    SEQUENCER_OT_generate_image,
    SEQEUNCER_PT_generate_movie,
    # SEQEUNCER_PT_generate_audio,
    GeneratorAddonPreferences,
    GENERATOR_OT_sound_notification,
    SEQUENCER_OT_strip_to_generatorAI,
    GENERATOR_OT_install,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.generate_movie_prompt = bpy.props.StringProperty(
        name="generate_movie_prompt", default=""
    )
    bpy.types.Scene.generate_movie_negative_prompt = bpy.props.StringProperty(
        name="generate_movie_negative_prompt",
        default="text, watermark, copyright, blurry, grainy, copyright",
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x",
        default=512,
        step=64,
        min=192,
        max=1024,
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=256,
        step=64,
        min=192,
        max=1024,
    )
    # The number of frames to be generated.
    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_y",
        default=16,
        min=1,
        max=125,
    )
    # The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.
    bpy.types.Scene.movie_num_inference_steps = bpy.props.IntProperty(
        name="movie_num_inference_steps",
        default=25,
        min=1,
        max=100,
    )
    # The number of videos to generate.
    bpy.types.Scene.movie_num_batch = bpy.props.IntProperty(
        name="movie_num_batch",
        default=1,
        min=1,
        max=100,
    )
    # The seed number.
    bpy.types.Scene.movie_num_seed = bpy.props.IntProperty(
        name="movie_num_seed",
        default=1,
        min=1,
        max=2147483647,
    )

    # The seed number.
    bpy.types.Scene.movie_use_random = bpy.props.BoolProperty(
        name="movie_use_random",
        default=0,
    )

    # The seed number.
    bpy.types.Scene.movie_num_guidance = bpy.props.IntProperty(
        name="movie_num_guidance",
        default=17,
        min=1,
        max=100,
    )

    # The frame ausio duration.
    bpy.types.Scene.audio_length_in_f = bpy.props.IntProperty(
        name="audio_length_in_f",
        default=80,
        min=1,
        max=10000,
    )

    bpy.types.Scene.generatorai_typeselect = bpy.props.EnumProperty(
        name="Sound",
        items={
            ("movie", "Video", "Generate Video"),
            ("image", "Image", "Generate Image"),
            ("audio", "Audio", "Generate Audio"),
        },
        default="movie",
    )

    bpy.types.SEQUENCER_MT_add.append(panel_text_to_generatorAI)


def unregister():
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
    bpy.types.SEQUENCER_MT_add.remove(panel_text_to_generatorAI)


if __name__ == "__main__":
    register()
