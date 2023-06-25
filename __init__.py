# https://modelscope.cn/models/damo/text-to-video-synthesis/summary

bl_info = {
    "name": "Generative AI",
    "author": "tintwotin",
    "version": (1, 2),
    "blender": (3, 4, 0),
    "location": "Video Sequence Editor > Sidebar > Generative AI",
    "description": "Generate media in the VSE",
    "category": "Sequencer",
}

import bpy, ctypes, random
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, FloatProperty
import site, platform
import subprocess
import sys, os, aud, re
import string
from os.path import dirname, realpath, isdir, join, basename
import shutil
os_platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'


# not working
def get_active_device_vram():
    active_scene = bpy.context.scene
    active_view_layer = active_scene.view_layers.active
    active_view_layer.use_gpu_select = True  # Enable GPU selection in the view layer

    # Iterate over available GPU devices
    for gpu_device in bpy.context.preferences.system.compute_device:
        if gpu_device.type == 'CUDA':  # Only consider CUDA devices
            if gpu_device.use:
                return gpu_device.memory_total

    return None


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


def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.,\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still within the max length
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.,':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    return rv


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
    filename = filename[:50]
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    clean_filename = "".join(c if c in valid_chars else "_" for c in filename)
    clean_filename = clean_filename.replace('\n', ' ')
    clean_filename = clean_filename.replace('\r', ' ')

    return clean_filename.strip()


def create_folder(folderpath):
    if not isdir(folderpath):
        os.makedirs(folderpath, exist_ok=True)
    return folderpath


def clean_path(full_path):
    preferences = bpy.context.preferences
    addon_prefs = preferences.addons[__name__].preferences
    name, ext = os.path.splitext(full_path)
    dir_path, filename = os.path.split(name)
    dir_path = create_folder(addon_prefs.generator_ai)
    cleaned_filename = clean_filename(filename)
    new_filename = cleaned_filename + ext
    i = 1
    while os.path.exists(os.path.join(dir_path, new_filename)):
        name, ext = os.path.splitext(new_filename)
        new_filename = f"{name.rsplit('(', 1)[0]}({i}){ext}"
        i += 1
    return os.path.join(dir_path, new_filename)


def limit_string(my_string):
    if len(my_string) > 77:
        print("Warning: String is longer than 77 characters. Excessive string:", my_string[77:])
        return my_string[:77]
    else:
        return my_string


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
            import_module(self, "torchvision", "torchvision")
            import_module(self, "torchaudio", "torchaudio")
    if os_platform == 'Darwin' or os_platform == 'Linux':
        import_module(self, "sox", "sox")
    else:
        import_module(self, "soundfile", "PySoundFile")
    import_module(self, "diffusers", "diffusers") #git+https://github.com/huggingface/diffusers.git")
    import_module(self, "accelerate", "accelerate")
    import_module(self, "transformers", "transformers")
    import_module(self, "sentencepiece", "sentencepiece")
    import_module(self, "safetensors", "safetensors")
    import_module(self, "cv2", "opencv_python")
    import_module(self, "scipy", "scipy")
    import_module(self, "IPython", "IPython")
    import_module(self, "bark", "git+https://github.com/suno-ai/bark.git")
    import_module(self, "xformers", "xformers")
    #subprocess.check_call([pybin,"-m","pip","install","force-reinstall","no-deps","pre xformers"])
    subprocess.check_call([pybin,"-m","pip","install","numpy","--upgrade"])
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


def get_module_dependencies(module_name):
    """
    Get the list of dependencies for a given module.
    """

    app_path = site.USER_SITE
    if app_path not in sys.path:
        sys.path.append(app_path)
    pybin = sys.executable

    result = subprocess.run([pybin,'-m' , 'pip', 'show', module_name], capture_output=True, text=True)
    output = result.stdout.strip()

    dependencies = []
    for line in output.split('\n'):
        if line.startswith('Requires:'):
            dependencies = line.split(':')[1].strip().split(', ')
            break
    return dependencies


def uninstall_module_with_dependencies(module_name):
    """
    Uninstall a module and its dependencies.
    """

    show_system_console(True)
    set_system_console_topmost(True)

    app_path = site.USER_SITE
    if app_path not in sys.path:
        sys.path.append(app_path)
    pybin = sys.executable

    dependencies = get_module_dependencies(module_name)

    # Uninstall the module
    subprocess.run([pybin, '-m', 'pip', 'uninstall', '-y', module_name])

    # Uninstall the dependencies
    for dependency in dependencies:
        subprocess.run([pybin, '-m', 'pip', 'uninstall', '-y', dependency])
        
    subprocess.check_call([pybin,"-m","pip","install","numpy"])


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

    movie_model_card: bpy.props.EnumProperty(
        name="Video Model Card",
        items=[
            ("strangeman3107/animov-0.1.1", "Animov (448x384)", "Animov (448x384)"),
            ("strangeman3107/animov-512x", "Animov (512x512)", "Animov (512x512)"),
            ("polyware-ai/longscope", "Longscope (384x216x94)", "Longscope ( 384x216x94)"),
            #("vdo/potat1-lotr-25000/", "LOTR (1024x576x24)", "LOTR (1024x576x24)"),
            ("damo-vilab/text-to-video-ms-1.7b", "Modelscope (256x256)", "Modelscope (256x256)"),
            ("polyware-ai/text-to-video-ms-stable-v1", "Polyware 1.7b (384x384)", "Polyware 1.7b (384x384)"),
            ("camenduru/potat1", "Potat v1 (1024x576)", "Potat (1024x576)"),
            # ("cerspense/zeroscope_v1-1_320s", "Zeroscope v1.1 (320x320)", "Zeroscope (320x320)"),
            ("cerspense/zeroscope_v2_dark_30x448x256", "Zeroscope (448x256x30)", "Zeroscope (448x256x30)"),
            ("cerspense/zeroscope_v2_576w", "Zeroscope (576x320x24)", "Zeroscope (576x320x24)"),
            ("cerspense/zeroscope_v2_XL", "Zeroscope XL (1024x576x24)", "Zeroscope XL (1024x576x24)"),
            #("vdo/potat1-50000", "Potat v1 50000 (1024x576)", "Potat (1024x576)"),
        ],
        default="cerspense/zeroscope_v2_dark_30x448x256",
    )

    image_model_card: bpy.props.EnumProperty(
        name="Image Model Card",
        items=[
            ("runwayml/stable-diffusion-v1-5", "Stable Diffusion 1.5 (512x512)", "Stable Diffusion 1.5"),
            ("stabilityai/stable-diffusion-2", "Stable Diffusion 2 (768x768)", "Stable Diffusion 2"),
            ("DeepFloyd/IF-I-M-v1.0", "DeepFloyd/IF-I-M-v1.0", "DeepFloyd"),
        ],
        default="stabilityai/stable-diffusion-2",
    )

    audio_model_card: bpy.props.EnumProperty(
        name="Audio Model Card",
        items=[
            ("cvssp/audioldm-s-full-v2", "AudioLDM S Full v2", "AudioLDM Small Full v2"),
            #("cvssp/audioldm", "AudioLDM", "AudioLDM"),
            ("bark", "Bark", "Bark"),
        ],
        default="bark",
    )

    hugginface_token: bpy.props.StringProperty(
        name="Hugginface Token",
        default="hugginface_token",
        subtype = "PASSWORD",
    )

    generator_ai: StringProperty(
        name = "Filepath",
        description = "Path to the folder where the generated files are stored",
        subtype = 'DIR_PATH',
        default = join(bpy.utils.user_resource('DATAFILES'), "Generator AI")
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.operator("sequencer.install_generator")
        row.operator("sequencer.uninstall_generator")
        box.prop(self, "movie_model_card")
        box.prop(self, "image_model_card")
        if self.image_model_card == "DeepFloyd/IF-I-M-v1.0":
            row = box.row(align=True)
            row.prop(self, "hugginface_token")
            row.operator("wm.url_open", text="", icon='URL').url = "https://huggingface.co/settings/tokens"
        box.prop(self, "audio_model_card")
        box.prop(self, "generator_ai")
        row = box.row(align=True)
        row.label(text="Notification:")
        row.prop(self, "playsound", text="")
        sub_row = row.row()
        sub_row.prop(self, "soundselect", text="")
        if self.soundselect == "user":
            sub_row.prop(self, "usersound", text="")
        sub_row.operator("renderreminder.play_notification", text="", icon="PLAY")
        sub_row.active = self.playsound


class GENERATOR_OT_install(Operator):
    """Install all dependencies"""

    bl_idname = "sequencer.install_generator"
    bl_label = "Install Dependencies"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        install_modules(self)
        self.report(
            {"INFO"},
            "Installation of dependencies is finished.",
        )
        return {"FINISHED"}


class GENERATOR_OT_uninstall(Operator):
    """Unnstall all dependencies"""

    bl_idname = "sequencer.uninstall_generator"
    bl_label = "Uninstall Dependencies"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences

        uninstall_module_with_dependencies("torch")
        uninstall_module_with_dependencies("torchvision")
        uninstall_module_with_dependencies("torchaudio")

        if os_platform == 'Darwin' or os_platform == 'Linux':
            uninstall_module_with_dependencies("sox")
        else:
            uninstall_module_with_dependencies("PySoundFile")
        uninstall_module_with_dependencies("diffusers")
        uninstall_module_with_dependencies("accelerate")
        uninstall_module_with_dependencies("transformers")
        uninstall_module_with_dependencies("sentencepiece")
        uninstall_module_with_dependencies("safetensors")
        uninstall_module_with_dependencies("opencv_python")
        uninstall_module_with_dependencies("scipy")
        uninstall_module_with_dependencies("IPython")
        uninstall_module_with_dependencies("bark")
        uninstall_module_with_dependencies("xformers")
        
        self.report(
            {"INFO"},
            "\nRemove AI Models manually: \nOn Linux and macOS: ~/.cache/huggingface/transformers\nOn Windows: %userprofile%.cache\\huggingface\\transformers",
        )
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


class SEQEUNCER_PT_generate_ai(Panel):
    """Generate Media using AI"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_movie_panel"
    bl_label = "Generative AI"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generative AI"

    def draw(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        audio_model_card = addon_prefs.audio_model_card

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

        if type == "audio" and audio_model_card == "bark":
            pass
        else:
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
        if type == "audio" and audio_model_card != "bark":
            col.prop(context.scene, "audio_length_in_f", text="Frames")

        if type == "audio" and audio_model_card == "bark":
            col = layout.column(align=True)
            col.prop(context.scene, "speakers", text="Speaker")
            col.prop(context.scene, "languages", text="Language")
        else:
            col.prop(context.scene, "movie_num_inference_steps", text="Quality Steps")
            col.prop(context.scene, "movie_num_guidance", text="Word Power")

            col = layout.column()
            row = col.row(align=True)
            sub_row = row.row(align=True)
            sub_row.prop(context.scene, "movie_num_seed", text="Seed")
            row.prop(context.scene, "movie_use_random", text="", icon="QUESTION")
            sub_row.active = not context.scene.movie_use_random

        col.prop(context.scene, "movie_num_batch", text="Batch Count")

        row = layout.row(align=True)
        row.scale_y = 1.1
        if type == "movie":
            row.operator("sequencer.generate_movie", text="Generate")
        if type == "image":
            row.operator("sequencer.generate_image", text="Generate")
        if type == "audio":
            row.operator("sequencer.generate_audio", text="Generate")


class SEQUENCER_OT_generate_movie(Operator):
    """Generate Video"""

    bl_idname = "sequencer.generate_movie"
    bl_label = "Prompt"
    bl_description = "Convert text to video"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        scene = context.scene
        if not scene.generate_movie_prompt:
            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
            return {"CANCELLED"}

        show_system_console(True)
        set_system_console_topmost(True)

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

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_frame = scene.frame_current
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt + " nsfw nude nudity"
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

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        movie_model_card = addon_prefs.movie_model_card

        # Options: https://huggingface.co/docs/diffusers/api/pipelines/text_to_video
        pipe = DiffusionPipeline.from_pretrained(
            movie_model_card,
            #"strangeman3107/animov-0.1.1",
            #"damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # memory optimization
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        #pipe.enable_xformers_memory_efficient_attention()

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
                else random.randint(0, 999999)
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
            if not os.path.isfile(dst_path):
                print("No resulting file found.")
                return {"CANCELLED"}

            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == "SEQUENCE_EDITOR":
                        from bpy import context

                        with context.temp_override(window=window, area=area):
                            bpy.ops.sequencer.movie_strip_add(filepath=dst_path,
                                                              frame_start=start_frame,
                                                              channel=empty_channel,
                                                              fit_method="FIT",
                                                              adjust_playback_rate=True,
                                                              sound=False,
                                                              use_framerate = False,
                                                              )
                            strip = scene.sequence_editor.active_strip
                            strip.transform.filter = 'SUBSAMPLING_3x3'
                            scene.sequence_editor.active_strip = strip
                            strip.use_proxy = True
                            strip.name = str(seed)+"_"+prompt
                            bpy.ops.sequencer.rebuild_proxy()
                            if i > 0:
                                scene.frame_current = (
                                    scene.sequence_editor.active_strip.frame_final_start
                                )
                            # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
                            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                            break

            # clear the VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        bpy.ops.renderreminder.play_notification()
        #wm.progress_end()
        scene.frame_current = current_frame

        return {"FINISHED"}


class SEQUENCER_OT_generate_audio(Operator):
    """Generate Audio"""

    bl_idname = "sequencer.generate_audio"
    bl_label = "Prompt"
    bl_description = "Convert text to audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        if not scene.generate_movie_prompt:
            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
            return {"CANCELLED"}

        if not scene.sequence_editor:
            scene.sequence_editor_create()

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences

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
            #from bark import SAMPLE_RATE, generate_audio, preload_models
            from IPython.display import Audio
            from scipy.io.wavfile import write as write_wav
            import xformers

            if addon_prefs.audio_model_card == "bark":
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                import numpy as np
                from bark.generation import (
                    generate_text_semantic,
                    preload_models,
                )
                from bark.api import semantic_to_waveform
                from bark import generate_audio, SAMPLE_RATE
        except ModuleNotFoundError:
            print("Dependencies needs to be installed in the add-on preferences.")
            self.report(
                {"INFO"},
                "Dependencies needs to be installed in the add-on preferences.",
            )
            return {"CANCELLED"}

        show_system_console(True)
        set_system_console_topmost(True)

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if addon_prefs.audio_model_card != "bark":
            repo_id = addon_prefs.audio_model_card
            pipe = AudioLDMPipeline.from_pretrained(repo_id)  # , torch_dtype=torch.float16z

            # Use cuda if possible
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
        else: #bark
            preload_models(
            text_use_small=True,
            coarse_use_small=True,
            fine_use_gpu=True,
            fine_use_small=True,
        )

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
                    100000000000000000000,
                )
                start_frame = scene.frame_current

            if addon_prefs.audio_model_card == "bark":

                rate = 24000
                GEN_TEMP = 0.6
                SPEAKER = "v2/"+scene.languages + "_" + scene.speakers #"v2/"+
                silence = np.zeros(int(0.25 * rate))  # quarter second of silence

                prompt = context.scene.generate_movie_prompt
                prompt = prompt.replace("\n", " ").strip()

                sentences = split_and_recombine_text(prompt, desired_length=90, max_length=150)

                pieces = []
                for sentence in sentences:
                    print(sentence)
                    semantic_tokens = generate_text_semantic(
                        sentence,
                        history_prompt=SPEAKER,
                        temp=GEN_TEMP,
                        #min_eos_p=0.1,  # this controls how likely the generation is to end
                    )

                    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
                    pieces += [audio_array, silence.copy()]

                audio = np.concatenate(pieces) #Audio(np.concatenate(pieces), rate=rate)
                filename = clean_path(dirname(realpath(__file__)) + "/" + prompt + ".wav")

                # Write the combined audio to a file
                write_wav(filename, rate, audio.transpose())

            else: # AudioLDM
                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 999999)
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
                rate = 16000

                filename = clean_path(dirname(realpath(__file__)) + "/" + prompt + ".wav")
                write_wav(filename, rate, audio.transpose()) #.transpose()

            filepath = filename
            if os.path.isfile(filepath):
                empty_channel = empty_channel
                strip = scene.sequence_editor.sequences.new_sound(
                    name = prompt,
                    filepath=filepath,
                    channel=empty_channel,
                    frame_start=start_frame,
                )
                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = (
                        scene.sequence_editor.active_strip.frame_final_start
                    )
                # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            else:
                print("No resulting file found!")

            # clear the VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        bpy.ops.renderreminder.play_notification()

        return {"FINISHED"}


class SEQUENCER_OT_generate_image(Operator):
    """Generate Image"""

    bl_idname = "sequencer.generate_image"
    bl_label = "Prompt"
    bl_description = "Convert text to image"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        if scene.generate_movie_prompt == "":
            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
            return {"CANCELLED"}

        show_system_console(True)
        set_system_console_topmost(True)

        scene = context.scene
        seq_editor = scene.sequence_editor

        if not seq_editor:
            scene.sequence_editor_create()

        try:
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import pt_to_pil
            import torch
        except ModuleNotFoundError:
            print("Dependencies needs to be installed in the add-on preferences.")
            self.report(
                {"INFO"},
                "Dependencies needs to be installed in the add-on preferences.",
            )
            return {"CANCELLED"}

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_frame = scene.frame_current
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt + " nsfw nude nudity"
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

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        image_model_card = addon_prefs.image_model_card

        if image_model_card == "DeepFloyd/IF-I-M-v1.0":
            from huggingface_hub.commands.user import login
            result = login(token = addon_prefs.hugginface_token)
            
            torch.cuda.set_per_process_memory_fraction(0.85)  # 6 GB VRAM

            # stage 1
            stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16)
            # stage_1.enable_model_cpu_offload()
            stage_1.enable_sequential_cpu_offload() # 6 GB VRAM

            # stage 2
            stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
            )
            stage_2.enable_model_cpu_offload()

            # stage 3
            safety_modules = {
                "feature_extractor": stage_1.feature_extractor,
                "safety_checker": stage_1.safety_checker,
                "watermarker": stage_1.watermarker,
            }
            stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
            )
            stage_3.enable_model_cpu_offload()
            
        else: # stable Diffusion
            pipe = DiffusionPipeline.from_pretrained(
                image_model_card,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            # memory optimization
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.enable_xformers_memory_efficient_attention()

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
                else random.randint(0, 999999)
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

            if image_model_card == "DeepFloyd/IF-I-M-v1.0":
                prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, negative_prompt)
                
                # stage 1
                image = stage_1(
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
                ).images
                pt_to_pil(image)[0].save("./if_stage_I.png")

                # stage 2
                image = stage_2(
                    image=image,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images
                pt_to_pil(image)[0].save("./if_stage_II.png")

                # stage 3
                image = stage_3(prompt=prompt, image=image, noise_level=100, generator=generator).images
                # image[0].save("./if_stage_III.png")
                image = image[0]

            else: # Stable Diffusion
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
            filename = clean_filename(str(seed)+"_"+context.scene.generate_movie_prompt)
            out_path = clean_path(dirname(realpath(__file__))+"/"+filename+".png")
            image.save(out_path)

            # Add strip
            if os.path.isfile(out_path):
                strip = scene.sequence_editor.sequences.new_image(
                    name = str(seed)+"_"+context.scene.generate_movie_prompt,
                    frame_start=start_frame,
                    filepath=out_path,
                    channel=empty_channel,
                    fit_method="FIT",
                )
                strip.frame_final_duration = scene.generate_movie_frames
                strip.transform.filter = 'SUBSAMPLING_3x3'

                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = (
                        scene.sequence_editor.active_strip.frame_final_start
                    )
                strip.use_proxy = True
                bpy.ops.sequencer.rebuild_proxy()

                # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            else:
                print("No resulting file found.")
                
            # clear the VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        bpy.ops.renderreminder.play_notification()
        #wm.progress_end()
        scene.frame_current = current_frame

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"FINISHED"}


class SEQUENCER_OT_strip_to_generatorAI(Operator):
    """Convert selected text strips to Generative AI"""

    bl_idname = "sequencer.text_to_generator"
    bl_label = "Convert Text Strips to Generative AI"
    bl_options = {"INTERNAL"}
    bl_description = "Adds selected text strips as Generative AI strips"

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
                    print("Processing: " + strip.text+", "+prompt)
                    scene.generate_movie_prompt = strip.text+", "+prompt
                    scene.frame_current = strip.frame_final_start
                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()
                    scene.generate_movie_prompt = prompt
                    
        scene.frame_current = current_frame
        scene.generate_movie_prompt = prompt
        addon_prefs.playsound = play_sound
        bpy.ops.renderreminder.play_notification()

        return {"FINISHED"}


def panel_text_to_generatorAI(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(
        "sequencer.text_to_generator", text="Text to Generative AI", icon="SHADERFX"
    )


classes = (
    GeneratorAddonPreferences,
    SEQUENCER_OT_generate_movie,
    SEQUENCER_OT_generate_audio,
    SEQUENCER_OT_generate_image,
    SEQEUNCER_PT_generate_ai,
    GENERATOR_OT_sound_notification,
    SEQUENCER_OT_strip_to_generatorAI,
    GENERATOR_OT_install,
    GENERATOR_OT_uninstall,
)


def register():

    bpy.types.Scene.generate_movie_prompt = bpy.props.StringProperty(
        name="generate_movie_prompt", default="high quality, masterpiece, slow motion, 4k"
    )
    bpy.types.Scene.generate_movie_negative_prompt = bpy.props.StringProperty(
        name="generate_movie_negative_prompt",
        default="low quality, windy, flicker, jitter",
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x",
        default=448,
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
        name="generate_movie_frames",
        default=18,
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
        default=1,
    )

    # The seed number.
    bpy.types.Scene.movie_num_guidance = bpy.props.FloatProperty(
        name="movie_num_guidance",
        default=15.0,
        min=1,
        max=100,
    )

    # The frame audio duration.
    bpy.types.Scene.audio_length_in_f = bpy.props.IntProperty(
        name="audio_length_in_f",
        default=80,
        min=1,
        max=10000,
    )

    bpy.types.Scene.generatorai_typeselect = bpy.props.EnumProperty(
        name="Sound",
        items=[
            ("movie", "Video", "Generate Video"),
            ("image", "Image", "Generate Image"),
            ("audio", "Audio", "Generate Audio"),
        ],
        default="movie",
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
        default="en"
    )

    for cls in classes:
        bpy.utils.register_class(cls)

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
