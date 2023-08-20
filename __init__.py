# https://modelscope.cn/models/damo/text-to-video-synthesis/summary

bl_info = {
    "name": "Generative AI",
    "author": "tintwotin",
    "version": (1, 4),
    "blender": (3, 4, 0),
    "location": "Video Sequence Editor > Sidebar > Generative AI",
    "description": "Generate media in the VSE",
    "category": "Sequencer",
}

import bpy, ctypes, random
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
)
import site, platform, json
import subprocess
import sys, os, aud, re
import string
from os.path import dirname, realpath, isdir, join, basename
import shutil
from datetime import date



os_platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'


def show_system_console(show):
    if os_platform == "Windows":
        # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
        SW_HIDE = 0
        SW_SHOW = 5

        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW  # if show else SW_HIDE
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
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

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
                while c not in "!?.,\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in "!?\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?.,"
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv


def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    if numbers:
        return int(numbers[0])
    else:
        return None


def load_styles(json_filename):
    styles_array = []

    try:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"JSON file '{json_filename}' not found.")
        data = []

    for item in data:
        name = item["name"]
        prompt = item["prompt"]
        negative_prompt = item["negative_prompt"]
        styles_array.append((negative_prompt.lower().replace(" ", "_"), name.title(), prompt))

    return styles_array


def style_prompt(prompt):
    selected_entry_key = bpy.context.scene.generatorai_styles

    return_array = []
   
    if selected_entry_key:
        styles_array = load_styles(os.path.dirname(os.path.abspath(__file__))+"/styles.json")

        if selected_entry_key:
            selected_entry = next((item for item in styles_array if item[0] == selected_entry_key), None)

            if selected_entry:
                selected_entry_list = list(selected_entry)
                return_array.append(selected_entry_list[2].replace("{prompt}", prompt))
                return_array.append(selected_entry_list[0].replace("_", " "))
                return return_array

    return_array.append(bpy.context.scene.generate_movie_prompt)
    return_array.append(bpy.context.scene.generate_movie_negative_prompt)
    return return_array


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


#def ensure_divisible_by_64(value):
#    remainder = value % 64
#    if remainder != 0:
#        value += 64 - remainder
#    return value


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
    valid_chars = "-_,.() %s%s" % (string.ascii_letters, string.digits)
    clean_filename = "".join(c if c in valid_chars else "_" for c in filename)
    clean_filename = clean_filename.replace("\n", " ")
    clean_filename = clean_filename.replace("\r", " ")
    clean_filename = clean_filename.replace(" ", "_")

    return clean_filename.strip()


def create_folder(folderpath):
    try:
        os.makedirs(folderpath)
        return True
    except FileExistsError:
       # directory already exists
        pass
        return False

def solve_path(full_path):
    preferences = bpy.context.preferences
    addon_prefs = preferences.addons[__name__].preferences
    name, ext = os.path.splitext(full_path)
    dir_path, filename = os.path.split(name)
    dir_path = addon_prefs.generator_ai+"\\"+str(date.today())
    create_folder(dir_path)
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
        print(
            "Warning: String is longer than 77 characters. Excessive string:",
            my_string[77:],
        )
        return my_string[:77]
    else:
        return my_string


# Function to load a video as a NumPy array
def load_video_as_np_array(video_path):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Error opening video file")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def process_frames(frame_folder_path, target_width):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import cv2

    processed_frames = []

    # List all image files in the folder
    image_files = sorted(
        [f for f in os.listdir(frame_folder_path) if f.endswith(".png")]
    )

    for image_file in image_files:
        image_path = os.path.join(frame_folder_path, image_file)
        img = Image.open(image_path)

        # Process the image (resize and convert to RGB)
        frame_width, frame_height = img.size
        #target_width = 512
        target_height = int((target_width / frame_width) * frame_height)

        # Ensure width and height are divisible by 64
        target_width = closest_divisible_64(target_width)
        target_height = closest_divisible_64(target_height)

        img = img.resize((target_width, target_height), Image.ANTIALIAS)
        img = img.convert("RGB")

        processed_frames.append(img)
    return processed_frames


def process_video(input_video_path, output_video_path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import cv2
    import shutil

    # Create a temporary folder for storing frames
    temp_image_folder = solve_path("temp_images")
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Save each loaded frame as an image in the temp folder
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Save the frame as an image in the temp folder
        temp_image_path = os.path.join(temp_image_folder, f"frame_{i:04d}.png")
        cv2.imwrite(temp_image_path, frame)
    cap.release()

    # Process frames using the separate function
    processed_frames = process_frames(temp_image_folder, 1024)
    # print("Temp folder: "+temp_image_folder)

    # Clean up: Delete the temporary image folder
    shutil.rmtree(temp_image_folder)

    return processed_frames


def process_image(image_path, frames_nr):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import cv2, shutil

    img = cv2.imread(image_path)

    # Create a temporary folder for storing frames
    temp_image_folder = solve_path("/temp_images")
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)

    # Add zoom motion to the image and save frames
    zoom_factor = 1.0
    for i in range(frames_nr):
        zoomed_img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        output_path = os.path.join(temp_image_folder, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, zoomed_img)
        zoom_factor += 1.0

    # Process frames using the separate function
    processed_frames = process_frames(temp_image_folder, 1024)

    # Clean up: Delete the temporary image folder
    shutil.rmtree(temp_image_folder)

    return processed_frames


def low_vram():
    import torch

    total_vram = 0
    for i in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(i)
        total_vram += properties.total_memory
    return (total_vram / (1024**3)) < 8.1  # Y/N under 6.1 GB?


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
    if os_platform == "Darwin" or os_platform == "Linux":
        import_module(self, "sox", "sox")
    else:
        import_module(self, "soundfile", "PySoundFile")
    #import_module(self, "diffusers", "diffusers")
    import_module(self, "diffusers", "git+https://github.com/huggingface/diffusers.git@v0.19.3")
    # import_module(self, "diffusers", "git+https://github.com/huggingface/diffusers.git")
    import_module(self, "accelerate", "accelerate")
    import_module(self, "transformers", "transformers")
    # import_module(self, "optimum", "optimum")
    import_module(self, "sentencepiece", "sentencepiece")
    import_module(self, "safetensors", "safetensors")
    import_module(self, "cv2", "opencv_python")
    import_module(self, "PIL", "pillow")
    import_module(self, "scipy", "scipy")
    import_module(self, "IPython", "IPython")
    import_module(self, "bark", "git+https://github.com/suno-ai/bark.git")
    import_module(self, "xformers", "xformers")
    import_module(self, "imageio", "imageio")
    import_module(self, "imwatermark", "invisible-watermark>=0.2.0")
    # import_module(self, "triton", "C://Users//45239//Downloads//triton-2.0.0-cp310-cp310-win_amd64.whl")
    # import_module(self, "audiocraft", "git+https://github.com/facebookresearch/audiocraft.git")
    # subprocess.check_call([pybin,"-m","pip","install","force-reinstall","no-deps","pre xformers"])
    subprocess.check_call([pybin, "-m", "pip", "install", "numpy", "--upgrade"])
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

    result = subprocess.run(
        [pybin, "-m", "pip", "show", module_name], capture_output=True, text=True
    )
    output = result.stdout.strip()

    dependencies = []
    for line in output.split("\n"):
        if line.startswith("Requires:"):
            dependencies = line.split(":")[1].strip().split(", ")
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
    subprocess.run([pybin, "-m", "pip", "uninstall", "-y", module_name])

    # Uninstall the dependencies
    for dependency in dependencies:
        subprocess.run([pybin, "-m", "pip", "uninstall", "-y", dependency])
    subprocess.check_call([pybin, "-m", "pip", "install", "numpy"])


def input_strips_updated(self, context):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences
    movie_model_card = addon_prefs.movie_model_card

    scene = context.scene
    input = scene.input_strips
    
    if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
        scene.input_strips = "input_strips"


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
        name="Video Model",
        items=[
            ("strangeman3107/animov-0.1.1", "Animov (448x384)", "Animov (448x384)"),
            ("strangeman3107/animov-512x", "Animov (512x512)", "Animov (512x512)"),
            ("camenduru/potat1", "Potat v1 (1024x576)", "Potat (1024x576)"),
            (
                "cerspense/zeroscope_v2_dark_30x448x256",
                "Zeroscope (448x256x30)",
                "Zeroscope (448x256x30)",
            ),
            (
                "cerspense/zeroscope_v2_576w",
                "Zeroscope (576x320x24)",
                "Zeroscope (576x320x24)",
            ),
            (
                "cerspense/zeroscope_v2_XL",
                "Zeroscope XL (1024x576x24)",
                "Zeroscope XL (1024x576x24)",
            ),
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                "Img2img SD XL 1.0 Refine (1024x1024)",
                "Stable Diffusion XL 1.0",
            ),
            (
                "576-b2g8f5x4-36-18000/18000",
                "576-b2g8f5x4-36-18000 (576x320)",
                "576-b2g8f5x4-36-18000",
            ),
            # ("camenduru/AnimateDiff/", "AnimateDiff", "AnimateDiff"),
            # ("polyware-ai/longscope", "Longscope (384x216x94)", "Longscope ( 384x216x94)"),
            # ("vdo/potat1-lotr-25000/", "LOTR (1024x576x24)", "LOTR (1024x576x24)"),
            # ("damo-vilab/text-to-video-ms-1.7b", "Modelscope (256x256)", "Modelscope (256x256)"),
            # ("polyware-ai/text-to-video-ms-stable-v1", "Polyware 1.7b (384x384)", "Polyware 1.7b (384x384)"),
            # ("vdo/potat1-50000", "Potat v1 50000 (1024x576)", "Potat (1024x576)"),
            # ("cerspense/zeroscope_v1-1_320s", "Zeroscope v1.1 (320x320)", "Zeroscope (320x320)"),
        ],
        default="cerspense/zeroscope_v2_XL",
        update=input_strips_updated,
    )

    image_model_card: bpy.props.EnumProperty(
        name="Image Model",
        items=[
            (
                "runwayml/stable-diffusion-v1-5",
                "Stable Diffusion 1.5 (512x512)",
                "Stable Diffusion 1.5",
            ),
            (
                "stabilityai/stable-diffusion-2",
                "Stable Diffusion 2 (768x768)",
                "Stable Diffusion 2",
            ),
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                "Stable Diffusion XL 1.0 (1024x1024)",
                "Stable Diffusion XL 1.0",
            ),
#            (
#                "segmind/tiny-sd",
#                "Stable Diffusion Tiny (512x512)",
#                "Stable Diffusion Tiny",
#            ),
#            (
#                "nota-ai/bk-sdm-small-2m",
#                "BK SDM Small (512×512)",
#                "BK SDM Small (512×512)",
#            ),
            ("DeepFloyd/IF-I-M-v1.0", "DeepFloyd/IF-I-M-v1.0", "DeepFloyd"),
            # ("stabilityai/stable-diffusion-xl-base-0.9", "Stable Diffusion XL Base 0.9", "Stable Diffusion XL Base 0.9"),
            # ("kandinsky-community/kandinsky-2-1", "Kandinsky 2.1 (768x768)", "Kandinsky 2.1 (768x768)"),
        ],
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )

    audio_model_card: bpy.props.EnumProperty(
        name="Audio Model",
        items=[
            (
                "cvssp/audioldm-s-full-v2",
                "AudioLDM S Full v2",
                "AudioLDM Small Full v2",
            ),
            ("bark", "Bark", "Bark"),
            # ("facebook/audiogen-medium", "AudioGen", "AudioGen"), #I do not have enough VRAM to test if this is working...
            # ("cvssp/audioldm", "AudioLDM", "AudioLDM"),
        ],
        default="bark",
    )

    hugginface_token: bpy.props.StringProperty(
        name="Hugginface Token",
        default="hugginface_token",
        subtype="PASSWORD",
    )

    generator_ai: StringProperty(
        name="Filepath",
        description="Path to the folder where the generated files are stored",
        subtype="DIR_PATH",
        default=join(bpy.utils.user_resource("DATAFILES"), "Generator AI"),
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
            row.operator(
                "wm.url_open", text="", icon="URL"
            ).url = "https://huggingface.co/settings/tokens"
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
    """Uninstall all dependencies"""

    bl_idname = "sequencer.uninstall_generator"
    bl_label = "Uninstall Dependencies"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences

        uninstall_module_with_dependencies("torch")
        uninstall_module_with_dependencies("torchvision")
        uninstall_module_with_dependencies("torchaudio")

        if os_platform == "Darwin" or os_platform == "Linux":
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
        uninstall_module_with_dependencies("imageio")
        uninstall_module_with_dependencies("invisible-watermark")
        uninstall_module_with_dependencies("pillow")

        self.report(
            {"INFO"},
            "\nRemove AI Models manually: \nLinux and macOS: ~/.cache/huggingface/transformers\nWindows: %userprofile%.cache\\huggingface\\transformers",
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


class SEQEUNCER_PT_generate_ai(Panel):  # UI
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
        movie_model_card = addon_prefs.movie_model_card
        image_model_card = addon_prefs.image_model_card

        scene = context.scene
        type = scene.generatorai_typeselect
        input = scene.input_strips

        layout = self.layout
        col = layout.column(align=True)
        col.use_property_split = True
        col.use_property_decorate = False
        col.scale_y = 1.2
        col.prop(context.scene, "generate_movie_prompt", text="", icon="ADD")

        if type == "audio" and audio_model_card == "bark":
            pass
        else:
            col.prop(
                context.scene, "generate_movie_negative_prompt", text="", icon="REMOVE"
            )
        
        col.prop(context.scene, "generatorai_styles", text="Style")
        
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

        if type == "movie" and (
            movie_model_card == "cerspense/zeroscope_v2_dark_30x448x256"
            or movie_model_card == "cerspense/zeroscope_v2_576w"
            or movie_model_card == "cerspense/zeroscope_v2_XL"
        ):
            col = layout.column(heading="Upscale", align=True)
            col.prop(context.scene, "video_to_video", text="2x")

        if type == "image" and (
            image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
        ):
            col = layout.column(heading="Refine", align=True)
            col.prop(context.scene, "refine_sd", text="Image")
            sub_col = col.row()
            sub_col.active = context.scene.refine_sd

        col = layout.column()
        col.prop(context.scene, "input_strips", text="Input")
        if input == "input_strips":
            col.prop(context.scene, "image_power", text="Strip Power")

        col = layout.column()
        col.prop(context.scene, "generatorai_typeselect", text="Output")
        col.prop(context.scene, "movie_num_batch", text="Batch Count")

        if input == "input_strips":
            row = layout.row(align=True)
            row.scale_y = 1.1
            row.operator("sequencer.text_to_generator", text="Generate from Strips")
        else:
            row = layout.row(align=True)
            row.scale_y = 1.1
            if type == "movie":
                if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    row.operator("sequencer.text_to_generator", text="Generate from Strips")
                else:
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

        try:
            import torch
            from diffusers import (
                DiffusionPipeline,
                StableDiffusionXLPipeline,
                DPMSolverMultistepScheduler,
                TextToVideoSDPipeline,
                VideoToVideoSDPipeline,
            )
            from diffusers.utils import export_to_video
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            import numpy as np
        except ModuleNotFoundError:
            print("In the add-on preferences, install dependencies.")
            self.report(
                {"INFO"},
                "In the add-on preferences, install dependencies.",
            )
            return {"CANCELLED"}

        show_system_console(True)
        set_system_console_topmost(True)

        seq_editor = scene.sequence_editor

        if not seq_editor:
            scene.sequence_editor_create()

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_frame = scene.frame_current
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        #print("Positive "+prompt)
        negative_prompt = scene.generate_movie_negative_prompt +", "+ style_prompt(scene.generate_movie_prompt)[1] +", nsfw nude nudity"
        #print("Negative "+negative_prompt)
        movie_x = scene.generate_movie_x
        movie_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_64(movie_x)
        y = scene.generate_movie_y = closest_divisible_64(movie_y)
        duration = scene.generate_movie_frames
        movie_num_inference_steps = scene.movie_num_inference_steps
        movie_num_guidance = scene.movie_num_guidance
        input = scene.input_strips

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        movie_model_card = addon_prefs.movie_model_card
        image_model_card = addon_prefs.image_model_card

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        # LOADING MODULES

        # Models for refine imported image or movie
        if (scene.movie_path or scene.image_path) and input == "input_strips":

            if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                import torch
                from diffusers import StableDiffusionXLImg2ImgPipeline

                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    #image_model_card,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )

                if low_vram:
                    torch.cuda.set_per_process_memory_fraction(0.95)
                    pipe.enable_model_cpu_offload()
                    # pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
                    #pipe.unet.added_cond_kwargs={}
                    pipe.enable_vae_slicing()
                    #pipe.enable_xformers_memory_efficient_attention()
                else:
                    pipe.to("cuda")


                from diffusers import StableDiffusionXLImg2ImgPipeline
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    #"stabilityai/stable-diffusion-xl-base-1.0",
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    text_encoder_2=pipe.text_encoder_2,
                    vae=pipe.vae,
                    torch_dtype=torch.float16,
                    #use_safetensors=True,
                    variant="fp16",
                )

                if low_vram:
                    torch.cuda.set_per_process_memory_fraction(0.95)
                    refiner.enable_model_cpu_offload()
                    # refiner.unet.enable_forward_chunking(chunk_size=1, dim=1)
                    #refiner.unet.added_cond_kwargs={}
                    refiner.enable_vae_slicing()
                    #refiner.enable_xformers_memory_efficient_attention()
                else:
                    refiner.to("cuda")

            else:
                if movie_model_card == "cerspense/zeroscope_v2_dark_30x448x256" or movie_model_card == "cerspense/zeroscope_v2_576w":
                    card = "cerspense/zeroscope_v2_XL"
                else:
                    card = movie_model_card

                upscale = VideoToVideoSDPipeline.from_pretrained(
                    # "cerspense/zeroscope_v2_576w",
                    #"cerspense/zeroscope_v2_XL",
                    card,
                    torch_dtype=torch.float16,
                    #text_encoder=upscale.text_encoder,
                    #vae=upscale.vae,
                    #"cerspense/zeroscope_v2_XL", torch_dtype=torch.float16
                )

                upscale.scheduler = DPMSolverMultistepScheduler.from_config(upscale.scheduler.config)

                if low_vram:
                    torch.cuda.set_per_process_memory_fraction(0.95)  # 6 GB VRAM
                    upscale.enable_model_cpu_offload()

                    upscale.unet.enable_forward_chunking(chunk_size=1, dim=1)
                    # upscale.unet.added_cond_kwargs={}
                    upscale.enable_vae_slicing()
                    #pscale.enable_xformers_memory_efficient_attention()
                else:
                    upscale.to("cuda")

        # Models for movie generation
        else:
            # Options: https://huggingface.co/docs/diffusers/api/pipelines/text_to_video
            pipe = TextToVideoSDPipeline.from_pretrained(
            #pipe = DiffusionPipeline.from_pretrained(
                movie_model_card,
                torch_dtype=torch.float16,
                # variant="fp16",
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            if low_vram:
                pipe.enable_model_cpu_offload()
                # pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
                #pipe.unet.added_cond_kwargs={}
                pipe.enable_vae_slicing()
                #pipe.enable_xformers_memory_efficient_attention()
            else:
                pipe.to("cuda")

            # Model for upscale generated movie
            if scene.video_to_video:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # torch.cuda.set_per_process_memory_fraction(0.85)  # 6 GB VRAM

                # upscale = VideoToVideoSDPipeline.from_pretrained(
                upscale = DiffusionPipeline.from_pretrained(
                    #"cerspense/zeroscope_v2_576w", torch_dtype=torch.float16
                    "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16
                )

                # upscale = VideoToVideoSDPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
                upscale.scheduler = DPMSolverMultistepScheduler.from_config(
                    upscale.scheduler.config
                )

                if low_vram:
                    upscale.enable_model_cpu_offload()
                    upscale.unet.enable_forward_chunking(chunk_size=1, dim=1)
                    #upscale.unet.added_cond_kwargs={}
                    upscale.enable_vae_slicing()
                    #upscale.enable_xformers_memory_efficient_attention()
                else:
                    upscale.to("cuda")

        # GENERATING

        # Main Loop
        for i in range(scene.movie_num_batch):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

            # Get seed
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

            # Process batch input
            if (scene.movie_path or scene.image_path) and input == "input_strips":
                # Path to the video file
                video_path = scene.movie_path

                # img2img
                if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    print("Frame by frame video with SD XL")

                    input_video_path = video_path
                    output_video_path = solve_path("temp_images")

                    if scene.movie_path:
                        frames = process_video(input_video_path, output_video_path)
                    elif scene.image_path:
                        frames = process_image(scene.image_path, int(scene.generate_movie_frames))

                    video_frames = []
                    # Iterate through the frames
                    for frame_idx, frame in enumerate(frames): # would love to get this flicker free
                        print(str(frame_idx+1) + "/" + str(len(frames)))
                        image = refiner(
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=movie_num_inference_steps,
                            strength = 1.00 - scene.image_power,
                            guidance_scale=movie_num_guidance,
                            image=frame,
                            generator=generator,
                        ).images[0]

                        video_frames.append(image)

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    video_frames = np.array(video_frames)

                # vid2vid / img2vid
                else:
                    if scene.movie_path:
                        video = load_video_as_np_array(video_path)
                        print("\nVid2vid processing")

                    elif scene.image_path:
                        print("\nImg2vid processing")
                        video = process_image(scene.image_path, int(scene.generate_movie_frames))
                        video = np.array(video)

                    # Upscale video
                    if scene.video_to_video:
                        video = [
                            Image.fromarray(frame).resize((closest_divisible_64(int(x * 2)), closest_divisible_64(int(y * 2))))
                            for frame in video
                        ]

                    video_frames = upscale(
                        prompt,
                        video=video,
                        strength=1.00 - scene.image_power,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        generator=generator,
                    ).frames

            # Generation of movie
            else:
                print("Generating Video")
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

                movie_model_card = addon_prefs.movie_model_card

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Upscale video
                if scene.video_to_video:
                    print("Upscale Video")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    video = [Image.fromarray(frame).resize((closest_divisible_64(x * 2), closest_divisible_64(y * 2))) for frame in video_frames]

                    video_frames = upscale(
                        prompt,
                        video=video,
                        strength=1.00 - scene.image_power,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        generator=generator,
                    ).frames

            # Move to folder
            src_path = export_to_video(video_frames)
            dst_path = solve_path(clean_filename(str(seed)+"_"+prompt)+".mp4")
            print(src_path)
            print(dst_path)
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
                            bpy.ops.sequencer.movie_strip_add(
                                filepath=dst_path,
                                frame_start=start_frame,
                                channel=empty_channel,
                                fit_method="FIT",
                                adjust_playback_rate=True,
                                sound=False,
                                use_framerate=False,
                            )
                            strip = scene.sequence_editor.active_strip
                            strip.transform.filter = "SUBSAMPLING_3x3"
                            scene.sequence_editor.active_strip = strip
                            strip.name = str(seed) + "_" + prompt
                            strip.use_proxy = True
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

        bpy.types.Scene.movie_path = ""
        bpy.ops.renderreminder.play_notification()
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
        audio_length_in_s = scene.audio_length_in_f / (
            scene.render.fps / scene.render.fps_base
        )

        try:
            import torch

            if addon_prefs.audio_model_card == "cvssp/audioldm-s-full-v2":
                from diffusers import AudioLDMPipeline
                import scipy

                # from bark import SAMPLE_RATE, generate_audio, preload_models
                from IPython.display import Audio
                from scipy.io.wavfile import write as write_wav
                import xformers

            if addon_prefs.audio_model_card == "facebook/audiogen-medium":
                import torchaudio
                from audiocraft.models import AudioGen
                from audiocraft.data.audio import audio_write
                from scipy.io.wavfile import write as write_wav

            if addon_prefs.audio_model_card == "bark":
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                import numpy as np
                from bark.generation import (
                    generate_text_semantic,
                    preload_models,
                )
                from bark.api import semantic_to_waveform
                from bark import generate_audio, SAMPLE_RATE
                from scipy.io.wavfile import write as write_wav

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

        if addon_prefs.audio_model_card == "cvssp/audioldm-s-full-v2":
            repo_id = addon_prefs.audio_model_card
            pipe = AudioLDMPipeline.from_pretrained(
                repo_id
            )  # , torch_dtype=torch.float16z

            if low_vram:
                pipe.enable_model_cpu_offload()
                # pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
                # pipe.unet.added_cond_kwargs={}
                pipe.enable_vae_slicing()
                #pipe.enable_xformers_memory_efficient_attention()
            else:
                pipe.to("cuda")

        elif addon_prefs.audio_model_card == "facebook/audiogen-medium":
            pipe = AudioGen.get_pretrained("facebook/audiogen-medium")
            pipe = pipe.to("cuda")

        else:  # bark
            preload_models(
                text_use_small=True,
                coarse_use_small=True,
                fine_use_gpu=True,
                fine_use_small=True,
            )

        for i in range(scene.movie_num_batch):

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
                SPEAKER = "v2/" + scene.languages + "_" + scene.speakers  # "v2/"+
                silence = np.zeros(int(0.25 * rate))  # quarter second of silence

                prompt = context.scene.generate_movie_prompt
                prompt = prompt.replace("\n", " ").strip()

                sentences = split_and_recombine_text(
                    prompt, desired_length=90, max_length=150
                )

                pieces = []
                for sentence in sentences:
                    print(sentence)
                    semantic_tokens = generate_text_semantic(
                        sentence,
                        history_prompt=SPEAKER,
                        temp=GEN_TEMP,
                        # min_eos_p=0.1,  # this controls how likely the generation is to end
                    )

                    audio_array = semantic_to_waveform(
                        semantic_tokens, history_prompt=SPEAKER
                    )
                    pieces += [audio_array, silence.copy()]
                audio = np.concatenate(
                    pieces
                )  # Audio(np.concatenate(pieces), rate=rate)
                filename = solve_path(clean_filename(prompt + ".wav"))

                # Write the combined audio to a file
                write_wav(filename, rate, audio.transpose())
            else:  # AudioLDM
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

                filename = solve_path(prompt + ".wav")

                write_wav(filename, rate, audio.transpose())  # .transpose()
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
            import requests
            from diffusers.utils import load_image
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
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        negative_prompt = scene.generate_movie_negative_prompt +", "+ style_prompt(scene.generate_movie_prompt)[1] +", nsfw nude nudity"
        image_x = scene.generate_movie_x
        image_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_64(image_x)
        y = scene.generate_movie_y = closest_divisible_64(image_y)
        duration = scene.generate_movie_frames
        image_num_inference_steps = scene.movie_num_inference_steps
        image_num_guidance = scene.movie_num_guidance

        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        image_model_card = addon_prefs.image_model_card
        do_refine = (scene.refine_sd and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0") or scene.image_path


        # LOADING MMODELS

        # Models for stable diffusion
        if not image_model_card == "DeepFloyd/IF-I-M-v1.0":
            pipe = DiffusionPipeline.from_pretrained(
                image_model_card,
                torch_dtype=torch.float16,
                variant="fp16",
                #use_safetensors=True,
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            if low_vram:
                torch.cuda.set_per_process_memory_fraction(0.95)  # 6 GB VRAM
                pipe.enable_model_cpu_offload()
                # pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
                #pipe.unet.added_cond_kwargs={}
                pipe.enable_vae_slicing()
                #pipe.enable_xformers_memory_efficient_attention()
            else:
                pipe.to("cuda")

        # DeepFloyd
        elif image_model_card == "DeepFloyd/IF-I-M-v1.0":
            from huggingface_hub.commands.user import login

            result = login(token=addon_prefs.hugginface_token)

            # torch.cuda.set_per_process_memory_fraction(0.85)  # 6 GB VRAM

            # stage 1
            stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16
            )
            if low_vram:
                stage_1.enable_model_cpu_offload()
                stage_1.unet.enable_forward_chunking(chunk_size=1, dim=1)
                stage_1.enable_vae_slicing()
                stage_1.enable_xformers_memory_efficient_attention()
            else:
                stage_1.to("cuda")
            # stage 2
            stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            if low_vram:
                stage_2.enable_model_cpu_offload()
                # stage_2.unet.enable_forward_chunking(chunk_size=1, dim=1)
                stage_2.enable_vae_slicing()
                stage_2.enable_xformers_memory_efficient_attention()
            else:
                stage_2.to("cuda")
            # stage 3
            safety_modules = {
                "feature_extractor": stage_1.feature_extractor,
                "safety_checker": stage_1.safety_checker,
                "watermarker": stage_1.watermarker,
            }
            stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                **safety_modules,
                torch_dtype=torch.float16,
            )
            if low_vram:
                stage_3.enable_model_cpu_offload()
                # stage_3.unet.enable_forward_chunking(chunk_size=1, dim=1)
                stage_3.enable_vae_slicing()
                stage_3.enable_xformers_memory_efficient_attention()
            else:
                stage_3.to("cuda")


        # Add refiner model if chosen.
        if do_refine:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=pipe.text_encoder_2,
                vae=pipe.vae,
                torch_dtype=torch.float16,
                #use_safetensors=True,
                variant="fp16",
            )

            if low_vram:
                refiner.enable_model_cpu_offload()
                # refiner.unet.enable_forward_chunking(chunk_size=1, dim=1)
                #refiner.unet.added_cond_kwargs={}
                refiner.enable_vae_slicing()
                #refiner.enable_xformers_memory_efficient_attention()
            else:
                refiner.to("cuda")

        # Main Generate Loop:
        for i in range(scene.movie_num_batch):
            
            # Find free space for the strip in the timeline.
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
                
            # Generate seed.
            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(0, 999999)
            )
            context.scene.movie_num_seed = seed

            # Use cuda if possible.
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

            # DeepFloyd process: 
            if image_model_card == "DeepFloyd/IF-I-M-v1.0":
                print("DeepFloyd")
                prompt_embeds, negative_embeds = stage_1.encode_prompt(
                    prompt, negative_prompt
                )

                # stage 1
                image = stage_1(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
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
                image = stage_3(
                    prompt=prompt, image=image, noise_level=100, generator=generator
                ).images
                # image[0].save("./if_stage_III.png")
                image = image[0]

            # Img2img
            elif scene.image_path:
                print("Img2img")
                init_image = load_image(scene.image_path).convert("RGB")
                image = refiner(
                    prompt=prompt,
                    image=init_image,
                    strength = 1.00 - scene.image_power,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    generator=generator,
                    # output_type="latent" if scene.refine_sd else "pil",
                ).images[0]
 
            # Generate
            else:
                print("Generating ")
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            # Add refiner
            if scene.refine_sd: # and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0") or scene.image_path:
                print("Refining")
                image = refiner(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    denoising_start=0.8,
                    guidance_scale=image_num_guidance,
                    image=image,
                    #image=image[None, :], 
                ).images[0]

            # Move to folder
            filename = clean_filename(
                str(seed) + "_" + context.scene.generate_movie_prompt
            )
            out_path = solve_path(filename+".png")
            
            image.save(out_path)

            # Add strip
            if os.path.isfile(out_path):
                strip = scene.sequence_editor.sequences.new_image(
                    name=str(seed) + "_" + context.scene.generate_movie_prompt,
                    frame_start=start_frame,
                    filepath=out_path,
                    channel=empty_channel,
                    fit_method="FIT",
                )
                strip.frame_final_duration = scene.generate_movie_frames
                strip.transform.filter = "SUBSAMPLING_3x3"

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
        scene.frame_current = current_frame

        # clear the VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"FINISHED"}


class SEQUENCER_OT_strip_to_generatorAI(Operator):
    """Convert selected text strips to Generative AI"""

    bl_idname = "sequencer.text_to_generator"
    bl_label = "Generative AI"
    bl_options = {"INTERNAL"}
    bl_description = "Adds selected strips as inputs to the Generative AI process"

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
        seed = scene.movie_num_seed
        use_random = scene.movie_use_random

        if not strips:
            self.report({"INFO"}, "Select strips for batch processing.")
            return {"CANCELLED"}
        else:
            print("\nBatch processing started (ctrl+c to cancel).")

        for count, strip in enumerate(strips):
            if strip.type == "TEXT":
                if strip.text:
                    print("\n" + str(count+1) + "/"+ str(len(strips)) + " Prompt: " + strip.text + ", " + prompt)
                    scene.generate_movie_prompt = strip.text + ", " + prompt
                    scene.frame_current = strip.frame_final_start
                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()
                    scene.generate_movie_prompt = prompt
            if strip.type == "IMAGE":
                strip_dirname = os.path.dirname(strip.directory)
                image_path = bpy.path.abspath(
                    os.path.join(strip_dirname, strip.elements[0].filename)
                )
                bpy.types.Scene.image_path = image_path
                if strip.name:
                    strip_prompt = os.path.splitext(strip.name)[0]
                    seed_nr = extract_numbers(str(strip_prompt))
                    if seed_nr:
                        file_seed = int(seed_nr)
                        if file_seed:
                            strip_prompt = (strip_prompt.replace(str(file_seed)+"_", ""))
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed
                            
                    styled_prompt = style_prompt(strip_prompt + ", " + prompt)[0]
                    print("\n" + str(count+1) + "/"+ str(len(strips)) + " Prompt: " + styled_prompt)
                    scene.generate_movie_prompt = styled_prompt
                    scene.frame_current = strip.frame_final_start

                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()

                    context.scene.generate_movie_prompt = prompt
                    context.scene.movie_use_random = use_random
                    context.scene.movie_num_seed = seed

                bpy.types.Scene.image_path = ""

            if strip.type == "MOVIE":
                movie_path = bpy.path.abspath(
                    strip.filepath
                )
                bpy.types.Scene.movie_path = movie_path
                if strip.name:
                    strip_prompt = os.path.splitext(strip.name)[0]

                    seed_nr = extract_numbers(str(strip_prompt))
                    if seed_nr:
                        file_seed = int(seed_nr)
                        if file_seed:
                            strip_prompt = (strip_prompt.replace(str(file_seed)+"_", ""))
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed

                    styled_prompt = style_prompt(strip_prompt + ", " + prompt)[0]
                    print("\n" + str(count+1) + "/"+ str(len(strips)) + " Prompt: " + styled_prompt)
                    scene.generate_movie_prompt = styled_prompt
                    scene.generate_movie_prompt = prompt
                    scene.frame_current = strip.frame_final_start

                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()

                    scene.generate_movie_prompt = prompt
                    context.scene.movie_use_random = use_random
                    context.scene.movie_num_seed = seed

                bpy.types.Scene.movie_path = ""

        scene.frame_current = current_frame

        scene.generate_movie_prompt = prompt
        context.scene.movie_use_random = use_random
        context.scene.movie_num_seed = seed

        addon_prefs.playsound = play_sound
        bpy.ops.renderreminder.play_notification()
        
        print("Batch processing finished.")

        return {"FINISHED"}


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
        name="generate_movie_prompt",
        default="",
    )
    bpy.types.Scene.generate_movie_negative_prompt = bpy.props.StringProperty(
        name="generate_movie_negative_prompt",
        default="",
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x",
        default=1024,
        step=64,
        min=192,
        max=1536,
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=512,
        step=64,
        min=192,
        max=1536,
    )
    # The number of frames to be generated.
    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_frames",
        default=6,
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

    # The guidance number.
    bpy.types.Scene.movie_num_guidance = bpy.props.FloatProperty(
        name="movie_num_guidance",
        default=9.0,
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
        default="image",
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

    # Upscale
    bpy.types.Scene.video_to_video = bpy.props.BoolProperty(
        name="video_to_video",
        default=0,
    )

    # Refine SD
    bpy.types.Scene.refine_sd = bpy.props.BoolProperty(
        name="refine_sd",
        default=1,
    )

    # movie path
    bpy.types.Scene.movie_path = bpy.props.StringProperty(name="movie_path", default="")
    bpy.types.Scene.movie_path = ""

    # image path
    bpy.types.Scene.image_path = bpy.props.StringProperty(name="image_path", default="")
    bpy.types.Scene.image_path = ""

    bpy.types.Scene.input_strips = bpy.props.EnumProperty(
        items=[
            ("generate", "No Input", "No Input"),
            ("input_strips", "Strips", "Selected Strips"),
        ],
        default="generate",
        update=input_strips_updated,
    )

    bpy.types.Scene.image_power = bpy.props.FloatProperty(
        name="image_power",
        default=0.50,
        min=0.05,
        max=0.95,
    )

    styles_array = load_styles(os.path.dirname(os.path.abspath(__file__))+"/styles.json")
    if styles_array:
        bpy.types.Scene.generatorai_styles = bpy.props.EnumProperty(
            name="Generator AI Styles",
            items=[("no_style", "No Style", "No Style")] + styles_array,
            default="no_style",
        )

    for cls in classes:
        bpy.utils.register_class(cls)


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
    del bpy.types.Scene.movie_path
    del bpy.types.Scene.image_path
    del bpy.types.Scene.refine_sd
    # del bpy.types.Scene.denoising_strength
    del bpy.types.Scene.generatorai_styles

    #bpy.types.SEQUENCER_MT_add.remove(panel_text_to_generatorAI)


if __name__ == "__main__":
    unregister()
    register()
