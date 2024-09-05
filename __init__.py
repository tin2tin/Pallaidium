# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


bl_info = {
    "name": "Pallaidium - Generative AI",
    "author": "tintwotin",
    "version": (2, 0),
    "blender": (3, 4, 0),
    "location": "Video Sequence Editor > Sidebar > Generative AI",
    "description": "AI Generate media in the VSE",
    "category": "Sequencer",
}

# TO DO: Long prompts, Move prints, FLUX Controlnet & LoRA, refactor, clean-up. 

import bpy
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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Python: " + sys.version)

# if os_platform == "Windows":
#    # Temporarily modify pathlib.PosixPath for Windows compatibility
#    temp = pathlib.PosixPath
#    pathlib.PosixPath = pathlib.WindowsPath


try:
    exec("import torch")
    if torch.cuda.is_available():
        gfx_device = "cuda"
    elif torch.backends.mps.is_available():
        gfx_device = "mps"
    else:
        gfx_device = "cpu"
    print("GFX Device: "+gfx_device)
except:
    print("Pallaidium dependencies needs to be installed and Blender needs to be restarted.")

os_platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'
if os_platform == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


def show_system_console(show):
    if os_platform == "Windows":
        # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow

        SW_HIDE = 0
        SW_SHOW = 5
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW)  # if show else SW_HIDE


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
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii


def format_time(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{int(milliseconds):03d}"


def timer():
    start_time = time.time()
    return start_time


def print_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    formatted_time = format_time(elapsed_time * 1000)  # Convert to milliseconds
    print(f"Total time: {formatted_time}\n\n")


def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“[“”]”]", '"', text)
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

            while pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?.,":
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
    numbers = re.findall(r"\d+", input_string)
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
        styles_array = load_styles(os.path.dirname(os.path.abspath(__file__)) + "/styles.json")
        if styles_array:
            selected_entry = next((item for item in styles_array if item[0] == selected_entry_key), None)
            if selected_entry:
                selected_entry_list = list(selected_entry)
                return_array.append(selected_entry_list[2].replace("{prompt}", prompt))
                return_array.append(bpy.context.scene.generate_movie_negative_prompt + ", " + selected_entry_list[0].replace("_", " "))
                return return_array
    return_array.append(prompt)
    return_array.append(bpy.context.scene.generate_movie_negative_prompt)
    return return_array


def closest_divisible_16(num):
    # Determine the remainder when num is divided by 64

    remainder = num % 16
    # If the remainder is less than or equal to 16, return num - remainder,
    # but ensure the result is not less than 192

    if remainder <= 8:
        result = num - remainder
        return max(result, 192)
    # Otherwise, return num + (32 - remainder)

    else:
        return max(num + (16 - remainder), 192)


def closest_divisible_128(num):
    # Determine the remainder when num is divided by 128

    remainder = num % 128
    # If the remainder is less than or equal to 64, return num - remainder,
    # but ensure the result is not less than 256

    if remainder <= 64:
        result = num - remainder
        return max(result, 256)
    # Otherwise, return num + (32 - remainder)

    else:
        return max(num + (64 - remainder), 256)


def find_first_empty_channel(start_frame, end_frame):
    for ch in range(1, len(bpy.context.scene.sequence_editor.sequences_all) + 1):
        for seq in bpy.context.scene.sequence_editor.sequences_all:
            if seq.channel == ch and seq.frame_final_start < end_frame and (seq.frame_final_start + seq.frame_final_duration) > start_frame:
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
    dir_path = os.path.join(addon_prefs.generator_ai, str(date.today()))
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


def delete_strip(input_strip):
    if input_strip is None:
        return
    original_selection = [strip for strip in bpy.context.scene.sequence_editor.sequences_all if strip.select]
    bpy.ops.sequencer.select_all(action="DESELECT")
    input_strip.select = True
    bpy.ops.sequencer.delete()
    for strip in original_selection:
        strip.select = True


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


def load_first_frame(file_path):
    import cv2, PIL, os
    from diffusers.utils import load_image

    extension = os.path.splitext(file_path)[-1].lower()  # Convert to lowercase for case-insensitive comparison
    valid_image_extensions = {
        ".sgi",
        ".rgb",
        ".bw",
        ".cin",
        ".dpx",
        ".png",
        ".jpg",
        ".jpeg",
        ".jp2",
        ".jp2",
        ".j2c",
        ".tga",
        ".exr",
        ".hdr",
        ".tif",
        ".tiff",
        ".webp",
    }
    valid_video_extensions = {
        ".avi",
        ".flc",
        ".mov",
        ".movie",
        ".mp4",
        ".m4v",
        ".m2v",
        ".m2t",
        ".m2ts",
        ".mts",
        ".ts",
        ".mv",
        ".avs",
        ".wmv",
        ".ogv",
        ".ogg",
        ".r3d",
        ".dv",
        ".mpeg",
        ".mpg",
        ".mpg2",
        ".vob",
        ".mkv",
        ".flv",
        ".divx",
        ".xvid",
        ".mxf",
        ".webm",
    }
    if extension in valid_image_extensions:
        image = cv2.imread(file_path)
        # if image is not None:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)
    if extension in valid_video_extensions:
        # Try to open the file as a video

        cap = cv2.VideoCapture(file_path)
        # Check if the file was successfully opened as a video

        if cap.isOpened():
            # Read the first frame from the video

            ret, frame = cap.read()
            cap.release()  # Release the video capture object
            if ret:
                # If the first frame was successfully read, it's a video

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return PIL.Image.fromarray(frame)
    # If neither video nor image worked, return None

    return None


def process_frames(frame_folder_path, target_width):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    import cv2

    processed_frames = []
    # List all image files in the folder

    image_files = sorted([f for f in os.listdir(frame_folder_path) if f.endswith(".png")])
    for image_file in image_files:
        image_path = os.path.join(frame_folder_path, image_file)
        img = Image.open(image_path)
        # Process the image (resize and convert to RGB)

        frame_width, frame_height = img.size
        # Calculate the target height to maintain the original aspect ratio

        target_height = int((target_width / frame_width) * frame_height)
        # Ensure width and height are divisible by 64
        target_width = closest_divisible_16(target_width)
        target_height = closest_divisible_16(target_height)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        processed_frames.append(img)
    return processed_frames


def process_video(input_video_path, output_video_path):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    import cv2
    import shutil

    scene = bpy.context.scene
    movie_x = scene.generate_movie_x
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

    processed_frames = process_frames(temp_image_folder, movie_x)
    # Clean up: Delete the temporary image folder

    shutil.rmtree(temp_image_folder)
    return processed_frames


# Define the function for zooming effect


def zoomPan(img, zoom=1, angle=0, coord=None):
    import cv2

    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    rot = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    res = cv2.warpAffine(img, rot, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return res


def process_image(image_path, frames_nr):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    import cv2, shutil

    scene = bpy.context.scene
    movie_x = scene.generate_movie_x
    img = cv2.imread(image_path)
    height, width, layers = img.shape
    # Create a temporary folder for storing frames

    temp_image_folder = solve_path("/temp_images")
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)
    max_zoom = 2.0  # Maximum Zoom level (should be > 1.0)
    max_rot = 30  # Maximum rotation in degrees, set '0' for no rotation
    # Make the loop for Zooming-in

    i = 1
    while i < frames_nr:
        zLvl = 1.0 + ((i / (1 / (max_zoom - 1)) / frames_nr) * 0.005)
        angle = 0  # i * max_rot / frames_nr
        zoomedImg = zoomPan(img, zLvl, angle, coord=None)
        output_path = os.path.join(temp_image_folder, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, zoomedImg)
        i = i + 1
    # Process frames using the separate function

    processed_frames = process_frames(temp_image_folder, movie_x)
    # Clean up: Delete the temporary image folder

    shutil.rmtree(temp_image_folder)
    return processed_frames


#def low_vram():
#    import torch

#    total_vram = 0
#    for i in range(torch.cuda.device_count()):
#        properties = torch.cuda.get_device_properties(i)
#        total_vram += properties.total_memory
#    return (total_vram / (1024**3)) <= 16  # Y/N under 16 GB?


def low_vram():
    try:
        if sys.platform == "darwin":  # macOS
            vram_info = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], 
                text=True
            )
            vram_match = re.search(r"(\d+)\s?MB", vram_info)        
            if vram_match:
                vram_mb = int(vram_match.group(1))
                vram_gb = vram_mb / 1024
            print("Vram: " + str(vram_gb / (1024**3)))
            vram_gb = ((vram_gb / (1024**3)) <= 16)  # Y/N under 16 GB?
        else:
            import torch
            vram_gb = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(i)
                    vram_gb += properties.total_memory
                print("Vram: " + str(vram_gb / (1024**3)))
                vram_gb = ((vram_gb / (1024**3)) <= 16)  # Y/N under 16 GB?
            else:
               vram_gb = 0 
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return vram_gb # Y/N under 16 GB?


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def isWindows():
    return os.name == "nt"


def isMacOS():
    return os.name == "posix" and platform.system() == "Darwin"


def isLinux():
    return os.name == "posix" and platform.system() == "Linux"


def python_exec():
    import sys

    if isWindows():
        return os.path.join(sys.prefix, "bin", "python.exe")
    elif isMacOS():
        try:
            # 2.92 and older

            path = bpy.app.binary_path_python
        except AttributeError:
            # 2.93 and later

            import sys

            path = sys.executable
        return os.path.abspath(path)
    elif isLinux():
        return os.path.join(sys.prefix, "bin", "python")
    else:
        print("sorry, still not implemented for ", os.name, " - ", platform.system)


def find_strip_by_name(scene, name):
    for sequence in scene.sequence_editor.sequences:
        if sequence.name == name:
            return sequence
    return None


def get_strip_path(strip):
    if strip.type == "IMAGE":
        strip_dirname = os.path.dirname(strip.directory)
        image_path = bpy.path.abspath(os.path.join(strip_dirname, strip.elements[0].filename))
        return image_path
    if strip.type == "MOVIE":
        movie_path = bpy.path.abspath(strip.filepath)
        return movie_path
    return None


def clamp_value(value, min_value, max_value):
    # Ensure value is within the specified range

    return max(min(value, max_value), min_value)


def find_overlapping_frame(strip, current_frame):
    # Calculate the end frame of the strip

    strip_end_frame = strip.frame_final_start + strip.frame_duration
    # Check if the strip's frame range overlaps with the current frame

    if strip.frame_final_start <= current_frame <= strip_end_frame:
        # Calculate the overlapped frame by subtracting strip.frame_start from the current frame

        return current_frame - strip.frame_start
    else:
        return None  # Return None if there is no overlap


def ensure_unique_filename(file_name):
    # Check if the file already exists

    if os.path.exists(file_name):
        base_name, extension = os.path.splitext(file_name)
        index = 1
        # Keep incrementing the index until a unique filename is found

        while True:
            unique_file_name = f"{base_name}_{index}{extension}"
            if not os.path.exists(unique_file_name):
                return unique_file_name
            index += 1
    else:
        # File doesn't exist, return the original name

        return file_name


def import_module(self, module, install_module):
    show_system_console(True)
    set_system_console_topmost(True)
    module = str(module)
    python_exe = python_exec()

#    try:
#        #exec("import " + module)
#        subprocess.call([python_exe, "import ", module])
#    except:
    self.report({"INFO"}, "Installing: " + module + " module.")
    print("\nInstalling: " + module + " module")
    subprocess.call([python_exe, "-m", "pip", "install", install_module, "--no-warn-script-location", "--upgrade"])

#    try:
#        exec("import " + module)
#    except ModuleNotFoundError:
#        print("Module not found: " + module)
#        return False
    return True


def parse_python_version(version_info):
    major, minor = version_info[:2]
    return f"{major}.{minor}"


def install_modules(self):
    os_platform = platform.system()
    app_path = site.USER_SITE

    pybin = python_exec()
    print("Ensuring: pip")

    try:
        subprocess.call([pybin, "-m", "ensurepip"])
        #subprocess.call([pybin, "-m", "pip", "install", "--upgrade", "pip"])
    except ImportError:
        subprocess.call([pybin, "-m", "pip", "install", "--upgrade", "pip"])
        pass

    import_module(self, "huggingface_hub", "huggingface_hub")
    import_module(self, "transformers", "transformers==4.43.0")
    #import_module(self, "transformers", "git+https://github.com/huggingface/transformers.git")
    if os_platform != "Linux":
        subprocess.call([pybin, "-m", "pip", "install", "git+https://github.com/suno-ai/bark.git", "--upgrade"])
    if os_platform != "Linux":
        import_module(self, "whisperspeech", "WhisperSpeech==0.8")
    import_module(self, "pydub", "pydub")
    
    if os_platform == "Windows":
        import_module(self, "deepspeed", "https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu121-cp311-cp311-win_amd64.whl")
        # resemble-enhance:
        subprocess.call(
            [pybin, "-m", "pip", "install", "git+https://github.com/tin2tin/resemble-enhance-windows.git", "--upgrade"] #"--no-dependencies",
        )
        #deep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepspeed/deepspeed-0.12.4+unknown-py3-none-any.whl")
        #deep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepspeed/deepspeed-0.14.4-py3-none-any.whl")
        #print("deep_speed_path: " + deep_path)
        #import_module(self, "deepspeed", deep_path)
        #subprocess.call([pybin, "-m", deep_path, "--upgrade"])
#        import_module(self, "librosa", "librosa")
#        import_module(self, "celluloid", "celluloid")
#        import_module(self, "omegaconf", "omegaconf")
#        import_module(self, "pandas", "pandas")
#        subprocess.check_call([pybin, "-m", "pip", "install", "git+https://github.com/sovrasov/flops-counter.pytorch.git", "--upgrade"])
#        import_module(self, "rich", "rich")
#        import_module(self, "resampy", "resampy")
#        import_module(self, "tabulate", "tabulate")
    else:
        import_module(self, "deepspeed", "deepspeed==0.14.4")
        import_module(self, "resemble_enhance", "resemble-enhance")

    # import_module(self, "diffusers", "diffusers")
    import_module(self, "diffusers", "git+https://github.com/huggingface/diffusers.git")
    subprocess.check_call([pybin, "-m", "pip", "install", "tensorflow", "--upgrade"])

    if os_platform == "Windows":
        import_module(self, "soundfile", "PySoundFile")
    else:
        import_module(self, "soundfile", "soundfile")
    import_module(self, "sentencepiece", "sentencepiece")
    import_module(self, "safetensors", "safetensors")
    import_module(self, "cv2", "opencv_python")
    import_module(self, "PIL", "pillow")
    import_module(self, "scipy", "scipy")
    import_module(self, "IPython", "IPython")
    import_module(self, "omegaconf", "omegaconf")
    import_module(self, "protobuf", "protobuf")
    #import_module(self, "hidiffusion", "hidiffusion")
    import_module(self, "aura_sr", "aura-sr")

    import_module(self, "stable_audio_tools", "stable-audio-tools")
    if os_platform == "Windows":
        pass
        import_module(self, "flash_attn", "https://github.com/oobabooga/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.3.1cxx11abiFALSE-cp311-cp311-win_amd64.whl")
    else:
        import_module(self, "flash_attn", "flash-attn")

    import_module(self, "controlnet_aux", "controlnet-aux")
    subprocess.call([pybin, "-m", "pip", "install", "controlnet-aux", "--upgrade"])

    import_module(self, "beautifulsoup4", "beautifulsoup4")
    import_module(self, "ftfy", "ftfy")

    python_version_info = sys.version_info
    python_version_str = parse_python_version(python_version_info)

    #import_module(self, "imageio", "imageio==2.4.1")
    import_module(self, "imageio", "imageio[ffmpeg]==2.4.1")
    import_module(self, "imageio", "imageio-ffmpeg")
    import_module(self, "imWatermark", "imWatermark")
    import_module(self, "parler_tts", "git+https://github.com/huggingface/parler-tts.git")
    if os_platform == "Windows":
        subprocess.call([pybin, "-m", "pip", "install", "https://hf-mirror.com/LightningJay/triton-2.1.0-python3.11-win_amd64-wheel/resolve/main/triton-2.1.0-cp311-cp311-win_amd64.whl", "--upgrade", '--user'])
    else:
        try:
            exec("import triton")
        except ModuleNotFoundError:
            import_module(self, "triton", "triton")
#    if os_platform == "Windows":
#        if python_version_str == "3.10":
#            subprocess.check_call(
#                [
#                    pybin,
#                    "-m",
#                    "pip",
#                    "install",
#                    "https://files.pythonhosted.org/packages/e2/a9/98e0197b24165113ac551aae5646005205f88347fb13ac59a75a9864e1d3/mediapipe-0.10.9-cp310-cp310-win_amd64.whl",
#                    "--no-warn-script-location",
#                ]
#            )
#        else:
#            subprocess.check_call(
#                [
#                    pybin,
#                    "-m",
#                    "pip",
#                    "install",
#                    "https://files.pythonhosted.org/packages/e9/7b/cd671c5067a56e1b4a9b70d0e42ac8cdb9f63acdc186589827cf213802a5/mediapipe-0.10.9-cp311-cp311-win_amd64.whl",
#                    "--no-warn-script-location",
#                ]
#            )
#    else:
    import_module(self, "mediapipe", "mediapipe")
#    if os_platform == "Windows":
#        print("")
#        if python_version_str == "3.10":
#            subprocess.check_call(
#                [
#                    pybin,
#                    "-m",
#                    "pip",
#                    "install",
#                    "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl",
#                    "--no-warn-script-location",
#                ]
#            )
#        else:
#            subprocess.check_call(
#                [
#                    pybin,
#                    "-m",
#                    "pip",
#                    "install",
#                    "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl",
#                    "--no-warn-script-location",
#                ]
#            )
#    else:
#    import_module(self, "insightface", "insightface")
#    import_module(self, "onnxruntime", "onnxruntime")
    #import_module(self, "asdff", "git+https://github.com/theblackhatmagician/adetailer_sdxl.git")
    #subprocess.call([pybin, "-m", "pip", "install", "git+https://github.com/theblackhatmagician/adetailer_sdxl.git", "--upgrade"])
    #import_module(self, "ultralytics", "ultralytics")

    subprocess.call([pybin, "-m", "pip", "install", "ultralytics", "--upgrade"])
    subprocess.call([pybin, "-m", "pip", "install", "git+https://github.com/theblackhatmagician/adetailer_sdxl.git", "--upgrade"])
    subprocess.call([pybin, "-m", "pip", "install", "lmdb", "--upgrade"])
    subprocess.call([pybin, "-m", "pip", "install", "git+https://github.com/huggingface/accelerate.git", "--upgrade"])
    #import_module(self, "accelerate", "git+https://github.com/huggingface/accelerate.git")
    #import_module(self, "accelerate", "accelerate")
    subprocess.check_call([pybin, "-m", "pip", "install", "peft", "--upgrade"])


    self.report({"INFO"}, "Installing: torch module.")
    print("\nInstalling: torch module")
    if os_platform == "Windows":
        subprocess.call([pybin, "-m", "pip", "uninstall", "-y", "torch"])
        subprocess.call([pybin, "-m", "pip", "uninstall", "-y", "torchvision"])
        subprocess.call([pybin, "-m", "pip", "uninstall", "-y", "torchaudio"])
        subprocess.call([pybin, "-m", "pip", "uninstall", "-y", "xformers"])

#         subprocess.call([pybin, "-m", "pip", "install", "torch torchvision torchaudio xformers", "--index-url", "https://download.pytorch.org/whl/cu121"])
#        subprocess.check_call(
#            [
#                pybin,
#                "-m",
#                "pip",
#                "install",
#                #"torchvision==0.17.0+cu121",
#                "torchvision==0.18.0+cu121",
#                "--index-url",
#                "https://download.pytorch.org/whl/cu121",
#                "--no-warn-script-location",
#                #"--user",
#                "--upgrade",
#            ]
#        )
#        subprocess.check_call(
#            [
#                pybin,
#                "-m",
#                "pip",
#                "install",
#                "xformers==0.0.26.post1",
#                "--index-url",
#                "https://download.pytorch.org/whl/cu121",
#                "--no-warn-script-location",
#                #"--user",
#                "--upgrade",
#            ]
#        )
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                "torch==2.3.1+cu121",
                "xformers",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "--no-warn-script-location",
                #"--user",
                "--upgrade",
            ]
        )
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                "torchaudio==2.3.1+cu121",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "--no-warn-script-location",
                #"--user",
                "--upgrade",
            ]
        )
#        subprocess.check_call(
#            [
#                pybin,
#                "-m",
#                "pip",
#                "install",
#                "torchao",
#                "--index-url",
#                "https://download.pytorch.org/whl/cu124",
#                #"https://download.pytorch.org/whl/cu121",
#                "--no-warn-script-location",
#                #"--user",
#                "--upgrade",
#            ]
#
#        )
    else:
        import_module(self, "torch", "torch")
        import_module(self, "torchvision", "torchvision")
        import_module(self, "torchaudio", "torchaudio")
        import_module(self, "xformers", "xformers")
        import_module(self, "torchao", "torchao")

    print("Dir: " + str(subprocess.check_call([pybin, "-m", "pip", "cache", "purge"])))


def get_module_dependencies(module_name):
    """
    Get the list of dependencies for a given module.
    """
    pybin = python_exec()
    result = subprocess.run([pybin, "-m", "pip", "show", module_name], capture_output=True, text=True)
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
    pybin = python_exec()
    dependencies = get_module_dependencies(module_name)
    # Uninstall the module

    subprocess.run([pybin, "-m", "pip", "uninstall", "-y", module_name])
    # Uninstall the dependencies

    for dependency in dependencies:
        print("\n ")
        if len(dependency) > 5 and str(dependency[5].lower) != "numpy":
            subprocess.run([pybin, "-m", "pip", "uninstall", "-y", dependency])


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
        pybin = python_exec()
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        uninstall_module_with_dependencies("torch")
        uninstall_module_with_dependencies("torchvision")
        uninstall_module_with_dependencies("torchaudio")
        uninstall_module_with_dependencies("soundfile")
        uninstall_module_with_dependencies("diffusers")
        uninstall_module_with_dependencies("transformers")
        uninstall_module_with_dependencies("sentencepiece")
        uninstall_module_with_dependencies("safetensors")
        uninstall_module_with_dependencies("opencv_python")
        uninstall_module_with_dependencies("scipy")
        uninstall_module_with_dependencies("IPython")
        uninstall_module_with_dependencies("bark")
        uninstall_module_with_dependencies("xformers")
        uninstall_module_with_dependencies("imageio")
        uninstall_module_with_dependencies("imWatermark")
        uninstall_module_with_dependencies("pillow")
        uninstall_module_with_dependencies("libtorrent")
        uninstall_module_with_dependencies("accelerate")
        uninstall_module_with_dependencies("triton")
        uninstall_module_with_dependencies("cv2")
        uninstall_module_with_dependencies("protobuf")
        uninstall_module_with_dependencies("resemble-enhance")
        uninstall_module_with_dependencies("mediapipe")
        uninstall_module_with_dependencies("flash_attn")

        uninstall_module_with_dependencies("controlnet-aux")

        uninstall_module_with_dependencies("stable-audio-tools")

        uninstall_module_with_dependencies("beautifulsoup4")
        uninstall_module_with_dependencies("ftfy")
        uninstall_module_with_dependencies("albumentations")
        uninstall_module_with_dependencies("datasets")
        uninstall_module_with_dependencies("deepspeed")
        uninstall_module_with_dependencies("gradio-client")
        uninstall_module_with_dependencies("insightface")
        uninstall_module_with_dependencies("suno-bark")
        uninstall_module_with_dependencies("aura-sr")
        uninstall_module_with_dependencies("peft")
        uninstall_module_with_dependencies("ultralytics")
        uninstall_module_with_dependencies("aura-sr")
        uninstall_module_with_dependencies("parler-tts")

        # "resemble-enhance":

        uninstall_module_with_dependencies("celluloid")
        uninstall_module_with_dependencies("omegaconf")
        uninstall_module_with_dependencies("pandas")
        uninstall_module_with_dependencies("ptflops")
        uninstall_module_with_dependencies("rich")
        uninstall_module_with_dependencies("resampy")
        uninstall_module_with_dependencies("tabulate")
        uninstall_module_with_dependencies("gradio")

        # WhisperSpeech

        uninstall_module_with_dependencies("ruamel.yaml.clib")
        uninstall_module_with_dependencies("fastprogress")
        uninstall_module_with_dependencies("fastcore")
        uninstall_module_with_dependencies("ruamel.yaml")
        uninstall_module_with_dependencies("hyperpyyaml")
        uninstall_module_with_dependencies("speechbrain")
        uninstall_module_with_dependencies("vocos")
        uninstall_module_with_dependencies("WhisperSpeech")
        uninstall_module_with_dependencies("pydub")

        subprocess.check_call([pybin, "-m", "pip", "cache", "purge"])

        self.report(
            {"INFO"},
            "\nRemove AI Models manually: \nLinux and macOS: ~/.cache/huggingface/hub\nWindows: %userprofile%\\.cache\\huggingface\\hub",
        )
        return {"FINISHED"}


def lcm_updated(self, context):
    scene = context.scene
    if scene.use_lcm:
        scene.movie_num_guidance = 0


def filter_updated(self, context):
    scene = context.scene
    if (scene.aurasr or scene.adetailer) and scene.movie_num_batch > 1 :
        scene.movie_num_batch = 1
        print("INFO: Aura SR and ADetailer will only allow for 1 batch for memory reasons.")


def input_strips_updated(self, context):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences
    movie_model_card = addon_prefs.movie_model_card
    image_model_card = addon_prefs.image_model_card
    scene = context.scene
    type = scene.generatorai_typeselect
    input = scene.input_strips
    if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0" and type == "movie":
        scene.input_strips = "input_strips"
    if type == "movie" or type == "audio" or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0":
        scene.inpaint_selected_strip = ""
    if (
        type == "image"
        and scene.input_strips != "input_strips"
        and (
            image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            or image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
            or image_model_card == "Salesforce/blipdiffusion"
            # or image_model_card == "h94/IP-Adapter"
        )
    ):
        scene.input_strips = "input_strips"

    if context.scene.lora_folder:
        bpy.ops.lora.refresh_files()
    if type == "text":
        scene.input_strips = "input_strips"

    if (type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid") or (
        type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
    ):
        scene.input_strips = "input_strips"
#    if movie_model_card == "guoyww/animatediff-motion-adapter-sdxl-beta" and type == "movie":
#        scene.input_strips = "input_prompt"
    if scene.input_strips == "input_prompt":
        bpy.types.Scene.movie_path = ""
        bpy.types.Scene.image_path = ""
#    if movie_model_card == "wangfuyun/AnimateLCM" and type == "movie":
#        scene.input_strips = "input_prompt"
    if (movie_model_card == "THUDM/CogVideoX-5b" or movie_model_card == "THUDM/CogVideoX-2b") and type == "movie":
        #scene.input_strips = "input_prompt"
        scene.generate_movie_x = 720
        scene.generate_movie_y = 480
        scene.generate_movie_frames = 48
        scene.movie_num_inference_steps = 50
        scene.movie_num_guidance = 6

        #scene.input_strips = "input_prompt"


    if (image_model_card == "dataautogpt3/OpenDalleV1.1") and type == "image":
        bpy.context.scene.use_lcm = False
    if (image_model_card == "Kwai-Kolors/Kolors-diffusers") and type == "image":
        bpy.context.scene.use_lcm = False
#    if (image_model_card == "Vargol/auraflow0.2-fp16-diffusers") and type == "image":
#        bpy.context.scene.use_lcm = False
    if (image_model_card == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers") and type == "image":
        bpy.context.scene.use_lcm = False
    if movie_model_card == "cerspense/zeroscope_v2_XL" and type == "movie":
        scene.upscale = False
    if (image_model_card == "black-forest-labs/FLUX.1-schnell") and type == "image":
        bpy.context.scene.movie_num_inference_steps = 4
    if (image_model_card == "ChuckMcSneed/FLUX.1-dev") and type == "image":
        bpy.context.scene.movie_num_inference_steps = 25
        bpy.context.scene.movie_num_guidance = 4
    if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0" and type == "audio":
        bpy.context.scene.movie_num_inference_steps = 200


def output_strips_updated(self, context):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences
    movie_model_card = addon_prefs.movie_model_card
    image_model_card = addon_prefs.image_model_card
    scene = context.scene
    type = scene.generatorai_typeselect
    input = scene.input_strips
    if type == "movie" or type == "audio" or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0":
        scene.inpaint_selected_strip = ""
        if context.scene.lora_folder:
            bpy.ops.lora.refresh_files()
    if (
        image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
        or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
        or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
        or image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
        or image_model_card == "Salesforce/blipdiffusion"
        # or image_model_card == "h94/IP-Adapter"
    ) and type == "image":
        scene.input_strips = "input_strips"

    if type == "text":
        scene.input_strips = "input_strips"
    if (type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid") or (
        type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
    ):
        scene.input_strips = "input_strips"

#    if movie_model_card == "guoyww/animatediff-motion-adapter-sdxl-beta" and type == "movie":
#        scene.input_strips = "input_prompt"
    if (image_model_card == "dataautogpt3/OpenDalleV1.1") and type == "image":
        bpy.context.scene.use_lcm = False
    if (image_model_card == "Kwai-Kolors/Kolors-diffusers") and type == "image":
        bpy.context.scene.use_lcm = False
#    if (image_model_card == "Vargol/auraflow0.2-fp16-diffusers") and type == "image":
#        bpy.context.scene.use_lcm = False
    if (image_model_card == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers") and type == "image":
        bpy.context.scene.use_lcm = False
    if movie_model_card == "cerspense/zeroscope_v2_XL" and type == "movie":
        scene.upscale = False


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
    default_folder = os.path.join(__file__,"sounds", "*.wav")
    #default_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds", "*.wav")
#    if default_folder not in sys.path:
#        sys.path.append(default_folder)
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
            (
                "stabilityai/stable-video-diffusion-img2vid-xt",
                "Stable Video Diffusion XT (576x1024x24) ",
                "stabilityai/stable-video-diffusion-img2vid-xt",
            ),
            (
                "stabilityai/stable-video-diffusion-img2vid",
                "Stable Video Diffusion (576x1024x14)",
                "stabilityai/stable-video-diffusion-img2vid",
            ),
            # Frame by Frame - disabled
            #            (
            #                "stabilityai/stable-diffusion-xl-base-1.0",
            #                "Img2img SD XL 1.0 Refine (1024x1024)",
            #                "Stable Diffusion XL 1.0",
            #            ),
            #            (
            #                "stabilityai/sd-turbo",
            #                "Img2img SD Turbo (512x512)",
            #                "stabilityai/sd-turbo",
            #            ),
            # ("VideoCrafter/Image2Video-512", "VideoCrafter v1 (512x512)", "VideoCrafter/Image2Video-512"),
            ("wangfuyun/AnimateLCM", "AnimateLCM", "wangfuyun/AnimateLCM"),
            ("THUDM/CogVideoX-2b", "CogVideoX-2b (720x480x48)", "THUDM/CogVideoX-2b"),
            ("THUDM/CogVideoX-5b", "CogVideoX-5b (720x480x48)", "THUDM/CogVideoX-5b"),
            (
                "cerspense/zeroscope_v2_XL",
                "Zeroscope XL (1024x576x24)",
                "Zeroscope XL (1024x576x24)",
            ),
#            (
#                "cerspense/zeroscope_v2_576w",
#                "Zeroscope (576x320x24)",
#                "Zeroscope (576x320x24)",
#            ),
            #            (
            #                "cerspense/zeroscope_v2_dark_30x448x256",
            #                "Zeroscope (448x256x30)",
            #                "Zeroscope (448x256x30)",
            #            ),
#            (
#                "guoyww/animatediff-motion-adapter-sdxl-beta",
#                "AnimateDiff",
#                "AnimateDiff",
#            ),
            # ("hotshotco/Hotshot-XL", "Hotshot-XL (512x512)", "Hotshot-XL (512x512)"),
            #            ("strangeman3107/animov-512x", "Animov (512x512)", "Animov (512x512)"),
            #            ("strangeman3107/animov-0.1.1", "Animov (448x384)", "Animov (448x384)"),
        ],
        default="stabilityai/stable-video-diffusion-img2vid",
        update=input_strips_updated,
    )
    image_model_card: bpy.props.EnumProperty(
        name="Image Model",
        items=[
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                "Stable Diffusion XL 1.0 (1024x1024)",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ),
            ("ByteDance/SDXL-Lightning", "Stable Diffusion XL Lightning (1024x1024)", "ByteDance/SDXL-Lightning"),
            ("stabilityai/stable-diffusion-3-medium-diffusers", "Stable Diffusion 3", "stabilityai/stable-diffusion-3-medium-diffusers"),
#            (
#                "Lykon/dreamshaper-8",
#                "Dreamshaper v8 (1024x1024)",
#                "Lykon/dreamshaper-8",
#            ),
#            ("Lykon/dreamshaper-xl-lightning", "Dreamshaper XL-Lightning (1024x1024)", "Lykon/dreamshaper-xl-lightning"),
            #            ("thibaud/sdxl_dpo_turbo", "SDXL DPO TURBO (1024x1024)", "thibaud/sdxl_dpo_turbo"),
            #            (
            #                "stabilityai/sdxl-turbo",
            #                "Stable Diffusion XL Turbo (512x512)",
            #                "stabilityai/sdxl-turbo",
            #            ),
            #            (
            #                "stabilityai/sd-turbo",
            #                "Stable Diffusion Turbo (512x512)",
            #                "stabilityai/sd-turbo",
            #            ),
            #            (
            #                "stabilityai/stable-diffusion-2",
            #                "Stable Diffusion 2 (768x768)",
            #                "stabilityai/stable-diffusion-2",
            #            ),
            #            (
            #                "runwayml/stable-diffusion-v1-5",
            #                "Stable Diffusion 1.5 (512x512)",
            #                "runwayml/stable-diffusion-v1-5",
            #            ),
            ("black-forest-labs/FLUX.1-schnell", "Flux Schnell (24 GB VRAM)", "black-forest-labs/FLUX.1-schnell"),
            ("ChuckMcSneed/FLUX.1-dev", "Flux 1 Dev (24 GB VRAM)", "ChuckMcSneed/FLUX.1-dev"),
            ("youknownothing/Fluently-XL-Final", "Fluently (1024x1024)", "youknownothing/Fluently-XL-Final"),
            ("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", "HunyuanDiT-v1.2", "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"),
#            ("RunDiffusion/Juggernaut-X-Hyper", "Juggernaut X Hyper (1024x1024)", "RunDiffusion/Juggernaut-X-Hyper"),
            ("Kwai-Kolors/Kolors-diffusers", "Kolors", "Kwai-Kolors/Kolors-diffusers"),
#            Not working:
#            ("Vargol/auraflow0.2-fp16-diffusers", "AuraFlow", "Vargol/auraflow0.2-fp16-diffusers"),
            ("Corcelio/mobius", "Mobius (1024x1024)", "Corcelio/mobius"),
#            ("Corcelio/openvision", "OpenVision (1280x1280)", "Corcelio/openvision"),
            ("dataautogpt3/OpenDalleV1.1", "OpenDalle (1024 x 1024)", "dataautogpt3/OpenDalleV1.1"),
#            (
#                "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
#                "PixArt Sigma T5 (1024x1024)",
#                "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
#            ),
            (
                "Vargol/PixArt-Sigma_16bit",
                "PixArt Sigma XL 16 bit(1024x1024)",
                "Vargol/PixArt-Sigma_16bit",
            ),
            (
                "Vargol/PixArt-Sigma_2k_16bit",
                "PixArt Sigma 2K 16 bit (2560x1440)",
                "Vargol/PixArt-Sigma_2k_16bit",
            ),
#            ("playgroundai/playground-v2.5-1024px-aesthetic", "Playground v2.5 (1024x1024)", "playgroundai/playground-v2.5-1024px-aesthetic"),
            (
                "Vargol/ProteusV0.4",
                "Proteus 0.4 (1024x1024)",
                "Vargol/ProteusV0.4",
            ),
#            ("dataautogpt3/ProteusV0.5", "Proteus 0.5 (1024 x 1024)", "dataautogpt3/ProteusV0.5"),
            (
                "SG161222/RealVisXL_V4.0",
                "RealVisXL_V4 (1024x1024)",
                "SG161222/RealVisXL_V4.0",
            ),  #
#            (
#                "segmind/SSD-1B",
#                "Segmind SSD-1B (1024x1024)",
#                "segmind/SSD-1B",
#            ),
            # ("h94/IP-Adapter", "IP-Adapter (512 x 512)", "h94/IP-Adapter"),
            # ("Vargol/PixArt-Sigma_16bit", "PixArt (1024 x 1024)", "Vargol/PixArt-Sigma_16bit"),
            ### ("ptx0/terminus-xl-gamma-v1", "Terminus XL Gamma v1", "ptx0/terminus-xl-gamma-v1"),
            #            ("warp-ai/wuerstchen", "Würstchen (1024x1024)", "warp-ai/wuerstchen"),
            #            (
            #                "playgroundai/playground-v2-1024px-aesthetic",
            #                "Playground v2 (1024x1024)",
            #                "playgroundai/playground-v2-1024px-aesthetic",
            #            ),
            (
                "Salesforce/blipdiffusion",
                "Blip Subject Driven (512x512)",
                "Salesforce/blipdiffusion",
            ),
            (
                "diffusers/controlnet-canny-sdxl-1.0-small",
                "Canny ControlNet (1024 x 1024)",
                "diffusers/controlnet-canny-sdxl-1.0-small",
            ),
            # Disabled - has log-in code.
            # ("DeepFloyd/IF-I-M-v1.0", "DeepFloyd/IF-I-M-v1.0", "DeepFloyd/IF-I-M-v1.0"),
            #            (
            #                "monster-labs/control_v1p_sdxl_qrcode_monster",
            #                "Illusion ControlNet",
            #                "monster-labs/control_v1p_sdxl_qrcode_monster",
            #            ),
            (
                "xinsir/controlnet-openpose-sdxl-1.0",
                "OpenPose ControlNet (1024 x 1024)",
                "xinsir/controlnet-openpose-sdxl-1.0",
            ),
            (
                "xinsir/controlnet-scribble-sdxl-1.0",
                "Scribble (1024x1024)",
                "xinsir/controlnet-scribble-sdxl-1.0",
            ),
        ],
        default="dataautogpt3/OpenDalleV1.1",
        update=input_strips_updated,
    )
    if low_vram():
        parler = ("parler-tts/parler-tts-mini-v1", "Speech: Parler TTS Mini", "parler-tts/parler-tts-mini-v1")
    else:
        parler = ("parler-tts/parler-tts-large-v1", "Speech: Parler TTS Large", "parler-tts/parler-tts-large-v1")

    if os_platform != "Linux":
        items=[
            ("stabilityai/stable-audio-open-1.0", "Stable Audio Open", "stabilityai/stable-audio-open-1.0"),
            (
                "facebook/musicgen-stereo-melody-large",
                "Music: MusicGen Stereo Melody",
                "facebook/musicgen-stereo-melody-large",
            ),
            (
                "cvssp/audioldm2-large",
                "Audio LDM 2 Large",
                "cvssp/audioldm2-large",
            ),
            parler,    
            ("bark", "Speech: Bark", "Bark"),
            ("WhisperSpeech", "Speech: WhisperSpeech", "WhisperSpeech")
        ]
    else:
        items=[
            ("stabilityai/stable-audio-open-1.0", "Stable Audio Open", "stabilityai/stable-audio-open-1.0"),
            (
                "facebook/musicgen-stereo-melody-large",
                "Music: MusicGen Stereo Melody",
                "facebook/musicgen-stereo-melody-large",
            ),
            (
                "cvssp/audioldm2-large",
                "Audio LDM 2 Large",
                "cvssp/audioldm2-large",
            ),
            parler,
        ]

    audio_model_card: bpy.props.EnumProperty(
        name="Audio Model",
        items=items,
        default="stabilityai/stable-audio-open-1.0",
        update=input_strips_updated,
    )
    # For DeepFloyd

    hugginface_token: bpy.props.StringProperty(
        name="Hugginface Token",
        default="hugginface_token",
        subtype="PASSWORD",
    )
    text_model_card: EnumProperty(
        name="Text Model",
        items={
            (
                "Salesforce/blip-image-captioning-large",
                "Blip Image Captioning",
                "Salesforce/blip-image-captioning-large",
            ),
            (
                "gokaygokay/Florence-2-Flux-Large",
                "Florence-2 Image Captioning",
                "gokaygokay/Florence-2-Flux-Large",
            ),            
            
        },
        default="Salesforce/blip-image-captioning-large",
    )
    generator_ai: StringProperty(
        name="Filepath",
        description="Path to the folder where the generated files are stored",
        subtype="DIR_PATH",
        default=join(bpy.utils.user_resource("DATAFILES"), "Generator AI"),
    )
    use_strip_data: BoolProperty(
        name="Use Input Strip Data",
        default=True,
    )
    local_files_only: BoolProperty(
        name="Use Local Files Only",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.operator("sequencer.install_generator")
        row.operator("sequencer.uninstall_generator")
        box.prop(self, "movie_model_card")
        box.prop(self, "image_model_card")
        if self.image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers":
            row = box.row(align=True)
            row.prop(self, "hugginface_token")
            row.operator("wm.url_open", text="", icon="URL").url = "https://huggingface.co/settings/tokens"
        box.prop(self, "audio_model_card")
        box.prop(self, "generator_ai")
        row = box.row(align=True)
        row.label(text="Notification:")
        row.prop(self, "playsound", text="")
        sub_row = row.row()
        sub_row.prop(self, "soundselect", text="")
        if self.soundselect == "user":
            sub_row.prop(self, "usersound", text="")
        sub_row.operator("renderreminder.pallaidium_play_notification", text="", icon="PLAY")
        sub_row.active = self.playsound

        row_row = box.row(align=True)
        row_row.label(text="Use Input Strip Data:")
        row_row.prop(self, "use_strip_data", text="")
        row_row.label(text="")
        row_row.label(text="")
        row_row.label(text="")

        row_row = box.row(align=True)
        row_row.label(text="Use Local Files Only:")
        row_row.prop(self, "local_files_only", text="")
        row_row.label(text="")
        row_row.label(text="")
        row_row.label(text="")


class GENERATOR_OT_sound_notification(Operator):
    """Test your notification settings"""

    bl_idname = "renderreminder.pallaidium_play_notification"
    bl_label = "Test Notification"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        if addon_prefs.playsound:
            device = aud.Device()

            def coinSound():
                sound = aud.Sound("")
                handle = device.play(sound.triangle(1000).highpass(20).lowpass(2000).ADSR(0, 0.5, 1, 0).fadeout(0.1, 0.1).limit(0, 1))
                handle = device.play(sound.triangle(1500).highpass(20).lowpass(2000).ADSR(0, 0.5, 1, 0).fadeout(0.2, 0.2).delay(0.1).limit(0, 1))

            def ding():
                sound = aud.Sound("")
                handle = device.play(sound.triangle(3000).highpass(20).lowpass(1000).ADSR(0, 0.5, 1, 0).fadeout(0, 1).limit(0, 1))

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


def copy_struct(source, target):
    if not source or not target:
        return
    for name, prop in source.bl_rna.properties.items():
        if name in ("rna_type", "name", "name_full", "original", "is_evaluated"):
            continue
        try:
            setattr(target, name, getattr(source, name))
        except AttributeError:
            new_source = getattr(source, name)
            new_target = getattr(target, name)
            if hasattr(new_source, "bl_rna"):
                copy_struct(new_source, new_target)
        except TypeError:
            pass


def get_render_strip(self, context, strip):
    """Render selected strip to hard-disk"""
    # Check for the context and selected strips

    if not context or not context.scene or not context.scene.sequence_editor:
        self.report({"ERROR"}, "No valid context or selected strips")
        return {"CANCELLED"}
    bpy.context.preferences.system.sequencer_proxy_setup = "MANUAL"
    current_scene = context.scene
    sequencer = current_scene.sequence_editor
    current_frame_old = bpy.context.scene.frame_current
    selected_sequences = strip

    # Get the first empty channel above all strips
    insert_channel_total = 1
    for s in sequencer.sequences_all:
        if s.channel >= insert_channel_total:
            insert_channel_total = s.channel + 1

    print("Strip type: "+str(strip.type))

    if strip.type in {
        "MOVIE",
        "IMAGE",
        "SOUND",
        "SCENE",
        "TEXT",
        "COLOR",
        "META",
        "MASK",
    }:
        # Deselect all strips in the current scene
        for s in sequencer.sequences_all:
            s.select = False

        # Select the current strip in the current scene
        strip.select = True

        # Store current frame for later
        bpy.context.scene.frame_current = int(strip.frame_start)

        #        if strip.type == "SCENE":
        #            bpy.data.scenes["Scene"].name

        if strip.type != "SCENE":
            # make_meta to keep transforms
            #bpy.ops.sequencer.meta_make()

            # Copy the strip to the clipboard
            bpy.ops.sequencer.copy()

            # unmeta
            #bpy.ops.sequencer.meta_separate()
#        else:
#            print("hep")
#            bpy.ops.sequencer.scene_strip_add(frame_start=0, channel=5, replace_sel=True)
#            new_strip = bpy.context.selected_sequences[0]
#            copy_struct(strip, new_strip)

        # Create a new scene
        # new_scene = bpy.data.scenes.new(name="New Scene")
        # Create a new scene
        new_scene = bpy.ops.scene.new(type="EMPTY")

        # Get the newly created scene
        new_scene = bpy.context.scene

        # Add a sequencer to the new scene
        new_scene.sequence_editor_create()

        # Set the new scene as the active scene
        context.window.scene = new_scene

        # Copy the scene properties from the current scene to the new scene
        new_scene.render.resolution_x = current_scene.render.resolution_x
        new_scene.render.resolution_y = current_scene.render.resolution_y
        new_scene.render.resolution_percentage = current_scene.render.resolution_percentage
        new_scene.render.pixel_aspect_x = current_scene.render.pixel_aspect_x
        new_scene.render.pixel_aspect_y = current_scene.render.pixel_aspect_y
        new_scene.render.fps = current_scene.render.fps
        new_scene.render.fps_base = current_scene.render.fps_base
        new_scene.render.sequencer_gl_preview = current_scene.render.sequencer_gl_preview
        new_scene.render.use_sequencer_override_scene_strip = current_scene.render.use_sequencer_override_scene_strip
        new_scene.world = current_scene.world

        area = [area for area in context.screen.areas if area.type == "SEQUENCE_EDITOR"][0]
        with bpy.context.temp_override(area=area):
            if strip.type == "SCENE":
                bpy.ops.sequencer.scene_strip_add(frame_start=0, channel=8, replace_sel=True)
            else:
                # Paste the strip from the clipboard to the new scene
                bpy.ops.sequencer.paste()
                #bpy.ops.sequencer.meta_separate()

        # Get the new strip in the new scene
        new_strip = new_scene.sequence_editor.active_strip = bpy.context.selected_sequences[0]

        copy_struct(strip, new_strip)

        # Set the range in the new scene to fit the pasted strip
        new_scene.frame_start = int(new_strip.frame_final_start)
        new_scene.frame_end = int(new_strip.frame_final_start + new_strip.frame_final_duration) - 1

        # Set the render settings for rendering animation with FFmpeg and MP4 with sound
        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.ffmpeg.format = "MPEG4"
        bpy.context.scene.render.ffmpeg.audio_codec = "AAC"

        # Make dir
        preferences = bpy.context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        rendered_dir = os.path.join(addon_prefs.generator_ai, str(date.today()))
        rendered_dir = os.path.join(rendered_dir, "Rendered_Strips")

        # Set the name of the file
        src_name = strip.name
        src_dir = ""
        src_ext = ".mp4"

        # Create a new folder for the rendered files
        if not os.path.exists(rendered_dir):
            os.makedirs(rendered_dir)

        # Set the output path for the rendering
        output_path = os.path.join(rendered_dir, src_name + "_rendered" + src_ext)
        output_path = ensure_unique_filename(output_path)
        new_scene.render.filepath = output_path

        # Render the strip to hard disk
        bpy.ops.render.opengl(animation=True, sequencer=True)

        # Delete the new scene
        bpy.data.scenes.remove(new_scene, do_unlink=True)

        if not os.path.exists(output_path):
            print("Render failed: " + output_path)
            bpy.context.preferences.system.sequencer_proxy_setup = "AUTOMATIC"
            return {"CANCELLED"}

        # Set the original scene as the active scene
        context.window.scene = current_scene

        # Reset to total top channel
        insert_channel = insert_channel_total
        area = [area for area in context.screen.areas if area.type == "SEQUENCE_EDITOR"][0]
        with bpy.context.temp_override(area=area):
            insert_channel = find_first_empty_channel(
                strip.frame_final_start,
                strip.frame_final_start + strip.frame_final_duration,
            )
            if strip.type == "SOUND":
                # Insert the rendered file as a sound strip in the original scene without video.

                bpy.ops.sequencer.sound_strip_add(
                    channel=insert_channel,
                    filepath=output_path,
                    frame_start=int(strip.frame_final_start),
                    overlap=0,
                )
#            elif strip.type == "SCENE":
#                # Insert the rendered file as a scene strip in the original scene.

#                bpy.ops.sequencer.movie_strip_add(
#                    channel=insert_channel,
#                    filepath=output_path,
#                    frame_start=int(strip.frame_final_start),
#                    overlap=0,
#                    sound=False,
#                )
            #            elif strip.type == "IMAGE":
            #                # Insert the rendered file as an image strip in the original scene.
            #                bpy.ops.sequencer.image_strip_add(
            #                    channel=insert_channel,
            #                    filepath=output_path,
            #                    frame_start=int(strip.frame_final_start),
            #                    overlap=0,
            #                    sound=False,
            #                )

            else:
                # Insert the rendered file as a movie strip in the original scene without sound.
                bpy.ops.sequencer.movie_strip_add(
                    channel=insert_channel,
                    filepath=output_path,
                    frame_start=int(strip.frame_final_start),
                    overlap=0,
                    sound=False,
                )
        resulting_strip = sequencer.active_strip
        resulting_strip.use_proxy = False

        # Reset current frame
        bpy.context.scene.frame_current = current_frame_old
        bpy.context.preferences.system.sequencer_proxy_setup = "AUTOMATIC"
    return resulting_strip


# LoRA.
class LORABrowserFileItem(PropertyGroup):
    name: bpy.props.StringProperty()
    enabled: bpy.props.BoolProperty(default=True)
    weight_value: bpy.props.FloatProperty(default=1.0)
    index: bpy.props.IntProperty(name="Index", default=0)


class LORABROWSER_UL_files(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        split = row.split(factor=0.7)
        split.label(text=item.name)
        split.prop(item, "weight_value", text="", emboss=False)


def update_folder_callback(self, context):
    if context.scene.lora_folder:
        bpy.ops.lora.refresh_files()


class LORA_OT_RefreshFiles(Operator):
    bl_idname = "lora.refresh_files"
    bl_label = "Refresh Files"

    def execute(self, context):
        scene = context.scene
        directory = bpy.path.abspath(scene.lora_folder)
        if not directory:
            self.report({"ERROR"}, "No folder selected")
            return {"CANCELLED"}
        lora_files = scene.lora_files
        lora_files.clear()
        for filename in os.listdir(directory):
            if filename.endswith(".safetensors"):
                file_item = lora_files.add()
                file_item.name = filename.replace(".safetensors", "")
                file_item.enabled = False
                file_item.weight_value = 1.0
        return {"FINISHED"}


class SEQUENCER_PT_pallaidium_panel(Panel):  # UI
    """Generate Media using AI"""

    bl_idname = "SEQUENCER_PT_sequencer_generate_movie_panel"
    bl_label = "Generative AI"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Generative AI"

    @classmethod
    def poll(cls, context):
        return context.area.type == "SEQUENCE_EDITOR"

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
        col = layout.column(align=False)
        col.use_property_split = True
        col.use_property_decorate = False
        col = col.box()
        col = col.column()
        # Input

        if image_model_card == "Salesforce/blipdiffusion" and type == "image":
            col.prop(context.scene, "input_strips", text="Source Image")
            col.prop(context.scene, "blip_cond_subject", text="Source Subject")
            # col.prop(context.scene, "blip_subject_image", text="Target Image")

            col.prop_search(
                scene,
                "blip_subject_image",
                scene.sequence_editor,
                "sequences",
                text="Target Image",
                icon="SEQ_STRIP_DUPLICATE",
            )
            col.prop(context.scene, "blip_tgt_subject", text="Target Subject")
        else:
            col.prop(context.scene, "input_strips", text="Input")
        if type != "text":
            if type != "audio":
                if (type == "movie") or ( # and movie_model_card != "guoyww/animatediff-motion-adapter-sdxl-beta"
                    type == "image"
                    # and image_model_card != "diffusers/controlnet-canny-sdxl-1.0-small"
                    and image_model_card != "xinsir/controlnet-openpose-sdxl-1.0"
                    # and image_model_card != "h94/IP-Adapter"
                    and image_model_card != "xinsir/controlnet-scribble-sdxl-1.0"
                    # and image_model_card!= "monster-labs/control_v1p_sdxl_qrcode_monster"
                    and image_model_card != "Salesforce/blipdiffusion"
                ):
                    if input == "input_strips" and not scene.inpaint_selected_strip:
                        col = col.column(heading="Use", align=True)
                        col.prop(addon_prefs, "use_strip_data", text=" Name & Seed")
                        col.prop(context.scene, "image_power", text="Strip Power")
                        if (type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid") or (
                            type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
                        ):
                            col.prop(context.scene, "svd_motion_bucket_id", text="Motion")
                            col.prop(
                                context.scene,
                                "svd_decode_chunk_size",
                                text="Decode Frames",
                            )
                    if (
                        bpy.context.scene.sequence_editor is not None
                        and image_model_card != "diffusers/controlnet-canny-sdxl-1.0-small"
                        and image_model_card != "ByteDance/SDXL-Lightning"
                    ):
                        if len(bpy.context.scene.sequence_editor.sequences) > 0:
                            if input == "input_strips" and type == "image":
                                col.prop_search(
                                    scene,
                                    "inpaint_selected_strip",
                                    scene.sequence_editor,
                                    "sequences",
                                    text="Inpaint Mask",
                                    icon="SEQ_STRIP_DUPLICATE",
                                )
            if image_model_card == "xinsir/controlnet-openpose-sdxl-1.0" and type == "image":
                col = col.column(heading="Read as", align=True)
                col.prop(context.scene, "openpose_use_bones", text="OpenPose Rig Image")
            if image_model_card == "xinsir/controlnet-scribble-sdxl-1.0" and type == "image":
                col = col.column(heading="Read as", align=True)
                col.prop(context.scene, "use_scribble_image", text="Scribble Image")

            # IPAdapter.
            if (
                (
                    image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                    or image_model_card == "runwayml/stable-diffusion-v1-5"
                    or image_model_card == "stabilityai/sdxl-turbo"
                    # or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
                    # or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
                    # or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
                )
                and type == "image"
                # and input != "input_strips"
            ):
                # col = layout.column(align=True)
                # col = col.box()
                # col = col.column(align=True)
                # col.use_property_split = True
                # col.use_property_decorate = False
                # Folder selection and refresh button

                row = col.row(align=True)
                row.prop(scene, "ip_adapter_face_folder", text="Adapter Face")
                row.operator("ip_adapter_face.file_browser", text="", icon="FILE_FOLDER")
                # if not scene.inpaint_selected_strip:

                row = col.row(align=True)
                row.prop(scene, "ip_adapter_style_folder", text="Adapter Style")
                row.operator("ip_adapter_style.file_browser", text="", icon="FILE_FOLDER")
            # Prompts

            col = layout.column(align=True)
            col = col.box()
            col = col.column(align=True)
            col.use_property_split = True
            col.use_property_decorate = False
            if (type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid") or (
                type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
            ):
                pass
            else:
                col.use_property_split = False
                col.use_property_decorate = False
                col.prop(context.scene, "generate_movie_prompt", text="", icon="ADD")
                if (type == "audio" and audio_model_card == "bark") or (type == "audio" and audio_model_card == "stabilityai/stable-audio-open-1.0") or (
                    type == "image" and image_model_card == "black-forest-labs/FLUX.1-schnell") or (
                    type == "image" and image_model_card == "ChuckMcSneed/FLUX.1-dev") or (
                    type == "audio" and audio_model_card == "facebook/musicgen-stereo-melody-large" and audio_model_card == "WhisperSpeech"
                ):
                    pass
                elif type == "audio" and (audio_model_card == "parler-tts/parler-tts-large-v1" or audio_model_card == "parler-tts/parler-tts-mini-v1"):
                    layout = col.column()
                    col = layout.column(align=True)
                    col.use_property_split = True
                    col.use_property_decorate = False
                    col.prop(
                        context.scene,
                        "parler_direction_prompt",
                        text="Direction",
                    )                
                else:
                    col.prop(
                        context.scene,
                        "generate_movie_negative_prompt",
                        text="",
                        icon="REMOVE",
                    )
                layout = col.column()
                col = layout.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
                if type != "audio":
                    col.prop(context.scene, "generatorai_styles", text="Style")
            layout = col.column()
            if type == "movie" or type == "image":
                col = layout.column(align=True)
                col.prop(context.scene, "generate_movie_x", text="X")
                col.prop(context.scene, "generate_movie_y", text="Y")
            col = layout.column(align=True)
            if type == "movie" or type == "image":
                col.prop(context.scene, "generate_movie_frames", text="Frames")
            if type == "audio" and audio_model_card != "bark" and audio_model_card != "WhisperSpeech"and audio_model_card != "parler-tts/parler-tts-large-v1":
                col.prop(context.scene, "audio_length_in_f", text="Frames")
            if type == "audio" and audio_model_card == "bark":
                col = layout.column(align=True)
                col.prop(context.scene, "speakers", text="Speaker")
                col.prop(context.scene, "languages", text="Language")
            elif type == "audio" and audio_model_card == "WhisperSpeech":
                row = col.row(align=True)
                row.prop(context.scene, "audio_path", text="Speaker")
                row.operator("sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER")
                col.prop(context.scene, "audio_speed", text="Speed")
#            elif type == "audio" and audio_model_card == "parler-tts/parler-tts-large-v1":
#                row = col.row(align=True)
#                row.prop(context.scene, "audio_path", text="Speaker")
#                # Speaker / Instruction

            elif type == "audio" and (addon_prefs.audio_model_card == "facebook/musicgen-stereo-melody-large" or addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0"):
                col.prop(context.scene, "movie_num_inference_steps", text="Quality Steps")
            else:
                if (
                    type == "image"
                    and image_model_card == "ByteDance/SDXL-Lightning"
                    or type == "image"
                    and image_model_card == "Lykon/dreamshaper-xl-lightning"
                ):
                    pass
                else:
                    col.prop(context.scene, "movie_num_inference_steps", text="Quality Steps")
                if (
                    (type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid")
                    or (type == "movie" and movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt")
                    or (type == "image" and image_model_card == "black-forest-labs/FLUX.1-schnell")
                    #or (type == "image" and image_model_card == "ChuckMcSneed/FLUX.1-dev")
                    or (
                        scene.use_lcm
                        and not (type == "image" and image_model_card == "Lykon/dreamshaper-8")
                        and not (
                            type == "image"
                            and image_model_card == image_model_card == "ByteDance/SDXL-Lightning"
#                            or type == "image"
#                            and image_model_card == "dataautogpt3/ProteusV0.5"
                            or type == "image"
                            and image_model_card == "Lykon/dreamshaper-xl-lightning"
                        )
                    )
                ):
                    pass
                else:
                    col.prop(context.scene, "movie_num_guidance", text="Word Power")
            col = col.column()
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
                col = col.column(heading="Upscale", align=True)
                col.prop(context.scene, "video_to_video", text="2x")
            if type == "image":
                col = col.column(heading="Enhance", align=True)
                row = col.row()
                row.prop(context.scene, "refine_sd", text="Quality")
                sub_col = col.row()
                sub_col.active = context.scene.refine_sd
                #            if type != "audio":
                #                row = col.row()
                ##                if type == "movie" or (
                ##                    type == "image"
                ##                    and image_model_card != "diffusers/controlnet-canny-sdxl-1.0-small"
                ##                    and image_model_card != "xinsir/controlnet-openpose-sdxl-1.0"
                ##                    and image_model_card != "xinsir/controlnet-scribble-sdxl-1.0"
                ##                    and image_model_card
                ##                    != "monster-labs/control_v1p_sdxl_qrcode_monster"
                ##                    and image_model_card != "Salesforce/blipdiffusion"
                ##                ):
                ##                    row.prop(context.scene, "use_freeU", text="FreeU")
                #                if type == "image":

                if (
                    (type == "image" and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0")
                    # or (type == "image" and image_model_card == "segmind/SSD-1B")
                    or (type == "image" and image_model_card == "xinsir/controlnet-openpose-sdxl-1.0")
                    or (type == "image" and image_model_card == "xinsir/controlnet-scribble-sdxl-1.0")
                    or (type == "image" and image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small")
                    or (type == "image" and image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster")
                    or (type == "image" and image_model_card == "segmind/Segmind-Vega")
                    or (type == "image" and image_model_card == "runwayml/stable-diffusion-v1-5")
                    or (type == "image" and image_model_card == "Lykon/dreamshaper-8")
                    or (type == "image" and image_model_card == "Vargol/PixArt-Sigma_16bit")
                    or (type == "image" and image_model_card == "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers")
                    or (type == "image" and image_model_card == "Vargol/PixArt-Sigma_2k_16bit")
                ):
                    row.prop(context.scene, "use_lcm", text="Speed")
#                if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
#                    col = col.column(heading="Resolution", align=True)
#                    row = col.row()
#                    row.prop(context.scene, "hidiff", text="HiDiff")

#                # ADetailer - Switch off for release:
                if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    col = col.column(heading="Details", align=True)

                row = col.row()
                # enable this for Adetailer
                #row.prop(context.scene, "adetailer", text="Face")

#                # AuraSR
#                if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                #col = col.column(heading="Upscale", align=True)
                #row = col.row()
                row.prop(context.scene, "aurasr", text="Upscale 4x")

        # LoRA.
        if (
            (
                image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                or image_model_card == "runwayml/stable-diffusion-v1-5"
                or image_model_card == "stabilityai/sdxl-turbo"
                or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
                or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
                or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            )
            and type == "image"
            # and input != "input_strips"
        ):
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column(align=True)
            col = col.box()
            col = col.column(align=True)
            col.use_property_split = False
            col.use_property_decorate = False

            # Folder selection and refresh button
            row = col.row(align=True)
            row.prop(scene, "lora_folder", text="LoRA")
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
        # Output.

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.box()
        col = col.column(align=True)
        col.prop(context.scene, "generatorai_typeselect", text="Output")
        if type == "image":
            col.prop(addon_prefs, "image_model_card", text=" ")
            if addon_prefs.image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers":
                row = col.row(align=True)
                row.prop(addon_prefs, "hugginface_token")
                row.operator("wm.url_open", text="", icon="URL").url = "https://huggingface.co/settings/tokens"
        if type == "movie":
            col.prop(addon_prefs, "movie_model_card", text=" ")
        if type == "audio":
            col.prop(addon_prefs, "audio_model_card", text=" ")
        if type == "text":
            col.prop(addon_prefs, "text_model_card", text=" ")
        if type != "text":
            col = col.column()
            col.prop(context.scene, "movie_num_batch", text="Batch Count")

        # Generate.
        col = layout.column()
        col = col.box()
        if input == "input_strips":
            ed = scene.sequence_editor
            row = col.row(align=True)
            row.scale_y = 1.2
            row.operator("sequencer.text_to_generator", text="Generate from Strips")
        else:
            row = col.row(align=True)
            row.scale_y = 1.2
            if type == "movie":
                #                if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                #                    row.operator(
                #                        "sequencer.text_to_generator", text="Generate from Strips"
                #                    )
                #                else:

                if movie_model_card == "stabilityai/sd-turbo":
                    row.operator("sequencer.text_to_generator", text="Generate from Strips")
                else:
                    row.operator("sequencer.generate_movie", text="Generate")
            if type == "image":
                row.operator("sequencer.generate_image", text="Generate")
            if type == "audio":
                row.operator("sequencer.generate_audio", text="Generate")


class NoWatermark:
    def apply_watermark(self, img):
        return img


class SEQUENCER_OT_generate_movie(Operator):
    """Generate Video"""

    bl_idname = "sequencer.generate_movie"
    bl_label = "Prompt"
    bl_description = "Convert text to video"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

#        if not scene.generate_movie_prompt:
#            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
#            return {"CANCELLED"}
        try:
            import torch
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
        input = scene.input_strips

        if not seq_editor:
            scene.sequence_editor_create()
        # clear the VRAM

        clear_cuda_cache()

        current_frame = scene.frame_current
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        negative_prompt = scene.generate_movie_negative_prompt + ", " + style_prompt(scene.generate_movie_prompt)[1] + ", nsfw, nude, nudity"
        movie_x = scene.generate_movie_x
        movie_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_16(movie_x)
        y = scene.generate_movie_y = closest_divisible_16(movie_y)
        duration = scene.generate_movie_frames
        movie_num_inference_steps = scene.movie_num_inference_steps
        movie_num_guidance = scene.movie_num_guidance
        input = scene.input_strips
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        local_files_only = addon_prefs.local_files_only
        movie_model_card = addon_prefs.movie_model_card
        image_model_card = addon_prefs.image_model_card
        pipe = None
        flush()

        # LOADING MODELS

        print("Model:  " + movie_model_card)

        # Models for refine imported image or movie
        if (
            (scene.movie_path or scene.image_path)
            and input == "input_strips"
            #and movie_model_card != "guoyww/animatediff-motion-adapter-sdxl-beta"
            and movie_model_card != "wangfuyun/AnimateLCM"
            and movie_model_card != "THUDM/CogVideoX-5b"
            and movie_model_card != "THUDM/CogVideoX-2b"
        ):

            if movie_model_card == "stabilityai/sd-turbo":  # img2img
                from diffusers import AutoPipelineForImage2Image

                # from diffusers.utils import load_image

                pipe = AutoPipelineForImage2Image.from_pretrained(
                    "stabilityai/sd-turbo",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )

                from diffusers import DPMSolverMultistepScheduler

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)
            # img2img SDXL - disabled
            #                from diffusers import StableDiffusionXLImg2ImgPipeline
            #                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            #                    "stabilityai/stable-diffusion-xl-refiner-1.0",
            #                    text_encoder_2=pipe.text_encoder_2,
            #                    vae=pipe.vae,
            #                    torch_dtype=torch.float16,
            #                    variant="fp16",
            #                )
            #                if low_vram():
            #                    refiner.enable_model_cpu_offload()
            #                    # refiner.enable_vae_tiling()
            #                    # refiner.enable_vae_slicing()
            #                else:
            #                    refiner.to(gfx_device)
            #            if (
            #                movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
            #            ):  # img2img
            #                from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
            #                vae = AutoencoderKL.from_pretrained(
            #                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            #                )
            #                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            #                    movie_model_card,
            #                    torch_dtype=torch.float16,
            #                    variant="fp16",
            #                    vae=vae,
            #                )
            #                from diffusers import DPMSolverMultistepScheduler
            #                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            #                    pipe.scheduler.config
            #                )
            #                pipe.watermark = NoWatermark()
            #                if low_vram():
            #                    pipe.enable_model_cpu_offload()
            #                    # pipe.unet.enable_forward_chunking(chunk_size=1, dim=1) # Heavy
            #                    # pipe.enable_vae_slicing()
            #                else:
            #                    pipe.to(gfx_device)
            #                from diffusers import StableDiffusionXLImg2ImgPipeline
            #                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            #                    "stabilityai/stable-diffusion-xl-refiner-1.0",
            #                    text_encoder_2=pipe.text_encoder_2,
            #                    vae=pipe.vae,
            #                    torch_dtype=torch.float16,
            #                    variant="fp16",
            #                )
            #                if low_vram():
            #                    refiner.enable_model_cpu_offload()
            #                    # refiner.enable_vae_tiling()
            #                    # refiner.enable_vae_slicing()
            #                else:
            #                    refiner.to(gfx_device)

            elif (
                movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                or movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
            ):  # or movie_model_card == "vdo/stable-video-diffusion-img2vid-fp16"):
                from diffusers import StableVideoDiffusionPipeline
                from diffusers.utils import load_image, export_to_video

                #                from .lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
                #                noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
                #                    num_train_timesteps=40,
                #                    sigma_min=0.002,
                #                    sigma_max=700.0,
                #                    sigma_data=1.0,
                #                    s_noise=1.0,
                #                    rho=7,
                #                    clip_denoised=False,
                #                )

                if movie_model_card == "stabilityai/stable-video-diffusion-img2vid":
                    # Version 1.1 - too heavy
                    # refiner = StableVideoDiffusionPipeline.from_single_file(
                    # "https://huggingface.co/vdo/stable-video-diffusion-img2vid-fp16/blob/main/svd_image_decoder-fp16.safetensors",

                    refiner = StableVideoDiffusionPipeline.from_pretrained(
                        movie_model_card,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )
                if movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt":
                    # Version 1.1 - too heavy
                    # refiner = StableVideoDiffusionPipeline.from_single_file(
                    # "https://huggingface.co/vdo/stable-video-diffusion-img2vid-fp16/blob/main/svd_xt_image_decoder-fp16.safetensors",

                    refiner = StableVideoDiffusionPipeline.from_pretrained(
                        "vdo/stable-video-diffusion-img2vid-xt-1-1",
                        # movie_model_card,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )
                #                    model_select = "AnimateLCM-SVD-xt-1.1.safetensors"
                #                    refinere.unet.cpu()
                #                    file_path = os.path.join("./safetensors", model_select)
                #                    state_dict = {}
                #                    with safe_open(file_path, framework="pt", device="cpu") as f:
                #                        for key in f.keys():
                #                            state_dict[key] = f.get_tensor(key)
                #                    missing, unexpected = refiner.unet.load_state_dict(state_dict, strict=True)
                #                    pipe.unet.cuda()
                #                    del state_dict

                if low_vram():
                    refiner.enable_model_cpu_offload()
                    refiner.unet.enable_forward_chunking()
                else:
                    refiner.to(gfx_device)
            else:  # vid2vid / img2vid
                if (
                    movie_model_card == "cerspense/zeroscope_v2_dark_30x448x256"
                    or movie_model_card == "cerspense/zeroscope_v2_576w"
                    or scene.image_path
                ):
                    card = "cerspense/zeroscope_v2_XL"
                else:
                    card = movie_model_card

                from diffusers import VideoToVideoSDPipeline

                upscale = VideoToVideoSDPipeline.from_pretrained(
                    card,
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                )

                from diffusers import DPMSolverMultistepScheduler

                upscale.scheduler = DPMSolverMultistepScheduler.from_config(upscale.scheduler.config)
                if low_vram():
                    upscale.enable_model_cpu_offload()
                else:
                    upscale.to(gfx_device)

        # Models for movie generation
        else:

#            if movie_model_card == "guoyww/animatediff-motion-adapter-sdxl-beta":
#                print("Importing: AnimateDiff Pipeline")
#
#                #import torch
#                from diffusers.models import MotionAdapter
#                from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
#                from diffusers.utils import export_to_gif

#                adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)

#                model_id = "stabilityai/stable-diffusion-xl-base-1.0"
#                scheduler = DDIMScheduler.from_pretrained(
#                    model_id,
#                    subfolder="scheduler",
#                    clip_sample=False,
#                    timestep_spacing="linspace",
#                    beta_schedule="linear",
#                    steps_offset=1,
#                )
#                pipe = AnimateDiffSDXLPipeline.from_pretrained(
#                    model_id,
#                    motion_adapter=adapter,
#                    scheduler=scheduler,
#                    torch_dtype=torch.float16,
#                    variant="fp16",
#                )
#
#                if low_vram():
#                    pipe.enable_model_cpu_offload()
#                    pipe.enable_vae_slicing()
#                    # pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)  # heavy:
#                else:
#                    pipe.to(gfx_device)

            if movie_model_card == "wangfuyun/AnimateLCM":

                import torch
                from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
                from diffusers.utils import export_to_gif

                adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
                # pipe = AnimateDiffPipeline.from_pretrained("dataautogpt3/OpenDalleV1.1", motion_adapter=adapter, torch_dtype=torch.float16, variant="fp16",)
                # pipe = AnimateDiffPipeline.from_pretrained("lykon/dreamshaper-8", motion_adapter=adapter, torch_dtype=torch.float16, variant="fp16",)

                pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

                pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
                pipe.set_adapters(["lcm-lora"], [0.8])

                if low_vram():
                    pipe.enable_vae_slicing()
                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)

            elif movie_model_card == "THUDM/CogVideoX-5b" or movie_model_card == "THUDM/CogVideoX-2b":

                #vid2vid
                if ((scene.movie_path or scene.image_path)and input == "input_strips"):
                    from diffusers.utils import load_video
                    from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline
                    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(movie_model_card, torch_dtype=torch.bfloat16)

                #txt2vid
                else:
                    from diffusers import CogVideoXPipeline

                    pipe = CogVideoXPipeline.from_pretrained(
                        movie_model_card,
                        torch_dtype=torch.float16,
                    )

                if low_vram():
                    pipe.enable_model_cpu_offload()
                    pipe.enable_sequential_cpu_offload()
                    pipe.vae.enable_tiling()
                else:
                    if gfx_device == "cuda":
                        pipe.enable_model_cpu_offload()
                        #pipe.enable_sequential_cpu_offload()
                        #pipe.vae.enable_tiling()
                    else:    
                        pipe.to(gfx_device)
                if ((scene.movie_path or scene.image_path)and input == "input_strips"):
                    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
                scene.generate_movie_x = 720
                scene.generate_movie_y = 480

            elif movie_model_card == "VideoCrafter/Image2Video-512":
                from diffusers import StableDiffusionPipeline

                pipe = StableDiffusionPipeline.from_single_file(
                    "https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt",
                    torch_dtype=torch.float16,
                )
                from diffusers import DPMSolverMultistepScheduler

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                if low_vram():
                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)

            elif (
                movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                or movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
            ):
                print("Stable Video Diffusion needs image input")
                return {"CANCELLED"}

            else:
                from diffusers import TextToVideoSDPipeline
                import torch

                pipe = TextToVideoSDPipeline.from_pretrained(
                    movie_model_card,
                    torch_dtype=torch.float16,
                    use_safetensors=False,
                    local_files_only=local_files_only,
                )
                from diffusers import DPMSolverMultistepScheduler

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                if low_vram():
                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)

            # Model for upscale generated movie
            if scene.video_to_video:
                clear_cuda_cache()
                from diffusers import DiffusionPipeline

                upscale = DiffusionPipeline.from_pretrained(
                    "cerspense/zeroscope_v2_XL",
                    torch_dtype=torch.float16,
                    use_safetensors=False,
                    local_files_only=local_files_only,
                )
                upscale.scheduler = DPMSolverMultistepScheduler.from_config(upscale.scheduler.config)
                if low_vram():
                    upscale.enable_model_cpu_offload()
                else:
                    upscale.to(gfx_device)
        #        if scene.use_freeU and pipe:  # Free Lunch
        #            # -------- freeu block registration
        #            print("Process: FreeU")
        #            register_free_upblock3d(pipe)  # , b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        #            register_free_crossattn_upblock3d(pipe)  # , b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        #            # -------- freeu block registration

        # GENERATING - Main Loop Video
        for i in range(scene.movie_num_batch):

            start_time = timer()
            clear_cuda_cache()
            
            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = scene.sequence_editor.active_strip.frame_final_start + scene.sequence_editor.active_strip.frame_final_duration
                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * abs(duration)) + scene.frame_current,
                )
                start_frame = scene.frame_current

            # Get seed
            seed = context.scene.movie_num_seed
            seed = seed if not context.scene.movie_use_random else random.randint(-2147483647, 2147483647)
            print("Seed: " + str(seed))
            context.scene.movie_num_seed = seed

            # Use cuda if possible
            if (
                torch.cuda.is_available()
                and movie_model_card != "stabilityai/stable-video-diffusion-img2vid"
                and movie_model_card != "stabilityai/stable-video-diffusion-img2vid-xt"
            ):
                generator = torch.Generator(gfx_device).manual_seed(seed) if seed != 0 else None
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None

            # Process batch input for images
            if (
                (scene.movie_path or scene.image_path)
                and input == "input_strips"

                #and movie_model_card != "guoyww/animatediff-motion-adapter-sdxl-beta"
            ):
                video_path = scene.movie_path

                #                # img2img
                #                if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                #                    print("Process: Frame by frame (SD XL)")
                #                    input_video_path = video_path
                #                    output_video_path = solve_path("temp_images")
                #                    if scene.movie_path:
                #                        frames = process_video(input_video_path, output_video_path)
                #                    elif scene.image_path:
                #                        frames = process_image(
                #                            scene.image_path, int(scene.generate_movie_frames)
                #                        )
                #                    video_frames = []
                #                    # Iterate through the frames
                #                    for frame_idx, frame in enumerate(
                #                        frames
                #                    ):  # would love to get this flicker free
                #                        print(str(frame_idx + 1) + "/" + str(len(frames)))
                #                        image = refiner(
                #                            prompt,
                #                            negative_prompt=negative_prompt,
                #                            num_inference_steps=movie_num_inference_steps,
                #                            strength=1.00 - scene.image_power,
                #                            guidance_scale=movie_num_guidance,
                #                            image=frame,
                #                            generator=generator,
                #                        ).images[0]
                #                        video_frames.append(image)
                #                        if torch.cuda.is_available():
                #                            torch.cuda.empty_cache()
                #                    video_frames = np.array(video_frames)

                # img2img
                if movie_model_card == "stabilityai/sd-turbo":
                    print("Process: Frame by frame (SD Turbo)")
                    input_video_path = video_path
                    output_video_path = solve_path("temp_images")
                    if scene.movie_path:
                        frames = process_video(input_video_path, output_video_path)
                    elif scene.image_path:
                        frames = process_image(scene.image_path, int(scene.generate_movie_frames))
                    video_frames = []
                    # Iterate through the frames

                    for frame_idx, frame in enumerate(frames):  # would love to get this flicker free
                        print(str(frame_idx + 1) + "/" + str(len(frames)))
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=2,  # movie_num_inference_steps,
                            strength=0.5,  # scene.image_power,
                            guidance_scale=3.0,
                            image=frame,
                            generator=generator,
                        ).images[0]
                        video_frames.append(image)
                        clear_cuda_cache()
                    video_frames = np.array(video_frames)

                # vid2vid / img2vid
                elif (
                    movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                    or movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
                ):

                    if scene.movie_path:
                        print("Process: Video Image to SVD Video")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_first_frame(bpy.path.abspath(scene.movie_path))

                    elif scene.image_path:
                        print("Process: Image to SVD Video")
                        if not os.path.isfile(scene.image_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(bpy.path.abspath(scene.image_path))

                    image = image.resize((closest_divisible_16(int(x)), closest_divisible_16(int(y))))

                    video_frames = refiner(
                        image,
                        noise_aug_strength=1.00 - scene.image_power,
                        decode_chunk_size=scene.svd_decode_chunk_size,
                        motion_bucket_id=scene.svd_motion_bucket_id,
                        num_inference_steps=movie_num_inference_steps,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]

                # needs to input image
                elif movie_model_card == "wangfuyun/AnimateLCM":
                    video_frames = pipe(
                        prompt=prompt,
                        #image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]
                    
                #vid2vid
                elif movie_model_card == "THUDM/CogVideoX-5b" or movie_model_card == "THUDM/CogVideoX-2b":
                    if scene.movie_path:
                        print("Process: Video to video (CogVideoX)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        #video = load_video_as_np_array(video_path)
                        video = load_video(video_path)[:49]

                    elif scene.image_path:
                        print("Process: Image to video (CogVideoX)")
                        if not os.path.isfile(scene.image_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        video = process_image(scene.image_path, int(scene.generate_movie_frames))
                        video = np.array(video)
#                    if not video.any():
#                        print("Loading of file failed")
#                        return {"CANCELLED"}

                    video_frames = pipe(
                        video=video,
                        prompt=prompt,
                        strength=1.00 - scene.image_power,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        #num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]
                    #prompt=prompt, strength=0.8, guidance_scale=6, num_inference_steps=50

                else:# movie_model_card != "guoyww/animatediff-motion-adapter-sdxl-beta":
                    if scene.movie_path:
                        print("Process: Video to video")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        video = load_video_as_np_array(video_path)
                    elif scene.image_path:
                        print("Process: Image to video")
                        if not os.path.isfile(scene.image_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        video = process_image(scene.image_path, int(scene.generate_movie_frames))
                        video = np.array(video)
                    if not video.any():
                        print("Loading of file failed")
                        return {"CANCELLED"}

                    # Upscale video
                    if scene.video_to_video:
                        video = [
                            Image.fromarray(frame).resize(
                                (
                                    closest_divisible_16(int(x * 2)),
                                    closest_divisible_16(int(y * 2)),
                                )
                            )
                            for frame in video
                        ]
                    else:
                        video = [
                            Image.fromarray(frame).resize(
                                (
                                    closest_divisible_16(int(x)),
                                    closest_divisible_16(int(y)),
                                )
                            )
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
                    ).frames[0]

            # Prompt input for movies
            else:

                print("Generate: Video from text")
                
                if movie_model_card == "THUDM/CogVideoX-5b" or movie_model_card == "THUDM/CogVideoX-2b":
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        num_videos_per_prompt=1,
                        height=480,
                        width=720,
                        #
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]
                    scene.generate_movie_x = 720
                    scene.generate_movie_y = 480

                else:
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]
                movie_model_card = addon_prefs.movie_model_card

                clear_cuda_cache()
                # Upscale video.
                if scene.video_to_video:
                    print("Upscale: Video")
                    clear_cuda_cache()
                    video = [Image.fromarray(frame).resize((closest_divisible_16(x * 2), closest_divisible_16(y * 2))) for frame in video_frames]
                    video_frames = upscale(
                        prompt,
                        video=video,
                        strength=1.00 - scene.image_power,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        generator=generator,
                    ).frames[0]
#            if movie_model_card == "guoyww/animatediff-motion-adapter-sdxl-beta":
#                # from diffusers.utils import export_to_video
#                # Move to folder.

#                video_frames = np.array(video_frames)
#                src_path = export_to_video(video_frames)
#                dst_path = solve_path(clean_filename(str(seed) + "_" + prompt) + ".mp4")
#                shutil.move(src_path, dst_path)
#            else:
            # Move to folder.
            render = bpy.context.scene.render
            fps = round((render.fps / render.fps_base), 3)
            src_path = export_to_video(video_frames, fps=fps)
            dst_path = solve_path(clean_filename(str(seed) + "_" + prompt) + ".mp4")
            shutil.move(src_path, dst_path)

            # Add strip.
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
                            scene.sequence_editor.active_strip = strip
                            strip.name = str(seed) + "_" + prompt
                            strip.use_proxy = True
                            bpy.ops.sequencer.rebuild_proxy()
                            if i > 0:
                                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start

                            # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
                            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                            break
            print_elapsed_time(start_time)
        pipe = None
        refiner = None
        converter = None
        # clear the VRAM

        clear_cuda_cache()
        flush()

        bpy.types.Scene.movie_path = ""
        if input != "input_strips":
            bpy.ops.renderreminder.pallaidium_play_notification()
        scene.frame_current = current_frame
        return {"FINISHED"}


class SequencerOpenAudioFile(Operator, ImportHelper):
    bl_idname = "sequencer.open_audio_filebrowser"
    bl_label = "Open Audio File Browser"
    filter_glob: StringProperty(
        default="*.wav;",
        options={"HIDDEN"},
    )

    def execute(self, context):
        scene = context.scene
        # Check if the file exists

        if self.filepath and os.path.exists(self.filepath):
            valid_extensions = {".wav"}
            filename, extension = os.path.splitext(self.filepath)
            if extension.lower() in valid_extensions:
                print("Selected audio file:", self.filepath)
                scene.audio_path = bpy.path.abspath(self.filepath)
            else:
                print("Info: Only wav is allowed.")
        else:
            self.report({"ERROR"}, "Selected file does not exist.")
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class SEQUENCER_OT_generate_audio(Operator):
    """Generate Audio"""

    bl_idname = "sequencer.generate_audio"
    bl_label = "Prompt"
    bl_description = "Convert text to audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
#        if not scene.generate_movie_prompt:
#            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
#            return {"CANCELLED"}
        if not scene.sequence_editor:
            scene.sequence_editor_create()
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        local_files_only = addon_prefs.local_files_only
        current_frame = scene.frame_current
        active_strip = scene.sequence_editor.active_strip
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        movie_num_inference_steps = scene.movie_num_inference_steps
        movie_num_guidance = scene.movie_num_guidance
        audio_length_in_s = scene.audio_length_in_f / (scene.render.fps / scene.render.fps_base)
        pipe = None
        import torch
        import torchaudio
        import scipy
        import random
        from scipy.io.wavfile import write as write_wav

        if addon_prefs.audio_model_card == "facebook/musicgen-stereo-melody-large":
            #            if os_platform == "Darwin" or os_platform == "Linux":
            #                import sox
            #            else:

            import soundfile as sf

        if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0":
            try:
                import scipy
                import torch
                import soundfile as sf
                from diffusers import StableAudioPipeline
            except ModuleNotFoundError:
                print("Dependencies needs to be installed in the add-on preferences.")
                self.report(
                    {"INFO"},
                    "Dependencies needs to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}

        if addon_prefs.audio_model_card == "WhisperSpeech":
            import numpy as np

            try:
                from whisperspeech.pipeline import Pipeline
                from resemble_enhance.enhancer.inference import denoise, enhance
            except ModuleNotFoundError:
                print("Dependencies needs to be installed in the add-on preferences.")
                self.report(
                    {"INFO"},
                    "Dependencies needs to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}

        if addon_prefs.audio_model_card == "parler-tts/parler-tts-large-v1" or addon_prefs.audio_model_card == "parler-tts/parler-tts-mini-v1":
            import numpy as np

            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                from transformers import AutoTokenizer
                import soundfile as sf
            except ModuleNotFoundError:
                print("Dependencies needs to be installed in the add-on preferences.")
                self.report(
                    {"INFO"},
                    "Dependencies needs to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}

        if addon_prefs.audio_model_card == "bark":
            #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            try:
                import numpy as np
                from bark.generation import (
                    generate_text_semantic,
                    preload_models,
                )
                from bark.api import semantic_to_waveform
                from bark import generate_audio, SAMPLE_RATE

                from resemble_enhance.enhancer.inference import denoise, enhance
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
        clear_cuda_cache()

        # Load models Audio
        print("Model:  " + addon_prefs.audio_model_card)
        
        if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0":

            repo_id = "ylacombe/stable-audio-1.0"
            pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        elif addon_prefs.audio_model_card == "cvssp/audioldm2-large":
            repo_id = addon_prefs.audio_model_card
            from diffusers import AudioLDM2Pipeline

            pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Musicgen
        elif addon_prefs.audio_model_card == "facebook/musicgen-stereo-melody-large":
            from transformers import pipeline
            from transformers import set_seed

            pipe = pipeline(
                "text-to-audio",
                "facebook/musicgen-stereo-melody-large",
                device="cuda:0",
                torch_dtype=torch.float16,
            )

            if int(audio_length_in_s * 50) > 1503:
                self.report({"INFO"}, "Maximum output duration is 30 sec.")

        # Bark
        elif addon_prefs.audio_model_card == "bark":
            preload_models(
                text_use_small=True,
                coarse_use_small=True,
                fine_use_gpu=True,
                fine_use_small=True,
            )

        # WhisperSpeech
        elif addon_prefs.audio_model_card == "WhisperSpeech":
            from whisperspeech.pipeline import Pipeline

            pipe = Pipeline(s2a_ref="collabora/whisperspeech:s2a-q4-small-en+pl.model")

        # Parler
        elif addon_prefs.audio_model_card == "parler-tts/parler-tts-large-v1" or addon_prefs.audio_model_card == "parler-tts/parler-tts-mini-v1":
            pipe = ParlerTTSForConditionalGeneration.from_pretrained(addon_prefs.audio_model_card).to(gfx_device)
            tokenizer = AutoTokenizer.from_pretrained(addon_prefs.audio_model_card)

        # Deadend
        else:
            print("Audio model not found.")
            self.report({"INFO"}, "Audio model not found.")
            return {"CANCELLED"}

        # Main loop Audio
        for i in range(scene.movie_num_batch):

            start_time = timer()

            # Find free space for the strip in the timeline.

            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = scene.sequence_editor.active_strip.frame_final_start + scene.sequence_editor.active_strip.frame_final_duration
                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
            else:
                if input != "input_strips":
                    empty_channel = find_first_empty_channel(
                        scene.frame_current,
                        (scene.movie_num_batch * (len(prompt) * 4)) + scene.frame_current,
                    )
                else:
                    empty_channel = find_first_empty_channel(
                        active_strip.frame_final_start,
                        (scene.movie_num_batch * (len(prompt) * 4)) + scene.frame_current,
                    )
                start_frame = scene.frame_current

            # Get seed
            seed = context.scene.movie_num_seed
            seed = seed if not context.scene.movie_use_random else random.randint(-2147483647, 2147483647)
            print("Seed: " + str(seed))
            context.scene.movie_num_seed = seed

            # Use cuda if possible
            if (
                torch.cuda.is_available()
            ):
                generator = torch.Generator(gfx_device).manual_seed(seed) if seed != 0 else None
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None

            # Stable Open Audio
            if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0":
                import random
                print("Generate: Stable Open Audio")
                seed = context.scene.movie_num_seed
                seed = seed if not context.scene.movie_use_random else random.randint(0, 999999)
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed
                filename = solve_path(clean_filename(str(seed) + "_" + prompt) + ".wav")

                audio = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=movie_num_inference_steps,
                    audio_end_in_s=audio_length_in_s,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                ).audios

                output = audio[0].T.float().cpu().numpy()
                sf.write(filename, output, pipe.vae.sampling_rate)

#                # Rearrange audio batch to a single sequence
#                output = rearrange(output, "b d n -> d (b n)")

#                # Peak normalize, clip, convert to int16, and save to file
#                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

#                # Ensure the output tensor has the right shape
#                if output.ndim == 1:
#                    output = output.unsqueeze(0)  # Make it 2D: (channels x samples)

#                max_length = int(sample_rate * audio_length_in_s)
#                if output.shape[1] > max_length:
#                    output = output[:, :max_length]

#                torchaudio.save(filename, output, sample_rate)

            # Bark.
            elif addon_prefs.audio_model_card == "bark":
                print("Generate: Speech (Bark)")

                rate = SAMPLE_RATE
                GEN_TEMP = 0.6
                SPEAKER = "v2/" + scene.languages + "_" + scene.speakers
                silence = np.zeros(int(0.28 * rate))  # quarter second of silence
                prompt = context.scene.generate_movie_prompt
                prompt = prompt.replace("\n", " ").strip()
                sentences = split_and_recombine_text(prompt, desired_length=120, max_length=150)
                pieces = []
                for sentence in sentences:
                    print("Sentence: " + sentence)
                    semantic_tokens = generate_text_semantic(
                        sentence,
                        history_prompt=SPEAKER,
                        temp=GEN_TEMP,
                        # min_eos_p=0.1,  # this controls how likely the generation is to end
                    )
                    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
                    pieces += [audio_array, silence.copy()]
                audio = np.concatenate(pieces)
                filename = solve_path(clean_filename(prompt) + ".wav")

                # Write the combined audio to a file
                write_wav(filename, rate, audio.transpose())

                # resemble_enhance
                dwav, sr = torchaudio.load(filename)
                # print("sr_load " + str(sr))

                dwav = dwav.mean(dim=0)

                wav2, new_sr = enhance(
                    dwav=dwav, sr=sr, device=gfx_device, nfe=64, chunk_seconds=10, chunks_overlap=1, solver="midpoint", lambd=0.1, tau=0.5
                )

                wav2 = wav2.cpu().numpy()

                # Write the combined audio to a file
                write_wav(filename, new_sr, wav2)

            # WhisperSpeech

            elif addon_prefs.audio_model_card == "WhisperSpeech":

                prompt = context.scene.generate_movie_prompt
                prompt = prompt.replace("\n", " ").strip()
                filename = solve_path(clean_filename(prompt) + ".wav")
                if scene.audio_path:
                    speaker = scene.audio_path
                else:
                    speaker = None
                #                sentences = split_and_recombine_text(
                #                    prompt, desired_length=250, max_length=320
                #                )
                #                pieces = []
                #                #pieces.append(silence.copy())
                #                for sentence in sentences:
                #                    print("Sentence: " + sentence)
                ##                    semantic_tokens = generate_text_semantic(
                ##                        sentence,
                ##                        history_prompt=SPEAKER,
                ##                        temp=GEN_TEMP,
                ##                        # min_eos_p=0.1,  # this controls how likely the generation is to end
                ##                    )
                ##                    audio_array = semantic_to_waveform(
                ##                        semantic_tokens, history_prompt=SPEAKER
                ##                    )
                #                    audio_array = pipe.generate(sentence, speaker=speaker, lang='en', cps=int(scene.audio_speed))
                #                    audio_piece = (audio_array.cpu().numpy() * 32767).astype(np.int16)
                #                    #pieces += [np.expand_dims(audio_piece, axis=0), np.expand_dims(silence.copy(), axis=0)]

                #                    #pieces += [audio_array.cpu().numpy().astype(np.int16)]
                #                    #pieces.append(audio_piece)
                #                    pieces += [silence.copy(), audio_piece]
                #                audio = pieces.numpy()#np.concatenate(pieces)
                #                filename = solve_path(clean_filename(prompt) + ".wav")
                #                # Write the combined audio to a file
                #                write_wav(filename, rate, audio.transpose())

                pipe.generate_to_file(filename, prompt, speaker=speaker, lang="en", cps=int(scene.audio_speed))

            # Musicgen.

            elif addon_prefs.audio_model_card == "facebook/musicgen-stereo-melody-large":

                print("Generate: MusicGen Stereo")
                #print("Prompt: " + prompt)
                seed = context.scene.movie_num_seed
                seed = seed if not context.scene.movie_use_random else random.randint(0, 999999)
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed
                set_seed(seed)
                music = pipe(
                    prompt,
                    forward_params={"max_new_tokens": int(min(audio_length_in_s * 50, 1503))},
                )
                filename = solve_path(clean_filename(str(seed) + "_" + prompt) + ".wav")
                rate = 48000
                #                if os_platform == "Darwin" or os_platform == "Linux":
                #                    tfm = sox.Transformer()
                #                    tfm.build_file(
                #                    input_array=music["audio"][0].T,
                #                    sample_rate_in=music["sampling_rate"],
                #                    output_filepath=filename
                #                    )
                #                else:

                sf.write(filename, music["audio"][0].T, music["sampling_rate"])

            # MusicLDM ZAC
            elif addon_prefs.audio_model_card == "cvssp/audioldm2-large":
                print("Generate: Audio LDM2 Large")
                seed = context.scene.movie_num_seed
                seed = seed if not context.scene.movie_use_random else random.randint(0, 999999)
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed
                # Use cuda if possible
                if torch.cuda.is_available():
                    generator = torch.Generator(gfx_device).manual_seed(seed) if seed != 0 else None
                else:
                    if seed != 0:
                        generator = torch.Generator()
                        generator.manual_seed(seed)
                    else:
                        generator = None

                music = pipe(
                    prompt,
                    num_inference_steps=movie_num_inference_steps,
                    negative_prompt=negative_prompt,
                    audio_length_in_s=audio_length_in_s,
                    guidance_scale=movie_num_guidance,
                    generator=generator,
                ).audios[0]
                
                filename = solve_path(clean_filename(str(seed) + "_" + prompt) + ".wav")
                rate = 16000
                
                write_wav(filename, rate, music.transpose())

            # Parler
            elif addon_prefs.audio_model_card == "parler-tts/parler-tts-large-v1" or addon_prefs.audio_model_card == "parler-tts/parler-tts-mini-v1":
                prompt = prompt
                seed = context.scene.movie_num_seed
                seed = seed if not context.scene.movie_use_random else random.randint(0, 999999)
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed
                description = context.scene.parler_direction_prompt
                input_ids = tokenizer(description, return_tensors="pt").input_ids.to(gfx_device)
                prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gfx_device)

                generation = pipe.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                audio_arr = generation.cpu().numpy().squeeze()
                filename = solve_path(str(seed) + "_" + prompt + ".wav")
                sf.write(filename, audio_arr, pipe.config.sampling_rate)

            # Add Audio Strip
            filepath = filename
            if os.path.isfile(filepath):
                #                empty_channel = find_first_empty_channel(
                #                    start_frame, start_frame + scene.audio_length_in_f
                #                )

                strip = scene.sequence_editor.sequences.new_sound(
                    name=prompt,
                    filepath=filepath,
                    channel=empty_channel,
                    frame_start=start_frame,
                )
                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
                # Redraw UI to display the new strip. Remove this if Blender crashes:
                # https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution

                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            else:
                print("No resulting file found!")
            print_elapsed_time(start_time)
        if pipe:
            pipe = None

        # clear the VRAM
        clear_cuda_cache()

        if input != "input_strips":
            bpy.ops.renderreminder.pallaidium_play_notification()
        return {"FINISHED"}


def scale_image_within_dimensions(image, target_width=None, target_height=None):
    import cv2
    import numpy as np

    # img = cv2.imread(image_path)
    # height, width, layers = img.shape

    # Get the original image dimensions

    height, width, layers = image.shape

    # Calculate the aspect ratio

    aspect_ratio = width / float(height)

    # Calculate the new dimensions based on the target width or height

    if target_width is not None:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    elif target_height is not None:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        # If neither target width nor height is provided, return the original image

        return image
    # Use the resize function to scale the image

    scaled_image = cv2.resize(image, (new_width, new_height))

    return scaled_image


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(gfx_device)
    with torch.no_grad(), torch.autocast(gfx_device):
        depth_map = depth_estimator(image).predicted_depth
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


class IPAdapterFaceProperties(bpy.types.PropertyGroup):
    files_to_import: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)


class IPAdapterFaceFileBrowserOperator(Operator):
    bl_idname = "ip_adapter_face.file_browser"
    bl_label = "Open IP Adapter Face File Browser"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    import_as_folder: bpy.props.BoolProperty(name="Import as Folder", default=False)

    def execute(self, context):
        valid_image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".hdr"}
        scene = context.scene

        if self.filepath:
            if self.import_as_folder:
                files_to_import = bpy.context.scene.ip_adapter_face_files_to_import
                files_to_import.clear()
                # self.filepath = os.path.dirname(self.filepath)

                print("Importing folder:", self.filepath)
                for file_path in glob.glob(os.path.join(self.filepath, "*")):
                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext in valid_image_extensions:
                            print("Found image file in folder:", os.path.basename(file_path))
                            new_file = files_to_import.add()
                            # new_file.name = os.path.basename(self.filepath)

                            new_file.path = os.path.abspath(self.filepath)
                scene.ip_adapter_face_folder = os.path.abspath(os.path.dirname(self.filepath))
                self.report({"INFO"}, f"{len(files_to_import)} image files found in folder.")
            else:
                print("Importing file:", self.filepath)
                valid_file_ext = os.path.splitext(self.filepath)[1].lower()
                if valid_file_ext in valid_image_extensions:
                    print("Adding image file:", os.path.basename(self.filepath))
                    files_to_import = bpy.context.scene.ip_adapter_face_files_to_import
                    new_file = files_to_import.add()
                    # new_file.name = os.path.basename(self.filepath)

                    new_file.name = os.path.abspath(self.filepath)
                    self.report({"INFO"}, "Image file added.")
                    scene.ip_adapter_face_folder = os.path.abspath(self.filepath)
                else:
                    self.report({"ERROR"}, "Selected file is not a valid image.")
        else:
            self.report({"ERROR"}, "No file selected.")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class IPAdapterStyleProperties(bpy.types.PropertyGroup):
    files_to_import: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)


class IPAdapterStyleFileBrowserOperator(Operator):
    bl_idname = "ip_adapter_style.file_browser"
    bl_label = "Open IP Adapter Style File Browser"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    import_as_folder: bpy.props.BoolProperty(name="Import as Folder", default=False)

    def execute(self, context):
        valid_image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".hdr"}
        scene = context.scene

        if self.filepath:
            if self.import_as_folder:
                files_to_import = bpy.context.scene.ip_adapter_style_files_to_import
                files_to_import.clear()  # Clear the list first
                self.filepath = os.path.dirname(self.filepath)
                print("Importing folder:", self.filepath)
                for file_path in glob.glob(os.path.join(self.filepath, "*")):
                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext in valid_image_extensions:
                            print("Found image file in folder:", os.path.basename(file_path))
                            new_file = files_to_import.add()
                            new_file.name = os.path.basename(file_path)
                            new_file.path = os.path.abspath(file_path)
                scene.ip_adapter_style_folder = os.path.abspath(self.filepath)
                self.report({"INFO"}, f"{len(files_to_import)} image files found in folder.")
            else:
                print("Importing file:", self.filepath)
                valid_file_ext = os.path.splitext(self.filepath)[1].lower()
                if valid_file_ext in valid_image_extensions:
                    print("Adding image file:", os.path.basename(self.filepath))
                    files_to_import = bpy.context.scene.ip_adapter_style_files_to_import
                    new_file = files_to_import.add()
                    new_file.name = os.path.basename(self.filepath)
                    new_file.path = os.path.abspath(self.filepath)
                    self.report({"INFO"}, "Image file added.")
                    scene.ip_adapter_style_folder = os.path.abspath(self.filepath)
                else:
                    self.report({"ERROR"}, "Selected file is not a valid image.")
        else:
            self.report({"ERROR"}, "No file selected.")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def load_images_from_folder(folder_path):
    from diffusers.utils import load_image

    # List to hold the loaded images

    loaded_images = []

    # Check if the path is a file

    if os.path.isfile(folder_path) and folder_path.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp")):
        # Load the image

        try:
            image = load_image(folder_path)
            loaded_images.append(image)
            print(f"Loaded image: {folder_path}")
        except Exception as e:
            print(f"Failed to load image {folder_path}: {e}")
        if len(loaded_images) == 1:
            return loaded_images[0]
        else:
            return None

    # Check if the folder exists

    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return None
    print(f"Loaded folder: {folder_path}")

    # Iterate through all files in the folder

    for filename in os.listdir(folder_path):
        # Build the full file path
        file_path = os.path.join(folder_path, filename)

        # Check if the current file is an image
        if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp")):
            # Load the image
            try:
                image = load_image(file_path)
                loaded_images.append(image)
            except Exception as e:
                print(f"Failed to load folder image {file_path}: {e}")
    if len(loaded_images) == 1:
        return loaded_images[0]
    elif len(loaded_images) > 1:
        return loaded_images
    else:
        return None


def flush():
    import torch
    import gc
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


class SEQUENCER_OT_generate_image(Operator):
    """Generate Image"""

    bl_idname = "sequencer.generate_image"
    bl_label = "Prompt"
    bl_description = "Convert text to image"
    bl_options = {"REGISTER", "UNDO"}


    def execute(self, context):
        scene = context.scene
        seq_editor = scene.sequence_editor
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        use_strip_data = addon_prefs.use_strip_data
        local_files_only = addon_prefs.local_files_only
        image_model_card = addon_prefs.image_model_card
        image_power = scene.image_power
        strips = context.selected_sequences
        type = scene.generatorai_typeselect

        pipe = None
        refiner = None
        converter = None
        guidance = scene.movie_num_guidance
        enabled_items = None

        lora_files = scene.lora_files
        enabled_names = []
        enabled_weights = []
        # Check if there are any enabled items before loading

        enabled_items = [item for item in lora_files if item.enabled]

#        if (
#            scene.generate_movie_prompt == ""
#            and not image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
#            and not image_model_card == "Salesforce/blipdiffusion"
#            and not image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
#        ):
#            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
#            return {"CANCELLED"}
        show_system_console(True)
        set_system_console_topmost(True)

        if not seq_editor:
            scene.sequence_editor_create()
        try:
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import pt_to_pil
            import torch
            import requests
            from diffusers.utils import load_image
            import numpy as np
            import PIL
            import cv2
            from PIL import Image
        #            from .free_lunch_utils import (
        #                register_free_upblock2d,
        #                register_free_crossattn_upblock2d,
        #            )

        # from compel import Compel

        except ModuleNotFoundError:
            print("Dependencies needs to be installed in the add-on preferences.")
            self.report(
                {"INFO"},
                "Dependencies needs to be installed in the add-on preferences.",
            )
            return {"CANCELLED"}
        # clear the VRAM

        clear_cuda_cache()

        current_frame = scene.frame_current
        type = scene.generatorai_typeselect
        input = scene.input_strips
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        negative_prompt = scene.generate_movie_negative_prompt + ", " + style_prompt(scene.generate_movie_prompt)[1] + ", nsfw, nude, nudity,"
        image_x = scene.generate_movie_x
        image_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_16(image_x)
        y = scene.generate_movie_y = closest_divisible_16(image_y)
        duration = scene.generate_movie_frames
        image_num_inference_steps = scene.movie_num_inference_steps
        image_num_guidance = scene.movie_num_guidance
        active_strip = context.scene.sequence_editor.active_strip
        do_inpaint = (
            input == "input_strips"
            and find_strip_by_name(scene, scene.inpaint_selected_strip)
            and type == "image"
            and not image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            and not image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            and not image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            # and not image_model_card == "h94/IP-Adapter"
            and not image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
            and not image_model_card == "Salesforce/blipdiffusion"
            and not image_model_card == "Lykon/dreamshaper-8"
            and not image_model_card == "Corcelio/mobius"
            and not image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers"
#            and not image_model_card == "Corcelio/openvision"
            and not image_model_card == "Vargol/ProteusV0.4"
            and not image_model_card == "Lykon/dreamshaper-xl-lightning"
            and not scene.ip_adapter_face_folder
            and not scene.ip_adapter_style_folder
        )
        do_convert = (
            (scene.image_path or scene.movie_path)
            and not image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            and not image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            and not image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            # and not image_model_card == "h94/IP-Adapter"
            and not image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
            and not image_model_card == "Salesforce/blipdiffusion"
            and not image_model_card == "ByteDance/SDXL-Lightning"
            and not image_model_card == "Vargol/ProteusV0.4"
            and not image_model_card == "Lykon/dreamshaper-xl-lightning"
            and not scene.ip_adapter_face_folder
            and not scene.ip_adapter_style_folder
            and not do_inpaint
        )
        do_refine = scene.refine_sd and not do_convert
        if (
            do_inpaint
            or do_convert
            or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            # or image_model_card == "h94/IP-Adapter"
            or image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
            or image_model_card == "Salesforce/blipdiffusion"
            and not scene.ip_adapter_face_folder
            and not scene.ip_adapter_style_folder
        ):
            if not strips:
                self.report({"INFO"}, "Select strip(s) for processing.")
                return {"CANCELLED"}
            for strip in strips:
                if strip.type in {"MOVIE", "IMAGE", "TEXT", "SCENE"}:
                    break
            else:
                self.report(
                    {"INFO"},
                    "None of the selected strips are movie, image, text or scene types.",
                )
                return {"CANCELLED"}

        # LOADING MODELS
        # models for inpaint

        if do_inpaint:
            print("Load: Inpaint Model")
            from diffusers import AutoPipelineForInpainting

            # from diffusers import StableDiffusionXLInpaintPipeline

            from diffusers.utils import load_image

            # clear the VRAM

            clear_cuda_cache()

            pipe = AutoPipelineForInpainting.from_pretrained(
                # pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only,
            ).to(gfx_device)

            # Set scheduler

            if scene.use_lcm:
                from diffusers import LCMScheduler

                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                if enabled_items:
                    enabled_names.append("lcm-lora-sdxl")
                    enabled_weights.append(1.0)
                    pipe.load_lora_weights(
                        "latent-consistency/lcm-lora-sdxl",
                        weight_name="pytorch_lora_weights.safetensors",
                        adapter_name=("lcm-lora-sdxl"),
                    )
                else:
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            else:
                from diffusers import DPMSolverMultistepScheduler

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.watermark = NoWatermark()

            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Conversion img2img/vid2img.
        elif (
            do_convert
            and image_model_card != "warp-ai/wuerstchen"
            and not scene.aurasr
            # and image_model_card != "h94/IP-Adapter"
        ):

            print("Load: img2img/vid2img Model")
            print("Conversion Model:  " + image_model_card)
            if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                )
                converter = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    # text_encoder_2=pipe.text_encoder_2,
                    vae=vae,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )
                if low_vram():
                    converter.enable_model_cpu_offload()
                else:
                    converter.to(gfx_device)

            elif image_model_card == "Kwai-Kolors/Kolors-diffusers":
                from diffusers import DPMSolverMultistepScheduler, KolorsImg2ImgPipeline
                from diffusers.utils import load_image
                converter = KolorsImg2ImgPipeline.from_pretrained(image_model_card, torch_dtype=torch.float16, variant="fp16", local_files_only=local_files_only)
                converter.scheduler = DPMSolverMultistepScheduler.from_config(converter.scheduler.config, use_karras_sigmas=True)
                if low_vram():
                    converter.enable_model_cpu_offload()
                else:
                    converter.to(gfx_device)
            else:
                from diffusers import AutoPipelineForImage2Image
                if image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers":
                    import torch
                    from huggingface_hub.commands.user import login
                    result = login(token=addon_prefs.hugginface_token, add_to_git_credential=True)

                try:
                    converter = AutoPipelineForImage2Image.from_pretrained(
                        image_model_card,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )
                except:
                    try:
                        converter = AutoPipelineForImage2Image.from_pretrained(
                            image_model_card,
                            torch_dtype=torch.float16,
                            local_files_only=local_files_only,
                        )
                    except:
                        print("The "+ image_model_card + " model does not work for a image to image pipeline!")
                        return {"CANCELLED"}

                if low_vram():
                    converter.enable_model_cpu_offload()
                else:
                    converter.to(gfx_device)
            if (
                enabled_items
                and input == "input_strips"
                and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                and (scene.image_path or scene.movie_path)
                and not do_inpaint
            ):
                print("LoRAs will be ignored for image or movie input.")
                enabled_items = False
            #            if enabled_items and input == "input_strips" and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0" and (scene.ip_adapter_face_folder or scene.ip_adapter_style_folder):
            #                print("Adapters will be ignored for image or movie input.")
            #                enabled_items = False


            if enabled_items:
                if scene.use_lcm:
                    from diffusers import LCMScheduler

                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                    if enabled_items:
                        enabled_names.append("lcm-lora-sdxl")
                        enabled_weights.append(1.0)
                        converter.load_lora_weights(
                            "latent-consistency/lcm-lora-sdxl",
                            weight_name="pytorch_lora_weights.safetensors",
                            adapter_name=("lcm-lora-sdxl"),
                        )
                    else:
                        converter.load_lora_weights("latent-consistency/lcm-lora-sdxl")

                    converter.watermark = NoWatermark()

                    if low_vram():
                        converter.enable_model_cpu_offload()
                    else:
                        converter.to(gfx_device)

        #        elif: # depth
        #            from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        #            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
        #            from diffusers.utils import load_image

        #            depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        #            feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        #            controlnet = ControlNetModel.from_pretrained(
        #                "diffusers/controlnet-depth-sdxl-1.0-small",
        #                variant="fp16",
        #                use_safetensors=True,
        #                torch_dtype=torch.float16,
        #            ).to(gfx_device)
        #            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        #            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        #                "stabilityai/stable-diffusion-xl-base-1.0",
        #                controlnet=controlnet,
        #                vae=vae,
        #                variant="fp16",
        #                use_safetensors=True,
        #                torch_dtype=torch.float16,
        #            ).to(gfx_device)
        #            pipe.enable_model_cpu_offload()


        # Canny & Illusion

        elif image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small" or image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster":
            if image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small":
                print("Load: Canny")
            else:
                print("Load: Illusion")
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

            if image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster":
                controlnet = ControlNetModel.from_pretrained(
                    "monster-labs/control_v1p_sdxl_qrcode_monster",
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                )
            else:
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0-small",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            pipe.watermark = NoWatermark()

            if scene.use_lcm:
                from diffusers import LCMScheduler

                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                if enabled_items:
                    enabled_names.append("lcm-lora-sdxl")
                    enabled_weights.append(1.0)
                    pipe.load_lora_weights(
                        "latent-consistency/lcm-lora-sdxl",
                        weight_name="pytorch_lora_weights.safetensors",
                        adapter_name=("lcm-lora-sdxl"),
                    )
                else:
                    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Blip

        elif image_model_card == "Salesforce/blipdiffusion":
            print("Load: Blip Model")
            from diffusers.utils import load_image
            import torch

            if not find_strip_by_name(scene, scene.blip_subject_image):
                from diffusers.pipelines import BlipDiffusionPipeline

                pipe = BlipDiffusionPipeline.from_pretrained(
                    "Salesforce/blipdiffusion",
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                ).to(gfx_device)
            else:
                from controlnet_aux import CannyDetector
                from diffusers.pipelines import BlipDiffusionControlNetPipeline

                pipe = BlipDiffusionControlNetPipeline.from_pretrained(
                    "Salesforce/blipdiffusion-controlnet",
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                ).to(gfx_device)

        # OpenPose

        elif image_model_card == "xinsir/controlnet-openpose-sdxl-1.0":
            print("Load: OpenPose Model")

            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
            from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
            from controlnet_aux import OpenposeDetector
            from PIL import Image
            import torch
            import numpy as np

            # import cv2

            controlnet_conditioning_scale = 1.0

            eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

            controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)

            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                vae=vae,
                # safety_checker=None,
                torch_dtype=torch.float16,
                scheduler=eulera_scheduler,
            )

            processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

            if scene.use_lcm:
                from diffusers import LCMScheduler

                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                scene.movie_num_guidance = 0
            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Scribble
        elif image_model_card == "xinsir/controlnet-scribble-sdxl-1.0":

            #https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0 #use this instead

            print("Load: Scribble Model")
            from controlnet_aux import PidiNetDetector, HEDdetector
            from diffusers import (
                ControlNetModel,
                StableDiffusionXLControlNetPipeline,
                EulerAncestralDiscreteScheduler,
                AutoencoderKL,
            )

            processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
            checkpoint = "xinsir/controlnet-scribble-sdxl-1.0"
            controlnet = ControlNetModel.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )

            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                vae=vae,
                local_files_only=local_files_only,
            )
            if scene.use_lcm:
                from diffusers import LCMScheduler

                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                pipe.fuse_lora()
                scene.movie_num_guidance = 0
            else:
                eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
            if low_vram():
                # torch.cuda.set_per_process_memory_fraction(0.95)  # 6 GB VRAM

                pipe.enable_model_cpu_offload()
                # pipe.enable_vae_slicing()
                # pipe.enable_forward_chunking(chunk_size=1, dim=1)
            else:
                pipe.to(gfx_device)

        # Dreamshaper

        elif do_convert == False and image_model_card == "Lykon/dreamshaper-8":
            print("Load: Dreamshaper Model")
            import torch
            from diffusers import AutoPipelineForText2Image

            if scene.use_lcm:
                from diffusers import LCMScheduler

                pipe = AutoPipelineForText2Image.from_pretrained("lykon/dreamshaper-8-lcm", torch_dtype=torch.float16)
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            else:
                from diffusers import DEISMultistepScheduler

                pipe = AutoPipelineForText2Image.from_pretrained("lykon/dreamshaper-8", torch_dtype=torch.float16, variant="fp16")
                pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # dreamshaper-xl-lightning
        elif do_convert == False and image_model_card == "Lykon/dreamshaper-xl-lightning":
            from diffusers import AutoPipelineForText2Image, AutoencoderKL
            from diffusers import DPMSolverMultistep

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            pipe = AutoPipelineForText2Image.from_pretrained("Lykon/dreamshaper-xl-lightning", torch_dtype=torch.float16, variant="fp16", vae=vae)
            pipe.scheduler = DPMSolverMultistep.from_config(pipe.scheduler.config)
            pipe = pipe.to(gfx_device)

        # HunyuanDiT
        elif do_convert == False and image_model_card == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers":
            from diffusers import HunyuanDiTPipeline
            pipe = HunyuanDiTPipeline.from_pretrained(image_model_card, torch_dtype=torch.float16)
            pipe = pipe.to(gfx_device)

        # SD3 Stable Diffusion 3
        elif image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers":
            print("Load: Stable Diffusion 3 Model")
            import torch
            from huggingface_hub.commands.user import login
            result = login(token=addon_prefs.hugginface_token, add_to_git_credential=True)
            print(str(result))
            from diffusers import StableDiffusion3Pipeline
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
#                "stabilityai/stable-diffusion-3-medium-diffusers",
#                revision="refs/pr/26",
                torch_dtype=torch.float16,
            )
#            pipe = StableDiffusion3Pipeline.from_single_file(
#                "https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips_t5xxlfp8.safetensors",
#                torch_dtype=torch.float16,
#            )
            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Mobius
        elif do_convert == False and image_model_card == "Corcelio/mobius":
            print("Load: Mobius Model")
            import torch
            from diffusers import StableDiffusionXLPipeline, KDPM2AncestralDiscreteScheduler, AutoencoderKL

            # Load VAE component

            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

            # Configure the pipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "Corcelio/mobius",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only,
            )
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)

            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

#        # OpenVision
#        elif do_convert == False and image_model_card == "Corcelio/openvision":
#            print("Load: OpenVision Model")
#            import torch
#            from diffusers import (
#                StableDiffusionXLPipeline,
#                KDPM2AncestralDiscreteScheduler,
#                AutoencoderKL,
#            )

#            # Load VAE component
#            vae = AutoencoderKL.from_pretrained(
#                "madebyollin/sdxl-vae-fp16-fix",
#                torch_dtype=torch.float16,
#                variant="fp16",
#                local_files_only=local_files_only,
#            )

#            # Configure the pipeline
#            pipe = StableDiffusionXLPipeline.from_pretrained("Corcelio/openvision", vae=vae, torch_dtype=torch.float16)
#            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#            if low_vram():
#                pipe.enable_model_cpu_offload()
#            else:
#                pipe.to(gfx_device)

        # Wuerstchen

        elif image_model_card == "warp-ai/wuerstchen":
            print("Load: Würstchen Model")
            if do_convert:
                print(image_model_card + " does not support img2img or img2vid. Ignoring input strip.")
            from diffusers import AutoPipelineForText2Image

            # from diffusers import DiffusionPipeline

            from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

            pipe = AutoPipelineForText2Image.from_pretrained(
                "warp-ai/wuerstchen",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # DeepFloyd

        elif image_model_card == "DeepFloyd/IF-I-M-v1.0":
            print("Load: DeepFloyd Model")
            if do_convert:
                print(image_model_card + " does not support img2img or img2vid. Ignoring input strip.")
            from huggingface_hub.commands.user import login

            result = login(token=addon_prefs.hugginface_token)
            # stage 1

            stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                variant="fp16",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            if low_vram():
                stage_1.enable_model_cpu_offload()
            else:
                stage_1.to(gfx_device)
            # stage 2

            stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            if low_vram():
                stage_2.enable_model_cpu_offload()
            else:
                stage_2.to(gfx_device)
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
                local_files_only=local_files_only,
            )
            if low_vram():
                stage_3.enable_model_cpu_offload()
            else:
                stage_3.to(gfx_device)

#        # playground
#        elif image_model_card == "playgroundai/playground-v2.5-1024px-aesthetic":
#            from diffusers import DiffusionPipeline

#            pipe = DiffusionPipeline.from_pretrained(
#                "playgroundai/playground-v2.5-1024px-aesthetic",
#                torch_dtype=torch.float16,
#                variant="fp16",
#            )

#            from diffusers import EDMDPMSolverMultistepScheduler
#            pipe.scheduler = EDMDPMSolverMultistepScheduler()

#            if low_vram():
#                pipe.enable_model_cpu_offload()
#            else:
#                pipe.to(gfx_device)

        # Flux Schnell
        elif image_model_card == "black-forest-labs/FLUX.1-schnell" or image_model_card == "ChuckMcSneed/FLUX.1-dev":

            from diffusers import FluxPipeline#, FluxTransformer2DModel, AutoencoderKL
            #from diffusers.image_processor import VaeImageProcessor
            #from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel

            flush()
            if low_vram():
                pipe = FluxPipeline.from_pretrained(image_model_card, torch_dtype=torch.bfloat16)
                pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
                pipe.enable_sequential_cpu_offload()
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            else:
                from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
                from diffusers.image_processor import VaeImageProcessor
                from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel
                ckpt_id = image_model_card
                if image_model_card == "black-forest-labs/FLUX.1-schnell":
                    #transformer = FluxTransformer2DModel.from_pretrained("sayakpaul/FLUX.1-merged", torch_dtype=torch.bfloat16)
                    #transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors")
                    #pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=transformer)
                    text_encoder = CLIPTextModel.from_pretrained(ckpt_id, revision="refs/pr/1", subfolder="text_encoder", torch_dtype=torch.bfloat16)
                    text_encoder_2 = T5EncoderModel.from_pretrained(ckpt_id, revision="refs/pr/1", subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
                    tokenizer = CLIPTokenizer.from_pretrained(ckpt_id, subfolder="tokenizer", revision="refs/pr/1")
                    tokenizer_2 = T5TokenizerFast.from_pretrained(ckpt_id, subfolder="tokenizer_2", revision="refs/pr/1")

                    pipe = FluxPipeline.from_pretrained(
                        ckpt_id, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer, tokenizer_2=tokenizer_2, vae=None, transformer=None, #transformer=transformer, #
                        revision="refs/pr/1"
                    )
                else:
                    #transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors")
                    text_encoder = CLIPTextModel.from_pretrained(ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
                    text_encoder_2 = T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
                    tokenizer = CLIPTokenizer.from_pretrained(ckpt_id, subfolder="tokenizer")
                    tokenizer_2 = T5TokenizerFast.from_pretrained(ckpt_id, subfolder="tokenizer_2")

                    pipe = FluxPipeline.from_pretrained(
                        ckpt_id, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer, tokenizer_2=tokenizer_2, transformer=None, vae=None, #transformer=transformer
                    )

                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

                #pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
                #pipe.enable_sequential_cpu_offload()
                #pipe.enable_vae_slicing()#                pipe.enable_vae_tiling()

                with torch.no_grad():
                    print("Encoding prompts.")
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)

                del text_encoder
                del text_encoder_2
                del tokenizer
                del tokenizer_2
                del pipe



#        # Flux Dev
#        elif image_model_card == "ChuckMcSneed/FLUX.1-dev":

#            from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
#            from diffusers.image_processor import VaeImageProcessor
#            from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel

#            flush()

#            ckpt_id = "ChuckMcSneed/FLUX.1-dev"


#            with torch.no_grad():
#                print("Encoding prompts.")
#                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)

#            del text_encoder
#            del text_encoder_2
#            del tokenizer
#            del tokenizer_2
#            del pipe


        # Fluently-XL
        elif image_model_card == "youknownothing/Fluently-XL-Final":
            from diffusers import DiffusionPipeline, DDIMScheduler

            pipe = DiffusionPipeline.from_pretrained(
                "youknownothing/Fluently-XL-Final",
                torch_dtype=torch.float16,
                scheduler=DDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                ),
            )

            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)


        # sdxl_dpo_turbo
        elif image_model_card == "thibaud/sdxl_dpo_turbo":
            from diffusers import StableDiffusionXLPipeline

            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLPipeline.from_single_file(
                "https://huggingface.co/thibaud/sdxl_dpo_turbo/blob/main/sdxl_dpo_turbo.safetensors",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            if low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Stable diffusion etc.
        else:
            print("Load: " + image_model_card + " Model")

            if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                if not (scene.ip_adapter_face_folder or scene.ip_adapter_style_folder):
                    from diffusers import AutoencoderKL

                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=torch.float16,
                        local_files_only=local_files_only,
                    )
                    pipe = DiffusionPipeline.from_pretrained(
                        image_model_card,
                        vae=vae,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )

                # IPAdapter
                else:
                    print("Loading: IP Adapter")
                    import torch
                    from diffusers import DDIMScheduler
                    from diffusers.utils import load_image

                    from transformers import CLIPVisionModelWithProjection

                    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                        "h94/IP-Adapter",
                        subfolder="models/image_encoder",
                        torch_dtype=torch.float16,
                        local_files_only=local_files_only,
                    )
                    if find_strip_by_name(scene, scene.inpaint_selected_strip):
                        from diffusers import AutoPipelineForInpainting

                        pipe = AutoPipelineForInpainting.from_pretrained(
                            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                            torch_dtype=torch.float16,
                            image_encoder=image_encoder,
                            local_files_only=local_files_only,
                        )
                    elif scene.input_strips == "input_strips" and (scene.image_path or scene.movie_path):
                        from diffusers import AutoPipelineForImage2Image

                        pipe = AutoPipelineForImage2Image.from_pretrained(
                            "stabilityai/stable-diffusion-xl-base-1.0",
                            torch_dtype=torch.float16,
                            image_encoder=image_encoder,
                            local_files_only=local_files_only,
                        )
                    else:
                        from diffusers import AutoPipelineForText2Image

                        pipe = AutoPipelineForText2Image.from_pretrained(
                            "stabilityai/stable-diffusion-xl-base-1.0",
                            torch_dtype=torch.float16,
                            image_encoder=image_encoder,
                            local_files_only=local_files_only,
                        )
                    if scene.ip_adapter_face_folder and scene.ip_adapter_style_folder:
                        pipe.load_ip_adapter(
                            "h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"],
                            local_files_only=local_files_only,
                        )
                        pipe.set_ip_adapter_scale([0.7, 0.5])
                    elif scene.ip_adapter_face_folder:
                        pipe.load_ip_adapter(
                            "h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"],
                            local_files_only=local_files_only,
                        )
                        pipe.set_ip_adapter_scale([0.8])
                    elif scene.ip_adapter_style_folder:
                        pipe.load_ip_adapter(
                            "h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"],
                            local_files_only=local_files_only,
                        )
                        pipe.set_ip_adapter_scale([1.0])
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

#                    scale = {
#                        "down": {"block_2": [0.0, 1.0]},
#                        "up": {"block_0": [0.0, 1.0, 0.0]},
#                    }
#                    pipe.set_ip_adapter_scale(scale)#[scale, scale])

            elif image_model_card == "runwayml/stable-diffusion-v1-5":
                from diffusers import StableDiffusionPipeline

                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,  # vae=vae,
                    local_files_only=local_files_only,
                )

            elif image_model_card == "Vargol/PixArt-Sigma_16bit":
                from diffusers import PixArtAlphaPipeline

                if scene.use_lcm:
                    pipe = PixArtAlphaPipeline.from_pretrained(
                        "PixArt-alpha/PixArt-LCM-XL-2-1024-MS", torch_dtype=torch.float16, local_files_only=local_files_only
                    )
                else:
                    pipe = PixArtAlphaPipeline.from_pretrained(
                        "Vargol/PixArt-Sigma_16bit",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )
                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif image_model_card == "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers":
                from diffusers import PixArtSigmaPipeline

                pipe = PixArtSigmaPipeline.from_pretrained(
                    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
                    torch_dtype=torch.float16,
                    #variant="fp16",
                    local_files_only=local_files_only,
                )
                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif image_model_card == "Vargol/PixArt-Sigma_2k_16bit":
                from diffusers import PixArtSigmaPipeline

                pipe = PixArtSigmaPipeline.from_pretrained(
                    "Vargol/PixArt-Sigma_2k_16bit",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )
                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif image_model_card == "ByteDance/SDXL-Lightning":
                import torch
                from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL
                from huggingface_hub import hf_hub_download

                base = "stabilityai/stable-diffusion-xl-base-1.0"
                repo = "ByteDance/SDXL-Lightning"
                ckpt = "sdxl_lightning_2step_lora.safetensors"  # Use the correct ckpt for your step setting!

                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

                # Load model.
                pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, vae=vae, variant="fp16").to(gfx_device)
                pipe.load_lora_weights(hf_hub_download(repo, ckpt))
                pipe.fuse_lora()

                # Ensure sampler uses "trailing" timesteps.
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

#            elif image_model_card == "dataautogpt3/ProteusV0.5":
#                import torch
#                from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL

#                # Load VAE component
#                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

#                # Configure the pipeline
#                pipe = StableDiffusionXLPipeline.from_pretrained("dataautogpt3/ProteusV0.5", vae=vae, torch_dtype=torch.float16)
#                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#                if low_vram():
#                    pipe.enable_model_cpu_offload()
#                else:
#                    pipe.to(gfx_device)

            elif image_model_card == "Vargol/ProteusV0.4":
                from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "Vargol/ProteusV0.4",
                    vae=vae,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif image_model_card == "Kwai-Kolors/Kolors-diffusers":
                import torch
                from diffusers import DPMSolverMultistepScheduler, KolorsPipeline

                pipe = KolorsPipeline.from_pretrained(image_model_card, torch_dtype=torch.float16, variant="fp16", local_files_only=local_files_only)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)
#            elif image_model_card == "Vargol/auraflow0.2-fp16-diffusers":
#                from diffusers import AuraFlowPipeline
#                pipe = AuraFlowPipeline.from_pretrained(image_model_card, torch_dtype=torch.float16, variant="fp16", local_files_only=local_files_only)
#                if low_vram():
#                    pipe.enable_model_cpu_offload()
#                else:
#                    pipe.to(gfx_device)
            else:
                try:
                    from diffusers import AutoPipelineForText2Image

                    pipe = AutoPipelineForText2Image.from_pretrained(
                        image_model_card,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )
                except:
                    from diffusers import AutoPipelineForText2Image

                    pipe = AutoPipelineForText2Image.from_pretrained(
                        image_model_card,
                        torch_dtype=torch.float16,
                        local_files_only=local_files_only,
                    )

            # LCM
            if scene.use_lcm:
                print("Use LCM: True")
                from diffusers import LCMScheduler

                if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    if enabled_items:
                        enabled_names.append("lcm-lora-sdxl")
                        enabled_weights.append(1.0)
                        pipe.load_lora_weights(
                            "latent-consistency/lcm-lora-sdxl",
                            weight_name="pytorch_lora_weights.safetensors",
                            adapter_name=("lcm-lora-sdxl"),
                        )
                    else:
                        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                    scene.movie_num_guidance = 0
                elif image_model_card == "runwayml/stable-diffusion-v1-5":
                    if enabled_items:
                        enabled_names.append("lcm-lora-sdv1-5")
                        enabled_weights.append(1.0)
                        pipe.load_lora_weights(
                            "latent-consistency/lcm-lora-sdv1-5",
                            weight_name="pytorch_lora_weights.safetensors",
                            adapter_name=("lcm-lora-sdv1-5"),
                        )
                    else:
                        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                    scene.movie_num_guidance = 0
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#                elif image_model_card == "segmind/SSD-1B":
#                    scene.movie_num_guidance = 0
#                    pipe.load_lora_weights("latent-consistency/lcm-lora-ssd-1b")
#                    pipe.fuse_lora()
                elif image_model_card == "segmind/Segmind-Vega":
                    scene.movie_num_guidance = 0
                    pipe.load_lora_weights("segmind/Segmind-VegaRT")
                    pipe.fuse_lora()

            elif image_model_card == "Lykon/dreamshaper-8":
                from diffusers import EulerAncestralDiscreteScheduler
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            elif (
                image_model_card != "Vargol/PixArt-Sigma_16bit"
                and image_model_card != "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
                and image_model_card != "Vargol/PixArt-Sigma_2k_16bit"
            ):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            if image_model_card != "Vargol/PixArt-Sigma_2k_16bit":
                pipe.watermark = NoWatermark()

                if low_vram():
                    # torch.cuda.set_per_process_memory_fraction(0.95)  # 6 GB VRAM

                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)
        #            # FreeU
        #            if scene.use_freeU and pipe:  # Free Lunch
        #                # -------- freeu block registration
        #                print("Process: FreeU")
        #                register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        #                register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        #                # -------- freeu block registration

        # LoRA

        if (
            (image_model_card == "stabilityai/stable-diffusion-xl-base-1.0" and ((not scene.image_path and not scene.movie_path) or do_inpaint))
            or image_model_card == "runwayml/stable-diffusion-v1-5"
            or image_model_card == "stabilityai/sdxl-turbo"
            or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
        ):
            scene = context.scene
            if enabled_items:
                for item in enabled_items:
                    enabled_names.append((clean_filename(item.name)).replace(".", ""))
                    enabled_weights.append(item.weight_value)
                    pipe.load_lora_weights(
                        bpy.path.abspath(scene.lora_folder),
                        weight_name=item.name + ".safetensors",
                        adapter_name=((clean_filename(item.name)).replace(".", "")),
                    )
                pipe.set_adapters(enabled_names, adapter_weights=enabled_weights)
                print("Load LoRAs: " + " ".join(enabled_names))

#        # HiDiffusion

#        if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0" and scene.hidiff:
#            from hidiffusion import apply_hidiffusion

#            apply_hidiffusion(pipe, apply_window_attn=False)

        # Refiner model - load if chosen.
        if do_refine:
            print("Load Refine Model:  " + "stabilityai/stable-diffusion-xl-refiner-1.0")
            from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only,
            )
            refiner.watermark = NoWatermark()
            if low_vram():
                refiner.enable_model_cpu_offload()
                # refiner.enable_vae_tiling()
                # refiner.enable_vae_slicing()
            else:
                refiner.to(gfx_device)
            #        # Allow longer prompts.
            #        if image_model_card == "runwayml/stable-diffusion-v1-5":
            #            if pipe:
            #                compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            #            if refiner:
            #                compel = Compel(tokenizer=refiner.tokenizer, text_encoder=refiner.text_encoder)
            #            if converter:
            #                compel = Compel(tokenizer=converter.tokenizer, text_encoder=converter.text_encoder)
            #            prompt_embed = compel.build_conditioning_tensor(prompt)

        # Main Generate Loop Image:
        import random
        for i in range(scene.movie_num_batch):

            start_time = timer()

            # Find free space for the strip in the timeline.

            if i > 0:
                empty_channel = scene.sequence_editor.active_strip.channel
                start_frame = scene.sequence_editor.active_strip.frame_final_start + scene.sequence_editor.active_strip.frame_final_duration
                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
            else:
                empty_channel = find_first_empty_channel(
                    scene.frame_current,
                    (scene.movie_num_batch * duration) + scene.frame_current,
                )
                start_frame = scene.frame_current
            # Generate seed.

            seed = context.scene.movie_num_seed
            seed = seed if not context.scene.movie_use_random else random.randint(-2147483647, 2147483647)
            print("Seed: " + str(seed))
            context.scene.movie_num_seed = seed

            # Use cuda if possible.
            if torch.cuda.is_available():
                generator = torch.Generator(gfx_device).manual_seed(seed) if seed != 0 else None
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None

            # DeepFloyd process:

            if image_model_card == "DeepFloyd/IF-I-M-v1.0":
                prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, negative_prompt)
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

                image = stage_3(prompt=prompt, image=image, noise_level=100, generator=generator).images
                # image[0].save("./if_stage_III.png")

                image = image[0]

            # Wuerstchen

            elif image_model_card == "warp-ai/wuerstchen":
                scene.generate_movie_y = y = closest_divisible_128(y)
                scene.generate_movie_x = x = closest_divisible_128(x)
                print("Generate: Image with Würstchen")
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    decoder_guidance_scale=0.0,
                    # image_embeddings=None,
                    prior_guidance_scale=image_num_guidance,
                    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            # Canny & Illusion

            elif (
                image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small" or image_model_card == "monster-labs/control_v1p_sdxl_qrcode_monster"
            ):

                init_image = None
                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    return {"CANCELLED"}
                image = scale_image_within_dimensions(np.array(init_image), x, None)

                if image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small":
                    print("Process: Canny")
                    image = np.array(init_image)
                    low_threshold = 100
                    high_threshold = 200
                    image = cv2.Canny(image, low_threshold, high_threshold)
                    image = image[:, :, None]
                    canny_image = np.concatenate([image, image, image], axis=2)
                    canny_image = Image.fromarray(canny_image)
                    # canny_image = np.array(canny_image)

                    image = pipe(
                        prompt=prompt,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,  # Should be around 50
                        controlnet_conditioning_scale=1.00 - scene.image_power,
                        image=canny_image,
                        #                    guidance_scale=clamp_value(
                        #                        image_num_guidance, 3, 5
                        #                    ),  # Should be between 3 and 5.
                        #                    # guess_mode=True, #NOTE: Maybe the individual methods should be selectable instead?
                        #                    height=y,
                        #                    width=x,
                        #                    generator=generator,
                    ).images[0]
                else:
                    print("Process: Illusion")
                    illusion_image = init_image

                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,  # Should be around 50
                        control_image=illusion_image,
                        controlnet_conditioning_scale=1.00 - scene.image_power,
                        generator=generator,
                        control_guidance_start=0,
                        control_guidance_end=1,
                        # output_type="latent"
                        #                    guidance_scale=clamp_value(
                        #                        image_num_guidance, 3, 5
                        #                    ),  # Should be between 3 and 5.
                        #                    # guess_mode=True, #NOTE: Maybe the individual methods should be selectable instead?
                        #                    height=y,
                        #                    width=x,
                    ).images[0]

            # DreamShaper

            elif image_model_card == "Lykon/dreamshaper-8":
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    lcm_origin_steps=50,
                    height=y,
                    width=x,
                    generator=generator,
                    output_type="pil",
                ).images[0]

            # dreamshaper-xl-lightning
            elif image_model_card == "Lykon/dreamshaper-xl-lightning":
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=5,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    generator=generator,
                    output_type="pil",
                ).images[0]

            # OpenPose
            elif image_model_card == "xinsir/controlnet-openpose-sdxl-1.0":
                image = None
                if scene.image_path:
                    image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    image = load_first_frame(scene.movie_path)
                if not image:
                    print("Loading strip failed!")
                    return {"CANCELLED"}
                image = image.resize((x, y))
                # image = scale_image_within_dimensions(np.array(init_image),x,None)

                # Make OpenPose bones from normal image

                if not scene.openpose_use_bones:
                    image = np.array(image)

                    image = processor(image, hand_and_face=False)
                    # Save pose image
                    filename = clean_filename(str(seed) + "_" + context.scene.generate_movie_prompt)
                    out_path = solve_path("Pose_" + filename + ".png")
                    print("Saving OpenPoseBone image: " + out_path)
                    image.save(out_path)
                # OpenPose from prompt
                # if not (scene.ip_adapter_face_folder or scene.ip_adapter_style_folder):

                print("Process: OpenPose")
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=image_num_inference_steps,
                    # guidance_scale=image_num_guidance,
                    generator=generator,
                ).images[0]
            #                # OpenPose & IPAdapter
            #                else:
            #                    mask_image = None
            #                    ip_adapter_image = None

            #                    if scene.ip_adapter_face_folder and scene.ip_adapter_style_folder:
            #                        face_images = load_images_from_folder((scene.ip_adapter_face_folder).replace('\\','/'))
            #                        style_images = load_images_from_folder((scene.ip_adapter_style_folder).replace('\\','/'))
            #                        ip_adapter_image=[style_images, face_images]
            #                    elif scene.ip_adapter_face_folder:
            #                        face_images = load_images_from_folder((scene.ip_adapter_face_folder).replace('\\','/'))
            #                        ip_adapter_image=[face_images]
            #                    elif scene.ip_adapter_style_folder:
            #                        style_images = load_images_from_folder((scene.ip_adapter_style_folder).replace('\\','/'))
            #                        ip_adapter_image=[style_images]

            #                    # Input strip + ip adapter
            #                    if scene.input_strips == "input_strips" and (scene.image_path or scene.movie_path):

            #                        image = pipe(
            #                            prompt=prompt,
            #                            image=image,
            #                            negative_prompt=negative_prompt,
            #                            ip_adapter_image=ip_adapter_image,
            #                            controlnet_conditioning_scale=controlnet_conditioning_scale,
            #                            num_inference_steps=image_num_inference_steps,
            #                            #guidance_scale=image_num_guidance,
            #                            #height=y,
            #                            #width=x,
            #                            #strength=max(1.00 - scene.image_power, 0.1),
            #                            generator=generator,
            #                        ).images[0]

            #                    # Inpaint - not exposed in panel
            #                    elif scene.inpaint_selected_strip:
            #                        print("Process: Inpaint")
            #                        mask_strip = find_strip_by_name(scene, scene.inpaint_selected_strip)
            #
            #                        if not mask_strip:
            #                            print("Selected mask not found!")
            #                            return {"CANCELLED"}
            #
            #                        if (
            #                            mask_strip.type == "MASK"
            #                            or mask_strip.type == "COLOR"
            #                            or mask_strip.type == "SCENE"
            #                            or mask_strip.type == "META"
            #                        ):
            #                            mask_strip = get_render_strip(self, context, mask_strip)

            #                        mask_path = get_strip_path(mask_strip)
            #                        mask_image = load_first_frame(mask_path)

            #                        if not mask_image:
            #                            print("Loading mask failed!")
            #                            return

            #                        mask_image = mask_image.resize((x, y))
            #                        mask_image = pipe.mask_processor.blur(mask_image, blur_factor=33)

            ##                        if scene.image_path:
            ##                            init_image = load_first_frame(scene.image_path)
            ##                        if scene.movie_path:
            ##                            init_image = load_first_frame(scene.movie_path)
            ##                        if not init_image:
            ##                            print("Loading strip failed!")
            ##                            return {"CANCELLED"}

            #                        image = pipe(
            #                            prompt,
            #                            negative_prompt=negative_prompt,
            #                            image=image,
            #                            mask_image=mask_image,
            #                            ip_adapter_image=ip_adapter_image,
            #                            num_inference_steps=image_num_inference_steps,
            #                            guidance_scale=image_num_guidance,
            #                            height=y,
            #                            width=x,
            #                            generator=generator,
            #                            #cross_attention_kwargs={"scale": 1.0},
            #                            #padding_mask_crop=42,
            #                            #strength=0.99,
            #                        ).images[0]

            #                    # No inpaint, but IP Adapter
            #                    else:
            #                        image = pipe(
            #                            prompt,
            #                            negative_prompt=negative_prompt,
            #                            ip_adapter_image=ip_adapter_image,
            #                            num_inference_steps=image_num_inference_steps,
            #                            guidance_scale=image_num_guidance,
            #                            height=y,
            #                            width=x,
            #                            generator=generator,
            #                        ).images[0]

            # Scribble
            elif image_model_card == "xinsir/controlnet-scribble-sdxl-1.0":
                print("Process: Scribble")
                init_image = None

                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    return {"CANCELLED"}
                image = scale_image_within_dimensions(np.array(init_image), x, None)

                if not scene.use_scribble_image:
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.bitwise_not(image)
                    image = cv2.GaussianBlur(image, (0, 0), 3)

                    # higher threshold, thiner line
                    random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
                    image[image > random_val] = 255
                    image[image < 255] = 0
                    image = Image.fromarray(image)
                    image = processor(image, scribble=True)
                else:
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.bitwise_not(image)
                    image = processor(image, scribble=True)

                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    controlnet_conditioning_scale=1.0,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            # Blip
            elif image_model_card == "Salesforce/blipdiffusion":
                print("Process: Subject Driven")
                text_prompt_input = prompt
                style_subject = str(scene.blip_cond_subject)
                tgt_subject = str(scene.blip_tgt_subject)
                init_image = None
                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    return {"CANCELLED"}
                init_image = init_image.resize((x, y))
                style_image = init_image
                subject_strip = find_strip_by_name(scene, scene.blip_subject_image)
                if subject_strip:
                    if subject_strip.type == "MASK" or subject_strip.type == "COLOR" or subject_strip.type == "SCENE" or subject_strip.type == "META":
                        subject_strip = get_render_strip(self, context, subject_strip)
                    subject_path = get_strip_path(subject_strip)
                    cldm_cond_image = load_first_frame(subject_path)
                    canny = CannyDetector()
                    cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
                    if cldm_cond_image:
                        cldm_cond_image = cldm_cond_image.resize((x, y))
                        image = pipe(
                            text_prompt_input,
                            style_image,
                            cldm_cond_image,
                            style_subject,
                            tgt_subject,
                            guidance_scale=image_num_guidance,
                            num_inference_steps=image_num_inference_steps,
                            neg_prompt=negative_prompt,
                            height=y,
                            width=x,
                            generator=generator,
                        ).images[0]
                    else:
                        print("Subject strip loading failed!")
                        subject_strip = ""
                if not subject_strip:
                    image = pipe(
                        text_prompt_input,
                        style_image,
                        style_subject,
                        tgt_subject,
                        guidance_scale=image_num_guidance,
                        num_inference_steps=image_num_inference_steps,
                        neg_prompt=negative_prompt,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]

            elif image_model_card == "ByteDance/SDXL-Lightning":
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=y,
                    width=x,
                    guidance_scale=0.0,
                    output_type="pil",
                    num_inference_steps=2,
                ).images[0]
                decoder = None

#            elif image_model_card == "dataautogpt3/ProteusV0.5":
#                image = pipe(
#                    prompt,
#                    negative_prompt=negative_prompt,
#                    num_inference_steps=image_num_inference_steps,
#                    guidance_scale=image_num_guidance,
#                    height=y,
#                    width=x,
#                    generator=generator,
#                ).images[0]
#                decoder = None

            elif image_model_card == "Vargol/ProteusV0.4":
                image = pipe(
                    # prompt_embeds=prompt, # for compel - long prompts
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            elif image_model_card == "Vargol/PixArt-Sigma_2k_16bit":
                image = pipe(
                    # prompt_embeds=prompt, # for compel - long prompts
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            # Inpaint
            elif do_inpaint:
                init_image = None
                print("Process: Inpaint")
                mask_strip = find_strip_by_name(scene, scene.inpaint_selected_strip)

                if not mask_strip:
                    print("Selected mask not found!")
                    return {"CANCELLED"}
                if mask_strip.type == "MASK" or mask_strip.type == "COLOR" or mask_strip.type == "SCENE" or mask_strip.type == "META":
                    mask_strip = get_render_strip(self, context, mask_strip)
                mask_path = get_strip_path(mask_strip)
                mask_image = load_first_frame(mask_path)

                if not mask_image:
                    print("Loading mask failed!")
                    return
                mask_image = mask_image.resize((x, y))
                mask_image = pipe.mask_processor.blur(mask_image, blur_factor=33)

                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    return {"CANCELLED"}
                init_image = init_image.resize((x, y))
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    generator=generator,
                    padding_mask_crop=42,
                    strength=0.99,
                ).images[0]

                #                # Limit inpaint to maske area:
                #                # Convert mask to grayscale NumPy array
                #                mask_image_arr = np.array(mask_image.convert("L"))

                #                # Add a channel dimension to the end of the grayscale mask
                #                mask_image_arr = mask_image_arr[:, :, None]
                #                mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
                #                mask_image_arr[mask_image_arr < 0.5] = 0
                #                mask_image_arr[mask_image_arr >= 0.5] = 1

                #                # Take the masked pixels from the repainted image and the unmasked pixels from the initial image
                #                unmasked_unchanged_image_arr = (
                #                    1 - mask_image_arr
                #                ) * init_image + mask_image_arr * image
                #                image = PIL.Image.fromarray(
                #                    unmasked_unchanged_image_arr.astype("uint8")
                #                )

                delete_strip(mask_strip)

            # Flux
            elif image_model_card == "black-forest-labs/FLUX.1-schnell" or image_model_card == "ChuckMcSneed/FLUX.1-dev":
                if low_vram():
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=image_num_guidance,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]
                    flush()
                else:
                    if image_model_card == "ChuckMcSneed/FLUX.1-dev":

                        pipe = FluxPipeline.from_pretrained(
                            ckpt_id, text_encoder=None, text_encoder_2=None,
                            tokenizer=None, tokenizer_2=None, vae=None,
                            torch_dtype=torch.bfloat16
                        ).to(gfx_device)

                        latents = pipe(
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=image_num_inference_steps, guidance_scale=image_num_guidance,
                            max_sequence_length=512,
                            height=y,
                            width=x,
                            generator=generator,
                            output_type="latent"
                        ).images
                        del pipe.transformer
                        del pipe

                        flush()

                        vae = AutoencoderKL.from_pretrained(
                            ckpt_id,
                            subfolder="vae",
                            torch_dtype=torch.bfloat16
                        ).to(gfx_device)

                    else:
                        pipe = FluxPipeline.from_pretrained(
                            ckpt_id, text_encoder=None, text_encoder_2=None,
                            tokenizer=None, tokenizer_2=None, vae=None,
                            revision="refs/pr/1",
                            torch_dtype=torch.bfloat16
                        ).to(gfx_device)
                        latents = pipe(
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=image_num_inference_steps, guidance_scale=0.0,
                            height=y,
                            width=x,
                            generator=generator,
                            output_type="latent"
                        ).images
                        # print(f"{latents.shape=}")
                        del pipe.transformer
                        del pipe

                        flush()

                        vae = AutoencoderKL.from_pretrained(
                            ckpt_id,
                            revision="refs/pr/1",
                            subfolder="vae",
                            torch_dtype=torch.bfloat16
                        ).to(gfx_device)

                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
                    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
#                    image = image_processor.postprocess(image, output_type="pil")
#                    image = image[0]

                    with torch.no_grad():
                        #print("Running decoding.")
                        pipe = FluxPipeline._unpack_latents(latents, y, x, vae_scale_factor)
                        pipe = (pipe / vae.config.scaling_factor) + vae.config.shift_factor

                        image = vae.decode(pipe, return_dict=False)[0]
                        image = image_processor.postprocess(image, output_type="pil")
                        image = image[0]

#            # Flux Dev
#            elif image_model_card == "ChuckMcSneed/FLUX.1-dev":

#                flush()


#                vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
#                image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

#                with torch.no_grad():
#                    #print("Running decoding.")
#                    pipe = FluxPipeline._unpack_latents(latents, y, x, vae_scale_factor)
#                    pipe = (pipe / vae.config.scaling_factor) + vae.config.shift_factor

#                    image = vae.decode(pipe, return_dict=False)[0]
#                    image = image_processor.postprocess(image, output_type="pil")
#                    image = image[0]

            # Img2img
            elif do_convert and not scene.aurasr:

                if enabled_items:
                    self.report(
                        {"INFO"},
                        "LoRAs are ignored for image to image processing.",
                    )
                if scene.movie_path:
                    print("Process: Image to Image")
                    init_image = load_first_frame(scene.movie_path)
                    init_image = init_image.resize((x, y))
                elif scene.image_path:
                    print("Process: Image to Image")
                    init_image = load_first_frame(scene.image_path)
                    init_image = init_image.resize((x, y))
                # init_image = load_image(scene.image_path).convert("RGB")
                print("X: " + str(x), "Y: " + str(y))

                # Turbo
                if (
                    image_model_card == "stabilityai/sdxl-turbo"
                    or image_model_card == "stabilityai/sd-turbo"
                    or image_model_card == "thibaud/sdxl_dpo_turbo"
                ):
                    image = converter(
                        prompt=prompt,
                        image=init_image,
                        strength=1.00 - scene.image_power,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=0.0,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]

                # Not Turbo
                else:
                    image = converter(
                        prompt=prompt,
                        image=init_image,
                        strength=1.00 - scene.image_power,
                        negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=image_num_guidance,
                        #height=y,
                        #width=x,
                        generator=generator,
                    ).images[0]

            # Generate Stable Diffusion etc.
            elif image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers":
                print("Generate: SD3 Image ")
                image = pipe(
                    prompt="",
                    prompt_3=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    max_sequence_length=512,
                    generator=generator,
                ).images[0]
            else:
                print("Generate: Image ")
                from diffusers.utils import load_image

                # IPAdapter
                if (scene.ip_adapter_face_folder or scene.ip_adapter_style_folder) and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":

                    mask_image = None
                    init_image = None
                    ip_adapter_image = None

                    if scene.ip_adapter_face_folder and scene.ip_adapter_style_folder:
                        face_images = load_images_from_folder((scene.ip_adapter_face_folder).replace("\\", "/"))
                        style_images = load_images_from_folder((scene.ip_adapter_style_folder).replace("\\", "/"))
                        ip_adapter_image = [style_images, face_images]
                    elif scene.ip_adapter_face_folder:
                        face_images = load_images_from_folder((scene.ip_adapter_face_folder).replace("\\", "/"))
                        ip_adapter_image = [face_images]
                    elif scene.ip_adapter_style_folder:
                        style_images = load_images_from_folder((scene.ip_adapter_style_folder).replace("\\", "/"))
                        ip_adapter_image = [style_images]

                    # Inpaint
                    if scene.inpaint_selected_strip:
                        print("Process: Inpaint")
                        mask_strip = find_strip_by_name(scene, scene.inpaint_selected_strip)

                        if not mask_strip:
                            print("Selected mask not found!")
                            return {"CANCELLED"}
                        if mask_strip.type == "MASK" or mask_strip.type == "COLOR" or mask_strip.type == "SCENE" or mask_strip.type == "META":
                            mask_strip = get_render_strip(self, context, mask_strip)
                        mask_path = get_strip_path(mask_strip)
                        mask_image = load_first_frame(mask_path)

                        if not mask_image:
                            print("Loading mask failed!")
                            return
                        mask_image = mask_image.resize((x, y))
                        mask_image = pipe.mask_processor.blur(mask_image, blur_factor=33)

                        if scene.image_path:
                            init_image = load_first_frame(scene.image_path)
                        if scene.movie_path:
                            init_image = load_first_frame(scene.movie_path)
                        if not init_image:
                            print("Loading strip failed!")
                            return {"CANCELLED"}
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt,
                            image=init_image,
                            mask_image=mask_image,
                            ip_adapter_image=ip_adapter_image,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            generator=generator,
                            # cross_attention_kwargs={"scale": 1.0},
                            # padding_mask_crop=42,
                            # strength=0.99,
                        ).images[0]

                    # Input strip + ip adapter
                    elif scene.input_strips == "input_strips" and (scene.image_path or scene.movie_path):
                        if scene.image_path:
                            init_image = load_first_frame(scene.image_path)
                        if scene.movie_path:
                            init_image = load_first_frame(scene.movie_path)
                        if not init_image:
                            print("Loading strip failed!")
                            return {"CANCELLED"}
                        image = pipe(
                            prompt,
                            image=init_image,
                            negative_prompt=negative_prompt,
                            ip_adapter_image=ip_adapter_image,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            # strength=max(1.00 - scene.image_power, 0.1),
                            generator=generator,
                        ).images[0]

                    # No inpaint, but IP Adapter
                    else:
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt,
                            ip_adapter_image=ip_adapter_image,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            generator=generator,
                        ).images[0]

                # SDXL Turbo
                elif image_model_card == "stabilityai/sdxl-turbo":  # or image_model_card == "thibaud/sdxl_dpo_turbo":

                    # LoRA.

                    if enabled_items:
                        image = pipe(
                            # prompt_embeds=prompt, # for compel - long prompts
                            prompt,
                            # negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=0.0,
                            height=y,
                            width=x,
                            cross_attention_kwargs={"scale": 1.0},
                            generator=generator,
                        ).images[0]

                    # No LoRA.

                    else:
                        image = pipe(
                            # prompt_embeds=prompt, # for compel - long prompts
                            prompt,
                            # negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=0.0,
                            height=y,
                            width=x,
                            generator=generator,
                        ).images[0]

                # Not Turbo
                else:

                    # LoRA.

                    if enabled_items:
                        image = pipe(
                            # prompt_embeds=prompt, # for compel - long prompts
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            cross_attention_kwargs={"scale": 1.0},
                            generator=generator,
                        ).images[0]

                    # No LoRA.
                    else:
                        image = pipe(
                            # prompt_embeds=prompt, # for compel - long prompts
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            generator=generator,
                        ).images[0]

            # Add refiner
            if do_refine:
                print("Refine: Image")
                image = refiner(
                    prompt=prompt,
                    image=image,
                    strength=max(1.00 - scene.image_power, 0.1),
                    negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=max(image_num_guidance, 1.1),
                    generator=generator,
                ).images[0]

            # ADetailer
            if scene.adetailer:

                from asdff.base import AdPipelineBase
                from huggingface_hub import hf_hub_download
                from diffusers import StableDiffusionXLPipeline, AutoencoderKL
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                )
                pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, variant="fp16", torch_dtype=torch.float16)
                if low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

                face_prompt = prompt+", face, (8k, RAW photo, best quality, masterpiece:1.2)"
                face_n_prompt = "nsfw, blurry, disfigured"
                face_mask_pad = 32
                mask_blur = 4
                mask_dilation = 4
                strength = 0.4
                ddim_steps = 20
                ad_images = image

                ad_components = pipe.components
                ad_pipe = AdPipelineBase(**ad_components)

                model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt",local_dir = "asdff/yolo_models", local_dir_use_symlinks = False)
                common = {"prompt": face_prompt,"n_prompt" : face_n_prompt, "num_inference_steps": int(image_num_inference_steps), "target_size" : (x,y)}
                inpaint_only = {'strength': strength}
                result = ad_pipe(common=common, inpaint_only=inpaint_only, images=ad_images, mask_dilation=mask_dilation, mask_blur=mask_blur, mask_padding=face_mask_pad, model_path=model_path)
                try:
                    image = result.images[0]
                except:
                    print("No images detected. ADetailer disabled.")

            # AuraSR
            if scene.aurasr:

                if do_convert:
                    if scene.movie_path:
                        print("Process: Image to Image")
                        init_image = load_first_frame(scene.movie_path)
                        init_image = init_image.resize((x, y))
                    elif scene.image_path:
                        print("Process: Image to Image")
                        init_image = load_first_frame(scene.image_path)
                        init_image = init_image.resize((x, y))
                    image = init_image

                from aura_sr import AuraSR
                aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR")
                image = aura_sr.upscale_4x(image)

            # Move to folder
            filename = clean_filename(str(seed) + "_" + context.scene.generate_movie_prompt)
            out_path = solve_path(filename + ".png")
            image.save(out_path)

            if input == "input_strips":
                old_strip = context.selected_sequences[0]

            # Add strip
            if os.path.isfile(out_path):
                strip = scene.sequence_editor.sequences.new_image(
                    name=str(seed) + "_" + context.scene.generate_movie_prompt,
                    frame_start=start_frame,
                    filepath=out_path,
                    channel=empty_channel,
                    fit_method="FIT",
                )
                if scene.generate_movie_frames == -1 and input == "input_strips":
                    strip.frame_final_duration = old_strip.frame_final_duration
                else:
                    strip.frame_final_duration = abs(scene.generate_movie_frames)
                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
                strip.use_proxy = True
                # bpy.ops.sequencer.rebuild_proxy()
            else:
                print("No resulting file found.")
            gc.collect()

            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == "SEQUENCE_EDITOR":
                        from bpy import context

                        with context.temp_override(window=window, area=area):
                            if i > 0:
                                scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
                            # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution

                            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                            break
            print_elapsed_time(start_time)
        try:
            if pipe:
                pipe = None
            if refiner:
                compel = None
            if converter:
                converter = None
        except:
            pass

        # clear the VRAM
        clear_cuda_cache()

        scene.movie_num_guidance = guidance
        if input != "input_strips":
            bpy.ops.renderreminder.pallaidium_play_notification()
        scene.frame_current = current_frame

        return {"FINISHED"}


# For generate text
def clean_string(input_string):
    # Words to be removed

    words_to_remove = ["araffe", "arafed", "there is", "there are "]
    for word in words_to_remove:
        input_string = input_string.replace(word, "")
    input_string = input_string.strip()
    # Capitalize the first letter

    input_string = input_string[:1].capitalize() + input_string[1:]
    # Add a full stop at the end

    input_string += "."
    return input_string


class SEQUENCER_OT_generate_text(Operator):
    """Generate Text"""

    bl_idname = "sequencer.generate_text"
    bl_label = "Prompt"
    bl_description = "Generate texts from strips"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        seq_editor = scene.sequence_editor
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        local_files_only = addon_prefs.local_files_only
        guidance = scene.movie_num_guidance
        current_frame = scene.frame_current
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        x = scene.generate_movie_x = closest_divisible_16(scene.generate_movie_x)
        y = scene.generate_movie_y = closest_divisible_16(scene.generate_movie_y)
        duration = scene.generate_movie_frames
        render = bpy.context.scene.render
        fps = render.fps / render.fps_base
        show_system_console(True)
        set_system_console_topmost(True)

        if not seq_editor:
            scene.sequence_editor_create()
        active_strip = context.scene.sequence_editor.active_strip

        if addon_prefs.text_model_card == "Salesforce/blip-image-captioning-large":
            try:
                import torch
                from PIL import Image
                from transformers import BlipProcessor, BlipForConditionalGeneration
            except ModuleNotFoundError:
                print("Dependencies need to be installed in the add-on preferences.")
                self.report(
                    {"INFO"},
                    "Dependencies need to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}
            
        elif addon_prefs.text_model_card == "gokaygokay/Florence-2-Flux-Large":
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
            except ModuleNotFoundError:
                print("Dependencies need to be installed in the add-on preferences.")
                self.report(
                    {"INFO"},
                    "Dependencies need to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}
            
        # clear the VRAM
        clear_cuda_cache()

        init_image = load_first_frame(scene.movie_path) if scene.movie_path else load_first_frame(scene.image_path)

        init_image = init_image.resize((x, y))

        if addon_prefs.text_model_card == "Salesforce/blip-image-captioning-large":
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                local_files_only=local_files_only,
            )

            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            ).to(gfx_device)

            text = ""
            inputs = processor(init_image, text, return_tensors="pt").to(gfx_device, torch.float16)

            out = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(out[0], skip_special_tokens=True)
            text = clean_string(text)
            print("Generated text: " + text)
            
        elif addon_prefs.text_model_card == "gokaygokay/Florence-2-Flux-Large":    

            model = AutoModelForCausalLM.from_pretrained("gokaygokay/Florence-2-Flux-Large", trust_remote_code=True).to(gfx_device).eval()
            processor = AutoProcessor.from_pretrained("gokaygokay/Florence-2-Flux-Large", trust_remote_code=True)

            # Ensure the image is in RGB mode
            if init_image.mode != "RGB":
                init_image = init_image.convert("RGB")

            inputs = processor(text="Describe", images=init_image, return_tensors="pt").to(gfx_device)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                repetition_penalty=1.10,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task="Describe", image_size=(init_image.width, init_image.height))
            text = parsed_answer["Describe"]
            print("Generated text: " + text)

        # Find free space for the strip in the timeline.
        
        

#        if active_strip.frame_final_start <= current_frame <= (active_strip.frame_final_start + active_strip.frame_final_duration):
##            empty_channel = find_first_empty_channel(
##                scene.frame_current,
##                (scene.sequence_editor.active_strip.frame_final_duration) + scene.frame_current,
##            )
#            start_frame = scene.frame_current
#            end_frame = frame_end=int(start_frame + ((len(text) / 20) * (fps/10)))
#        else:
        start_frame = int(scene.sequence_editor.active_strip.frame_start)
        #scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
        end_frame = start_frame + scene.sequence_editor.active_strip.frame_final_duration
        empty_channel = find_first_empty_channel(
            start_frame,
            end_frame,
        )

        # Add strip
        if text:
#            if input == "input_strips":
#                old_strip = context.selected_sequences[0]
            print(str(start_frame))
            strip = scene.sequence_editor.sequences.new_effect(
                name=text,
                type="TEXT",
                frame_start=start_frame,
                frame_end=end_frame,
                channel=empty_channel,
            )
            strip.text = text
            strip.wrap_width = 0.68
            strip.font_size = 44
            strip.location[0] = 0.5
            strip.location[1] = 0.2
            strip.align_x = "CENTER"
            strip.align_y = "TOP"
            strip.use_shadow = True
            strip.use_box = True
#            if scene.generate_movie_frames == -1 and input == "input_strips":
#                strip.frame_final_duration = old_strip.frame_final_duration
#            else:
#                strip.frame_final_duration = abs(scene.generate_movie_frames)
            scene.sequence_editor.active_strip = strip
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == "SEQUENCE_EDITOR":
                    from bpy import context

                    with context.temp_override(window=window, area=area):
                        if (
                            active_strip.frame_final_start
                            <= scene.frame_current
                            <= (active_strip.frame_final_start + active_strip.frame_final_duration)
                        ):
                            pass
                        else:
                            scene.frame_current = scene.sequence_editor.active_strip.frame_final_start
                        # Redraw UI to display the new strip.
                        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                        break
        scene.movie_num_guidance = guidance
        scene.frame_current = current_frame

        model = None

        # clear the VRAM
        clear_cuda_cache()

        return {"FINISHED"}


class SEQUENCER_OT_strip_to_generatorAI(Operator):
    """Convert selected text strips to Generative AI"""

    bl_idname = "sequencer.text_to_generator"
    bl_label = "Pallaidium"
    bl_options = {"INTERNAL"}
    bl_description = "Adds selected strips as inputs to the Generative AI process"

    @classmethod
    def poll(cls, context):
        return context.scene and context.scene.sequence_editor

    def execute(self, context):
        import torch
        import scipy

        bpy.types.Scene.movie_path = ""
        bpy.types.Scene.image_path = ""
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        play_sound = addon_prefs.playsound
        addon_prefs.playsound = False
        scene = context.scene
        sequencer = bpy.ops.sequencer
        sequences = bpy.context.sequences
        strips = context.selected_sequences
        active_strip = context.scene.sequence_editor.active_strip
        prompt = scene.generate_movie_prompt
        negative_prompt = scene.generate_movie_negative_prompt
        current_frame = scene.frame_current
        type = scene.generatorai_typeselect
        seed = scene.movie_num_seed
        use_random = scene.movie_use_random
        use_strip_data = addon_prefs.use_strip_data
        temp_strip = None
        input = scene.input_strips

        if not strips:
            self.report({"INFO"}, "Select strip(s) for processing.")
            return {"CANCELLED"}
        else:
            print("\nStrip input processing started...")
        for strip in strips:
            if strip.type in {"MOVIE", "IMAGE", "TEXT", "SCENE", "META"}:
                break
        else:
            self.report(
                {"INFO"},
                "None of the selected strips are movie, image, text, meta or scene types.",
            )
            return {"CANCELLED"}

        if type == "text":
            for strip in strips:
                if strip.type in {"MOVIE", "IMAGE"}:
                    print("Process: Image Captioning")
                    break
            else:
                self.report(
                    {"INFO"},
                    "None of the selected strips are movie or image.",
                )
                return {"CANCELLED"}

        if use_strip_data:
            print("Use file seed and prompt: Yes")
        else:
            print("Use file seed and prompt: No")

        total_vram = 0
        for i in range(torch.cuda.device_count()):
            if torch.cuda.is_available():
                properties = torch.cuda.get_device_properties(i)
                total_vram += properties.total_memory
                print("Total VRAM: " + str(total_vram))
                print("Total GPU Cards: " + str(torch.cuda.device_count()))

        for count, strip in enumerate(strips):
            for dsel_strip in bpy.context.scene.sequence_editor.sequences:
                dsel_strip.select = False
            strip.select = True

            # render intermediate mp4 file
            if strip.type == "SCENE" or strip.type == "MOVIE" or strip.type == "META":  # or strip.type == "IMAGE"

                # Make the current frame overlapped frame, the temp strip.
                if type == "image" or type == "text":
                    trim_frame = find_overlapping_frame(strip, current_frame)

                    if trim_frame and len(strips) == 1:
                        bpy.ops.sequencer.duplicate_move(SEQUENCER_OT_duplicate={}, TRANSFORM_OT_seq_slide={"value":(0, 1), "use_restore_handle_selection":False, "snap":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False})
                        intermediate_strip = bpy.context.selected_sequences[0]
                        intermediate_strip.frame_start = strip.frame_start
                        intermediate_strip.frame_offset_start = int(trim_frame)
                        intermediate_strip.frame_final_duration = 1
                        temp_strip = strip = get_render_strip(self, context, intermediate_strip)

                        if intermediate_strip is not None:
                            delete_strip(intermediate_strip)

                    elif type == "text":
                        bpy.ops.sequencer.copy()
                        bpy.ops.sequencer.paste(keep_offset=True)
                        intermediate_strip = bpy.context.selected_sequences[0]
                        intermediate_strip.frame_start = strip.frame_start

                        # intermediate_strip.frame_offset_start = int(trim_frame)
                        intermediate_strip.frame_final_duration = 1
                        temp_strip = strip = get_render_strip(self, context, intermediate_strip)

                        if intermediate_strip is not None:
                            delete_strip(intermediate_strip)
                    else:
                        temp_strip = strip = get_render_strip(self, context, strip)
#                    temp_strip.select = True
#                    bpy.ops.transform.seq_slide(value=(0, -1), snap=False, view2d_edge_pan=True)
                else:
                    temp_strip = strip = get_render_strip(self, context, strip)

            if strip.type == "TEXT":
                if strip.text:
                    print("\n" + str(count + 1) + "/" + str(len(strips)))
                    print("Prompt: " + strip.text + ", " + prompt)
                    print("Negative Prompt: " + negative_prompt)
                    scene.generate_movie_prompt = strip.text + ", " + prompt
                    scene.frame_current = strip.frame_final_start
                    context.scene.sequence_editor.active_strip = strip

                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()
                    # context.scene.generate_movie_prompt = prompt
                    # scene.generate_movie_negative_prompt = negative_prompt

                    context.scene.movie_use_random = use_random
                    context.scene.movie_num_seed = seed
                    # scene.generate_movie_prompt = prompt

                    scene.generate_movie_negative_prompt = negative_prompt

                    if use_strip_data:
                        scene.movie_use_random = use_random
                        scene.movie_num_seed = seed

            if strip.type == "IMAGE":
                strip_dirname = os.path.dirname(strip.directory)
                image_path = bpy.path.abspath(os.path.join(strip_dirname, strip.elements[0].filename))
                bpy.types.Scene.image_path = image_path

                if strip.name:
                    strip_prompt = os.path.splitext(strip.name)[0]
                    seed_nr = extract_numbers(str(strip_prompt))

                    if seed_nr:
                        file_seed = int(seed_nr)

                        if file_seed and use_strip_data:
                            strip_prompt = strip_prompt.replace(str(file_seed) + "_", "")
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed

                    if use_strip_data:
                        styled_prompt = style_prompt(strip_prompt + ", " + prompt)[0]
                        styled_negative_prompt = style_prompt(strip_prompt + ", " + prompt)[1]
                    else:
                        styled_prompt = style_prompt(prompt)[0]
                        styled_negative_prompt = style_prompt(prompt)[1]

                    print("\n" + str(count + 1) + "/" + str(len(strips)))

                    if type != "text":
                        print("Prompt: " + styled_prompt)
                        print("Negative Prompt: " + styled_negative_prompt)
                    scene.generate_movie_prompt = styled_prompt
                    scene.generate_movie_negative_prompt = styled_negative_prompt
                    scene.frame_current = strip.frame_final_start
                    context.scene.sequence_editor.active_strip = strip

                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()
                    if type == "text":
                        sequencer.generate_text()
                scene.generate_movie_prompt = prompt
                scene.generate_movie_negative_prompt = negative_prompt

                if use_strip_data:
                    scene.movie_use_random = use_random
                    scene.movie_num_seed = seed
                bpy.types.Scene.image_path = ""

            if strip.type == "MOVIE":
                movie_path = bpy.path.abspath(strip.filepath)
                bpy.types.Scene.movie_path = movie_path

                if strip.name:
                    strip_prompt = os.path.splitext(strip.name)[0]
                    seed_nr = extract_numbers(str(strip_prompt))

                    if seed_nr:
                        file_seed = int(seed_nr)

                        if file_seed and use_strip_data:
                            strip_prompt = strip_prompt.replace(str(file_seed) + "_", "")
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed

                    if use_strip_data:
                        styled_prompt = style_prompt(strip_prompt + ", " + prompt)[0]
                        styled_negative_prompt = style_prompt(strip_prompt + ", " + prompt)[1]
                    else:
                        styled_prompt = style_prompt(prompt)[0]
                        styled_negative_prompt = style_prompt(prompt)[1]
                    print("\n" + str(count + 1) + "/" + str(len(strips)))

                    if type != "text":
                        print("Prompt: " + styled_prompt)
                        print("Negative Prompt: " + styled_negative_prompt)

                    scene.generate_movie_prompt = styled_prompt
                    scene.generate_movie_negative_prompt = styled_negative_prompt
                    scene.frame_current = strip.frame_final_start
                    context.scene.sequence_editor.active_strip = strip

                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()
                    if type == "text":
                        sequencer.generate_text()

                scene.generate_movie_prompt = prompt
                scene.generate_movie_negative_prompt = negative_prompt

                if use_strip_data:
                    scene.movie_use_random = use_random
                    scene.movie_num_seed = seed
                if temp_strip is not None:
                    delete_strip(temp_strip)
#                    strip.select = True
#                    bpy.ops.transform.seq_slide(value=(0, -1), snap=False, view2d_edge_pan=True)
                bpy.types.Scene.movie_path = ""
            scene.generate_movie_prompt = prompt
            scene.generate_movie_negative_prompt = negative_prompt
            context.scene.movie_use_random = use_random
            context.scene.movie_num_seed = seed
        scene.frame_current = current_frame
        scene.generate_movie_prompt = prompt
        scene.generate_movie_negative_prompt = negative_prompt
        context.scene.movie_use_random = use_random
        context.scene.movie_num_seed = seed
        context.scene.sequence_editor.active_strip = active_strip

        addon_prefs.playsound = play_sound
        bpy.ops.renderreminder.pallaidium_play_notification()

        print("Processing finished.")

        return {"FINISHED"}


classes = (
    GeneratorAddonPreferences,
    SEQUENCER_OT_generate_movie,
    SEQUENCER_OT_generate_audio,
    SEQUENCER_OT_generate_image,
    SEQUENCER_OT_generate_text,
    SEQUENCER_PT_pallaidium_panel,
    GENERATOR_OT_sound_notification,
    SEQUENCER_OT_strip_to_generatorAI,
    LORABrowserFileItem,
    LORA_OT_RefreshFiles,
    LORABROWSER_UL_files,
    GENERATOR_OT_install,
    GENERATOR_OT_uninstall,
    SequencerOpenAudioFile,
    IPAdapterFaceProperties,
    IPAdapterFaceFileBrowserOperator,
    IPAdapterStyleProperties,
    IPAdapterStyleFileBrowserOperator,
)


def register():
    bpy.types.Scene.generate_movie_prompt = bpy.props.StringProperty(
        name="generate_movie_prompt",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.generate_movie_negative_prompt = bpy.props.StringProperty(
        name="generate_movie_negative_prompt",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.parler_direction_prompt = bpy.props.StringProperty(
        name="parler_direction_prompt",
        default="Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(name="generate_audio_prompt", default="")
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x",
        default=1024,
        step=64,
        min=256,
        max=4096,
        description="Use the power of 64",
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=576,
        step=64,
        min=256,
        max=4096,
        description="Use the power of 64",
    )

    # The number of frames to be generated.

    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_frames",
        default=6,
        min=-1,
        max=125,
        description="Number of frames to generate. NB. some models have fixed values.",
    )

    # The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.

    bpy.types.Scene.movie_num_inference_steps = bpy.props.IntProperty(
        name="movie_num_inference_steps",
        default=23,
        min=1,
        max=200,
        description="Number of inference steps to improve the quality",
    )

    # The number of videos to generate.

    bpy.types.Scene.movie_num_batch = bpy.props.IntProperty(
        name="movie_num_batch",
        default=1,
        min=1,
        max=100,
        description="Number of generated media files",
        update=filter_updated,
    )

    # The seed number.

    bpy.types.Scene.movie_num_seed = bpy.props.IntProperty(
        name="movie_num_seed",
        default=1,
        min=-2147483647,
        max=2147483647,
        description="Seed value",
    )

    # The seed number.

    bpy.types.Scene.movie_use_random = bpy.props.BoolProperty(
        name="movie_use_random",
        default=1,
        description="Randomize seed value. Switched off will give more consistency.",
    )

    # The guidance number.

    bpy.types.Scene.movie_num_guidance = bpy.props.FloatProperty(
        name="movie_num_guidance",
        default=4.0,
        min=0,
        max=100,
    )

    # The frame audio duration.

    bpy.types.Scene.audio_length_in_f = bpy.props.IntProperty(
        name="audio_length_in_f",
        default=80,
        min=-1,
        max=10000,
        description="Audio duration: Maximum 47 sec.",
    )
    bpy.types.Scene.generatorai_typeselect = bpy.props.EnumProperty(
        name="Sound",
        items=[
            ("movie", "Video", "Generate Video"),
            ("image", "Image", "Generate Image"),
            ("audio", "Audio", "Generate Audio"),
            ("text", "Text", "Generate Text"),
        ],
        default="image",
        update=output_strips_updated,
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

    # Inpaint

    bpy.types.Scene.inpaint_selected_strip = bpy.props.StringProperty(name="inpaint_selected_strip", default="")

    # Upscale

    bpy.types.Scene.video_to_video = bpy.props.BoolProperty(
        name="video_to_video",
        default=0,
    )

    # Refine SD

    bpy.types.Scene.refine_sd = bpy.props.BoolProperty(
        name="refine_sd",
        default=0,
        description="Add a refinement step",
    )

#    # hidiff

#    bpy.types.Scene.hidiff = bpy.props.BoolProperty(
#        name="hidiff",
#        default=0,
#        description="Use high resolution",
#    )


    # ADetailer
    bpy.types.Scene.adetailer = bpy.props.BoolProperty(
        name="adetailer",
        default=0,
        description="Add Face Details",
        update=filter_updated,
    )

    # AuraSR
    bpy.types.Scene.aurasr = bpy.props.BoolProperty(
        name="aurasr",
        default=0,
        description="4x Upscale (Aura SR)",
        update=filter_updated,
    )

    # movie path
    bpy.types.Scene.movie_path = bpy.props.StringProperty(name="movie_path", default="")
    bpy.types.Scene.movie_path = ""

    # image path
    bpy.types.Scene.image_path = bpy.props.StringProperty(name="image_path", default="")
    bpy.types.Scene.image_path = ""
    bpy.types.Scene.input_strips = bpy.props.EnumProperty(
        items=[
            ("input_prompt", "Prompts", "Prompts"),
            ("input_strips", "Strips", "Selected Strips"),
        ],
        default="input_prompt",
        update=input_strips_updated,
    )
    bpy.types.Scene.image_power = bpy.props.FloatProperty(
        name="image_power",
        default=0.50,
        min=0.05,
        max=0.82,
    )
    styles_array = load_styles(os.path.dirname(os.path.abspath(__file__)) + "/styles.json")
    if styles_array:
        bpy.types.Scene.generatorai_styles = bpy.props.EnumProperty(
            name="Generator AI Styles",
            items=[("no_style", "No Style", "No Style")] + styles_array,
            default="no_style",
            description="Add style prompts",
        )
    bpy.types.Scene.openpose_use_bones = bpy.props.BoolProperty(
        name="openpose_use_bones",
        default=0,
        description="Read as Open Pose rig image",
    )
    bpy.types.Scene.use_scribble_image = bpy.props.BoolProperty(
        name="use_scribble_image",
        default=0,
        description="Read as scribble image",
    )

    # Blip

    bpy.types.Scene.blip_cond_subject = bpy.props.StringProperty(
        name="blip_cond_subject",
        default="",
        description="Condition Image",
    )
    bpy.types.Scene.blip_tgt_subject = bpy.props.StringProperty(
        name="blip_tgt_subject",
        default="",
        description="Target Prompt",
    )
    bpy.types.Scene.blip_subject_image = bpy.props.StringProperty(
        name="blip_subject_image",
        default="",
        description="Subject Image",
    )
    #    bpy.types.Scene.use_freeU = bpy.props.BoolProperty(
    #        name="use_freeU",
    #        default=0,
    #    )

    bpy.types.Scene.use_lcm = bpy.props.BoolProperty(
        name="use_lcm",
        default=0,
        description="Higher Speed, lower quality. Try Quality Steps: 1-10",
        update=lcm_updated,
    )

    # SVD decode chunck

    bpy.types.Scene.svd_decode_chunk_size = bpy.props.IntProperty(
        name="svd_decode_chunk_size",
        default=2,
        min=1,
        max=100,
        description="Number of frames to decode",
    )

    # SVD motion_bucket_id

    bpy.types.Scene.svd_motion_bucket_id = bpy.props.IntProperty(
        name="svd_motion_bucket_id",
        default=1,
        min=1,
        max=512,
        description="A higher number: more camera movement. A lower number: more character movement",
    )

    for cls in classes:
        bpy.utils.register_class(cls)

    # LoRA

    bpy.types.Scene.lora_files = bpy.props.CollectionProperty(type=LORABrowserFileItem)
    bpy.types.Scene.lora_files_index = bpy.props.IntProperty(name="Index", default=0)
    bpy.types.Scene.lora_folder = bpy.props.StringProperty(
        name="Folder",
        description="Select a folder",
        subtype="DIR_PATH",
        default="",
        update=update_folder_callback,
    )
    bpy.types.Scene.audio_path = bpy.props.StringProperty(
        name="audio_path",
        default="",
        description="Path to speaker voice",
    )

    # The frame audio duration.

    bpy.types.Scene.audio_speed = bpy.props.IntProperty(
        name="audio_speed",
        default=13,
        min=1,
        max=20,
        description="Speech speed.",
    )

    bpy.types.Scene.ip_adapter_face_folder = bpy.props.StringProperty(
        name="File",
        description="Select a file or folder",
        default="",
        options={"TEXTEDIT_UPDATE"},
        # update=update_ip_adapter_face_callback,
    )
    bpy.types.Scene.ip_adapter_face_files_to_import = bpy.props.CollectionProperty(type=IPAdapterFaceProperties)
    bpy.types.Scene.ip_adapter_style_folder = bpy.props.StringProperty(
        name="File",
        description="Select a file or folder",
        default="",
        options={"TEXTEDIT_UPDATE"},
        # update=update_ip_adapter_style_callback,
    )
    bpy.types.Scene.ip_adapter_style_files_to_import = bpy.props.CollectionProperty(type=IPAdapterStyleProperties)


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
    #del bpy.types.Scene.hidiff
    del bpy.types.Scene.aurasr
    del bpy.types.Scene.adetailer
    del bpy.types.Scene.generatorai_styles
    del bpy.types.Scene.inpaint_selected_strip
    del bpy.types.Scene.openpose_use_bones
    del bpy.types.Scene.use_scribble_image
    del bpy.types.Scene.blip_cond_subject
    del bpy.types.Scene.blip_tgt_subject
    del bpy.types.Scene.blip_subject_image
    del bpy.types.Scene.lora_files
    del bpy.types.Scene.lora_files_index
    del bpy.types.Scene.ip_adapter_face_folder
    del bpy.types.Scene.ip_adapter_style_folder
    del bpy.types.Scene.ip_adapter_face_files_to_import
    del bpy.types.Scene.ip_adapter_style_files_to_import


if __name__ == "__main__":
    unregister()
    register()
