from __future__ import annotations

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
    "version": (2, 1),
    "blender": (4, 5, 0),
    "location": "Video Sequence Editor > Sidebar > Generative AI",
    "description": "AI Generate media in the VSE",
    "category": "Sequencer",
}

# TO DO: Move prints.
# Pop-up for audio
# Use a-z for batches

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
import base64
from io import BytesIO
import asyncio
import inspect
from fractions import Fraction


print("Python: " + sys.version)

# Get the path of the Python executable (e.g., python.exe)
python_exe_dir = os.path.dirname(os.__file__)

# Construct the path to the site-packages directory
site_packages_dir = os.path.join(python_exe_dir, "site-packages")

# Add the site-packages directory to the top of sys.path
sys.path.insert(0, site_packages_dir)

dir_path = os.path.join(bpy.utils.user_resource("DATAFILES"), "Pallaidium Media")
os.makedirs(dir_path, exist_ok=True)

# if os_platform == "Windows":
#    # Temporarily modify pathlib.PosixPath for Windows compatibility
#    temp = pathlib.PosixPath
#    pathlib.PosixPath = pathlib.WindowsPath

site_paths_to_move = set(site.getsitepackages() + [site.getusersitepackages()])

seen = set()
unique_ordered_paths = []
for path in sys.path:
    if path not in seen:
        unique_ordered_paths.append(path)
        seen.add(path)

non_site_paths = []
final_site_paths = []
for path in unique_ordered_paths:
    if path in site_paths_to_move or 'site-packages' in path:
        final_site_paths.append(path)
    else:
        non_site_paths.append(path)

sys.path[:] = non_site_paths + final_site_paths

#print("\n--- Modified Python Path (site-packages moved to the end) ---")
#for i, path in enumerate(sys.path):
#    print(f"{i}: {path}")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xformers.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="1Torch was not compiled"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
warnings.filterwarnings("ignore", category=UserWarning, message="FutureWarning: ")

# Disable certain warnings that are common with PyTorch on Apple Silicon
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="The operator.*is not current")
warnings.filterwarnings("ignore", category=UserWarning, message="Converting a tensor to a Python boolean")

import logging
logging.getLogger("xformers").setLevel(logging.ERROR)  # shutup triton
logging.getLogger("diffusers.models.modeling_utils").setLevel(logging.CRITICAL)


try:
    exec("import torch")
    if torch.cuda.is_available():
        gfx_device = "cuda"
    elif torch.backends.mps.is_available():
        gfx_device = "mps"
    else:
        gfx_device = "cpu"
    # Set environment variables for better MPS performance
    if os_platform == "Darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Disable oneDNN optimizations that can cause issues on Apple Silicon
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if gfx_device == 'mps' and not torch.backends.mps.is_available():
          raise Exception("Device set to MPS, but MPS is not available")
    elif gfx_device == 'cuda' and not torch.cuda.is_available():
          raise Exception("Device set to CUDA, but CUDA is not available") 
except:
    print(
        "Pallaidium dependencies needs to be installed and Blender needs to be restarted."
    )

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

os_platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'
if os_platform == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


DEBUG = False

def debug_print(*args):
    if DEBUG:
        print(*args)

def show_system_console(show):
    if os_platform == "Windows":
        # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
        SW_HIDE = 0
        SW_SHOW = 5
        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW
        )  # if show else SW_HIDE


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
        styles_array.append(
            (negative_prompt.lower().replace(" ", "_"), name.title(), prompt)
        )
    return styles_array


def style_prompt(prompt):
    selected_entry_key = bpy.context.scene.generatorai_styles
    return_array = []
    if selected_entry_key:
        styles_array = load_styles(
            os.path.dirname(os.path.abspath(__file__)) + "/styles.json"
        )
        if styles_array:
            selected_entry = next(
                (item for item in styles_array if item[0] == selected_entry_key), None
            )
            if selected_entry:
                selected_entry_list = list(selected_entry)
                return_array.append(selected_entry_list[2].replace("{prompt}", prompt))
                return_array.append(
                    bpy.context.scene.generate_movie_negative_prompt
                    + ", "
                    + selected_entry_list[0].replace("_", " ")
                )
                return return_array
    return_array.append(prompt)
    return_array.append(bpy.context.scene.generate_movie_negative_prompt)
    return return_array


def closest_divisible_8(num):
    # Determine the remainder when num is divided by 8
    remainder = num % 8
    # If the remainder is less than or equal to 16, return num - remainder,
    # but ensure the result is not less than 192
    if remainder <= 4:
        result = num - remainder
        return max(result, 192)
    # Otherwise, return num + (32 - remainder)
    else:
        return max(num + (8 - remainder), 192)


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


def closest_divisible_32(num):
    # Determine the remainder when num is divided by 64
    remainder = num % 32
    # If the remainder is less than or equal to 16, return num - remainder,
    # but ensure the result is not less than 192
    if remainder <= 16:
        result = num - remainder
        return max(result, 192)
    # Otherwise, return num + (32 - remainder)
    else:
        return max(num + (32 - remainder), 192)


### Ensure dimensions are divisible by 32
# def closest_divisible_32(value):
#    return max(32, (value // 32) * 32)  # Avoid zero or negative sizes


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
    original_selection = [
        strip
        for strip in bpy.context.scene.sequence_editor.sequences_all
        if strip.select
    ]
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

    extension = os.path.splitext(file_path)[
        -1
    ].lower()  # Convert to lowercase for case-insensitive comparison
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
    import numpy as np

    Image.MAX_IMAGE_PIXELS = None

    processed_frames = []
    image_files = sorted(
        [f for f in os.listdir(frame_folder_path) if f.endswith(".png")]
    )
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(frame_folder_path, image_file)
        img = Image.open(image_path)

        # Original dimensions
        frame_width, frame_height = img.size

        # Calculate target dimensions
        target_height = int((target_width / frame_width) * frame_height)
        target_width = closest_divisible_8(target_width)
        target_height = closest_divisible_8(target_height)

        # Validate dimensions
        if target_width <= 0 or target_height <= 0:
            print(
                f"Invalid dimensions for frame {idx + 1}: {target_width}x{target_height}"
            )
            continue

        # Resize and convert
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img = img.convert("RGB")

        # Validate image array
        img_array = np.array(img)
        if img_array.size == 0:
            print(f"Empty array for frame {idx + 1}. Skipping.")
            continue

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
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Dynamically get the video width (input size)
    movie_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Save each loaded frame as an image in the temp folder
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Save the frame as an image in the temp folder
        temp_image_path = os.path.join(temp_image_folder, f"frame_{i:04d}.png")
        print("Temp path: " + temp_image_path)
        cv2.imwrite(temp_image_path, frame)

    cap.release()

    # Process frames using the separate function
    processed_frames = process_frames(temp_image_folder, movie_x)

    # Clean up: Delete the temporary image folder (optional, commented for debugging)
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


def low_vram():
    try:
        if gfx_device == "mps":
            return True

        exec("import torch")

        total_vram = 0
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            total_vram += properties.total_memory
        return (total_vram / (1024**3)) <= 16  # Y/N under 16 GB?
    except:
        print("Torch not found!")
        return True


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
#
#def flush():
#    import torch
#    import gc

#    gc.collect()
#    torch.cuda.empty_cache()
#    torch.cuda.reset_max_memory_allocated()
#    # torch.cuda.reset_peak_memory_stats()

# def isWindows():
#    return os.name == "nt"


# def isMacOS():
#    return os.name == "posix" and platform.system() == "Darwin"


# def isLinux():
#    return os.name == "posix" and platform.system() == "Linux"


# def python_exec():
#    import sys

#    if isWindows():
#        return os.path.join(sys.prefix, "bin", "python.exe")
#    elif isMacOS():
#        try:
#            # 2.92 and older
#            path = bpy.app.binary_path_python
#        except AttributeError:
#            # 2.93 and later
#            import sys
#            path = sys.executable
#        return os.path.abspath(path)
#    elif isLinux():
#        return os.path.join(sys.prefix, "bin", "python")
#    else:
#        print("sorry, still not implemented for ", os.name, " - ", platform.system)


def python_exec():
    return sys.executable


def find_strip_by_name(scene, name):
    for sequence in scene.sequence_editor.sequences:
        if sequence.name == name:
            return sequence
    return None


def get_strip_path(strip):
    if strip.type == "IMAGE":
        strip_dirname = os.path.dirname(strip.directory)
        image_path = bpy.path.abspath(
            os.path.join(strip_dirname, strip.elements[0].filename)
        )
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
    if os.path.exists(file_name):
        base_name, extension = os.path.splitext(file_name)
        index = 1
        while True:
            unique_file_name = f"{base_name}_{index}{extension}"
            if not os.path.exists(unique_file_name):
                return unique_file_name
            index += 1
    else:
        return file_name


def parse_python_version(version_info):
    major, minor = version_info[:2]
    return f"{major}.{minor}"


def install_modules(self):
    os_platform = platform.system()
    pybin = python_exec()

    def ensure_pip():
        print("Ensuring: pip")
        try:
            subprocess.check_call([pybin, "-m", "pip", "install", "--upgrade", "pip"])
        except Exception as e:
            print(f"Error installing pip: {e}")
            return False
        return True

    def install_module(name, package=None, use_git=False):
        package = package if package else name
        try:
            subprocess.check_call([
                pybin, "-m", "pip", "install", "--disable-pip-version-check",
                "--use-deprecated=legacy-resolver", package,
                "--no-warn-script-location", "--upgrade"
            ])
            print(f"Successfully installed {name}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {name}: {e}")
            return False
        return True

    # Common modules
    common_modules = [
        ("requests", "requests"),
        #("huggingface_hub", "huggingface_hub"),
        ("huggingface_hub", "huggingface_hub[hf_xet]"), 
        ("gguf", "gguf"),
        ("pydub", "pydub"),
        ("sentencepiece", "sentencepiece"),
        ("safetensors", "safetensors"),
        ("cv2", "opencv_python"),
        ("PIL", "pillow"),
        ("IPython", "IPython"),
        ("omegaconf", "omegaconf"),
        ("aura_sr", "aura-sr"),
        ("stable_audio_tools", "stable-audio-tools"),
        ("beautifulsoup4", "beautifulsoup4"),
        ("ftfy", "ftfy"),
        ("librosa", "librosa"),
        ("imageio", "imageio[ffmpeg]==2.4.1"),
        ("imageio", "imageio-ffmpeg"),
        ("imWatermark", "imWatermark"),
        ("mediapipe", "mediapipe"),
        ("scipy", "scipy"), #scipy==1.12.0
        ("protobuf", "protobuf==3.20.1"),
        ("scikit_learn", "scikit-learn==1.2.2"),
        ("bitsandbytes", "bitsandbytes"),
        #("chatterbox", "--no-deps git+https://https://github.com/tin2tin/chatterbox.git"),
        ("chatterbox", "--no-deps chatterbox-tts"),
        ("numpy", "numpy==1.26.4"),
        ("jax", "jax"),
        #("jaxlib", "jaxlib>=0.5.0")
        ("tqdm", "tqdm"),
        ("tempfile", "tempfile"),
        ("f5_tts", "git+https://github.com/SWivid/F5-TTS.git"),
        ("resemble_perth", "resemble-perth==1.0.1"),
        ("s3tokenizer", "s3tokenizer"),
        ("conformer", "conformer"),
        ("spacy", "spacy"),
        ("hf_xet", "hf-xet"),
    ]

    show_system_console(True)
    set_system_console_topmost(True)
    ensure_pip()

    for module_name, package_name in common_modules:
        install_module(module_name, package_name)

    # Platform-specific installations
    if os_platform == "Windows":
        windows_modules = [
            # How to install a patch: git+https://github.com/huggingface/diffusers@integrations/ltx-097
            #("diffusers", "diffusers==0.34.0"),
            ("diffusers", "git+https://github.com/huggingface/diffusers.git"),
            ("mmaudio", "git+https://github.com/hkchengrex/MMAudio.git"),
            #("deepspeed", "https://www.piwheels.org/simple/deepspeed/deepspeed-0.16.5-py3-none-any.whl"),
            #("deepspeed", "https://github.com/agwosdz/DeepSpeed-Wheels-for-Windows/releases/download/DeepSpeed/deepspeed-0.16.1+unknown-cp311-cp311-win_amd64_cu124.whl"),
            #("deepspeed", "https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu121-cp311-cp311-win_amd64.whl"),
            #("deepspeed", "https://github.com/agwosdz/DeepSpeed-Wheels-for-Windows/releases/download/DeepSpeed/deepspeed-0.15.1+51c6eae-cp311-cp311-win_amd64_cu124.whl"),
            ("resemble_enhance", "git+https://github.com/tin2tin/resemble-enhance-windows.git"),
            ("flash_attn", "https://huggingface.co/lldacing/flash-attention-windows-wheel/blob/main/flash_attn-2.7.0.post2%2Bcu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl"),
            #("flash_attn", "git+https://github.com/ROCm/flash-attention.git"),
            #("flash_attn", "https://github.com/oobabooga/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.3.1cxx11abiFALSE-cp311-cp311-win_amd64.whl"),
            #("triton", "triton-windows"),
            #("sageattention", "https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu124torch2.5.1-cp311-cp311-win_amd64.whl"),
            #("triton", "https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp311-cp311-win_amd64.whl"),
            #("triton", "https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp311-cp311-win_amd64.whl"),
            # Use this for low cards/cuda?
            #("triton", "https://hf-mirror.com/LightningJay/triton-2.1.0-python3.11-win_amd64-wheel/resolve/main/triton-2.1.0-cp311-cp311-win_amd64.whl"),
        ]

        for module_name, package_name in windows_modules:
            install_module(module_name, package_name)
    else:
        other_modules = [
            ("diffusers", "git+https://github.com/huggingface/diffusers.git"),
            #("deepspeed", "deepspeed"), #==0.14.4
            ("resemble_enhance", "resemble-enhance"),
            ("flash_attn", "flash-attn"),
            ("triton", "triton"),
            ("sageattention","sageattention==1.0.6")
        ]

        for module_name, package_name in other_modules:
            install_module(module_name, package_name)
            
    if os_platform == "Darwin":
        install_module("mflux","--no-deps mflux")
        install_module("matplotlib","--no-deps matplotlib")
        install_module("mlx","--no-deps mlx")
        install_module("opencv_python","--no-deps opencv-python")
        install_module("piexif","--no-deps piexif")
        install_module("platformdirs","--no-deps platformdirs")
        install_module("toml","--no-deps toml")
    
    # Python version-specific installations
    from packaging import version
    python_version = sys.version_info
    if version.parse(".".join(map(str, python_version[:3]))) >= version.parse("3.8"):
        install_module("image_gen_aux", "git+https://github.com/huggingface/image_gen_aux")

    # Additional installations
#    subprocess.check_call([
#        pybin, "-m", "spacy", "download", "en_core_web_md",
#    ])
    subprocess.call(
        [
            pybin,
            "-m",
            "pip",
            "install",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
            "--no-deps",
            "--disable-pip-version-check",
            "--no-warn-script-location",
        ]
    )
#    subprocess.check_call([
#        pybin, "-m", "pip", "install", "--disable-pip-version-check",
#        "--use-deprecated=legacy-resolver", "tensorflow<2.11", "--upgrade"
#    ])
#    deepspeed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepspeed", "deepspeed-0.16.5-py3-none-any.whl")
#    print(deepspeed_path)
#    subprocess.call(
#        [
#            pybin,
#            "-m",
#            "pip",
#            "install",
#            "--disable-pip-version-check",
#            "--use-deprecated=legacy-resolver",
#            deepspeed_path,
#            "--no-warn-script-location",
#        ]
#    )
    install_module("controlnet_aux", "controlnet-aux")
    install_module("whisperspeech", "WhisperSpeech==0.8")
    install_module(
        "parler_tts", "git+https://github.com/huggingface/parler-tts.git"
    )
    install_module("laion_clap", "laion-clap==1.1.6")
    install_module("numpy", "numpy==1.26.4")
    subprocess.call(
        [
            pybin,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--use-deprecated=legacy-resolver",
            "ultralytics",
            "--no-warn-script-location",
            "--upgrade",
        ]
    )
    subprocess.call(
        [
            pybin,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--use-deprecated=legacy-resolver",
            "git+https://github.com/tin2tin/adetailer_sdxl.git",
        ]
    )
    #subprocess.call([pybin, "-m", "pip", "install", "--disable-pip-version-check", "--use-deprecated=legacy-resolver", "git+https://github.com/theblackhatmagician/adetailer_sdxl.git"])
    subprocess.call(
        [
            pybin,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--use-deprecated=legacy-resolver",
            "lmdb",
            "--no-warn-script-location",
            "--upgrade",
        ]
    )
    subprocess.call(
        [
            pybin,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--use-deprecated=legacy-resolver",
            "git+https://github.com/huggingface/accelerate.git",
            "--no-warn-script-location",
            "--upgrade",
        ]
    )
    subprocess.call(
        [
            pybin,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "git+https://github.com/tin2tin/bark.git",
            #"git+https://github.com/suno-ai/bark.git",
            "--no-warn-script-location",
            "--upgrade",
        ]
    )
#    uninstall_module_with_dependencies("timm")
#    subprocess.check_call([
#        pybin, "-m", "pip", "uninstall", "-y", "timm",
#    ])
    # Torch installations
    if os_platform == "Windows":
        subprocess.check_call([
            pybin, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio", "xformers"
        ])
        subprocess.check_call([
            pybin, "-m", "pip", "install",
            "torch", "xformers", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124",
            "--no-warn-script-location", "--upgrade"
        ])
#        subprocess.check_call([
#            pybin, "-m", "pip", "install",
#            "torch==2.3.1+cu121", "xformers", "torchvision",
#            "--index-url", "https://download.pytorch.org/whl/cu121",
#            "--no-warn-script-location", "--upgrade"
#        ])
#        subprocess.check_call([
#            pybin, "-m", "pip", "install",
#            "torchaudio==2.3.1+cu121",
#            "--index-url", "https://download.pytorch.org/whl/cu121",
#            "--no-warn-script-location", "--upgrade"
#        ])
    else:
        install_module("torch", "torch")
        install_module("torchvision", "torchvision")
        install_module("torchaudio", "torchaudio")
        install_module("xformers", "xformers")

    install_module("torcheval", "torcheval")
    install_module("torchao", "torchao")

    # Final tasks
    subprocess.check_call([
        pybin, "-m", "pip", "install", "--disable-pip-version-check",
        "peft", "--upgrade"
    ])
#    subprocess.check_call([
#        pybin, "pip", "install", "--disable-pip-version-check",
#        "--use-deprecated=legacy-resolver", "timm", "--upgrade"
#    ])

    install_module("sageattention","sageattention==1.0.6")
    install_module("timm", "git+https://github.com/rwightman/pytorch-image-models.git")
    install_module("protobuf", "protobuf==3.20.1")
    install_module("numpy", "numpy==1.26.4")
    #install_module("tokenizers", "tokenizers==0.21.1")
    install_module("tokenizers", "tokenizers==0.22.0")
    #install_module("transformers", "transformers==4.46.1")
    #install_module("transformers", "git+https://github.com/huggingface/transformers.git")
    install_module("transformers", "transformers==4.56.2")
    #print("Cleaning up cache...")
    #subprocess.check_call([pybin, "-m", "pip", "cache", "purge"])
    subprocess.check_call([pybin, "-m", "pip", "list"])

    self.report({"INFO"}, "All modules installed successfully.")


def get_module_dependencies(module_name):
    """
    Get the list of dependencies for a given module.
    """
    pybin = python_exec()
    result = subprocess.run(
        [pybin, "-m", "pip", "show", module_name], capture_output=True, text=True
    )
    dependencies = []
    if result.stdout:
        output = result.stdout.strip()
    else:
        return dependencies
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

    subprocess.run([pybin, "-m", "pip", "uninstall", "-y", module_name])

    for dependency in dependencies:
        if (
            len(dependency) > 5 and str(dependency[5].lower) != "numpy"
        ) and not dependency.find("requests"):
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

        # List of modules to uninstall grouped by category
        modules_to_uninstall = {
            "AI Tools": [
                "torch", "torchvision", "torchaudio", "diffusers", "transformers",
                "sentencepiece", "safetensors", "bark", "xformers", "imageio",
                "imWatermark", "controlnet-aux", "bitsandbytes"
            ],
            "ML Frameworks": [
                "opencv_python", "scipy", "IPython", "pillow", "libtorrent", "accelerate",
                "triton", "cv2", "protobuf", "tensorflow"
            ],
            "Model Tools": [
                "resemble-enhance", "mediapipe", "flash_attn", "stable-audio-tools",
                "beautifulsoup4", "ftfy", "deepspeed",
                "gradio-client" , "suno-bark", "peft", "ultralytics",
                "parler-tts"
            ], # "albumentations", "datasets", "insightface"
            "Utils": [
                "celluloid", "omegaconf", "pandas", "ptflops", "rich", "resampy",
                "tabulate", "gradio", "jax", "jaxlib", "sympy"
            ],
            "WhisperSpeech Components": [
                "ruamel.yaml.clib", "fastprogress", "fastcore", "ruamel.yaml",
                "hyperpyyaml", "speechbrain", "vocos", "WhisperSpeech", "pydub"
            ],
            "Speech Components": [
                "chatterbox-tts", "f5-tts", "resemble-perth", "s3tokenizer",
                "conformer"
            ],
        }

        # Uninstall all modules and their dependencies
        for category, modules in modules_to_uninstall.items():
            for module in modules:
                uninstall_module_with_dependencies(module)

        # Clear pip cache
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
    if (scene.aurasr or scene.adetailer) and scene.movie_num_batch > 1:
        scene.movie_num_batch = 1
        print(
            "INFO: Aura SR and ADetailer will only allow for 1 batch for memory reasons."
        )

def input_strips_updated(self, context):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences
    scene = context.scene
    scene_type = scene.generatorai_typeselect
    input_strips = scene.input_strips
    image_model = addon_prefs.image_model_card
    movie_model = addon_prefs.movie_model_card
    audio_model = addon_prefs.audio_model_card

    # Text Type Handling
    if scene_type == "text" and scene.input_strips != "input_strips":
        if addon_prefs.text_model_card != "ZuluVision/MoviiGen1.1_Prompt_Rewriter":
            scene.input_strips = "input_strips"
    # Image Type Handling
    if scene_type == "image":
#        if image_model == "Shitao/OmniGen-v1-diffusers": #crash
#            scene.input_strips = "input_prompt"
        if scene.input_strips != "input_strips" and image_model in {
            "diffusers/controlnet-canny-sdxl-1.0-small",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "xinsir/controlnet-scribble-sdxl-1.0",
            "ZhengPeng7/BiRefNet_HR",
            "Salesforce/blipdiffusion",
            "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora",
            "romanfratric234/FLUX.1-Depth-dev-lora",
            "Runware/FLUX.1-Redux-dev",
            "kontext-community/relighting-kontext-dev-lora-v3",
        }:
            scene.input_strips = "input_strips"

        # Handle specific image models
        if image_model in {
            "dataautogpt3/OpenDalleV1.1",
            "Kwai-Kolors/Kolors-diffusers",
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
        }:
            scene.use_lcm = False
        if image_model == "lzyvegetable/FLUX.1-schnell":
            scene.movie_num_inference_steps = 4
            scene.movie_num_guidance = 0
        elif image_model == "ChuckMcSneed/FLUX.1-dev":
            scene.movie_num_inference_steps = 25
            scene.movie_num_guidance = 4
        elif image_model == "ostris/Flex.2-preview":
            scene.movie_num_inference_steps = 28
            scene.movie_num_guidance = 3.5

    # Movie Type Handling
    elif scene_type == "movie":
        if movie_model == "hunyuanvideo-community/HunyuanVideo":
            #scene.generate_movie_x = 960
            #scene.generate_movie_y = 544
            #scene.generate_movie_frames = 49
            scene.movie_num_inference_steps = 40
            scene.movie_num_guidance = 4
        elif movie_model in {
            "THUDM/CogVideoX-5b",
            "THUDM/CogVideoX-2b"
        }:
            scene.generate_movie_x = 720
            scene.generate_movie_y = 480
            scene.generate_movie_frames = 49
            scene.movie_num_inference_steps = 50
            scene.movie_num_guidance = 6
        elif movie_model == "genmo/mochi-1-preview":
            scene.generate_movie_x = 848
            scene.generate_movie_y = 480
            scene.input_strips = "input_prompt"
        elif movie_model == "Skywork/SkyReels-V1-Hunyuan-T2V":
            #scene.generate_movie_x = 960
            #scene.generate_movie_y = 544
            #scene.generate_movie_frames = 49
            scene.movie_num_inference_steps = 40
            scene.movie_num_guidance = 1
#        elif movie_model == "cerspense/zeroscope_v2_XL":
#            scene.upscale = False

        # Handle specific input strips for movie types
        if (
            movie_model in {
                "stabilityai/stable-video-diffusion-img2vid",
                "stabilityai/stable-video-diffusion-img2vid-xt",
                "Hailuo/MiniMax/img2vid",
                "Hailuo/MiniMax/subject2vid"
            }
        ) and scene.input_strips != "input_strips":
            scene.input_strips = "input_strips"

    # Audio Type Handling
    elif scene_type == "audio":
        if audio_model == "stabilityai/stable-audio-open-1.0":
            scene.movie_num_inference_steps = 200
#        elif addon_prefs.audio_model_card == "MMAudio" and scene.input_strips != "input_strips":
#            scene.input_strips = "input_strips"

    # Common Handling for Selected Strip
    if scene_type in {"movie", "audio"} or image_model == "xinsir/controlnet-scribble-sdxl-1.0":
        scene.inpaint_selected_strip = ""

    # LORA Handling
    if scene.lora_folder:
        bpy.ops.lora.refresh_files()

    # Clear Paths if Input is Prompt
    if scene.input_strips == "input_prompt":
        bpy.types.Scene.movie_path = ""
        bpy.types.Scene.image_path = ""


def output_strips_updated(self, context):
    prefs = context.preferences
    addon_prefs = prefs.addons[__name__].preferences
    scene = context.scene

    image_model = addon_prefs.image_model_card
    movie_model = addon_prefs.movie_model_card
    audio_model = addon_prefs.audio_model_card

    type = scene.generatorai_typeselect
    #strip_input = scene.input_strips

    # Default values for movie generation settings
    movie_res_x = scene.generate_movie_x
    movie_res_y = scene.generate_movie_y
    movie_frames = scene.generate_movie_frames
    movie_inference = scene.movie_num_inference_steps
    movie_guidance = scene.movie_num_guidance

    # Text Type Handling
    if type == "text" and scene.input_strips != "input_strips":
        if addon_prefs.text_model_card != "ZuluVision/MoviiGen1.1_Prompt_Rewriter":
            scene.input_strips = "input_strips"

    # === IMAGE TYPE === #
    if type == "image":
        if image_model == "Shitao/OmniGen-v1-diffusers":
            scene.input_strips = "input_prompt"
        elif image_model in [
            "diffusers/controlnet-canny-sdxl-1.0",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "xinsir/controlnet-scribble-sdxl-1.0",
            "ZhengPeng7/BiRefNet_HR",
            "Salesforce/blipdiffusion",
            "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora",
            "romanfratric234/FLUX.1-Depth-dev-lora",
            "Runware/FLUX.1-Redux-dev",
            "kontext-community/relighting-kontext-dev-lora-v3",
        ]:
            scene.input_strips = "input_strips"
        elif image_model == "dataautogpt3/OpenDalleV1.1":
            scene.use_lcm = False
        elif image_model == "Kwai-Kolors/Kolors-diffusers":
            scene.use_lcm = False
        elif image_model == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers":
            scene.use_lcm = False
        elif image_model == "lzyvegetable/FLUX.1-schnell":
            movie_inference = 4
            movie_guidance = 0
        elif image_model == "ChuckMcSneed/FLUX.1-dev":
            movie_inference = 25
            movie_guidance = 4
        elif image_model == "ostris/Flex.2-preview":
            movie_inference = 28
            movie_guidance = 3.5

    # === MOVIE TYPE === #
    elif type == "movie":
        if movie_model == "hunyuanvideo-community/HunyuanVideo":
            movie_res_x = 960
            movie_res_y = 544
            movie_frames = 49
            movie_inference = 20
            movie_guidance = 4
        elif movie_model == "Skywork/SkyReels-V1-Hunyuan-T2V":
            movie_res_x = 960
            movie_res_y = 544
            movie_frames = 49
            movie_inference = 40
            movie_guidance = 1
#        elif movie_model == "cerspense/zeroscope_v2_XL":
#            scene.upscale = False
        elif movie_model in ["THUDM/CogVideoX-5b", "THUDM/CogVideoX-2b"]:
            movie_res_x = 720
            movie_res_y = 480
            movie_frames = 49
            movie_inference = 50
            movie_guidance = 6
        elif movie_model == "genmo/mochi-1-preview":
            movie_res_x = 848
            movie_res_y = 480
            movie_inference = 50
        elif movie_model in [
            "stabilityai/stable-video-diffusion-img2vid",
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "Hailuo/MiniMax/img2vid",
            "Hailuo/MiniMax/subject2vid"
        ]:
            scene.input_strips = "input_strips"

    # === AUDIO TYPE === #
    elif type == "audio":
        if audio_model == "stabilityai/stable-audio-open-1.0":
            movie_inference = 200
#        if addon_prefs.audio_model_card == "MMAudio":
#            scene.input_strips = "input_strips"

    # === COMMON SETTINGS === #
    if type in ["movie", "audio"] or image_model == "xinsir/controlnet-scribble-sdxl-1.0":
        scene.inpaint_selected_strip = ""
        if scene.lora_folder:
            bpy.ops.lora.refresh_files()

    if type == "movie":
        scene.generate_movie_x = movie_res_x
        scene.generate_movie_y = movie_res_y
        scene.generate_movie_frames = movie_frames
        scene.movie_num_inference_steps = movie_inference
        scene.movie_num_guidance = movie_guidance


# Relight:

# Illumination options mapping
ILLUMINATION_OPTIONS = {
    # Natural Daylight
    "natural lighting": "Neutral white color temperature with balanced exposure and soft shadows",
    "sunshine from window": "Bright directional sunlight with hard shadows and visible light rays",
    "golden time": "Warm golden hour lighting with enhanced warm colors and soft shadows",
    "sunrise in the mountains": "Warm backlighting with atmospheric haze and lens flare",
    "afternoon light filtering through trees": "Dappled sunlight patterns with green color cast from foliage",
    "early morning rays, forest clearing": "God rays through trees with warm color temperature",
    "golden sunlight streaming through trees": "Golden god rays with atmospheric particles in light beams",
    
    # Sunset & Evening
    "sunset over sea": "Warm sunset light with soft diffused lighting and gentle gradients",
    "golden hour in a meadow": "Golden backlighting with lens flare and rim lighting",
    "golden hour on a city skyline": "Golden lighting on buildings with silhouette effects",
    "evening glow in the desert": "Warm directional lighting with long shadows",
    "dusky evening on a beach": "Cool backlighting with horizon silhouettes",
    "mellow evening glow on a lake": "Warm lighting with water reflections",
    "warm sunset in a rural village": "Golden hour lighting with peaceful warm tones",
    
    # Night & Moonlight
    "moonlight through curtains": "Cool blue lighting with curtain shadow patterns",
    "moonlight in a dark alley": "Cool blue lighting with deep urban shadows",
    "midnight in the forest": "Very low brightness with minimal ambient lighting",
    "midnight sky with bright starlight": "Cool blue lighting with star point sources",
    "fireflies lighting up a summer night": "Small glowing points with warm ambient lighting",
    
    # Indoor & Cozy
    "warm atmosphere, at home, bedroom": "Very warm lighting with soft diffused glow",
    "home atmosphere, cozy bedroom illumination": "Warm table lamp lighting with pools of light",
    "cozy candlelight": "Warm orange flickering light with dramatic shadows",
    "candle-lit room, rustic vibe": "Multiple warm candlelight sources with atmospheric shadows",
    "night, cozy warm light from fireplace": "Warm orange-red firelight with flickering effects",
    "campfire light": "Warm orange flickering light from below with dancing shadows",
    
    # Urban & Neon
    "neon night, city": "Vibrant blue, magenta, and green neon lights with reflections",
    "blue neon light, urban street": "Blue neon lighting with urban glow effects",
    "neon, Wong Kar-wai, warm": "Warm amber and red neon with moody selective lighting",
    "red and blue police lights in rain": "Alternating red and blue strobing with wet reflections",
    "red glow, emergency lights": "Red emergency lighting with harsh shadows and high contrast",
    
    # Sci-Fi & Fantasy
    "sci-fi RGB glowing, cyberpunk": "Electric blue, pink, and green RGB lighting with glowing effects",
    "rainbow reflections, neon": "Chromatic rainbow patterns with prismatic reflections",
    "magic lit": "Colored rim lighting in purple and blue with soft ethereal glow",
    "mystical glow, enchanted forest": "Supernatural green and blue glowing with floating particles",
    "ethereal glow, magical forest": "Supernatural lighting with blue-green rim lighting",
    "underwater glow, deep sea": "Blue-green lighting with caustic patterns and particles",
    "underwater luminescence": "Blue-green bioluminescent glow with caustic light patterns",
    "aurora borealis glow, arctic landscape": "Green and purple dancing sky lighting",
    "crystal reflections in a cave": "Sparkle effects with prismatic light dispersion",
    
    # Weather & Atmosphere
    "foggy forest at dawn": "Volumetric fog with cool god rays through trees",
    "foggy morning, muted light": "Soft fog effects with reduced contrast throughout",
    "soft, diffused foggy glow": "Heavy fog with soft lighting and no harsh shadows",
    "stormy sky lighting": "Dramatic lighting with high contrast and rim lighting",
    "lightning flash in storm": "Brief intense white light with stark shadows",
    "rain-soaked reflections in city lights": "Wet surface reflections with streaking light effects",
    "gentle snowfall at dusk": "Cool blue lighting with snowflake particle effects",
    "hazy light of a winter morning": "Neutral lighting with atmospheric haze",
    "mysterious twilight, heavy mist": "Heavy fog with cool lighting and atmospheric depth",
    
    # Seasonal & Nature
    "vibrant autumn lighting in a forest": "Enhanced warm autumn colors with dappled sunlight",
    "purple and pink hues at twilight": "Warm lighting with soft purple and pink color grading",
    "desert sunset with mirage-like glow": "Warm orange lighting with heat distortion effects",
    "sunrise through foggy mountains": "Warm lighting through mist with atmospheric perspective",
    
    # Professional & Studio
    "soft studio lighting": "Multiple diffused sources with even illumination and minimal shadows",
    "harsh, industrial lighting": "Bright fluorescent lighting with hard shadows",
    "fluorescent office lighting": "Cool white overhead lighting with slight green tint",
    "harsh spotlight in a dark room": "Single intense directional light with dramatic shadows",
    
    # Special Effects & Drama
    "light and shadow": "Maximum contrast with sharp shadow boundaries",
    "shadow from window": "Window frame shadow patterns with geometric shapes",
    "apocalyptic, smoky atmosphere": "Orange-red fire tint with smoke effects",
    "evil, gothic, in a cave": "Low brightness with cool lighting and deep shadows",
    "flickering light in a haunted house": "Unstable flickering with cool and warm mixed lighting",
    "golden beams piercing through storm clouds": "Dramatic god rays with high contrast",
    "dim candlelight in a gothic castle": "Warm orange candlelight with stone texture enhancement",
    
    # Festival & Celebration
    "colorful lantern light at festival": "Multiple colored lantern sources with bokeh effects",
    "golden glow at a fairground": "Warm carnival lighting with colorful bulb effects",
    "soft glow through stained glass": "Colored light filtering with rainbow surface patterns",
    "glowing embers from a forge": "Orange-red glowing particles with intense heat effects"
}

DIRECTION_OPTIONS = {
    "auto": "",  
    "left side": "Position the light source from the left side of the frame, creating shadows falling to the right.",
    "right side": "Position the light source from the right side of the frame, creating shadows falling to the left.",
    "top": "Position the light source from directly above, creating downward shadows.",
    "top left": "Position the light source from the top left corner, creating diagonal shadows falling down and to the right.",
    "top right": "Position the light source from the top right corner, creating diagonal shadows falling down and to the left.",
    "bottom": "Position the light source from below, creating upward shadows and dramatic under-lighting.",
    "front": "Position the light source from the front, minimizing shadows and creating even illumination.",
    "back": "Position the light source from behind the subject, creating silhouette effects and rim lighting."
}


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
    default_folder = os.path.join(__file__, "sounds", "*.wav")

    # default_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds", "*.wav")
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
            ("Hailuo/MiniMax/txt2vid", "API MiniMax (txt2vid)", "Purchased API access needed!"),
            ("Hailuo/MiniMax/img2vid", "API MiniMax (img2vid)", "Purchased API access needed!"),
            (
                "Hailuo/MiniMax/subject2vid",
                "API MiniMax (subject2vid)",
                "Purchased API access needed!",
            ),
            ("THUDM/CogVideoX-2b", "CogVideoX-2b (720x480x48)", "THUDM/CogVideoX-2b"),
            ("THUDM/CogVideoX-5b", "CogVideoX-5b (720x480x48)", "THUDM/CogVideoX-5b"),
            (
                "hunyuanvideo-community/HunyuanVideo",
                "Hunyuan Video 960x544x(frames/4+1)",
                "hunyuanvideo-community/HunyuanVideo",
            ),
            (
                "lllyasviel/FramePackI2V_HY",
                "FramePack 960x544x(frames/4+1)",
                "lllyasviel/FramePackI2V_HY",
            ),
            (
                "Lightricks/LTX-Video",
                "LTX 0.9.7 (1280x720x257(frames/8+1))",
                "Lightricks/LTX-Video",
            ),
            (
                "Skywork/SkyReels-V1-Hunyuan-T2V",
                "SkyReels-V1-Hunyuan (960x544x97)",
                "Skywork/SkyReels-V1-Hunyuan-T2V",
            ),
#            ("wangfuyun/AnimateLCM", "AnimateLCM", "wangfuyun/AnimateLCM"),
#            (
#                "stabilityai/stable-video-diffusion-img2vid-xt",
#                "Stable Video Diffusion XT (1024x576x24) ",
#                "stabilityai/stable-video-diffusion-img2vid-xt",
#            ),
#            (
#                "stabilityai/stable-video-diffusion-img2vid",
#                "Stable Video Diffusion (1024x576x14)",
#                "stabilityai/stable-video-diffusion-img2vid",
#            ),
            #            ("genmo/mochi-1-preview", "Mochi-1", "genmo/mochi-1-preview"), #noot good enough yet!
            (
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "Wan2.1-T2V (832x480x81)",
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            ),
            (
                "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "Wan2.1-I2V-14B-480P (832x480x81)",
                "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            ),
            
#            (
#                "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
#                "Wan2.1-VACE 1.3B",
#                "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
#            ),

#            (
#                "cerspense/zeroscope_v2_XL",
#                "Zeroscope XL (1024x576x24)",
#                "Zeroscope XL (1024x576x24)",
#            ),
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                "Frame by Frame SDXL Turbo (1024x1024)",
                "Stable Diffusion XL 1.0",
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
        ],
        default="Lightricks/LTX-Video",
        update=input_strips_updated,
    )
    image_model_card: bpy.props.EnumProperty(
        name="Image Model",
        items=[
            ("ChuckMcSneed/FLUX.1-dev", "Flux 1 Dev", "ChuckMcSneed/FLUX.1-dev"),
            (
                "lzyvegetable/FLUX.1-schnell",
                "Flux Schnell",
                "lzyvegetable/FLUX.1-schnell",
            ),
            ("yuvraj108c/FLUX.1-Kontext-dev", "Flux.1 Kontext Dev", "yuvraj108c/FLUX.1-Kontext-dev"),
            ("kontext-community/relighting-kontext-dev-lora-v3", "Relight Flux.1 Kontext", "kontext-community/relighting-kontext-dev-lora-v3"),
            # Not ready for 4bit and depth has tensor problems
            ("fuliucansheng/FLUX.1-Canny-dev-diffusers-lora", "FLUX Canny", "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora"),
            ("romanfratric234/FLUX.1-Depth-dev-lora", "FLUX Depth", "romanfratric234/FLUX.1-Depth-dev-lora"),
            ("Runware/FLUX.1-Redux-dev", "FLUX Redux", "Runware/FLUX.1-Redux-dev"),

#            ("ostris/Flex.2-preview", "Flex 2 Preview", "ostris/Flex.2-preview"),
            ("Qwen/Qwen-Image", "Qwen-Image", "Qwen/Qwen-Image"),
            (
                "Qwen/Qwen-Image-Edit-2509",
                "Qwen Multi-image Edit",
                "Text and multiple images as input.",
            ),            
            ("lodestones/Chroma", "Chroma", "Chroma is a 8.9B parameter model based on FLUX.1-schnell"),
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                "Stable Diffusion XL 1.0 (1024x1024)",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ),
#            (
#                "ByteDance/SDXL-Lightning",
#                "Stable Diffusion XL Lightning (1024x1024)",
#                "ByteDance/SDXL-Lightning",
#            ),
            (
                "adamo1139/stable-diffusion-3.5-large-ungated",
                "Stable Diffusion 3.5 Large",
                "adamo1139/stable-diffusion-3.5-large-ungated",
            ),
            (
                "adamo1139/stable-diffusion-3.5-medium-ungated",
                "Stable Diffusion 3.5 Medium",
                "adamo1139/stable-diffusion-3.5-medium-ungated",
            ),
            ("Bercraft/Illustrious-XL-v2.0-FP16-Diffusers", "Illustrious XL", "Bercraft/Illustrious-XL-v2.0-FP16-Diffusers"),
            ("John6666/cyberrealistic-xl-v53-sdxl", "Cyberrealistic XL", "John6666/cyberrealistic-xl-v53-sdxl"),
#            (
#                "stabilityai/stable-diffusion-3-medium-diffusers",
#                "Stable Diffusion 3",
#                "stabilityai/stable-diffusion-3-medium-diffusers",
#            ),
#            (
#                "stabilityai/sdxl-turbo",
#                "Stable Diffusion XL Turbo (1024 x 1024)",
#                "stabilityai/sdxl-turbo",
#            ),
            (
                "Alpha-VLLM/Lumina-Image-2.0",
                "Lumina Image 2.0",
                "Alpha-VLLM/Lumina-Image-2.0",
            ),
            (
                "THUDM/CogView4-6B",
                "CogView4-6B (2048x2048)",
                "THUDM/CogView4-6B",
            ),
            (
                "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                "Sana 1600M 1024px",
                "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            ),
#            (
#                "fluently/Fluently-XL-Final",
#                "Fluently (1024x1024)",
#                "fluently/Fluently-XL-Final",
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
            # Must be optimized
            # ("shuttleai/shuttle-jaguar", "Shuttle Jaguar (1024x1024)", "shuttleai/shuttle-jaguar"),
#            (
#                "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
#                "HunyuanDiT-v1.2",
#                "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
#            ),
#            ("Kwai-Kolors/Kolors-diffusers", "Kolors", "Kwai-Kolors/Kolors-diffusers"),
            # ("Corcelio/mobius", "Mobius (1024x1024)", "Corcelio/mobius"),
            (
                "dataautogpt3/OpenDalleV1.1",
                "OpenDalle (1024 x 1024)",
                "dataautogpt3/OpenDalleV1.1",
            ),
            (
                "Vargol/ProteusV0.4",
                "Proteus 0.4 (1024x1024)",
                "Vargol/ProteusV0.4",
            ),
            (
                "SG161222/RealVisXL_V4.0",
                "RealVisXL_V4 (1024x1024)",
                "SG161222/RealVisXL_V4.0",
            ),
#            (
#                "Salesforce/blipdiffusion",
#                "Blip Subject (512x512)",
#                "Salesforce/blipdiffusion",
#            ),
            (
                "diffusers/controlnet-canny-sdxl-1.0-small",
                "Canny ControlNet XL (1024 x 1024)",
                "diffusers/controlnet-canny-sdxl-1.0-small",
            ),
            (
                "xinsir/controlnet-openpose-sdxl-1.0",
                "OpenPose ControlNet XL (1024 x 1024)",
                "xinsir/controlnet-openpose-sdxl-1.0",
            ),
            (
                "xinsir/controlnet-scribble-sdxl-1.0",
                "Scribble ControlNet XL (1024x1024)",
                "xinsir/controlnet-scribble-sdxl-1.0",
            ),
            (
                "Shitao/OmniGen-v1-diffusers",
                "OmniGen",
                "Text and image input.",
            ),
            (
                "ZhengPeng7/BiRefNet_HR",
                "BiRefNet Remove Background",
                "ZhengPeng7/BiRefNet_HR",
            ),
        ],
        default="stabilityai/stable-diffusion-xl-base-1.0",
        update=output_strips_updated,
    )
    if low_vram():
        parler = (
            "parler-tts/parler-tts-mini-v1",
            "Speech: Parler TTS Mini",
            "parler-tts/parler-tts-mini-v1",
        )
    else:
        parler = (
            "parler-tts/parler-tts-large-v1",
            "Speech: Parler TTS Large",
            "parler-tts/parler-tts-large-v1",
        )

    if os_platform != "Linux":
        items = [
            ("Chatterbox", "Speech: Chatterbox", "Zero shot TTS & voice conversion"),
            ("SWivid/F5-TTS", "Speech: F5-TTS", "Zero shot TTS"),
#            ("WhisperSpeech", "Speech: WhisperSpeech", "Zero shot TTS"),
            ("MMAudio", "Audio: Video to Audio", "Add sync audio to video"),
            (
                "stabilityai/stable-audio-open-1.0",
                "Audio: Stable Audio Open",
                "Text to sfx",
            ),
            # Not working!
#            (
#                "cvssp/audioldm2-large",
#                "Audio:Audio LDM 2 Large",
#                "cvssp/audioldm2-large",
#            ),
# Broken:
#            (
#                "facebook/musicgen-stereo-melody-large",
#                "MusicGen Stereo Melody",
#                "Generate music",
#            ),
            parler,
            #("bark", "Speech: Bark", "Bark"),
        ]
    else:
        items = [
            ("SWivid/F5-TTS", "Speech: F5-TTS", "SWivid/F5-TTS"),
            ("Chatterbox", "Chatterbox", "Zero shot txt2speech & voice cloning"),
            ("MMAudio", "Audio: Video to Audio", "Add sync audio to video"),
            (
                "stabilityai/stable-audio-open-1.0",
                "Stable Audio Open",
                "Text to sfx",
            ),
            (
                "facebook/musicgen-stereo-melody-large",
                "MusicGen Stereo Melody",
                "Generate music",
            ),
#            (
#                "cvssp/audioldm2-large",
#                "Audio LDM 2 Large",
#                "cvssp/audioldm2-large",
#            ),
            parler,
        ]

    audio_model_card: bpy.props.EnumProperty(
        name="Audio Model",
        items=items,
        default="stabilityai/stable-audio-open-1.0",
        update=output_strips_updated,
    )
    hugginface_token: bpy.props.StringProperty(
        name="Hugginface Token",
        default="hugginface_token",
        subtype="PASSWORD",
    )
    text_model_card: EnumProperty(
        name="Text Model",
        items=[
            (
                "Salesforce/blip-image-captioning-large",
                "Image Captioning: Blip",
                "Image Captioning",
            ),
            (
                "yownas/Florence-2-large",
                "Image Captioning: Florence-2",
                "Image Captioning",
            ),
            (
                "ZuluVision/MoviiGen1.1_Prompt_Rewriter",
                "Prompt Enhancer: MoviiGen",
                "MoviiGen Prompt Rewriter",
            ),
        ],
        default="yownas/Florence-2-large",
        update=output_strips_updated,
    )
    generator_ai: StringProperty(
        name="Filepath",
        description="Path to the folder where the generated files are stored",
        subtype="DIR_PATH",
        default=join(bpy.utils.user_resource("DATAFILES"), "Pallaidium_Media"),
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
        try:
            box.prop(self, "movie_model_card")
            box.prop(self, "image_model_card")
        except:
            pass
        if (
            self.image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers"
            or self.image_model_card == "adamo1139/stable-diffusion-3.5-large-ungated"
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

    print("Strip type: " + str(strip.type))

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
        for s in sequencer.sequences_all:
            s.select = False
        strip.select = True
        bpy.context.scene.frame_current = int(strip.frame_start)

        if strip.type != "SCENE":
            bpy.ops.sequencer.copy()

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
        new_scene.render.resolution_percentage = (
            current_scene.render.resolution_percentage
        )
        new_scene.render.pixel_aspect_x = current_scene.render.pixel_aspect_x
        new_scene.render.pixel_aspect_y = current_scene.render.pixel_aspect_y
        new_scene.render.fps = current_scene.render.fps
        new_scene.render.fps_base = current_scene.render.fps_base
        new_scene.render.sequencer_gl_preview = (
            current_scene.render.sequencer_gl_preview
        )
        new_scene.render.use_sequencer_override_scene_strip = (
            current_scene.render.use_sequencer_override_scene_strip
        )
        new_scene.world = current_scene.world

        area = [
            area for area in context.screen.areas if area.type == "SEQUENCE_EDITOR"
        ][0]
        with bpy.context.temp_override(area=area):
            if strip.type == "SCENE":
                bpy.ops.sequencer.scene_strip_add(
                    frame_start=0, channel=8, replace_sel=True
                )
            else:
                # Paste the strip from the clipboard to the new scene
                bpy.ops.sequencer.paste()
                # bpy.ops.sequencer.meta_separate()

        # Get the new strip in the new scene
        new_strip = new_scene.sequence_editor.active_strip = (
            bpy.context.selected_sequences[0]
        )

        copy_struct(strip, new_strip)

        # Set the range in the new scene to fit the pasted strip
        new_scene.frame_start = int(new_strip.frame_final_start)
        new_scene.frame_end = (
            int(new_strip.frame_final_start + new_strip.frame_final_duration) - 1
        )

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
        area = [
            area for area in context.screen.areas if area.type == "SEQUENCE_EDITOR"
        ][0]
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
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        split = row.split(factor=0.7)
        split.label(text=item.name)
        split.prop(item, "weight_value", text="", emboss=False)


def update_folder_callback(self, context):
    bpy.ops.lora.refresh_files()


class LORA_OT_RefreshFiles(Operator):
    bl_idname = "lora.refresh_files"
    bl_label = "Refresh Files"

    def execute(self, context):
        scene = context.scene
        directory = bpy.path.abspath(scene.lora_folder)
        lora_files = scene.lora_files
        lora_files.clear()
        if not directory:
            self.report({"ERROR"}, "No folder selected")
            return {"CANCELLED"}
        #        lora_files = scene.lora_files
        #        lora_files.clear()
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
    bl_label = "Pallaidium"
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
        text_model_card = addon_prefs.text_model_card
        scene = context.scene
        type = scene.generatorai_typeselect
        input = scene.input_strips
        layout = self.layout
        col = layout.column(align=False)
        col.use_property_split = True
        col.use_property_decorate = False
        col = col.box()
        col = col.column()

        if scene.sequence_editor is None:
            scene.sequence_editor_create()
            
        # OmniGen
        if image_model_card == "Shitao/OmniGen-v1-diffusers" and type == "image":
            col.prop(context.scene, "omnigen_prompt_1", text="", icon="ADD")
            row = col.row(align=True)
            row.prop_search(
                scene,
                "omnigen_strip_1",
                scene.sequence_editor,
                "sequences",
                text="",
                icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "omni_select1"

            col.prop(context.scene, "omnigen_prompt_2", text="", icon="ADD")
            row = col.row(align=True)
            row.prop_search(
                scene,
                "omnigen_strip_2",
                scene.sequence_editor,
                "sequences",
                text="",
                icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "omni_select2"

            col.prop(context.scene, "omnigen_prompt_3", text="", icon="ADD")
            row = col.row(align=True)
            row.prop_search(
                scene,
                "omnigen_strip_3",
                scene.sequence_editor,
                "sequences",
                text="",
                icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "omni_select3"

        elif image_model_card == "Salesforce/blipdiffusion" and type == "image":
            col.prop(context.scene, "input_strips", text="Source Image")
            col.prop(context.scene, "blip_cond_subject", text="Source Subject")

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
            try:
                col.prop(context.scene, "input_strips", text="Input")
            except:
                pass

        # Qwen multi-image
        if image_model_card == "Qwen/Qwen-Image-Edit-2509" and type == "image":
            row = col.row(align=True)
            row.prop_search(
                scene,
                "qwen_strip_1",
                scene.sequence_editor,
                "sequences",
                text="",
                icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "qwen_select1"

            row = col.row(align=True)
            row.prop_search(
                scene,
                "qwen_strip_2",
                scene.sequence_editor,
                "sequences",
                text="",
                icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "qwen_select2"

            row = col.row(align=True)
            row.prop_search(
                scene,
                "qwen_strip_3",
                scene.sequence_editor,
                "sequences",
                text="",
                icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "qwen_select3"

        if image_model_card == "kontext-community/relighting-kontext-dev-lora-v3" and type == "image":
            box = layout.box()
            box = box.column(align=True)
            box.use_property_split = True
            box.use_property_decorate = False
            box.prop(context.scene, "illumination_style", text="Relight Style")
            box.prop(context.scene, "light_direction", text="Direction")


        if type != "text":
            if type != "audio":
                if type == "movie" and "Hailuo/MiniMax/" in movie_model_card:
                    if movie_model_card == "Hailuo/MiniMax/subject2vid":
                        row = col.row(align=True)
                        row.prop_search(
                            scene,
                            "minimax_subject",
                            scene.sequence_editor,
                            "sequences",
                            text="Subject",
                            icon="USER",
                        )
                        row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "minimax_select"

                elif type == "movie" and movie_model_card == "lllyasviel/FramePackI2V_HY":
                    if input == "input_strips":
                        row = col.row(align=True)
                        row.prop_search(
                            scene,
                            "out_frame",
                            scene.sequence_editor,
                            "sequences",
                            text="End Frame",
                            icon="RENDER_RESULT",
                        )
                        row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "out_frame_select"

                elif (type == "movie") or (
                    type == "image"
                    and image_model_card != "xinsir/controlnet-openpose-sdxl-1.0"
                    and image_model_card != "xinsir/controlnet-scribble-sdxl-1.0"
                    and image_model_card != "Salesforce/blipdiffusion"
                    and image_model_card != "ZhengPeng7/BiRefNet_HR"
                    and image_model_card != "Shitao/OmniGen-v1-diffusers"
                    and image_model_card != "Qwen/Qwen-Image-Edit-2509"
                    and image_model_card != "Runware/FLUX.1-Redux-dev"
                    and image_model_card != "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora"
                    and image_model_card != "romanfratric234/FLUX.1-Depth-dev-lora"
                    #and image_model_card != "yuvraj108c/FLUX.1-Kontext-dev"
                    and image_model_card != "kontext-community/relighting-kontext-dev-lora-v3"
                ):
                    if input == "input_strips" and (not scene.inpaint_selected_strip or image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"):
                        col = col.column(heading="Use", align=True)
                        col.prop(addon_prefs, "use_strip_data", text=" Name & Seed")
                        if type == "movie" and os_platform != "Darwin" and (
                            movie_model_card == "lzyvegetable/FLUX.1-schnell"
                            or movie_model_card == "ChuckMcSneed/FLUX.1-dev"
                            #or movie_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                            #or movie_model_card == "ostris/Flex.2-preview"
                        ):
                            pass
                        else:
                            col.prop(context.scene, "image_power", text="Strip Power")
                        if (
                            type == "movie"
                            and movie_model_card
                            == "stabilityai/stable-video-diffusion-img2vid"
                        ) or (
                            type == "movie"
                            and movie_model_card
                            == "stabilityai/stable-video-diffusion-img2vid-xt"
                        ):
                            col.prop(
                                context.scene, "svd_motion_bucket_id", text="Motion"
                            )
                            col.prop(
                                context.scene,
                                "svd_decode_chunk_size",
                                text="Decode Frames",
                            )

                    if (
                        bpy.context.scene.sequence_editor is not None
                        and image_model_card
                        != "diffusers/controlnet-canny-sdxl-1.0-small"
                        and image_model_card != "ByteDance/SDXL-Lightning"
                    ):
                        if input == "input_strips" and type == "image":
                            row = col.row(align=True)
                            row.prop_search(
                                scene,
                                "inpaint_selected_strip",
                                scene.sequence_editor,
                                "sequences",
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
                    "sequences",
                    text="Reference Image",
                    icon="FILE_IMAGE",
                )
                row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "kontext_select1"

            if (
                image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
                and type == "image"
            ):
                col = col.column(heading="Read as", align=True)
                col.prop(context.scene, "openpose_use_bones", text="OpenPose Rig Image")
            if (
                image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
                and type == "image"
            ):
                col = col.column(heading="Read as", align=True)
                col.prop(context.scene, "use_scribble_image", text="Scribble Image")

            # IPAdapter.
            if (
                image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                or image_model_card == "stabilityai/sdxl-turbo"
                # or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
                # or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
                # or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            ) and type == "image":
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
            if not (type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR"):
                col = layout.column(align=True)
                col = col.box()
                col = col.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
            if (
                (
                    type == "movie"
                    and movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                )
                or (type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR")
                or (
                    type == "movie"
                    and movie_model_card
                    == "stabilityai/stable-video-diffusion-img2vid-xt"
                )
                or (
                    image_model_card == "Shitao/OmniGen-v1-diffusers"
                    and type == "image"
                )
                or (type == "image" and image_model_card == "Runware/FLUX.1-Redux-dev")
            ):
                pass
            else:
                col.use_property_split = False
                col.use_property_decorate = False
                col.prop(context.scene, "generate_movie_prompt", text="", icon="ADD")
                if (
                    (type == "audio" and audio_model_card == "bark")
                    or (
                        type == "audio"
                        and audio_model_card == "stabilityai/stable-audio-open-1.0"
                    )
                    or (
                        type == "image"
                        and image_model_card == "lzyvegetable/FLUX.1-schnell"
                    )
                    or (
                        type == "image"
                        and image_model_card == "ChuckMcSneed/FLUX.1-dev"
                    )
                    or (
                        type == "image"
                        and image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                    )
                    or (type == "image" and image_model_card == "kontext-community/relighting-kontext-dev-lora-v3")
                    or (type == "image" and image_model_card == "ostris/Flex.2-preview")
                    or (
                        type == "image"
                        and image_model_card
                        == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora"
                    )
                    or (
                        type == "image"
                        and image_model_card
                        == "romanfratric234/FLUX.1-Depth-dev-lora"
                    )
                    or (
                        type == "image"
                        and image_model_card == "Runware/FLUX.1-Redux-dev"
                    )
                    or (
                        type == "audio"
                        and (audio_model_card == "facebook/musicgen-stereo-melody-large"
                        or audio_model_card == "WhisperSpeech" or audio_model_card == "SWivid/F5-TTS" or audio_model_card == "Chatterbox")
                    )
                    or (type == "movie" and "Hailuo/MiniMax/" in movie_model_card)
                ):
                    pass
                elif type == "audio" and (
                    audio_model_card == "parler-tts/parler-tts-large-v1"
                    or audio_model_card == "parler-tts/parler-tts-mini-v1"
                ):
                    layout = col.column()
                    col = layout.column(align=True)
                    col.use_property_split = True
                    col.use_property_decorate = False
                    col.prop(
                        context.scene,
                        "parler_direction_prompt",
                        text="Instruction",
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
                if type != "audio" and not (
                    type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR"
                ):
                    col.prop(context.scene, "generatorai_styles", text="Style")
            if type == "movie" and "Hailuo/MiniMax/" in movie_model_card:
                pass
            else:
                layout = col.column()
                if (
                    type == "movie"
                    or type == "image"
                    and not (
                        type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR"
                    )
                ):
                    col = layout.column(align=True)
                    col.prop(context.scene, "generate_movie_x", text="X")
                    col.prop(context.scene, "generate_movie_y", text="Y")
                col = layout.column(align=True)
                if (
                    type == "movie"
                    or type == "image"
                    and not (
                        type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR"
                    )
                ):
                    col.prop(context.scene, "generate_movie_frames", text="Frames")
                if (
                    type == "audio"
                    and audio_model_card != "bark"
                    and audio_model_card != "WhisperSpeech"
                    and audio_model_card != "SWivid/F5-TTS"
                    and audio_model_card != "Chatterbox"
                    and audio_model_card != "parler-tts/parler-tts-large-v1"
                    and audio_model_card != "parler-tts/parler-tts-mini-v1"
                ):
                    col.prop(context.scene, "audio_length_in_f", text="Frames")
                if type == "audio" and (audio_model_card == "WhisperSpeech" or audio_model_card == "SWivid/F5-TTS" or audio_model_card == "Chatterbox"):
                    row = col.row(align=True)
                    row.prop(context.scene, "audio_path", text="Speaker")
                    row.operator(
                        "sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER"
                    )
                    if audio_model_card == "Chatterbox":
                        col.prop(context.scene, "chat_exaggeration")
                        col.prop(context.scene, "chat_pace")
                        col.prop(context.scene, "chat_temperature")
                    else:
                        if audio_model_card == "WhisperSpeech":
                            col.prop(context.scene, "audio_speed", text="Speed")
                        else:
                            col.prop(context.scene, "audio_speed_tts", text="Speed")

                if type == "audio" and audio_model_card == "bark":
                    col = layout.column(align=True)
                    col.prop(context.scene, "speakers", text="Speaker")
                    col.prop(context.scene, "languages", text="Language")
                elif type == "audio" and (audio_model_card == "WhisperSpeech"  or audio_model_card == "Chatterbox"):
                    pass

                elif type == "audio" and (
                    addon_prefs.audio_model_card
                    == "facebook/musicgen-stereo-melody-large"
                    or addon_prefs.audio_model_card
                    == "stabilityai/stable-audio-open-1.0"
                ):
                    col.prop(
                        context.scene, "movie_num_inference_steps", text="Quality Steps"
                    )
                else:
                    if (
                        type == "image"
                        and (image_model_card == "ByteDance/SDXL-Lightning")
                        or (
                            type == "audio"
                            and (
                                audio_model_card == "parler-tts/parler-tts-mini-v1"
                                or audio_model_card == "parler-tts/parler-tts-large-v1"
                            )
                            or (
                                type == "image"
                                and image_model_card == "ZhengPeng7/BiRefNet_HR"
                            )
                        )
                    ):
                        pass
                    else:
                        col.prop(
                            context.scene,
                            "movie_num_inference_steps",
                            text="Quality Steps",
                        )

                    if (
                        (
                            type == "movie"
                            and movie_model_card
                            == "stabilityai/stable-video-diffusion-img2vid"
                        )
                        or (
                            type == "movie"
                            and movie_model_card
                            == "stabilityai/stable-video-diffusion-img2vid-xt"
                        )
                        or (
                            type == "image"
                            and image_model_card == "lzyvegetable/FLUX.1-schnell"
                        )
                        or (
                            type == "image"
                            and image_model_card == "ZhengPeng7/BiRefNet_HR"
                        )
                        or (
                            type == "audio"
                            and (
                                audio_model_card == "parler-tts/parler-tts-mini-v1"
                                or audio_model_card == "parler-tts/parler-tts-large-v1"
                                or audio_model_card == "SWivid/F5-TTS"
                                or audio_model_card == "Chatterbox"
                            )
                        )
                        or (
                            scene.use_lcm
                            and not (
                                type == "image"
                                and image_model_card
                                == image_model_card
                                == "ByteDance/SDXL-Lightning"
                            )
                        )
                    ):
                        pass
                    else:
                        if (
                            image_model_card == "Shitao/OmniGen-v1-diffusers"
                            and type == "image"
                        ) or (
                            image_model_card == "Qwen/Qwen-Image-Edit-2509"
                            and type == "image"
                        ):
                            col.prop(
                                context.scene, "img_guidance_scale", text="Image Power"
                            )
                        col.prop(context.scene, "movie_num_guidance", text="Word Power")

                if not (
                    type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR"
                ):
                    col = col.column(align=True)
                    row = col.row(align=True)
                    sub_row = row.row(align=True)
                    row.prop(
                        context.scene, "movie_use_random", text="", icon="QUESTION"
                    )
                    sub_row.prop(context.scene, "movie_num_seed", text="Seed")
                    sub_row.active = not context.scene.movie_use_random   
                    
#                if type == "movie" and (
#                    movie_model_card == "cerspense/zeroscope_v2_dark_30x448x256"
#                    or movie_model_card == "cerspense/zeroscope_v2_576w"
#                    or movie_model_card == "cerspense/zeroscope_v2_XL"
#                ):
#                    col = col.column(heading="Upscale", align=True)
#                    col.prop(context.scene, "video_to_video", text="2x")
                if type == "image" and not (
                    type == "image" and image_model_card == "ZhengPeng7/BiRefNet_HR"
                ):
                    col = col.column(heading="Enhance", align=True)
                    row = col.row()
                    row.prop(context.scene, "refine_sd", text="Quality")
                    sub_col = col.row()
                    sub_col.active = context.scene.refine_sd

                    if (
                        (
                            type == "image"
                            and image_model_card
                            == "stabilityai/stable-diffusion-xl-base-1.0"
                        )
                        or (
                            type == "image"
                            and image_model_card
                            == "xinsir/controlnet-openpose-sdxl-1.0"
                        )
                        or (
                            type == "image"
                            and image_model_card
                            == "xinsir/controlnet-scribble-sdxl-1.0"
                        )
                        or (
                            type == "image"
                            and image_model_card
                            == "diffusers/controlnet-canny-sdxl-1.0-small"
                        )
                        or (
                            type == "image"
                            and image_model_card == "segmind/Segmind-Vega"
                        )
                        or (
                            type == "image"
                            and image_model_card == "Vargol/PixArt-Sigma_16bit"
                        )
                        or (
                            type == "image"
                            and image_model_card == "Vargol/PixArt-Sigma_2k_16bit"
                        )
                    ):
                        row.prop(context.scene, "use_lcm", text="Speed")

                    # ADetailer
                    if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                        col = col.column(heading="Details", align=True)

                    row = col.row()
                    if image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                        row.prop(context.scene, "adetailer", text="Faces")

                    # AuraSR
                    # col = col.column(heading="Upscale", align=True)
                    row.prop(context.scene, "aurasr", text="Upscale 4x")
                    # row = col.row()
                if (type == "movie") and (
                    movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                ):
                    col = layout.column(heading="Upscale", align=True)
                    col.prop(context.scene, "aurasr", text="4x")
                if type == "audio" and audio_model_card == "SWivid/F5-TTS":
                    col = layout.column(heading="Remove", align=True)
                    col.prop(context.scene, "remove_silence", text="Silence")

            # LoRA.
            if (
                (
                    image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                    or image_model_card == "stabilityai/sdxl-turbo"
                    or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
                    or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
                    or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
                    or image_model_card == "lzyvegetable/FLUX.1-schnell"
                    or image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                    or image_model_card == "ostris/Flex.2-preview"
                    or image_model_card == "lodestones/Chroma"
                    or image_model_card == "Qwen/Qwen-Image"
                    or image_model_card == "Qwen/Qwen-Image-Edit-2509"
                    or image_model_card == "ChuckMcSneed/FLUX.1-dev"
                    or image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora"
                    or image_model_card == "romanfratric234/FLUX.1-Depth-dev-lora"
                    or image_model_card == "Runware/FLUX.1-Redux-dev"
                    or image_model_card == "Bercraft/Illustrious-XL-v2.0-FP16-Diffusers"
                    or image_model_card == "John6666/cyberrealistic-xl-v53-sdxl"
                )
                and type == "image"
            ) or ((
                type == "movie")
                and (movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                or (movie_model_card == "hunyuanvideo-community/HunyuanVideo")
                or (movie_model_card == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
                or (movie_model_card == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
                or (movie_model_card == "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
            )):
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

        if type == "image":
            col.prop(addon_prefs, "image_model_card", text=" ")
            if (
                addon_prefs.image_model_card
                == "stabilityai/stable-diffusion-3-medium-diffusers"
                or addon_prefs.image_model_card
                == "adamo1139/stable-diffusion-3.5-large-ungated"
            ):
                row = col.row(align=True)
                row.prop(addon_prefs, "hugginface_token")
                row.operator(
                    "wm.url_open", text="", icon="URL"
                ).url = "https://huggingface.co/settings/tokens"

        if type == "movie":
            col.prop(addon_prefs, "movie_model_card", text=" ")
        if type == "audio":
            col.prop(addon_prefs, "audio_model_card", text=" ")
        if type == "text":
            col.prop(addon_prefs, "text_model_card", text=" ")
        if type != "text" and not (
            type == "movie" and "Hailuo/MiniMax/" in movie_model_card
        ):
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
                # Frame by Frame
                if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    row.operator(
                        "sequencer.text_to_generator", text="Generate from Strips"
                    )
                else:
                    row.operator("sequencer.generate_movie", text="Generate")
            if type == "image":
                row.operator("sequencer.generate_image", text="Generate")
            if type == "audio":
                row.operator("sequencer.generate_audio", text="Generate")
            if type == "text":
                row.operator("sequencer.generate_text", text="Generate")


class NoWatermark:
    def apply_watermark(self, img):
        return img


# MiniMax
def invoke_video_generation(prompt, api_key, image_url, movie_model_card):
    import requests
    import json
    import base64

    debug_print("-----------------Submit video generation task-----------------")
    url = "https://api.minimaxi.chat/v1/video_generation"
    # debug_print("Movie model card:", movie_model_card)
    debug_print("Prompt:", prompt)
    debug_print("Image URL:", image_url)

    if movie_model_card == "Hailuo/MiniMax/img2vid":
        with open(image_url, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")

        payload = json.dumps(
            {
                "model": "I2V-01-Director",
                #"model": "I2V-01",
                "prompt": prompt,
                # "prompt_optimizer": False,
                "first_frame_image": f"data:image/jpeg;base64,{data}",
            }
        )

    elif movie_model_card == "Hailuo/MiniMax/subject2vid":
        with open(image_url, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
        # debug_print("Base64 encoded image data:", data)

        payload = json.dumps(
            {
                "model": "S2V-01",
                "prompt": prompt,
                # "prompt_optimizer": False,
                "subject_reference": [
                    {"type": "character", "image": [f"data:image/jpeg;base64,{data}"]}
                ],
            }
        )
    else:
        payload = json.dumps(
            {
                "model": "T2V-01-Director",
                "prompt": prompt,
                # "prompt_optimizer": False,
            }
        )

    # debug_print("Payload:", payload)

    headers = {"authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # debug_print("Headers:", headers)

    response = requests.request("POST", url, headers=headers, data=payload)
    debug_print("Response text:", response.text)

    task_id = response.json()["task_id"]
    debug_print("Task ID:", task_id)
    print("Video generation task submitted successfully, task ID：" + task_id)
    return task_id


# MiniMax
def query_video_generation(task_id, api_key):
    debug_print("Task ID:", task_id)

    import requests

    url = f"https://api.minimaxi.chat/v1/query/video_generation?task_id={task_id}"
    debug_print("Query URL:", url)

    headers = {"authorization": f"Bearer {api_key}"}
    # debug_print("Headers:", headers)

    response = requests.request("GET", url, headers=headers)
    debug_print("Response text:", response.text)

    status = response.json()["status"]
    debug_print("Task Status:", status)

    if status == 'Preparing':
        print("...Preparing...")
        return "", 'Preparing'
    elif status == 'Queueing':
        print("...In the queue...")
        return "", 'Queueing'
    elif status == 'Processing':
        print("...Generating...")
        return "", 'Processing'
    elif status == 'Success':
        return response.json()['file_id'], "Finished"
    elif status == 'Fail':
        return "", "Fail"
    else:
        return "", "Unknown"


# MiniMax
def fetch_video_result(file_id, api_key, output_file_name):
    debug_print("File ID:", file_id)
    debug_print("Out file name:", output_file_name)
    import requests

    debug_print(
        "---------------Video generated successfully, downloading now---------------"
    )
    url = f"https://api.minimaxi.chat/v1/files/retrieve?file_id={file_id}"
    debug_print("Retrieve URL:", url)

    headers = {
        "authorization": f"Bearer {api_key}",
    }
    # debug_print("Headers:", headers)

    response = requests.request("GET", url, headers=headers)
    debug_print("Response text:", response.text)

    download_url = response.json()["file"]["download_url"]
    debug_print("Download URL:", download_url)

    print("Video download link：" + download_url)
    with open(output_file_name, "wb") as f:
        video_content = requests.get(download_url).content
        f.write(video_content)
    debug_print("Video content written to:", output_file_name)
    print("The video has been downloaded in：" + output_file_name)  # os.getcwd()+'/'+
    return output_file_name


def minimax_validate_image(file_path):
    """
    Validate an image based on the following criteria:
    - Format: JPG, JPEG, PNG
    - Aspect ratio: Greater than 2:5 and less than 5:2
    - Shorter side > 300 pixels
    - File size <= 20MB

    Args:
        file_path (str): Path to the local image file

    Returns:
        bool: True if the image is valid, False otherwise
    """
    MAX_FILE_SIZE_MB = 20
    MIN_SHORT_SIDE = 300
    MIN_ASPECT_RATIO = 2 / 5
    MAX_ASPECT_RATIO = 5 / 2
    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG"}
    from PIL import Image

    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            print("MiniMax Image Input Failure Reason: File size exceeds 20MB.")
            return False

        # Load image using PIL
        image = Image.open(file_path)

        # Check format
        if image.format not in SUPPORTED_FORMATS:
            print(f"Failure Reason: Unsupported image format: {image.format}.")
            return False

        # Check dimensions
        width, height = image.size
        shorter_side = min(width, height)
        aspect_ratio = width / height

        if shorter_side <= MIN_SHORT_SIDE:
            print("Failure Reason: Shorter side must exceed 300 pixels.")
            return False

        if not (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
            print("Failure Reason: Aspect ratio must be between 2:5 and 5:2.")
            return False

        # Passed all checks
        return True

    except Exception as e:
        print(f"Failure Reason: {str(e)}")
        return False


def read_file(path):
    try:
        with open(path, "r") as file:
            return file.read()
    except Exception as e:
        return str(e)


def progress_bar(duration):
    total_steps = 60
    for i in range(total_steps + 1):
        completed = int((i / total_steps) * 100)
        bar = f"[{'█' * i}{'.' * (total_steps - i)}] {completed}%"
        sys.stdout.write(f"\r{bar}")
        sys.stdout.flush()
        time.sleep(duration / total_steps)


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
        except ModuleNotFoundError as e:
            print("Dependencies needs to be installed in the add-on preferences. "+str(e.name))
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

        clear_cuda_cache()

        current_frame = scene.frame_current
        prompt = style_prompt(scene.generate_movie_prompt)[0]
        negative_prompt = (
            scene.generate_movie_negative_prompt
            + ", "
            + style_prompt(scene.generate_movie_prompt)[1]
            + ", nsfw, nude, nudity"
        )
        movie_x = scene.generate_movie_x
        movie_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_32(movie_x)
        y = scene.generate_movie_y = closest_divisible_32(movie_y)
        old_duration = duration = scene.generate_movie_frames
        movie_num_inference_steps = scene.movie_num_inference_steps
        movie_num_guidance = scene.movie_num_guidance
        input = scene.input_strips
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        local_files_only = addon_prefs.local_files_only
        movie_model_card = addon_prefs.movie_model_card
        image_model_card = addon_prefs.image_model_card
        pipe = None

        clear_cuda_cache()

        def ensure_skyreel(prompt: str) -> str:
            if not prompt.startswith("FPS-24,"):
                return "FPS-24, " + prompt
            return prompt

        # LOADING MODELS
        print("Model:  " + movie_model_card)

        # Models for refine imported image or movie
        if (
            (scene.movie_path or scene.image_path)
            and input == "input_strips"
            and movie_model_card != "wangfuyun/AnimateLCM"
            and movie_model_card != "THUDM/CogVideoX-5b"
            and movie_model_card != "THUDM/CogVideoX-2b"
            and movie_model_card != "Lightricks/LTX-Video"
            and movie_model_card != "hunyuanvideo-community/HunyuanVideo"
            and movie_model_card != "lllyasviel/FramePackI2V_HY"
            and movie_model_card != "genmo/mochi-1-preview"
            and movie_model_card != "Hailuo/MiniMax/txt2vid"
            and movie_model_card != "Hailuo/MiniMax/img2vid"
            and movie_model_card != "Hailuo/MiniMax/subject2vid"
            and movie_model_card != "Skywork/SkyReels-V1-Hunyuan-T2V"
            and movie_model_card != "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
            and movie_model_card != "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        ) or movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
            # Frame by Frame
            if (
                movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
            ):  # frame2frame
                from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
                from torchvision import transforms

                enabled_items = None

                lora_files = scene.lora_files
                enabled_names = []
                enabled_weights = []
                # Check if there are any enabled items before loading
                enabled_items = [item for item in lora_files if item.enabled]
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                )
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    movie_model_card,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    vae=vae,
                )
                #                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16")
                #                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
                pipe.watermark = NoWatermark()

                scene = context.scene

                if enabled_items:
                    for item in enabled_items:
                        enabled_names.append(
                            (clean_filename(item.name)).replace(".", "")
                        )
                        enabled_weights.append(item.weight_value)
                        pipe.load_lora_weights(
                            bpy.path.abspath(scene.lora_folder),
                            weight_name=item.name + ".safetensors",
                            adapter_name=((clean_filename(item.name)).replace(".", "")),
                        )
                    pipe.set_adapters(enabled_names, adapter_weights=enabled_weights)
                    print("Load LoRAs: " + " ".join(enabled_names))

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_sequential_cpu_offload()
                    # pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()

                refiner = pipe

            if (
                movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                or movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
            ):
                from diffusers import StableVideoDiffusionPipeline
                from diffusers.utils import load_image, export_to_video

                if movie_model_card == "stabilityai/stable-video-diffusion-img2vid":
                    refiner = StableVideoDiffusionPipeline.from_pretrained(
                        movie_model_card,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )
                if movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt":
                    refiner = StableVideoDiffusionPipeline.from_pretrained(
                        "vdo/stable-video-diffusion-img2vid-xt-1-1",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )

                if gfx_device == "mps":
                    refiner.to("mps")
                elif low_vram():
                    refiner.enable_model_cpu_offload()
                    refiner.unet.enable_forward_chunking()
                else:
                    refiner.to(gfx_device)

#            else:  # vid2vid / img2vid
##                if (
###                    movie_model_card == "cerspense/zeroscope_v2_dark_30x448x256"
###                    or movie_model_card == "cerspense/zeroscope_v2_576w"
##                    scene.image_path
##                ):
##                    card = "cerspense/zeroscope_v2_XL"
##                else:
#                card = movie_model_card

#                from diffusers import VideoToVideoSDPipeline

#                upscale = VideoToVideoSDPipeline.from_pretrained(
#                    card,
#                    torch_dtype=torch.float16,
#                    local_files_only=local_files_only,
#                )

#                from diffusers import DPMSolverMultistepScheduler

#                upscale.scheduler = DPMSolverMultistepScheduler.from_config(
#                    upscale.scheduler.config
#                )
#                if gfx_device == "mps":
#                    upscale.to("mps")
#                elif low_vram():
#                    upscale.enable_model_cpu_offload()
#                else:
#                    upscale.to(gfx_device)

        # Models for movie generation
        elif (
            movie_model_card != "Hailuo/MiniMax/txt2vid"
            and movie_model_card != "Hailuo/MiniMax/img2vid"
            and movie_model_card != "Hailuo/MiniMax/subject2vid"
        ):
            if movie_model_card == "wangfuyun/AnimateLCM":
                import torch
                from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
                from diffusers.utils import export_to_gif

                adapter = MotionAdapter.from_pretrained(
                    "wangfuyun/AnimateLCM", torch_dtype=torch.float16
                )

                pipe = AnimateDiffPipeline.from_pretrained(
                    "emilianJR/epiCRealism",
                    motion_adapter=adapter,
                    torch_dtype=torch.float16,
                )
                pipe.scheduler = LCMScheduler.from_config(
                    pipe.scheduler.config, beta_schedule="linear"
                )

                pipe.load_lora_weights(
                    "wangfuyun/AnimateLCM",
                    weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                    adapter_name="lcm-lora",
                )
                pipe.set_adapters(["lcm-lora"], [0.8])

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_vae_slicing()
                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)

            # CogVideoX
            elif (
                movie_model_card == "THUDM/CogVideoX-5b"
                or movie_model_card == "THUDM/CogVideoX-2b"
            ):
                # vid2vid
                if scene.movie_path and input == "input_strips":
                    from diffusers.utils import load_video
                    from diffusers import (
                        CogVideoXDPMScheduler,
                        CogVideoXVideoToVideoPipeline,
                    )

                    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
                        movie_model_card, torch_dtype=torch.bfloat16
                    )
                    pipe.scheduler = CogVideoXDPMScheduler.from_config(
                        pipe.scheduler.config
                    )

                # img2vid
                elif scene.image_path and input == "input_strips":
                    print("Load: Image to video (CogVideoX)")
                    from diffusers import CogVideoXImageToVideoPipeline
                    from diffusers.utils import load_image

                    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                        "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
                    )

                # txt2vid
                else:
                    print("Load: text to video (CogVideoX)")
                    from diffusers import CogVideoXPipeline

                    pipe = CogVideoXPipeline.from_pretrained(
                        movie_model_card,
                        torch_dtype=torch.float16,
                    )

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_sequential_cpu_offload()
                    # pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()

                scene.generate_movie_x = 720
                scene.generate_movie_y = 480

            # LTX
            elif movie_model_card == "Lightricks/LTX-Video":
                from transformers import T5EncoderModel, T5Tokenizer
                from diffusers import AutoencoderKLLTXVideo
                from diffusers import LTXPipeline, LTXVideoTransformer3DModel#, GGUFQuantizationConfig
                from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline, BitsAndBytesConfig, LTXVideoTransformer3DModel
                print("LTX Video: Load Model")

                import torch
                from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline, BitsAndBytesConfig, LTXVideoTransformer3DModel
                from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
                from diffusers.utils import export_to_video, load_video, load_image

                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

                transformer = LTXVideoTransformer3DModel.from_pretrained(
                    "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                    subfolder="transformer",
                )

                pipe = LTXConditionPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-diffusers", transformer=transformer, torch_dtype=torch.bfloat16)

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.vae.enable_tiling()
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.vae.enable_tiling()
                    pipe.enable_model_cpu_offload()

            # HunyuanVideo
            elif movie_model_card == "hunyuanvideo-community/HunyuanVideo":
                # vid2vid
                if scene.movie_path and input == "input_strips":
                    print("HunyuanVideo doesn't support vid2vid! Using img2vid instead...")

                # img2vid
                if (scene.image_path or scene.movie_path) and input == "input_strips":
                    print("HunyuanVideo: Load Image to Video Model")
                    from diffusers import HunyuanVideoImageToVideoPipeline
                    model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
                    if low_vram():
                        transformer_path = f"https://huggingface.co/city96/HunyuanVideo-I2V-gguf/blob/main/hunyuan-video-i2v-720p-Q4_K_S.gguf"
                    else:
                        transformer_path = f"https://huggingface.co/city96/HunyuanVideo-I2V-gguf/blob/main/hunyuan-video-i2v-720p-Q4_K_S.gguf"
                        #transformer_path = f"https://huggingface.co/city96/HunyuanVideo-I2V-gguf/blob/main/hunyuan-video-i2v-720p-Q5_K_S.gguf"
                # prompt to video
                else:
                    print("HunyuanVideo: Load Prompt to Video Model")
                    model_id = "hunyuanvideo-community/HunyuanVideo"
                    from diffusers import HunyuanVideoPipeline
                    if low_vram():
                        transformer_path = f"https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q3_K_S.gguf"
                    else:
                        transformer_path = f"https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q4_K_S.gguf"

                enabled_items = None
                lora_files = scene.lora_files
                enabled_names = []
                enabled_weights = []
                # Check if there are any enabled items before loading
                enabled_items = [item for item in lora_files if item.enabled]

                from diffusers.models import HunyuanVideoTransformer3DModel
                from diffusers.utils import export_to_video
                from diffusers import BitsAndBytesConfig
                from transformers import LlamaModel, CLIPTextModel
                from diffusers import GGUFQuantizationConfig

                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                transformer = HunyuanVideoTransformer3DModel.from_single_file(
                    transformer_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                )

                if (scene.image_path or scene.movie_path) and input == "input_strips":
                    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
                        model_id,
                        #text_encoder=text_encoder,
                        #text_encoder_2=text_encoder_2,
                        transformer=transformer,
                        torch_dtype=torch.float16,
                    )
                else:
                    text_encoder = LlamaModel.from_pretrained(
                        model_id,
                        #"hunyuanvideo-community/HunyuanVideo",
                        subfolder="text_encoder",
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16
                    )
                    text_encoder_2 = CLIPTextModel.from_pretrained(
                        model_id,
                        #"hunyuanvideo-community/HunyuanVideo",
                        subfolder="text_encoder_2",
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16
                    )
                    pipe = HunyuanVideoPipeline.from_pretrained(
                        model_id,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        transformer=transformer,
                        torch_dtype=torch.float16,
                    )

#                    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
#                    from diffusers import GGUFQuantizationConfig
#                    from diffusers.utils import export_to_video

#                    if low_vram():
#                        transformer_path = f"https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q3_K_S.gguf"
#                    else:
#                        transformer_path = f"https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q4_K_S.gguf"

#                    transformer = HunyuanVideoTransformer3DModel.from_single_file(
#                        transformer_path,
#                        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
#                        torch_dtype=torch.bfloat16,
#                    )

#                    pipe = HunyuanVideoPipeline.from_pretrained(
#                        movie_model_card,
#                        transformer=transformer,
#                        torch_dtype=torch.float16
#                    )

                if enabled_items:
                    for item in enabled_items:
                        enabled_names.append(
                            (clean_filename(item.name)).replace(".", "")
                        )
                        enabled_weights.append(item.weight_value)
                        pipe.load_lora_weights(
                            bpy.path.abspath(scene.lora_folder),
                            weight_name=item.name + ".safetensors",
                            adapter_name=((clean_filename(item.name)).replace(".", "")),
                        )
                    pipe.set_adapters(enabled_names, adapter_weights=enabled_weights)
                    print("Load LoRAs: " + " ".join(enabled_names))

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.vae.enable_tiling()
                    pipe.enable_model_cpu_offload()

                else:
                    #pipe.vae.enable_tiling()
                    pipe.enable_model_cpu_offload()

            # FramePack
            elif movie_model_card == "lllyasviel/FramePackI2V_HY":
                from diffusers import BitsAndBytesConfig, HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
                from diffusers.utils import export_to_video, load_image
                from transformers import SiglipImageProcessor, SiglipVisionModel

                # vid2vid
                if scene.movie_path and input == "input_strips":
                    print("FramePack doesn't support vid2vid! Using img2vid instead...")

                # img2vid
                if (scene.image_path or scene.movie_path) and input == "input_strips":
                    print("FramePack: Load Image to Video Model")

                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

                    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
                        "lllyasviel/FramePack_F1_I2V_HY_20250503",
                        #"lllyasviel/FramePackI2V_HY",
                        #"newgenai79/SkyReels-V1-Hunyuan-I2V-int4",
                        #subfolder="transformer",
                        quantization_config=nf4_config,
                        torch_dtype=torch.bfloat16,
                    )
                    feature_extractor = SiglipImageProcessor.from_pretrained(
                        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
                    )
                    image_encoder = SiglipVisionModel.from_pretrained(
                        "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
                    )

                    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
                        "hunyuanvideo-community/HunyuanVideo",
                        transformer=transformer,
                        feature_extractor=feature_extractor,
                        image_encoder=image_encoder,
                        torch_dtype=torch.float16,
                    )


                # prompt to video
                else:
                    print("FramePack: Prompt to Video is not supported!")
                    return {"CANCELLED"}
#                    model_id = "hunyuanvideo-community/HunyuanVideo"
##                    from diffusers import BitsAndBytesConfig, HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
##                    from diffusers.utils import export_to_video, load_image
#                    from transformers import SiglipImageProcessor, SiglipVisionModel

#                    nf4_config = BitsAndBytesConfig(
#                        load_in_4bit=True,
#                        bnb_4bit_quant_type="nf4",
#                        bnb_4bit_compute_dtype=torch.bfloat16,
#                    )

#                    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
#                        "lllyasviel/FramePackI2V_HY",
#                        quantization_config=nf4_config,
#                        torch_dtype=torch.bfloat16,
#                    )
#                    feature_extractor = SiglipImageProcessor.from_pretrained(
#                        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
#                    )
#                    image_encoder = SiglipVisionModel.from_pretrained(
#                        "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
#                    )
#                    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
#                        "hunyuanvideo-community/HunyuanVideo",
#                        transformer=transformer,
#                        feature_extractor=feature_extractor,
#                        image_encoder=image_encoder,
#                        torch_dtype=torch.float16,
#                    )

                if gfx_device == "mps":
                    pipe.to("mps")
                else:
                    pipe.vae.enable_tiling()
                    pipe.enable_model_cpu_offload()

            #Skyreel
            elif movie_model_card == "Skywork/SkyReels-V1-Hunyuan-T2V":

                prompt = ensure_skyreel(prompt)
                print("Corrected Prompt: "+prompt)

                # vid2vid
                if scene.movie_path and input == "input_strips":
                    print("SkyReels-V1-Hunyuan doesn't support vid2vid! Doing img2vid instead.")
                    #return {"CANCELLED"}

                # img2vid
                if (scene.image_path or scene.movie_path) and input == "input_strips":
                    print("Load: Image to video (SkyReels-V1-Hunyuan-I2V)")
                    #import torch._dynamo.config
                    from diffusers import HunyuanSkyreelsImageToVideoPipeline, HunyuanVideoTransformer3DModel
                    from diffusers.utils import load_image, export_to_video
#                    from diffusers.hooks import apply_group_offloading

                    #torch._dynamo.config.inline_inbuilt_nn_modules = True

                    model_id = "hunyuanvideo-community/HunyuanVideo"
                    transformer_model_id = "newgenai79/SkyReels-V1-Hunyuan-I2V-int4"

                    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                        transformer_model_id, torch_dtype=torch.bfloat16, subfolder="transformer",
                    )

#                    apply_group_offloading(
#                        transformer,
#                        onload_device=torch.device("cuda"),
#                        offload_device=torch.device("cpu"),
#                        offload_type="block_level",
#                        num_blocks_per_group=2,
#                        use_stream=True,
#                    )

                    pipe = HunyuanSkyreelsImageToVideoPipeline.from_pretrained(
                        model_id, transformer=transformer, torch_dtype=torch.float16
                    )

                # txt2vid
                else:
                    print("Load: text to video (SkyReels-V1-Hunyuan-T2V)")

                    #import torch._dynamo.config
                    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
                    from diffusers.utils import export_to_video

                    #torch._dynamo.config.inline_inbuilt_nn_modules = True

                    model_id = "newgenai79/HunyuanVideo-int4"
                    transformer_model_id = "newgenai79/SkyReels-V1-Hunyuan-T2V-int4"
                    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                        transformer_model_id,
                        subfolder="transformer",
                        torch_dtype=torch.bfloat16
                    )
                    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
                    pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    # pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()
                    #pipe.enable_sequential_cpu_offload()
                    #pipe.enable_xformers_memory_efficient_attention()
                    #pipe.to("cuda")

            # Mochi
            elif movie_model_card == "genmo/mochi-1-preview":
                from diffusers import MochiPipeline

                pipe = MochiPipeline.from_pretrained(
                    movie_model_card, variant="bf16", torch_dtype=torch.bfloat16
                )
                pipe.to("cuda")

                if gfx_device == "mps":
                    pipe.to("mps")
                else:
                    pipe.enable_sequential_cpu_offload()
                    pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()

                scene.generate_movie_x = 848
                scene.generate_movie_y = 480

            # stable-video-diffusion
            elif (
                movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                or movie_model_card == "stabilityai/stable-video-diffusion-img2vid-xt"
            ):
                print("Stable Video Diffusion needs image input")
                return {"CANCELLED"}

            elif movie_model_card == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers":
                if (scene.movie_path or scene.image_path) and input == "input_strips":
                    print("Wan2.1-T2V doesn't support img/vid2vid!")
                    return {"CANCELLED"}

                # Import all necessary classes
                from diffusers import WanPipeline
                from diffusers.quantizers import PipelineQuantizationConfig
                from diffusers.utils import export_to_video
                
                pipeline_quant_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16
                    },
                    # Specify which part of the pipeline to quantize. For Wan-AI, it's the transformer.
                    components_to_quantize=["transformer"],
                )

                print("Loading Wan-AI/Wan2.1-T2V-1.3B-Diffusers with 4-bit quantization API...")
                
                # Pass the new config object to from_pretrained
                pipe = WanPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    quantization_config=pipeline_quant_config,
                )

                lora_files = scene.lora_files
                enabled_names = []
                enabled_weights = []
                # Check if there are any enabled items before loading
                enabled_items = [item for item in lora_files if item.enabled]

                if enabled_items:
                    for item in enabled_items:
                        enabled_names.append(
                            (clean_filename(item.name)).replace(".", "")
                        )
                        enabled_weights.append(item.weight_value)
                        pipe.load_lora_weights(
                            bpy.path.abspath(scene.lora_folder),
                            weight_name=item.name + ".safetensors",
                            adapter_name=((clean_filename(item.name)).replace(".", "")),
                        )
                    pipe.set_adapters(enabled_names, adapter_weights=enabled_weights)
                    print("Load LoRAs: " + " ".join(enabled_names))

                if gfx_device == "mps":
                    # Note: bitsandbytes quantization typically requires a CUDA-enabled GPU.
                    # This line will likely fail on MPS. You may need to add logic
                    # to skip quantization if gfx_device is "mps".
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()


                lora_files = scene.lora_files
                enabled_names = []
                enabled_weights = []
                # Check if there are any enabled items before loading
                enabled_items = [item for item in lora_files if item.enabled]

                if enabled_items:
                    for item in enabled_items:
                        enabled_names.append(
                            (clean_filename(item.name)).replace(".", "")
                        )
                        enabled_weights.append(item.weight_value)
                        pipe.load_lora_weights(
                            bpy.path.abspath(scene.lora_folder),
                            weight_name=item.name + ".safetensors",
                            adapter_name=((clean_filename(item.name)).replace(".", "")),
                        )
                    pipe.set_adapters(enabled_names, adapter_weights=enabled_weights)
                    print("Load LoRAs: " + " ".join(enabled_names))

                if gfx_device == "mps":
                    # Note: bitsandbytes quantization typically requires a CUDA-enabled GPU.
                    # This line will likely fail on MPS. You may need to add logic
                    # to skip quantization if gfx_device is "mps".
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()

            elif movie_model_card == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
                if (not scene.movie_path and not scene.image_path) and not input == "input_strips":
                    print("Wan2.1-I2V doesn't support txt2vid!")
                    self.report({'ERROR'}, "Wan2.1-I2V requires an input image or video.")
                    return {"CANCELLED"}

                print(f"Load: {movie_model_card} with maximum memory optimization.")
        
                import torch
                from diffusers import WanImageToVideoPipeline
                from diffusers.utils import export_to_video, load_image
                from diffusers.quantizers import PipelineQuantizationConfig

                model_id = movie_model_card

                pipeline_quant_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16
                    },
                    components_to_quantize=["transformer", "text_encoder"],
                )

                print("Loading pipeline with 4-bit quantization to minimize RAM/VRAM usage...")

                try:
                    pipe = WanImageToVideoPipeline.from_pretrained(
                        model_id,
                        quantization_config=pipeline_quant_config,
                    )
                    
                    print("Pipeline loaded successfully in quantized state.")

                except Exception as e:
                    print(f"An error occurred during quantized model loading: {e}")
                    self.report({'ERROR'}, f"Failed to load model. Check console: {e}")
                    return {'CANCELLED'}

                lora_files = scene.lora_files
                enabled_names = []
                enabled_weights = []
                enabled_items = [item for item in lora_files if item.enabled]

                if enabled_items:
                    for item in enabled_items:
                        enabled_names.append(
                            (clean_filename(item.name)).replace(".", "")
                        )
                        enabled_weights.append(item.weight_value)
                        pipe.load_lora_weights(
                            bpy.path.abspath(scene.lora_folder),
                            weight_name=item.name + ".safetensors",
                            adapter_name=((clean_filename(item.name)).replace(".", "")),
                        )
                    pipe.set_adapters(enabled_names, adapter_weights=enabled_weights)
                    print("Load LoRAs: " + " ".join(enabled_names))

                if gfx_device == "mps":
                    pipe.to("mps")
                else:
                    torch.cuda.empty_cache() 
                    pipe.enable_model_cpu_offload()

            # Wan Vace - Refactored and Optimized
            elif movie_model_card == "Wan-AI/Wan2.1-VACE-1.3B-diffusers":
                print("Model: " + movie_model_card)

                # The loading logic is the same for both t2v and i2v, so we remove the redundant if/else.
                # We just print the mode for user feedback.
                if not ((scene.movie_path or scene.image_path) and input == "input_strips"):
                    print("Mode: Text-to-Video")
                else:
                    print("Mode: Image-to-Video")

                import torch
                from diffusers import AutoencoderKLWan, WanVACEPipeline
                from diffusers.quantizers import PipelineQuantizationConfig
                from diffusers.schedulers import UniPCMultistepScheduler
                from diffusers.utils import export_to_video, load_image

                # 1. Define the quantization configuration for the main transformer model.
                pipeline_quant_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16
                    },
                    components_to_quantize=["transformer"],
                )
                
                # 2. Load the VAE separately in full float32 precision to ensure maximum quality.
                # This is an important step for VACE models.
                print("Loading VAE in float32 for maximum quality...")
                vae = AutoencoderKLWan.from_pretrained(movie_model_card, subfolder="vae", torch_dtype=torch.float32)

                # 3. Load the main pipeline, passing both the quantization config and the pre-loaded VAE.
                # This applies 4-bit quantization to the transformer while using our high-quality VAE.
                print("Loading main pipeline with 4-bit quantization...")
                pipe = WanVACEPipeline.from_pretrained(
                    movie_model_card,
                    vae=vae, # Use the high-precision VAE we just loaded
                    quantization_config=pipeline_quant_config
                )
                print("Pipeline loaded successfully.")

                # 4. Set up the scheduler as before. This is done after the pipeline is loaded.
                flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
                print(f"Scheduler set to UniPCMultistep with flow_shift={flow_shift}")

                # 5. Apply memory management for inference. This is still a good safety measure.
                if gfx_device == "mps":
                    # Quantization is not supported on MPS, so this path assumes a non-quantized model.
                    print("Moving model to MPS.")
                    pipe.to("mps")
                # For CUDA devices, offloading is the final step for memory-safe inference.
                elif low_vram():
                    print("Low VRAM mode: Enabling model CPU offload.")
                    pipe.enable_model_cpu_offload()
                else:
                    print("Defaulting to model CPU offload for stability.")
                    #pipe.enable_sequential_cpu_offload()
                    pipe.enable_model_cpu_offload()

#            # Wan Vace
#            elif movie_model_card == "Wan-AI/Wan2.1-VACE-1.3B-diffusers":
#                print("Model: "+movie_model_card)
#                # t2i
#                if not ((scene.movie_path or scene.image_path) and input == "input_strips"):
#                    from diffusers import AutoencoderKLWan, WanVACEPipeline
#                    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
#                    from diffusers.utils import export_to_video

#                    vae = AutoencoderKLWan.from_pretrained(movie_model_card, subfolder="vae", torch_dtype=torch.float32)
#                    pipe = WanVACEPipeline.from_pretrained(movie_model_card, vae=vae, torch_dtype=torch.bfloat16)
#                    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
#                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
#                    
#                #i2v
#                else:
#                    import PIL.Image
#                    from diffusers import AutoencoderKLWan, WanVACEPipeline
#                    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
#                    from diffusers.utils import export_to_video, load_image

#                    vae = AutoencoderKLWan.from_pretrained(movie_model_card, subfolder="vae", torch_dtype=torch.float32)
#                    pipe = WanVACEPipeline.from_pretrained(movie_model_card, vae=vae, torch_dtype=torch.bfloat16)
#                    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
#                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)                 

#                if gfx_device == "mps":
#                    pipe.to("mps")
#                elif low_vram():
#                    # pipe.enable_vae_slicing()
#                    pipe.enable_model_cpu_offload()
#                else:
#                    #pipe.enable_sequential_cpu_offload()
#                    #pipe.vae.enable_tiling()
#                    pipe.enable_model_cpu_offload()

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

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )
                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)

#            # Model for upscale generated movie
#            if scene.video_to_video:
#                if torch.cuda.is_available():
#                    torch.cuda.empty_cache()
#                from diffusers import DiffusionPipeline

#                upscale = DiffusionPipeline.from_pretrained(
#                    "cerspense/zeroscope_v2_XL",
#                    torch_dtype=torch.float16,
#                    use_safetensors=False,
#                    local_files_only=local_files_only,
#                )
#                upscale.scheduler = DPMSolverMultistepScheduler.from_config(
#                    upscale.scheduler.config
#                )
#                if gfx_device == "mps":
#                    upscale.to("mps")
#                elif low_vram():
#                    upscale.enable_model_cpu_offload()
#                else:
#                    upscale.to(gfx_device)

        # GENERATING - Main Loop Video
        for i in range(scene.movie_num_batch):
            if duration == -1 and input == "input_strips":
                strip = scene.sequence_editor.active_strip
                if strip:
                    duration = scene.generate_movie_frames = (
                        strip.frame_final_duration + 1
                    )
                    print(str(strip.frame_final_duration))

            start_time = timer()

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
                    (scene.movie_num_batch * abs(duration)) + scene.frame_current,
                )
                start_frame = scene.frame_current

            # Get seed
            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(-2147483647, 2147483647)
            )
            print("Seed: " + str(seed))
            context.scene.movie_num_seed = seed

            # Use cuda if possible
            if (
                torch.cuda.is_available()
                and movie_model_card != "stabilityai/stable-video-diffusion-img2vid"
                and movie_model_card != "stabilityai/stable-video-diffusion-img2vid-xt"
            ):
                generator = (
                    torch.Generator("cuda").manual_seed(seed) if seed != 0 else None
                )
            else:
                if seed != 0:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                else:
                    generator = None

            # Process batch input for images
            if (scene.movie_path or scene.image_path) and input == "input_strips":
                video_path = scene.movie_path

                # frame2frame
                if movie_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    input_video_path = video_path
                    output_video_path = solve_path("temp_images")
                    if scene.movie_path:
                        print("Process: Frame by frame (SD XL) - from Movie strip")
                        frames = process_video(input_video_path, output_video_path)
                    elif scene.image_path:
                        print("Process: Frame by frame (SD XL) - from Image strip")
                        frames = process_image(
                            scene.image_path, int(scene.generate_movie_frames)
                        )

                    from torchvision import transforms

                    pil_to_tensor = transforms.ToTensor()

                    video_frames = []

                    for frame_idx, frame in enumerate(frames):
                        try:
                            if frame is None:
                                print(f"Frame {frame_idx} is None. Skipping.")
                                continue

                            if not isinstance(frame, Image.Image):
                                print(
                                    f"Frame {frame_idx} is not a valid PIL image. Type: {type(frame)}. Skipping."
                                )
                                continue

                            width, height = frame.size
                            print(
                                f"Processing frame {frame_idx + 1}/{len(frames)}, size: {width}x{height}"
                            )

                            if width == 0 or height == 0:
                                print(
                                    f"Frame {frame_idx} has invalid dimensions {width}x{height}. Skipping."
                                )
                                continue

                            new_width = closest_divisible_8(width)
                            new_height = closest_divisible_8(height)

                            if (new_width, new_height) != (width, height):
                                print(
                                    f"Resizing frame {frame_idx} to {new_width}x{new_height}"
                                )
                                frame = frame.resize(
                                    (new_width, new_height), Image.Resampling.LANCZOS
                                )

                            frame = transforms.functional.invert(frame)

                            frame_tensor = pil_to_tensor(frame)
                            frame_tensor = frame_tensor.float()

                            print(
                                f"Frame {frame_idx} - Tensor shape: {frame_tensor.shape}, Total elements: {frame_tensor.numel()}"
                            )
                            print(
                                f"Frame {frame_idx} - Tensor data type: {frame_tensor.dtype}"
                            )

                            if frame_tensor.numel() == 0:
                                print(
                                    f"Frame {frame_idx}: Tensor has zero elements. Skipping."
                                )
                                continue

                            if frame_tensor.ndim == 3:
                                frame_tensor = frame_tensor.unsqueeze(0)
                                print(
                                    f"After adding batch dimension - Tensor shape: {frame_tensor.shape}"
                                )

                            print(
                                f"Before processing - Tensor shape: {frame_tensor.shape}, Elements: {frame_tensor.numel()}"
                            )

                            try:
                                print(f"Frame {frame_idx}: Running Frame by Frame...")
                                image = refiner(
                                    prompt,
                                    image=frame_tensor,
                                    strength=1.00 - scene.image_power,
                                    num_inference_steps=movie_num_inference_steps,
                                    guidance_scale=2.8,  # movie_num_guidance,
                                    generator=generator,
                                ).images[0]

                                if image is None or not isinstance(image, Image.Image):
                                    print(
                                        f"Frame {frame_idx}: Output is INVALID. Skipping."
                                    )
                                    continue

                                print(f"Frame {frame_idx}: Is a valid image.")

                            except Exception as e:
                                print(f"Frame {frame_idx}: ERROR in refiner - {e}")
                                continue

                        except Exception as e:
                            print(f"Frame {frame_idx}: General error - {e}")
                            continue

                        video_frames.append(image)

                    video_frames = np.array(video_frames)

                # vid2vid / img2vid
                elif (
                    movie_model_card == "stabilityai/stable-video-diffusion-img2vid"
                    or movie_model_card
                    == "stabilityai/stable-video-diffusion-img2vid-xt"
                ):
                    if scene.movie_path:
                        print("Process: Video Image to SVD Video")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_first_frame(bpy.path.abspath(scene.movie_path))

                    elif scene.image_path:
                        print("Process: Image to SVD Video")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)

                    image = image.resize(
                        (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
                    )

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
                        # image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]

                # CogVideoX img/vid2vid
                elif (
                    movie_model_card == "THUDM/CogVideoX-5b"
                    or movie_model_card == "THUDM/CogVideoX-2b"
                ):
                    if scene.movie_path:
                        print("Process: Video to video (CogVideoX)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        # video = load_video_as_np_array(video_path)
                        video = load_video(video_path)[:49]
                        video_frames = pipe(
                            video=video,
                            prompt=prompt,
                            strength=1.00 - scene.image_power,
                            negative_prompt=negative_prompt,
                            num_inference_steps=movie_num_inference_steps,
                            guidance_scale=movie_num_guidance,
                            height=480,
                            width=720,
                            # num_frames=abs(duration),
                            generator=generator,
                        ).frames[0]

                    elif scene.image_path:
                        print("Process: Image to video (CogVideoX)")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)
                        image = image.resize(
                            (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
                        )
                        video_frames = pipe(
                            image=image,
                            prompt=prompt,
                            # strength=1.00 - scene.image_power,
                            # negative_prompt=negative_prompt,
                            num_inference_steps=movie_num_inference_steps,
                            guidance_scale=movie_num_guidance,
                            height=480,
                            width=720,
                            # num_frames=abs(duration),
                            generator=generator,
                            use_dynamic_cfg=True,
                        ).frames[0]

                # LTX
                elif movie_model_card == "Lightricks/LTX-Video":
                    if scene.movie_path:
                        print("Process: Video to Video")
                        if not os.path.isfile(bpy.path.abspath(scene.movie_path)):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_video(bpy.path.abspath(scene.movie_path))
                        #image = load_first_frame(bpy.path.abspath(scene.movie_path))
                    if scene.image_path:
                        strip = scene.sequence_editor.active_strip
                        print("Process: Image to video (LTX)")
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        print("Path: "+img_path)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)
                    #                    image = image.resize(
                    #                        (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
                    #                    )
                    video_frames = pipe(
                        image=image,
                        prompt=prompt,
                        # strength=1.00 - scene.image_power,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                        max_sequence_length=512,
                        decode_timestep=0.05,
                        image_cond_noise_scale=0.025,
                    ).frames[0]

                #Skyreel
                elif movie_model_card == "Skywork/SkyReels-V1-Hunyuan-T2V":
                    from diffusers.utils import load_image, export_to_video
                    if scene.movie_path:
                        print("Process: Video Image to Video (SkyReels-V1-Hunyuan-T2V)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_first_frame(bpy.path.abspath(scene.movie_path))
                    if scene.image_path:
                        print("Process: Image to video (SkyReels-V1-Hunyuan-T2V)")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        print("Path: "+img_path)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)
                    #                    image = image.resize(
                    #                        (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
                    #                    )
                    video_frames = pipe(
                        image=image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                        max_sequence_length=512,
                    ).frames[0]

                elif movie_model_card == "hunyuanvideo-community/HunyuanVideo":

                    from diffusers.utils import load_image, export_to_video
                    if scene.movie_path:
                        print("Process: Video Image to Video (Hunyuan-I2V)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_first_frame(bpy.path.abspath(scene.movie_path))
                    if scene.image_path:
                        print("Process: Image to video (Hunyuan-I2V)")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)
                    #                    image = image.resize(
                    #                        (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
                    #                    )
                    video_frames = pipe(
                        image=image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                        max_sequence_length=512,
                    ).frames[0]

                elif movie_model_card == "lllyasviel/FramePackI2V_HY":
                    from diffusers.utils import load_image, export_to_video
                    if scene.movie_path:
                        print("Process: Video Image to Video (FramePack)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}

                        #from diffusers.utils import load_video
                        #image=load_video(bpy.path.abspath(scene.movie_path))

                        image = load_first_frame(bpy.path.abspath(scene.movie_path))
                    if scene.image_path:
                        print("Process: Image to video (FramePack)")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)
#                        image = image.resize(
#                            (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
#                        )

                    if scene.out_frame:
                        subject_strip = find_strip_by_name(scene, scene.out_frame)
                        print("image_strip from find_strip_by_name:", subject_strip)

                        if subject_strip.type == "IMAGE":
                            print("image_strip type is IMAGE")
                            image_path_chk = bpy.path.abspath(
                                os.path.join(
                                    subject_strip.directory,
                                    subject_strip.elements[0].filename,
                                )
                            )
                            if not os.path.isfile(bpy.path.abspath(image_path_chk)):
                                print("No End Frame file found.")
                                return {"CANCELLED"}
                            else:
                                print("Load image path: "+bpy.path.abspath(image_path_chk))
                                last_image = load_image(bpy.path.abspath(image_path_chk))
                                last_image = last_image.resize(image.size)
                                print("Last Frame loaded.")
                        else:
                            print("image_strip type is not IMAGE:", image_strip.type)
                            return {"CANCELLED"}
                    else:
                        last_image = None

                    video_frames = pipe(
                        image=image,
                        last_image=last_image,
                        prompt=prompt,
                        #negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                        sampling_type="vanilla",
                        #max_sequence_length=512,
                    ).frames[0]

                elif movie_model_card == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
                    from diffusers.utils import load_image, export_to_video
                    import numpy as np
                    if scene.movie_path:
                        print("Process: Video Image to Video (Wan2.1-I2V-14B-480P-Diffusers)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_first_frame(bpy.path.abspath(scene.movie_path))
                    if scene.image_path:
                        print("Process: Image to video (Wan2.1-I2V-14B-480P-Diffusers)")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)
                    #                    image = image.resize(
                    #                        (closest_divisible_32(int(x)), closest_divisible_32(int(y)))
                    #                    )
#                    max_area = 480 * 832
#                    aspect_ratio = image.height / image.width
#                    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
#                    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
#                    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
#                    image = image.resize((width, height))
                    video_frames = pipe(
                        image=image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                        max_sequence_length=512,
                    ).frames[0]
                elif movie_model_card == "Wan-AI/Wan2.1-VACE-1.3B-diffusers" and input == "input_strips":
                    from diffusers.utils import load_image, export_to_video
                    import numpy as np
                    import PIL
                    if scene.movie_path:
                        print("Process: Video Image to Video (Wan2.1-I2V-14B-480P-Diffusers)")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_first_frame(bpy.path.abspath(scene.movie_path))
                    if scene.image_path:
                        print("Process: Image to video (Wan2.1-I2V-14B-480P-Diffusers)")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)

                        img = image.resize((x, y))
                        frames = [img]
                        # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
                        # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
                        # match the original code.
                        frames.extend([PIL.Image.new("RGB", (x, y), (128, 128, 128))] * (abs(duration) - 1))
                        mask_black = PIL.Image.new("L", (x, y), 0)
                        mask_white = PIL.Image.new("L", (x, y), 255)
                        mask = [mask_black, *[mask_white] * (abs(duration) - 1)]

                    video_frames = pipe(
                        video=frames,
                        mask=mask,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        #num_frames=abs(duration),
                        generator=generator,
                        max_sequence_length=512,
                    ).frames[0]
                elif (
                    movie_model_card != "Hailuo/MiniMax/txt2vid"
                    and movie_model_card != "Hailuo/MiniMax/img2vid"
                    and movie_model_card != "Hailuo/MiniMax/subject2vid"
                    and movie_model_card != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                ): #something is broken here?
                    if scene.movie_path:
                        print("Process: Video to video")
                        if not os.path.isfile(scene.movie_path):
                            print("No file found.")
                            return {"CANCELLED"}
                    elif scene.image_path:
                        print("Process: Image to video")
                        strip = scene.sequence_editor.active_strip
                        img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                        if not os.path.isfile(img_path):
                            print("No file found.")
                            return {"CANCELLED"}
                        image = load_image(img_path)

                    video = load_video_as_np_array(video_path)
                    video = process_image(
                        scene.image_path, int(scene.generate_movie_frames)
                    )
                    video = np.array(video)

                    # Upscale video
                    if scene.video_to_video:
                        video = [
                            Image.fromarray(frame).resize(
                                (
                                    closest_divisible_32(int(x * 2)),
                                    closest_divisible_32(int(y * 2)),
                                )
                            )
                            for frame in video
                        ]
                    else:
                        video = [
                            Image.fromarray(frame).resize(
                                (
                                    closest_divisible_32(int(x)),
                                    closest_divisible_32(int(y)),
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

                elif movie_model_card == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers":
                    if (scene.movie_path or scene.image_path) and input == "input_strips":
                        print("Wan2.1-T2V doesn't support img/vid2vid!")
                        return {"CANCELLED"}


            # Prompt input for movies
            elif (
                movie_model_card != "Hailuo/MiniMax/txt2vid"
                and movie_model_card != "Hailuo/MiniMax/img2vid"
                and movie_model_card != "Hailuo/MiniMax/subject2vid"
            ):
                print("Generate: Video from text")

                if (
                    movie_model_card == "THUDM/CogVideoX-5b"
                    or movie_model_card == "THUDM/CogVideoX-2b"
                ):
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

                # HunyuanVideo
                elif movie_model_card == "hunyuanvideo-community/HunyuanVideo":
                    video_frames = pipe(
                        prompt=prompt,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        num_videos_per_prompt=1,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]
                # FramePack
                elif movie_model_card == "lllyasviel/FramePackI2V_HY":
                    video_frames = pipe(
                        prompt=prompt,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        num_videos_per_prompt=1,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                    ).frames[0]
                # Skyreel
                elif movie_model_card == "Skywork/SkyReels-V1-Hunyuan-T2V":
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=movie_num_inference_steps,
                        guidance_scale=movie_num_guidance,
                        height=y,
                        width=x,
                        num_frames=abs(duration),
                        generator=generator,
                        max_sequence_length=512,
                        #true_cfg_scale=6.0,
                        # use_dynamic_cfg=True,
                    ).frames[0]
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
                        max_sequence_length=256,
                    ).frames[0]
                movie_model_card = addon_prefs.movie_model_card

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Upscale video.
                if scene.video_to_video:
                    print("Upscale: Video")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    video = [
                        Image.fromarray(frame).resize(
                            (closest_divisible_32(x * 2), closest_divisible_32(y * 2))
                        )
                        for frame in video_frames
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

            # MiniMax
            if (
                movie_model_card == "Hailuo/MiniMax/txt2vid"
                or movie_model_card == "Hailuo/MiniMax/img2vid"
                or movie_model_card == "Hailuo/MiniMax/subject2vid"
            ):
                current_dir = os.path.dirname(__file__)
                init_file_path = os.path.join(current_dir, "MiniMax_API.txt")
                api_key = str(read_file(init_file_path))
                if api_key == "":
                    print("API key is missing!")
                    return {"CANCELLED"}

                image_path = None

                if movie_model_card == "Hailuo/MiniMax/img2vid":
                    if scene.image_path and minimax_validate_image(
                        bpy.path.abspath(scene.image_path)
                    ):
                        image_path = bpy.path.abspath(scene.image_path)
                        print("Image Path: " + image_path)
                    else:
                        print("Image path not found: " + bpy.path.abspath(scene.image_path))
                        return {"CANCELLED"}

                elif movie_model_card == "Hailuo/MiniMax/subject2vid":
                    print("Entered movie_model_card == 'Hailuo/MiniMax/subject2vid'")
                    print("scene.minimax_subject:", scene.minimax_subject)

                    if scene.minimax_subject:
                        subject_strip = find_strip_by_name(scene, scene.minimax_subject)
                        print("image_strip from find_strip_by_name:", subject_strip)

                        if subject_strip.type == "IMAGE":
                            print("image_strip type is IMAGE")
                            image_path_chk = bpy.path.abspath(
                                os.path.join(
                                    subject_strip.directory,
                                    subject_strip.elements[0].filename,
                                )
                            )
                            # subject_strip = bpy.path.abspath(get_render_strip(self, context, subject_strip))
                            print("image_strip after get_render_strip:", image_path_chk)

                            # image_path_chk = bpy.path.abspath(get_strip_path(image_strip))
                            # print("image_path_chk (validated path):", image_path_chk)

                            if minimax_validate_image(image_path_chk):
                                print("Image path is valid")
                                image_path = image_path_chk
                                # print("Image Path:", image_path)
                            else:
                                print("Image path failed validation:", image_path_chk)
                                return {"CANCELLED"}
                        else:
                            print("image_strip type is not IMAGE:", image_strip.type)
                            return {"CANCELLED"}
                    else:
                        print("Subject is empty!")
                        return {"CANCELLED"}

                if not image_path and not movie_model_card == "Hailuo/MiniMax/txt2vid":
                    print("Loading strip failed!")
                    return {"CANCELLED"}

                task_id = invoke_video_generation(
                    prompt[:2000], api_key, image_path, movie_model_card
                )
                src_path = solve_path(clean_filename(prompt[:20]) + ".mp4")
                print("Task ID: "+str(task_id))
                print("Generating: " + src_path)
                print(
                    "-----------------Video generation task submitted to MiniMax-----------------"
                )
                while True:
                    #progress_bar(10)

                    file_id, status = query_video_generation(task_id, api_key)
                    if file_id != "":
                        print("Image Path: " + src_path)
                        dst_path = fetch_video_result(file_id, api_key, src_path)
                        if os.path.exists(dst_path):
                            print("---------------Successful---------------")
                            break
                        else:
                            print("---------------Failed---------------")
                            return {"CANCELLED"}
                    elif status == "Fail" or status == "Unknown":
                        print("---------------Failed---------------")
                        return {"CANCELLED"}

                print("Result: " + dst_path)
            else:
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
                                scene.frame_current = (
                                    scene.sequence_editor.active_strip.frame_final_start
                                )

                            # Redraw UI to display the new strip. Remove this if Blender crashes: https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
                            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                            break
            print_elapsed_time(start_time)
        if old_duration == -1 and input == "input_strips":
            scene.generate_movie_frames = -1
        pipe = None
        refiner = None
        converter = None

        clear_cuda_cache()

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

###### F5-TTS

## Configuration for default F5-TTS model
#DEFAULT_F5TTS_CFG = [
#    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
#    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
#    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
#]

## E2-TTS model config
#E2TTS_CKPT_PATH = "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
#E2TTS_MODEL_CFG = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)

### Global variables to hold loaded models and vocoder for reuse
##global vocoder
##vocoder = None
##F5TTS_ema_model = None
##E2TTS_ema_model = None
##custom_ema_model = None
##pre_custom_path = ""
##dependencies_loaded = False


def split_text_for_tts(full_text: str) -> list[str]:
    """
    Splits text into manageable and natural-sounding chunks for TTS.
    Uses spaCy if available, otherwise falls back to a simple splitter.
    """
    MAX_CHUNK_LENGTH = 285
    print("Full text: "+full_text)
    try:
        import spacy
        # Define an alias for the type hint to use later
        from spacy.tokens.span import Span as SpacySpan
        SPACY_AVAILABLE = True
        print("spaCy library found. Advanced text splitting is enabled.")
    except ImportError:
        # If spacy is not installed, create placeholder variables
        spacy = None
        SpacySpan = None # This is needed so type hints don't break
        SPACY_AVAILABLE = False
        print("Warning: spaCy library not found. Using simple text splitting. For more natural TTS, please install it.")
        
    if not SPACY_AVAILABLE:
        # Assuming simple_fallback_splitter and SPACY_AVAILABLE flag from previous answer
        return simple_fallback_splitter(full_text, MAX_CHUNK_LENGTH)

    # --- spaCy-powered logic (only runs if the import succeeded) ---
    # NOTE: The spaCy model should be loaded only once if possible for performance.
    # If this function is called many times, consider loading `nlp` outside.
    nlp = spacy.load("en_core_web_md")
    doc = nlp(full_text.replace("\n", " ")) # Replace newlines with spaces for better sentence detection
    chunks = []
    current_chunk = ""

    # Iterate through all sentences provided by spaCy
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        # If the sentence itself is too long, it must be split into sub-parts
        if len(sentence_text) > MAX_CHUNK_LENGTH:
            # First, if there's anything in current_chunk, finalize it and add to the list.
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Split the oversized sentence into smaller, manageable parts
            sub_parts = split_long_sentence(sent, MAX_CHUNK_LENGTH)

            # Add all the new sub-parts directly to the chunks list
            chunks.extend(sub_parts)
            
            # Continue to the next sentence, as this one has been fully processed
            continue

        # --- Logic for sentences that are NOT too long ---

        # If adding the new sentence would make the current chunk too long...
        # (add 1 for the space that will join them)
        if len(current_chunk) + len(sentence_text) + 1 > MAX_CHUNK_LENGTH:
            # ...finalize the current chunk...
            if current_chunk:
                chunks.append(current_chunk)
            # ...and start a new chunk with the new sentence.
            current_chunk = sentence_text
        else:
            # Otherwise, append the new sentence to the current chunk.
            if current_chunk:
                current_chunk += " " + sentence_text
            else:
                current_chunk = sentence_text

    # After the loop, add any remaining text in current_chunk to the list.
    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if c.strip()]


def split_long_sentence(spacy_sentence_span: spacy.tokens.span.Span, max_len: int) -> list[str]:
    """
    Splits a single spaCy sentence Span that is longer than max_len
    into smaller pieces, aiming for natural breaks.
    Returns a list of strings.
    """
    import spacy
    parts = []
    tokens = list(spacy_sentence_span) # Get all tokens from the sentence span
    current_pos = 0 # Index in the tokens list

    while current_pos < len(tokens):
        # Determine the text of the remaining part of the sentence
        remaining_doc = tokens[current_pos].doc
        span_start_index = tokens[current_pos].i
        span_end_index = tokens[-1].i + 1 # up to the end of the last token
        text_to_split = remaining_doc[span_start_index:span_end_index].text

        if len(text_to_split) <= max_len:
            parts.append(text_to_split)
            break # All remaining tokens fit

        # Find the best break point *within* the first max_len characters of text_to_split
        # This requires careful token-level iteration and checking linguistic features
        
        # Iterate tokens from current_pos up to where cumulative length approaches max_len
        potential_break_token_idx_in_sentence = -1 # Absolute index in the original sentence doc
        current_length_chars = 0
        
        # Iterate tokens starting from current_pos to find a segment <= max_len
        last_safe_break_token_offset = -1 # Relative to current_pos

        for i_offset, token in enumerate(tokens[current_pos:]):
            # Consider space before token, unless it's the first in this potential part
            token_text_to_add = token.text_with_ws if (current_length_chars > 0 or i_offset > 0) else token.text
            
            if current_length_chars + len(token_text_to_add.lstrip()) > max_len: # lstrip to avoid counting leading space if it's the start
                break # This token makes it too long

            current_length_chars += len(token_text_to_add.lstrip())
            last_safe_break_token_offset = i_offset # This token still fits

            # Check for good break points (punctuation, conjunctions)
            # Prefer breaks *after* punctuation if it's part of the current segment.
            # Prefer breaks *before* conjunctions.
            # This is a place for sophisticated logic. For simplicity:
            if token.is_punct and token.text in [',', ';', ':', '—']:
                # This could be a good place to note, will be handled by scan_back
                pass
            if token.dep_ == 'cc': # Coordinating conjunction
                # This could be a good place to note
                pass
        
        # If nothing fit (e.g., first token itself > max_len, highly unlikely with sane text)
        if last_safe_break_token_offset == -1:
             # Fallback: Take up to MAX_LEN characters and find last space (crude)
            slice_text = text_to_split[:max_len]
            last_space = slice_text.rfind(' ')
            if last_space != -1:
                parts.append(slice_text[:last_space].strip())
                # This requires updating current_pos based on character count, which is fiddly.
                # A token-based approach is cleaner.
                # For now, let's assume the token iteration handles this better.
                # This fallback needs to be robust or avoided by good token logic.
                # For this example, we'll rely on token iteration to find a split point.
                # If the first token itself is too long, this function has a problem.
                # It's better to ensure 'potential_break_token_idx_in_sentence' gets set properly.
                
                # Simplified: If the loop above found at least one token that fits.
                if last_safe_break_token_offset >=0:
                    actual_break_idx_relative_to_current_pos = last_safe_break_token_offset
                else: # First token already too long - should not happen if max_len is reasonable
                    parts.append(tokens[current_pos].text) # Take first token only
                    current_pos +=1
                    continue

            else: # No space, hard cut (worst case)
                parts.append(slice_text.strip())
                # Update current_pos...
                current_pos +=1 # very simplified, needs to advance by tokens in slice_text
                continue

        # Now, scan backward from tokens[current_pos + last_safe_break_token_offset]
        # to find the *best* break point (e.g., punctuation, conjunction).
        
        best_split_offset = last_safe_break_token_offset # Default to the furthest fitting token

        for i in range(last_safe_break_token_offset, 0, -1): # Scan back, but not before the first token of this sub-segment
            token_at_i = tokens[current_pos + i]
            prev_token_at_i = tokens[current_pos + i -1]

            # Ideal: split AFTER these punctuations
            if prev_token_at_i.text in [';', ':', '—']:
                best_split_offset = i -1 # The punctuation (prev_token) will be the last in the segment
                break
            # Good: split AFTER a comma
            if prev_token_at_i.text == ',':
                best_split_offset = i -1
                break
            # Good: split BEFORE a conjunction (token_at_i is the conjunction)
            if token_at_i.dep_ == 'cc' and token_at_i.pos_ == 'CCONJ':
                best_split_offset = i -1 # Break before the conjunction (segment ends with prev_token_at_i)
                break
        
        # Extract the part based on best_split_offset
        part_tokens = tokens[current_pos : current_pos + best_split_offset + 1]
        if part_tokens:
            part_doc = part_tokens[0].doc
            span_start = part_tokens[0].i
            span_end = part_tokens[-1].i + 1
            parts.append(part_doc[span_start:span_end].text.strip())
        
        current_pos += best_split_offset + 1


    return [p for p in parts if p] # Filter out any empty strings


def simple_fallback_splitter(full_text: str, max_len: int) -> list[str]:
    # ... (implementation of the simple splitter) ...
    print("Using simple fallback text splitter.")
    chunks = []
    while len(full_text) > max_len:
        break_point = full_text.rfind(' ', 0, max_len)
        if break_point == -1: break_point = max_len
        chunks.append(full_text[:break_point].strip())
        full_text = full_text[break_point:].strip()
    if full_text: chunks.append(full_text)
    return chunks



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
        strip = scene.sequence_editor.active_strip
        input = scene.input_strips
        pipe = None
        strips = context.selected_sequences
        if strip in strips:
            duration = scene.audio_length_in_f = (
                strip.frame_final_duration + 1
            )
            audio_length_in_s = duration = duration / (
                scene.render.fps / scene.render.fps_base
            )
        else:
            duration = scene.audio_length_in_f
            audio_length_in_s = duration = duration / (
                scene.render.fps / scene.render.fps_base
            )            

        import torch
        import torchaudio
        import scipy
        import random
        import os
        from scipy.io.wavfile import write as write_wav

        if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0":
            try:
                import scipy
                import torch
                from diffusers import StableAudioPipeline
            except ModuleNotFoundError as e:
                print("Dependencies needs to be installed in the add-on preferences: "+str(e.name))
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
                print("All required modules (whisperspeech, resemble_enhance) imported successfully.") # Optional: for confirmation

            except ModuleNotFoundError as e:
                missing_module_name = e.name

                error_message = (
                    f"Module '{missing_module_name}' not found. "
                    "This dependency needs to be installed. "
                    "Please check the add-on preferences to install missing dependencies."
                )

                print(error_message)

                if hasattr(self, 'report'):
                    self.report({"ERROR"}, error_message)

                return {"CANCELLED"}

        if addon_prefs.audio_model_card == "SWivid/F5-TTS":
            try:
                # tqdm
                #tempfile
                import torcheval
                import numpy as np
                import soundfile as sf
                import torch
                import torchaudio
                from cached_path import cached_path

                # Check if f5_tts is actually importable
                from f5_tts.infer.utils_infer import (
                    infer_process,
                    load_model,
                    load_vocoder,
                    preprocess_ref_audio_text,
                    remove_silence_for_generated_wav,
                )
                from f5_tts.model import DiT, UNetT

                import tempfile

            except ImportError as e:
                print("\n--------------------------------------------------")
                print(f"WARNING: TTS dependencies not found or failed to import: {e}")
                print("Please install required libraries in Blender's Python environment:")
                print("  Example: <Blender Install Dir>/4.1/python/bin/python.exe -m pip install f5-tts transformers torchaudio soundfile cached_path numpy torch torcheval") # Added torcheval to install list
                print("--------------------------------------------------\n")

                # Define dummy functions/classes to prevent errors if imports fail
                class DummyModule:
                    def __getattr__(self, name):
                        # Delay the error until synthesis is attempted
                        def dummy_func(*args, **kwargs):
                             raise RuntimeError(f"TTS dependency missing. Cannot access '{name}'. Install f5-tts, torch, etc.")
                        return dummy_func
                np = DummyModule()
                sf = DummyModule()
                torch = DummyModule()
                torch.cuda = DummyModule() # Ensure cuda access also raises error
                torcheval = DummyModule() # Dummy for torcheval
                torchaudio = DummyModule()
                cached_path = DummyModule()
                class DummyModel: pass
                # Assign dummy functions/classes directly to the expected names
                infer_process = DummyModule().infer_process
                load_model = DummyModule().load_model
                load_vocoder = DummyModule().load_vocoder
                preprocess_ref_audio_text = DummyModule().preprocess_ref_audio_text
                remove_silence_for_generated_wav = DummyModule().remove_silence_for_generated_wav
                DiT = DummyModel
                UNetT = DummyModel
                dependencies_loaded = False # Ensure flag is False
                return {"CANCELLED"}
        if (
            addon_prefs.audio_model_card == "Chatterbox"
        ):
            import numpy as np

            try:
                import torchaudio as ta
                from chatterbox.tts import ChatterboxTTS
                from chatterbox.vc import ChatterboxVC
                
                import spacy

            except ModuleNotFoundError as e:
                missing_module_name = e.name
                error_message = (
                    f"Module '{missing_module_name}' not found. "
                    "This dependency needs to be installed. "
                    "Please check the add-on preferences to install missing dependencies."
                )
                print(error_message)
                if hasattr(self, 'report'):
                    self.report({"ERROR"}, error_message)

                return {"CANCELLED"}
        if (
            addon_prefs.audio_model_card == "parler-tts/parler-tts-large-v1"
            or addon_prefs.audio_model_card == "parler-tts/parler-tts-mini-v1"
        ):
            import numpy as np

            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                from transformers import AutoTokenizer
            except ModuleNotFoundError as e:
                missing_module_name = e.name
                error_message = (
                    f"Module '{missing_module_name}' not found. "
                    "This dependency needs to be installed. "
                    "Please check the add-on preferences to install missing dependencies."
                )
                print(error_message)
                if hasattr(self, 'report'):
                    self.report({"ERROR"}, error_message)

                return {"CANCELLED"}

        if addon_prefs.audio_model_card == "bark":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            try:
                import numpy as np
                from bark.generation import (
                    generate_text_semantic,
                    preload_models,
                )
                from bark.api import semantic_to_waveform
                from bark import generate_audio, SAMPLE_RATE

                from resemble_enhance.enhancer.inference import denoise, enhance
            except ModuleNotFoundError as e:
                missing_module_name = e.name
                error_message = (
                    f"Module '{missing_module_name}' not found. "
                    "This dependency needs to be installed. "
                    "Please check the add-on preferences to install missing dependencies."
                )
                print(error_message)
                if hasattr(self, 'report'):
                    self.report({"ERROR"}, error_message)

                return {"CANCELLED"}

        if addon_prefs.audio_model_card == "MMAudio":
            try:
                #import spaces
                #import logging
                from datetime import datetime
                from pathlib import Path
                import librosa

                import gradio as gr
                import torch
                import torchaudio
                import os
                import numpy as np
                import mmaudio

                from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, load_image, make_video, VideoInfo,
                                                setup_eval_logging)
                from mmaudio.model.flow_matching import FlowMatching
                from mmaudio.model.networks import MMAudio, get_my_mmaudio
                from mmaudio.model.sequence_config import SequenceConfig
                from mmaudio.model.utils.features_utils import FeaturesUtils
                import tempfile

                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except ModuleNotFoundError as e:
                missing_module_name = e.name
                error_message = (
                    f"Module '{missing_module_name}' not found. "
                    "This dependency needs to be installed. "
                    "Please check the add-on preferences to install missing dependencies."
                )
                print(error_message)
                if hasattr(self, 'report'):
                    self.report({"ERROR"}, error_message)

                return {"CANCELLED"}

        show_system_console(True)
        set_system_console_topmost(True)

        # clear the VRAM
        clear_cuda_cache()

        # Load models Audio
        print("Model:  " + addon_prefs.audio_model_card)

        if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0":
            repo_id = "ylacombe/stable-audio-1.0"
            pipe = StableAudioPipeline.from_pretrained(
                repo_id, torch_dtype=torch.float16
            )
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        elif addon_prefs.audio_model_card == "cvssp/audioldm2-large":
            repo_id = addon_prefs.audio_model_card
            from diffusers import AudioLDM2Pipeline

            pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
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

        #F5-TTS
        elif addon_prefs.audio_model_card == "SWivid/F5-TTS":
             ##### F5-TTS
            # Configuration for default F5-TTS model
            DEFAULT_F5TTS_CFG = [
                "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
                "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
            ]
            # E2-TTS model config
            E2TTS_CKPT_PATH = "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
            E2TTS_MODEL_CFG = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
            ## Global variables to hold loaded models and vocoder for reuse
            #global vocoder
            #vocoder = None
            #F5TTS_ema_model = None
            #E2TTS_ema_model = None
            #custom_ema_model = None
            #pre_custom_path = ""
            #dependencies_loaded = False
            print("Loading vocoder...")
            try:
             # load_vocoder should handle device placement internally
                vocoder = load_vocoder()
                print("Vocoder loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Vocoder failed to load: {e}") from e

            ckpt_path_str = str(cached_path(DEFAULT_F5TTS_CFG[0]))

            # load_model should handle device placement
            F5TTS_model_cfg_dict = json.loads(DEFAULT_F5TTS_CFG[2])
            pipe = load_model(DiT, F5TTS_model_cfg_dict, ckpt_path_str).to(gfx_device)
            print("F5-TTS model loaded.")

            if pipe is None:
                 raise RuntimeError(f"Failed to load or get model F5-TTS'.")

        # Chatterbox
        elif addon_prefs.audio_model_card == "Chatterbox":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            print(f"Using device: {device}")

        # Parler
        elif (
            addon_prefs.audio_model_card == "parler-tts/parler-tts-large-v1"
            or addon_prefs.audio_model_card == "parler-tts/parler-tts-mini-v1"
        ):
#            pipe = ParlerTTSForConditionalGeneration.from_pretrained(
#                "parler-tts/parler-tts-large-v1", #revision="refs/pr/9"
#            ).to(gfx_device)
#            tokenizer = AutoTokenizer.from_pretrained(addon_prefs.audio_model_card)

#            # --- Start of Final Corrected Code ---
#            from parler_tts import ParlerTTSForConditionalGeneration
#            from transformers import AutoTokenizer

#            # Your existing setup
#            addon_prefs.audio_model_card = "parler-tts/parler-tts-large-v1"
#            #gfx_device = "cuda"  # Or your detected device

#            # Load the model using trust_remote_code=True
#            # This is the definitive fix that uses the author's own code to build the model,
#            # guaranteeing the architecture matches the saved weights.
#            pipe = ParlerTTSForConditionalGeneration.from_pretrained(
#                addon_prefs.audio_model_card,
#                trust_remote_code=True,  # This is the key
#            ).to(gfx_device)

#            # The tokenizer can be loaded as before
#            tokenizer = AutoTokenizer.from_pretrained(addon_prefs.audio_model_card)

#            # --- End of Final Corrected Code ---

            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            pipe = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(gfx_device)
            tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")


        #MMAudio
        elif addon_prefs.audio_model_card == "MMAudio":

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            #logger = logging.getLogger()

            device = gfx_device
            dtype = torch.bfloat16

            model_config: ModelConfig = all_model_cfg['large_44k_v2']
            model_config.download_if_needed()
            #setup_eval_logging()

            scheduler_config = model_config.seq_cfg
            model: MMAudio = get_my_mmaudio(model_config.model_name).to(device, dtype).eval()
            model.load_weights(torch.load(model_config.model_path, map_location=device, weights_only=True))
            print(f'Loaded weights from {model_config.model_path}')

            feature_extractor = FeaturesUtils(
                tod_vae_ckpt=model_config.vae_path,
                synchformer_ckpt=model_config.synchformer_ckpt,
                enable_conditions=True,
                mode=model_config.mode,
                bigvgan_vocoder_ckpt=model_config.bigvgan_16k_path,
                need_vae_encoder=False
            ).to(device, dtype)#.eval()


        # Deadend
        else:
            print("Audio model not found.")
            self.report({"INFO"}, "Audio model not found.")
            return {"CANCELLED"}

        old_duration = duration = scene.audio_length_in_f

        # Main loop Audio
        for i in range(scene.movie_num_batch):
            start_time = timer()
            strip = scene.sequence_editor.active_strip
            if strip and input == "input_strips" and duration == -1:
                duration = scene.audio_length_in_f = (
                    strip.frame_final_duration + 1
                )
                print("Input duration: "+str(strip.frame_final_duration))

                audio_length_in_s = duration = duration / (
                    scene.render.fps / scene.render.fps_base
                )
#
            else:
                duration = scene.audio_length_in_f
                audio_length_in_s = duration / (
                    scene.render.fps / scene.render.fps_base
                )
#                print("No input strip found!")
#                return {"CANCELLED"}

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
                if input != "input_strips":
                    empty_channel = find_first_empty_channel(
                        scene.frame_current,
                        (scene.movie_num_batch * (len(prompt) * 4))
                        + scene.frame_current,
                    )
                else:
                    empty_channel = find_first_empty_channel(
                        active_strip.frame_final_start,
                        (duration
                        + scene.frame_current)
                    )
                start_frame = scene.frame_current

            # Get seed
            seed = context.scene.movie_num_seed
            seed = (
                seed
                if not context.scene.movie_use_random
                else random.randint(-2147483647, 2147483647)
            )
            print("Seed: " + str(seed))
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

            # Stable Open Audio
            if addon_prefs.audio_model_card == "stabilityai/stable-audio-open-1.0":
                import random

                print("Generate: Stable Open Audio")
                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 999999)
                )
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
                write_wav(filename, pipe.vae.sampling_rate, output)

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
                sentences = split_and_recombine_text(
                    prompt, desired_length=120, max_length=150
                )
                pieces = []
                for sentence in sentences:
                    print("Sentence: " + sentence)
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
                audio = np.concatenate(pieces)
                filename = solve_path(clean_filename(prompt) + ".wav")
                # Write the combined audio to a file

                write_wav(filename, rate, audio.transpose())

                # resemble_enhance
                dwav, sr = torchaudio.load(filename)
                # print("sr_load " + str(sr))

                dwav = dwav.mean(dim=0)
                # transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
                # dwav = transform(dwav)
                #                dwav = audio
                # sr = rate

                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                #                wav1, new_sr = denoise(dwav, sr, device)

                wav2, new_sr = enhance(
                    dwav=dwav,
                    sr=sr,
                    device=device,
                    nfe=64,
                    chunk_seconds=10,
                    chunks_overlap=1,
                    solver="midpoint",
                    lambd=0.1,
                    tau=0.5,
                )
                # print("sr_save " + str(new_sr))
                # wav1 = wav1.cpu().numpy()

                wav2 = wav2.cpu().numpy()
                # Write the combined audio to a file

                write_wav(filename, new_sr, wav2)

            # WhisperSpeech
            elif addon_prefs.audio_model_card == "WhisperSpeech":
                prompt = context.scene.generate_movie_prompt
                prompt = prompt.replace("\n", " ").strip()
                filename = solve_path(clean_filename(prompt) + ".wav")
                if scene.audio_path:
                    speaker = bpy.path.abspath(scene.audio_path)
                else:
                    speaker = None

                pipe.generate_to_file(
                    filename,
                    prompt,
                    speaker=speaker,
                    lang="en",
                    cps=int(scene.audio_speed),
                )

            #F5-TTS
            elif addon_prefs.audio_model_card == "SWivid/F5-TTS":

                if scene.audio_path:
                    speaker = bpy.path.abspath(scene.audio_path)
                else:
                    speaker = None
                    print("No speaker file found. Cancelled...")
                    return {"CANCELLED"}
                print("Speaker: "+speaker)

                # Preprocess reference audio and text (blocking, happens in the thread)
                ref_audio_processed = None
                ref_text_used = ""#ref_text.strip() # Use stripped ref text

                # Ensure vocoder is loaded
                if vocoder is None:
                     print("Loading vocoder...")
                     try:
                         vocoder = load_vocoder()
                         print("Vocoder loaded successfully.")
                     except Exception as e:
                         raise RuntimeError(f"Vocoder failed to load: {e}") from e

                # Convert cached_path result to string path
                ckpt_path_str = str(cached_path(DEFAULT_F5TTS_CFG[0]))

                # load_model should handle device placement
                F5TTS_model_cfg_dict = json.loads(DEFAULT_F5TTS_CFG[2])

                pipe = load_model(DiT, F5TTS_model_cfg_dict, ckpt_path_str)
                print("F5-TTS model loaded.")

                prompt = context.scene.generate_movie_prompt
                prompt = prompt.replace("\n", " ").strip()
                filename = solve_path(clean_filename(prompt) + ".wav")

                ref_audio_processed, ref_text_used = preprocess_ref_audio_text(
                    speaker, # Pass the path string
                    ref_text_used, # Pass the stripped ref text
                    show_info=print, # Use print instead of gr.Info
                )

                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 2147483647)
                )

                torch.manual_seed(seed)
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed

                final_wave = None
                final_sample_rate = None

                final_wave, final_sample_rate, _ = infer_process(
                    ref_audio_processed, # Pass the processed tuple from preprocess
                    ref_text_used,       # Pass the potentially auto-transcribed text from preprocess
                    prompt,   # Pass the stripped generation text
                    pipe,
                    vocoder, # vocoder is loaded globally, accessed here
                    cross_fade_duration=0.15,#cross_fade_duration,
                    nfe_step=movie_num_inference_steps,
                    speed=(scene.audio_speed_tts), #speed,
                    show_info=print,
                    #progress=None, # <--- FIX: Pass None to disable f5-tts internal tqdm progress
                )
                filename = solve_path(clean_filename(str(seed) + "_" + prompt) + ".wav")

                # Remove silence
                if scene.remove_silence and final_wave is not None and len(final_wave) > 0:
                    print("Attempting to remove silence...")
                    tmp_wav_path = None
                    try:
                        # Use a more robust way to ensure the temp file exists and is closed before remove_silence_for_generated_wav opens it
                        tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
                        os.close(tmp_fd) # Close the file descriptor immediately

                        sf.write(tmp_wav_path, final_wave, final_sample_rate)

                        remove_silence_for_generated_wav(tmp_wav_path)

                        # Reload the potentially modified audio from the temporary file
                        loaded_audio, loaded_sr = torchaudio.load(tmp_wav_path)
                        final_wave = loaded_audio.squeeze().cpu().numpy() # Ensure 1D numpy array
                        final_sample_rate = loaded_sr # Update sample rate if it changed (unlikely but safe)

                        print("Silence removal successful.")
                    except Exception as e:
                        #_status_callback(f"Warning during silence removal: {e}", icon='WARNING') # Use warning icon?
                        print(f"Error during silence removal: {e}") # Print to console
                        #traceback.print_exc()
                        # Continue with the original wave if silence removal fails
                    finally:
                         # Ensure temp file is removed even if silence removal fails
                         if tmp_wav_path and os.path.exists(tmp_wav_path):
                             try:
                                 os.remove(tmp_wav_path)
                                 # print(f"Cleaned up temporary file: {tmp_wav_path}") # Optional: verbose cleanup log
                             except OSError as e:
                                 print(f"Warning: Could not remove temporary file {tmp_wav_path}: {e}")

                # Save the final audio (blocking, happens in the thread)
                # Check if final_wave is valid before attempting to save
                if final_wave is not None and len(final_wave) > 0:
                    try:
                        output_audio_path = filename = solve_path(clean_filename(str(seed) + "_" + prompt) + ".wav")
                        # sf.write expects numpy array, ensure correct dtype
                        sf.write(output_audio_path, final_wave.astype(np.float32), final_sample_rate)
                        #print(f"Synthesized audio saved to {output_audio_path}")
                        #result = (output_audio_path, ref_text_used, used_seed) # Set result tuple

                    except Exception as e:
                        # Catch save errors or directory creation errors
                        exception = e # Store exception
                        print(f"Error saving output audio to {output_audio_path}: {e}") # Print to console
                else:
                     # No audio data to save
                     exception = RuntimeError("Synthesis failed, no audio data generated.")
                     print("Synthesis failed, no audio data to save.")

            # Chatterbox
            elif (
                addon_prefs.audio_model_card == "Chatterbox"
            ):
                output_audio_path = filename = solve_path(clean_filename(str(seed) + "_" + prompt) + ".wav")
                strip = scene.sequence_editor.active_strip
                if scene.audio_path:
                    speaker = speaker = bpy.path.abspath(scene.audio_path)
                else:
                    speaker = None
                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 2147483647)
                )
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed)
                pace = scene.chat_pace
                exaggeration = scene.chat_exaggeration
                temperature = scene.chat_temperature

                if input and input == "input_strips" and strip.type == "SOUND": # Voice clone
                    AUDIO_PROMPT_PATH = os.path.join(bpy.path.abspath(strip.sound.filepath))#, strip.elements[0].filename)
                    print("Voice cloning: "+strip.sound.name)
                    model = ChatterboxVC.from_pretrained(device)
                    wav = model.generate(audio=AUDIO_PROMPT_PATH,target_voice_path=speaker)
                    ta.save(output_audio_path, wav, model.sr)
                else: # Text-to-Speech
                    try:
                        print(f"Starting Text-to-Speech for prompt: '{prompt}'")
                        
                        # TTS-specific parameters
                        pace = scene.chat_pace
                        exaggeration = scene.chat_exaggeration
                        temperature = scene.chat_temperature

                        # Load the model once
                        model = ChatterboxTTS.from_pretrained(device=device)
                        
                        # Split the full prompt into smaller, manageable chunks
                        chunks = split_text_for_tts(prompt)
                        
                        # List to hold the 1D audio waveform of each chunk
                        all_wav_chunks = [] 
                        
                        for i, chunk_text in enumerate(chunks):
                            if not chunk_text.strip():
                                continue
                                
                            print(f"Synthesizing chunk {i+1}/{len(chunks)}: '{chunk_text}...'")
                            try:
                                # Generate audio for the current chunk
                                wav_chunk_tensor = model.generate(
                                    chunk_text, 
                                    audio_prompt_path=speaker, # Use the path as the API requires
                                    exaggeration=exaggeration,
                                    cfg_weight=pace,
                                    temperature=temperature
                                )
                                all_wav_chunks.append(wav_chunk_tensor.flatten())

                            except Exception as e:
                                print(f"Error synthesizing chunk {i+1}: {e}")

                        # Concatenate and Save
                        if all_wav_chunks:
                            # Concatenate the list of 1D tensors along their only dimension (dim=0)
                            final_wav = torch.cat(all_wav_chunks, dim=0)
                            
                            # torchaudio.save needs a 2D tensor: (channels, length)
                            # Add the "channel" dimension back in with .unsqueeze(0)
                            ta.save(output_audio_path, final_wav.unsqueeze(0), model.sr)
                            
                            print(f"Successfully saved combined audio to {output_audio_path}")
                        else:
                            print("No audio was generated. The prompt might have been empty or resulted in errors.")

                    except Exception as e:
                        print(f"An unexpected error occurred in the TTS process: {e}")

            # Musicgen.
            elif (
                addon_prefs.audio_model_card == "facebook/musicgen-stereo-melody-large"
            ):
                print("Generate: MusicGen Stereo")
                # print("Prompt: " + prompt)
                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 999999)
                )
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed
                set_seed(seed)
                music = pipe(
                    prompt,
                    forward_params={
                        "max_new_tokens": int(min(audio_length_in_s * 50, 1503))
                    },
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

                write_wav(filename, music["sampling_rate"], music["audio"][0].T)

            # MusicLDM ZAC
            elif addon_prefs.audio_model_card == "cvssp/audioldm2-large":
                print("Generate: Audio LDM2 Large")
                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 999999)
                )
                print("Seed: " + str(seed))
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
            elif (
                addon_prefs.audio_model_card == "parler-tts/parler-tts-large-v1"
                or addon_prefs.audio_model_card == "parler-tts/parler-tts-mini-v1"
            ):
                prompt = prompt
                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 999999)
                )
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed
                description = context.scene.parler_direction_prompt
                input_ids = tokenizer(description, return_tensors="pt").input_ids.to(
                    gfx_device
                )
                prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    gfx_device
                )

                generation = pipe.generate(
                    input_ids=input_ids, prompt_input_ids=prompt_input_ids
                )
                audio_arr = generation.cpu().numpy().squeeze()
                filename = solve_path(str(seed) + "_" + prompt + ".wav")
                write_wav(filename, pipe.config.sampling_rate, audio_arr)

            #MMAudio
            if addon_prefs.audio_model_card == "MMAudio":

                scheduler = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=movie_num_inference_steps)

                seed = context.scene.movie_num_seed
                seed = (
                    seed
                    if not context.scene.movie_use_random
                    else random.randint(0, 999999)
                )
                print("Seed: " + str(seed))
                context.scene.movie_num_seed = seed

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
                generated_audio = None
                if scene.movie_path:
                    print("Process: Video to audio")
                    if not os.path.isfile(scene.movie_path):
                        print("No file found.")
                        return {"CANCELLED"}
                    video_path = scene.movie_path
                    video_data = load_video(video_path, audio_length_in_s)#duration)
                    print("Video Path: "+video_path)
                    print("audio_length_in_s: "+str(audio_length_in_s))
                    video_frames = video_data.clip_frames.unsqueeze(0)
                    sync_frames = video_data.sync_frames.unsqueeze(0)
                    duration = video_data.duration_sec
                    scheduler_config.duration = video_data.duration_sec
                    model.update_seq_lengths(scheduler_config.latent_seq_len, scheduler_config.clip_seq_len, scheduler_config.sync_seq_len)
                    with torch.no_grad():
                        generated_audio = generate(
                            video_frames, sync_frames, [prompt],
                            negative_text=[negative_prompt],
                            feature_utils=feature_extractor,
                            net=model, fm=scheduler, rng=generator,
                            cfg_strength=movie_num_guidance,
                        )

                elif scene.image_path:
                    print("Process: Image to audio")
                    strip = scene.sequence_editor.active_strip
                    img_path = os.path.join(bpy.path.abspath(strip.directory), strip.elements[0].filename)
                    if not os.path.isfile(img_path):
                        print("No file found.")
                        return {"CANCELLED"}
                        image = load_image(img_path)
                    video_path = img_path
                    image_data = load_image(scene.image_path)
                    clip_frames = image_data.clip_frames
                    sync_frames = image_data.sync_frames
                    clip_frames = clip_frames.unsqueeze(0)
                    sync_frames = sync_frames.unsqueeze(0)
                    blender_fps_num = bpy.context.scene.render.fps
                    blender_fps_den = bpy.context.scene.render.fps_base

                    if blender_fps_den == 0: # Avoid division by zero
                        effective_fps_float = 0.0
                    else:
                        effective_fps_float = blender_fps_num / blender_fps_den

                    # Create Fraction from a single float.
                    # limit_denominator is useful to get common video fractions like 30000/1001.
                    if effective_fps_float == 0.0:
                        fps_as_fraction = Fraction(24, 1) # Default to a sensible FPS if calculated is 0
                    else:
                        fps_as_fraction = Fraction(effective_fps_float).limit_denominator(1001)

                    video_data = VideoInfo.from_image_info(image_data, audio_length_in_s, fps=fps_as_fraction)
                    scheduler_config.duration = audio_length_in_s
                    model.update_seq_lengths(scheduler_config.latent_seq_len, scheduler_config.clip_seq_len, scheduler_config.sync_seq_len)
                    with torch.no_grad():
                        generated_audio = generate(clip_frames,
                                          sync_frames, [prompt],
                                          negative_text=[negative_prompt],
                                          feature_utils=feature_extractor,
                                          net=model, fm=scheduler, rng=generator,
                                          cfg_strength=movie_num_guidance,
                                          image_input=True)

                elif strip.type != "MOVIE" and strip.type != "IMAGE":
                    if scene.audio_length_in_f == -1:
                        scene.audio_length_in_f = 25
                    clip_frames = sync_frames = None
                    scheduler_config.duration = audio_length_in_s
                    model.update_seq_lengths(scheduler_config.latent_seq_len, scheduler_config.clip_seq_len, scheduler_config.sync_seq_len)
                    with torch.no_grad():
                        generation = generate(clip_frames,
                                          sync_frames, [prompt],
                                          negative_text=[negative_prompt],
                                          feature_utils=feature_extractor,
                                          net=model, fm=scheduler, rng=generator,
                                          cfg_strength=movie_num_guidance,
                                          image_input=True)
                                          
                    audio_output = generation.float().cpu()[0]
                    target_sr = int((context.preferences.system.audio_sample_rate).split('_')[1])
                    filename = solve_path(str(seed) + "_" + prompt + ".wav")
                    torchaudio.save(filename, audio_output, target_sr)

                if generated_audio != None:
                    audio_output = generated_audio.float().cpu()[0]
                    target_sr = int((context.preferences.system.audio_sample_rate).split('_')[1])
                    filename = video_output_path = solve_path(str(seed) + "_" + prompt + ".mp4")
                    make_video(video_data, video_output_path, audio_output, sampling_rate=target_sr)
                    print(f'Saved video to {video_output_path}')

            # Add Audio Strip
            filepath = filename
            if os.path.isfile(filepath):

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

                # Redraw UI to display the new strip. Remove this if Blender crashes:
                # https://docs.blender.org/api/current/info_gotcha.html#can-i-redraw-during-script-execution
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            else:
                print("No resulting file found!")
            print_elapsed_time(start_time)
            if old_duration == -1 and input == "input_strips":
                scene.audio_length_in_f = scene.generate_movie_frames = -1
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


#def get_depth_map(image):
#    from PIL import Image
#    from transformers import SiglipImageProcessor, SiglipVisionModel
#    feature_extractor = SiglipImageProcessor.from_pretrained(
#        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
#    )
#    image_encoder = SiglipVisionModel.from_pretrained(
#        "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
#    )

#    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
#        "hunyuanvideo-community/HunyuanVideo",
#        transformer=transformer,
#        feature_extractor=feature_extractor,
#        image_encoder=image_encoder,
#        torch_dtype=torch.float16,
#    )
#    feature_extractor = SiglipImageProcessor.from_pretrained(
#        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
#    )
#    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
#    with torch.no_grad(), torch.autocast("cuda"):
#        depth_map = depth_estimator(image).predicted_depth
#    depth_map = torch.nn.functional.interpolate(
#        depth_map.unsqueeze(1),
#        size=(1024, 1024),
#        mode="bicubic",
#        align_corners=False,
#    )
#    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
#    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
#    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#    image = torch.cat([depth_map] * 3, dim=1)
#    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
#    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
#    return image


class IPAdapterFaceProperties(bpy.types.PropertyGroup):
    files_to_import: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)


class IPAdapterFaceFileBrowserOperator(Operator):
    bl_idname = "ip_adapter_face.file_browser"
    bl_label = "Open IP Adapter Face File Browser"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    import_as_folder: bpy.props.BoolProperty(name="Import as Folder", default=False)

    def execute(self, context):
        valid_image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
            ".gif",
            ".hdr",
        }
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
                            print(
                                "Found image file in folder:",
                                os.path.basename(file_path),
                            )
                            new_file = files_to_import.add()
                            # new_file.name = os.path.basename(self.filepath)

                            new_file.path = os.path.abspath(self.filepath)
                scene.ip_adapter_face_folder = os.path.abspath(
                    os.path.dirname(self.filepath)
                )
                self.report(
                    {"INFO"}, f"{len(files_to_import)} image files found in folder."
                )
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
        valid_image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
            ".gif",
            ".hdr",
        }
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
                            print(
                                "Found image file in folder:",
                                os.path.basename(file_path),
                            )
                            new_file = files_to_import.add()
                            new_file.name = os.path.basename(file_path)
                            new_file.path = os.path.abspath(file_path)
                scene.ip_adapter_style_folder = os.path.abspath(self.filepath)
                self.report(
                    {"INFO"}, f"{len(files_to_import)} image files found in folder."
                )
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

    if os.path.isfile(folder_path) and folder_path.lower().endswith(
        (".png", ".jpg", ".jpeg", ".tga", ".bmp")
    ):
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
        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tga", ".bmp")
        ):
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

        inference_parameters = None
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
        #        ):
        #            self.report({"INFO"}, "Text prompt in the Generative AI tab is empty!")
        #            return {"CANCELLED"}
        show_system_console(True)
        set_system_console_topmost(True)

        if not seq_editor:
            scene.sequence_editor_create()

        try:
#            if os_platform != "Darwin":
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import pt_to_pil
            import torch
            from diffusers.utils import load_image
            import requests
            import numpy as np
            import PIL
            import cv2
            from PIL import Image

        # from compel import Compel

        except ModuleNotFoundError as e:
            print("Dependencies needs to be installed in the add-on preferences: "+str(e.name))
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
        negative_prompt = (
            scene.generate_movie_negative_prompt
            + ", "
            + style_prompt(scene.generate_movie_prompt)[1]
            + ", nsfw, nude, nudity,"
        )
        image_x = scene.generate_movie_x
        image_y = scene.generate_movie_y
        x = scene.generate_movie_x = closest_divisible_32(image_x)
        y = scene.generate_movie_y = closest_divisible_32(image_y)
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
            and not image_model_card == "Salesforce/blipdiffusion"
            # and not image_model_card == "Corcelio/mobius"
            and not image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers"
            and not image_model_card == "adamo1139/stable-diffusion-3.5-large-ungated"
            and not image_model_card == "adamo1139/stable-diffusion-3.5-medium-ungated"
            and not image_model_card == "Vargol/ProteusV0.4"
            and not image_model_card == "ZhengPeng7/BiRefNet_HR"
            and not image_model_card == "Shitao/OmniGen-v1-diffusers"
            and not image_model_card == "Qwen/Qwen-Image-Edit-2509"
#            and (not scene.ip_adapter_face_folder and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0")
#            and (not scene.ip_adapter_style_folder and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0")
        )
        do_convert = (
            (scene.image_path or scene.movie_path)
            and not image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            and not image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            and not image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            and not image_model_card == "Salesforce/blipdiffusion"
            and not image_model_card == "ByteDance/SDXL-Lightning"
            and not image_model_card == "Vargol/ProteusV0.4"
            and not image_model_card == "ZhengPeng7/BiRefNet_HR"
            and not image_model_card == "Shitao/OmniGen-v1-diffusers"
            and not image_model_card == "Qwen/Qwen-Image-Edit-2509"
#            and (not scene.ip_adapter_face_folder and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0")
#            and (not scene.ip_adapter_style_folder and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0")
            and not do_inpaint
        )
        do_refine = scene.refine_sd and not do_convert
        if (
            do_inpaint
            or do_convert
            or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            or image_model_card == "Salesforce/blipdiffusion"
            and not scene.ip_adapter_face_folder
            and not scene.ip_adapter_style_folder
            and not image_model_card == "Shitao/OmniGen-v1-diffusers"
            and not image_model_card == "Qwen/Qwen-Image-Edit-2509"
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

        print("do_inpaint: "+str(do_inpaint))
        print("do_convert: "+str(do_convert))
        print("do_refine: "+str(do_refine))


        # LOADING MODELS
        # models for inpaint
        if do_inpaint:
            from diffusers import AutoPipelineForInpainting
            from diffusers.utils import load_image

            # clear the VRAM
            clear_cuda_cache()
            
            if image_model_card == "yuvraj108c/FLUX.1-Kontext-dev":
                print("Load Inpaint: " + image_model_card)
                import torch 
                from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
                from diffusers import FluxKontextInpaintPipeline
                from diffusers.utils import load_image

                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_nf4 = FluxTransformer2DModel.from_pretrained(
                    image_model_card,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                )

                pipe = FluxKontextInpaintPipeline.from_pretrained(
                    image_model_card,
                    transformer=model_nf4,
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                )  

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    # torch.cuda.set_per_process_memory_fraction(0.99)
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)              
                
                
            elif image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                print("Load Inpaint: " + image_model_card)
                pipe = AutoPipelineForInpainting.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )

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

                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config
                    )

                pipe.watermark = NoWatermark()
                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    # torch.cuda.set_per_process_memory_fraction(0.99)
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif (
                image_model_card == "lzyvegetable/FLUX.1-schnell"
                or image_model_card == "ChuckMcSneed/FLUX.1-dev"
                or image_model_card == "ostris/Flex.2-preview"
            ):
                print("Load Inpaint: " + image_model_card)
                from diffusers import (
                    DiffusionPipeline,
                    FluxFillPipeline,
                    FluxTransformer2DModel,
                )
                from transformers import T5EncoderModel

                orig_pipeline = DiffusionPipeline.from_pretrained(
                    image_model_card, torch_dtype=torch.bfloat16
                )

                transformer = FluxTransformer2DModel.from_pretrained(
                    "sayakpaul/FLUX.1-Fill-dev-nf4",
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                )
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    "sayakpaul/FLUX.1-Fill-dev-nf4",
                    subfolder="text_encoder_2",
                    torch_dtype=torch.bfloat16,
                )
                pipe = FluxFillPipeline.from_pipe(
                    orig_pipeline,
                    transformer=transformer,
                    text_encoder_2=text_encoder_2,
                    torch_dtype=torch.bfloat16,
                )

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_sequential_cpu_offload()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()

        # Conversion img2img/vid2img.
        elif do_convert:  # and not scene.aurasr:
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
                    "thingthatis/stable-diffusion-xl-refiner-1.0",
                    # text_encoder_2=pipe.text_encoder_2,
                    vae=vae,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )
                if gfx_device == "mps":
                    converter.to("mps")
                elif low_vram():
                    converter.enable_model_cpu_offload()
                else:
                    converter.to(gfx_device)

            elif image_model_card == "Kwai-Kolors/Kolors-diffusers":
                from diffusers import DPMSolverMultistepScheduler, KolorsImg2ImgPipeline
                from diffusers.utils import load_image

                converter = KolorsImg2ImgPipeline.from_pretrained(
                    image_model_card,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )
                converter.scheduler = DPMSolverMultistepScheduler.from_config(
                    converter.scheduler.config, use_karras_sigmas=True
                )
                if gfx_device == "mps":
                    converter.to("mps")
                elif low_vram():
                    converter.enable_model_cpu_offload()
                else:
                    converter.to(gfx_device)
            else:
                from diffusers import AutoPipelineForImage2Image
                if (
                    image_model_card
                    == "stabilityai/stable-diffusion-3-medium-diffusers"
                    or os_platform == "Darwin"
                ):  # or image_model_card == "adamo1139/stable-diffusion-3.5-large-ungated":
                    from huggingface_hub.commands.user import login

                    result = login(
                        token=addon_prefs.hugginface_token, add_to_git_credential=True
                    )
                    print(str(result))

                # FLUX MacOS
                if image_model_card == "ChuckMcSneed/FLUX.1-dev" and os_platform == "Darwin":
                    from mflux import Flux1, Config
                    converter = Flux1.from_name(
                       model_name="dev",  # "schnell" or "dev"
                       quantize=4,            # 4 or 8
                    )
                elif image_model_card == "lzyvegetable/FLUX.1-schnell" and os_platform == "Darwin":
                    from mflux import Flux1, Config
                    converter = Flux1.from_name(
                       model_name="schnell",  # "schnell" or "dev"
                       quantize=4,            # 4 or 8
                    )                
                # Win                  
                elif (
                    image_model_card == "lzyvegetable/FLUX.1-schnell"
                    or image_model_card == "ChuckMcSneed/FLUX.1-dev"
                    or image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                    or image_model_card == "kontext-community/relighting-kontext-dev-lora-v3"
                    or image_model_card == "ostris/Flex.2-preview"
                ):
                    relight = False
                    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
                    
                    if image_model_card == "yuvraj108c/FLUX.1-Kontext-dev" or image_model_card == "kontext-community/relighting-kontext-dev-lora-v3":
                        from diffusers import FluxKontextPipeline
                        
                    if image_model_card == "kontext-community/relighting-kontext-dev-lora-v3":
                        image_model_card = "yuvraj108c/FLUX.1-Kontext-dev"
                        relight = True

                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    model_nf4 = FluxTransformer2DModel.from_pretrained(
                        image_model_card,
                        subfolder="transformer",
                        quantization_config=nf4_config,
                        torch_dtype=torch.bfloat16,
                    )

                    if image_model_card == "yuvraj108c/FLUX.1-Kontext-dev":
                        converter = FluxKontextPipeline.from_pretrained(
                            image_model_card,
                            transformer=model_nf4,
                            torch_dtype=torch.bfloat16,
                            local_files_only=local_files_only,
                        )                       
                    else:
                        converter = AutoPipelineForImage2Image.from_pretrained(
                            image_model_card,
                            transformer=model_nf4,
                            torch_dtype=torch.bfloat16,
                            local_files_only=local_files_only,
                        )

                    if relight == True:
                        print("AI Relight: Loading and applying Relighting LoRA...")
                        converter.load_lora_weights(
                            "kontext-community/relighting-kontext-dev-lora-v3", 
                            weight_name="relighting-kontext-dev-lora-v3.safetensors", 
                            adapter_name="lora"
                        )
                        converter.set_adapters(["lora"], adapter_weights=[0.75])
                        image_model_card = "kontext-community/relighting-kontext-dev-lora-v3"
                                          
                    if gfx_device == "mps":
                        converter.to("mps")
                    elif low_vram():
                        converter.enable_sequential_cpu_offload()
                        #converter.enable_model_cpu_offload()
                        converter.enable_vae_slicing()
                        converter.vae.enable_tiling()
                    else:
                        converter.enable_model_cpu_offload()
                    

                # FLUX ControlNets
                elif (
                    image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora"
                ) or (image_model_card == "romanfratric234/FLUX.1-Depth-dev-lora"):
                    from diffusers import FluxControlPipeline
                    from diffusers.utils import load_image
                    if image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora":
                        pipecard = "fuliucansheng/FLUX.1-Canny-dev-diffusers"
                    else:
                        pipecard = "ChuckMcSneed/FLUX.1-dev"

                    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    model_nf4 = FluxTransformer2DModel.from_pretrained(
                        pipecard,
                        subfolder="transformer",
                        quantization_config=nf4_config,
                        torch_dtype=torch.bfloat16,
                    )
                    converter = FluxControlPipeline.from_pretrained(
                        pipecard,
                        transformer=model_nf4,
                        torch_dtype=torch.bfloat16,
                        local_files_only=local_files_only,
                    )

                    if gfx_device == "mps":
                        converter.to("mps")
                    elif low_vram():
                        #pipe.enable_sequential_cpu_offload()
                        converter.enable_model_cpu_offload()
                        converter.enable_vae_slicing()
                        converter.vae.enable_tiling()
                    else:
                        #pipe.enable_sequential_cpu_offload()
                        converter.enable_model_cpu_offload()
                        converter.enable_vae_slicing()
                        converter.vae.enable_tiling()

                    if pipecard == "ChuckMcSneed/FLUX.1-dev":
                        converter.load_lora_weights(image_model_card)

                    if image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora":
                        from controlnet_aux import CannyDetector
                        processor = CannyDetector()
                    else:
                        from image_gen_aux import DepthPreprocessor
                        processor = DepthPreprocessor.from_pretrained(
                            "LiheYoung/depth-anything-large-hf"
                        )

                # redux
                elif image_model_card == "Runware/FLUX.1-Redux-dev":
                    from transformers import SiglipImageProcessor, SiglipVisionModel
#                    feature_extractor = SiglipImageProcessor.from_pretrained(
#                        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
#                    )
#                    image_encoder = SiglipVisionModel.from_pretrained(
#                        "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
#                    )

                    from diffusers import FluxPriorReduxPipeline, FluxPipeline
                    from diffusers.utils import load_image
#                    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

#                    nf4_config = BitsAndBytesConfig(
#                        load_in_4bit=True,
#                        bnb_4bit_quant_type="nf4",
#                        bnb_4bit_compute_dtype=torch.bfloat16,
#                    )
#                    model_nf4 = FluxTransformer2DModel.from_pretrained(
#                        "ChuckMcSneed/FLUX.1-dev",
#                        subfolder="transformer",
#                        quantization_config=nf4_config,
#                        torch_dtype=torch.bfloat16,
#                    )
                    converter = FluxPipeline.from_pretrained(
                        "ChuckMcSneed/FLUX.1-dev" , 
                        text_encoder=None,
                        text_encoder_2=None,
                        torch_dtype=torch.bfloat16,
                        #transformer=model_nf4, 
                    )
                    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("Runware/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to("cuda")

#                    converter = FluxPipeline.from_pretrained(
#                        "ChuckMcSneed/FLUX.1-dev" ,
#                        feature_extractor=feature_extractor,
#                        image_encoder=image_encoder,
#                        text_encoder=None,
#                        text_encoder_2=None,
#                        torch_dtype=torch.bfloat16,
#                        transformer=model_nf4,
#                    )

                    if gfx_device == "mps":
                        converter.to("mps")
                    elif low_vram():
                        converter.enable_model_cpu_offload()
                        converter.enable_vae_slicing()
                        converter.vae.enable_tiling()
                    else:
                        converter.enable_sequential_cpu_offload()
                        #converter.enable_model_cpu_offload() # too slow
                        converter.enable_vae_slicing()
                        converter.vae.enable_tiling()
                        
                elif image_model_card == "Qwen/Qwen-Image":
                    print("Load: Qwen-Image - img2img")

                    from diffusers.utils import load_image
                    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                    from diffusers import QwenImageImg2ImgPipeline, QwenImageTransformer2DModel

                    model_id = "Qwen/Qwen-Image"
                    torch_dtype = torch.bfloat16
                    device = gfx_device

                    quantization_config_transformer = DiffusersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                    )

                    quantization_config_text_encoder = TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

                    transformer = QwenImageTransformer2DModel.from_pretrained(
                        model_id,
                        subfolder="transformer",
                        quantization_config=quantization_config_transformer,
                        torch_dtype=torch_dtype,
                    )
                    transformer = transformer.to("cpu")

                    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        subfolder="text_encoder",
                        quantization_config=quantization_config_text_encoder,
                        torch_dtype=torch_dtype,
                    )
                    text_encoder = text_encoder.to("cpu")

                    converter = QwenImageImg2ImgPipeline.from_pretrained(
                        model_id,
                        transformer=transformer,
                        text_encoder=text_encoder,
                        torch_dtype=torch_dtype
                    )

                    if gfx_device == "mps":
                        converter.to("mps")
                    elif low_vram():
                        converter.enable_model_cpu_offload()
                        converter.enable_vae_slicing()
                        converter.vae.enable_tiling()
                    else:
                        converter.enable_model_cpu_offload()   
                    
                else:
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
                            print(
                                "The "
                                + image_model_card
                                + " model does not work for a image to image pipeline!"
                            )
                            return {"CANCELLED"}
                    if gfx_device == "mps":
                        converter.to("mps")
                    elif low_vram():
                        converter.enable_model_cpu_offload()
                    else:
                        converter.to(gfx_device)

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

                    if gfx_device == "mps":
                        converter.to("mps")
                    elif low_vram():
                        converter.enable_model_cpu_offload()
                    else:
                        converter.to(gfx_device)

#            elif: # depth
#                from transformers import DPTFeatureExtractor, DPTForDepthEstimation
#                from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
#                from diffusers.utils import load_image

#                depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
#                feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
#                controlnet = ControlNetModel.from_pretrained(
#                    "diffusers/controlnet-depth-sdxl-1.0-small",
#                    variant="fp16",
#                    use_safetensors=True,
#                    torch_dtype=torch.float16,
#                ).to(gfx_device)
#                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
#                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
#                    "stabilityai/stable-diffusion-xl-base-1.0",
#                    controlnet=controlnet,
#                    vae=vae,
#                    variant="fp16",
#                    use_safetensors=True,
#                    torch_dtype=torch.float16,
#                ).to(gfx_device)
#                pipe.enable_model_cpu_offload()

        # Canny & Illusion
        elif image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small":
            if image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small":
                print("Load: Canny")
            else:
                print("Load: Illusion")
            from diffusers import (
                ControlNetModel,
                StableDiffusionXLControlNetPipeline,
                AutoencoderKL,
            )

            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0-small",
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only,
            )
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            )
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

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
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

            from diffusers import (
                ControlNetModel,
                StableDiffusionXLControlNetPipeline,
                AutoencoderKL,
            )
            from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
            from controlnet_aux import OpenposeDetector
            from PIL import Image
            import torch
            import numpy as np

            # import cv2

            controlnet_conditioning_scale = 1.0

            eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
            )

            controlnet = ControlNetModel.from_pretrained(
                "xinsir/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
            )

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            )

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

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Scribble
        elif image_model_card == "xinsir/controlnet-scribble-sdxl-1.0":
            # https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0 #use this instead

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

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            )

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
                eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
                )

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Remove Background
        elif image_model_card == "ZhengPeng7/BiRefNet_HR":
            print("Load: Remove Background")

            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms
            from PIL import Image, ImageFilter
            import torch

            pipe = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_HR", trust_remote_code=True
            )
            
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # HunyuanDiT
        elif (
            do_convert == False
            and image_model_card == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
        ):
            from diffusers import HunyuanDiTPipeline

            pipe = HunyuanDiTPipeline.from_pretrained(
                image_model_card, torch_dtype=torch.float16
            )
            pipe = pipe.to(gfx_device)

        # SD3 Stable Diffusion 3
        elif (
            image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers"
        ):
            print("Load: Stable Diffusion 3 Model")
            import torch
            from huggingface_hub.commands.user import login

            result = login(
                token=addon_prefs.hugginface_token, add_to_git_credential=True
            )
            print(str(result))
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(
                image_model_card,
                torch_dtype=torch.float16,
            )

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # SD3 Stable Diffusion 3
        elif (
            image_model_card == "adamo1139/stable-diffusion-3.5-medium-ungated"
        ):
            print("Load: Stable Diffusion 3.5 Medium Model")
            from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
            from diffusers import StableDiffusion3Pipeline
            import torch

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                image_model_card,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16,
            )

            pipe = StableDiffusion3Pipeline.from_pretrained(
                image_model_card, transformer=model_nf4, torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()

        # SD3.5 Stable Diffusion 3.5
        elif image_model_card == "adamo1139/stable-diffusion-3.5-large-ungated":
            print("Load: Stable Diffusion 3.5 large Model")
            from huggingface_hub.commands.user import login

            result = login(
                token=addon_prefs.hugginface_token, add_to_git_credential=True
            )
            print(str(result))

            import torch

            if not do_inpaint and not enabled_items and not do_convert:
                from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
                from diffusers import StableDiffusion3Pipeline

                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_nf4 = SD3Transformer2DModel.from_pretrained(
                    image_model_card,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                )

                pipe = StableDiffusion3Pipeline.from_pretrained(
                    image_model_card,
                    transformer=model_nf4,
                    torch_dtype=torch.bfloat16,
                )
                # pipe.enable_model_cpu_offload()
            else:
                from diffusers import StableDiffusion3Pipeline

                pipe = StableDiffusion3Pipeline.from_pretrained(
                    image_model_card,
                    torch_dtype=torch.float16,
                )
            if gfx_device == "mps":
                pipe.to("mps")
            else:
                pipe.enable_model_cpu_offload()

        # FLUX MACOS
        elif image_model_card == "ChuckMcSneed/FLUX.1-dev" and os_platform == "Darwin":
            from huggingface_hub.commands.user import login

            result = login(
                token=addon_prefs.hugginface_token, add_to_git_credential=True
            )
            print(str(result))
            from mflux import Flux1, Config
            pipe = Flux1.from_name(
               model_name="dev",  # "schnell" or "dev"
               quantize=4,            # 4 or 8
            )
        elif image_model_card == "lzyvegetable/FLUX.1-schnell" and os_platform == "Darwin":
            from huggingface_hub.commands.user import login

            result = login(
                token=addon_prefs.hugginface_token, add_to_git_credential=True
            )
            print(str(result))
            from mflux import Flux1, Config
            pipe = Flux1.from_name(
               model_name="schnell",  # "schnell" or "dev"
               quantize=4,            # 4 or 8
            )  

        # Flux
        elif (
            image_model_card == "lzyvegetable/FLUX.1-schnell"
            or image_model_card == "ChuckMcSneed/FLUX.1-dev"
        ):
            print("Load: Flux Model")
            clear_cuda_cache()
            import torch
            #from diffusers import FluxPipeline
            sys.path.append(os.path.dirname(__file__))
            #from pipelines.pipeline_flux_de_distill import FluxPipeline

            if not do_inpaint and not enabled_items and not do_convert:
                sys.path.append(os.path.dirname(__file__))
                from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
                #print("De-destilled")

                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_nf4 = FluxTransformer2DModel.from_pretrained(
                    image_model_card,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                )
#                model_nf4 = FluxTransformer2DModel.from_pretrained(
#                    "InstantX/flux-dev-de-distill-diffusers",
#                    quantization_config=nf4_config,
#                    torch_dtype=torch.bfloat16
#                )

                pipe = FluxPipeline.from_pretrained(
                    image_model_card, transformer=model_nf4, torch_dtype=torch.bfloat16
                )

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    #pipe.enable_sequential_cpu_offload()
                    pipe.enable_model_cpu_offload()
                    pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()
            else:  # LoRA + img2img
                from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline

                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_nf4 = FluxTransformer2DModel.from_pretrained(
                    image_model_card,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                )

                pipe = FluxPipeline.from_pretrained(
                    image_model_card, transformer=model_nf4, torch_dtype=torch.bfloat16
                )

                # pipe = FluxPipeline.from_pretrained(image_model_card, torch_dtype=torch.bfloat16)

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                    pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()

        # FLUX Kontext
        elif image_model_card == "yuvraj108c/FLUX.1-Kontext-dev":
            from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
            from diffusers import FluxKontextPipeline

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_nf4 = FluxTransformer2DModel.from_pretrained(
                image_model_card,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16,
            )

            converter = FluxKontextPipeline.from_pretrained(
                image_model_card,
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
            )                       

            if gfx_device == "mps":
                converter.to("mps")
            elif low_vram():
                converter.enable_sequential_cpu_offload()
                #converter.enable_model_cpu_offload()
                converter.enable_vae_slicing()
                converter.vae.enable_tiling()
            else:
                converter.enable_model_cpu_offload()

        # FLEX
        elif image_model_card == "ostris/Flex.2-preview":
            print("Load: Flex Model")
            clear_cuda_cache()

            if not do_inpaint and not enabled_items and not do_convert:
                import torch
                #image_model_card = "ostris/Flex.1-alpha"
                image_model_card = "ostris/Flex.2-preview"

                from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
                sys.path.append(os.path.dirname(__file__))

                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
#                    model_nf4 = FluxTransformer2DModel.from_pretrained(
#                        image_model_card,
#                        subfolder="transformer",
#                        quantization_config=nf4_config,
#                        torch_dtype=torch.bfloat16,
#                    )
                model_nf4 = FluxTransformer2DModel.from_pretrained(
                    "ChuckMcSneed/FLUX.1-dev",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16
                )
                #flex_pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipelines", "flex_pipeline.py")
                #print(flex_pipeline_path)
                pipe = FluxPipeline.from_pretrained(
                    image_model_card,
                    #custom_pipeline=flex_pipeline_path,
                    #trust_remote_code=True,
                    transformer=model_nf4,
                    torch_dtype=torch.bfloat16,
                )

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                    pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()


        # Qwen-Image
        elif image_model_card == "Qwen/Qwen-Image":
                clear_cuda_cache()
                
                if not do_inpaint and not do_convert:
                    print("Load: Qwen-Image")

                    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
                    from transformers import Qwen2_5_VLForConditionalGeneration

                    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel


                    model_id = "Qwen/Qwen-Image"
                    torch_dtype = torch.bfloat16
                    device = gfx_device

                    quantization_config = DiffusersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                    )

                    transformer = QwenImageTransformer2DModel.from_pretrained(
                        model_id,
                        subfolder="transformer",
                        quantization_config=quantization_config,
                        torch_dtype=torch_dtype,
                    )
                    transformer = transformer.to("cpu")

                    quantization_config = TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

                    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        subfolder="text_encoder",
                        quantization_config=quantization_config,
                        torch_dtype=torch_dtype,
                    )
                    text_encoder = text_encoder.to("cpu")

                    pipe = QwenImagePipeline.from_pretrained(
                        model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
                    )
                    
                else:
                    print("Load: Qwen-Image - img2img")

                    from diffusers.utils import load_image
                    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                    from diffusers import QwenImageImg2ImgPipeline, QwenImageTransformer2DModel

                    model_id = "Qwen/Qwen-Image"
                    torch_dtype = torch.bfloat16
                    device = gfx_device

                    quantization_config_transformer = DiffusersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                    )

                    quantization_config_text_encoder = TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

                    transformer = QwenImageTransformer2DModel.from_pretrained(
                        model_id,
                        subfolder="transformer",
                        quantization_config=quantization_config_transformer,
                        torch_dtype=torch_dtype,
                    )
                    transformer = transformer.to("cpu")

                    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        subfolder="text_encoder",
                        quantization_config=quantization_config_text_encoder,
                        torch_dtype=torch_dtype,
                    )
                    text_encoder = text_encoder.to("cpu")

                    pipe = QwenImageImg2ImgPipeline.from_pretrained(
                        model_id,
                        transformer=transformer,
                        text_encoder=text_encoder,
                        torch_dtype=torch_dtype
                    )

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                    pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()
                else:
                    pipe.enable_model_cpu_offload()                   
                    #pipe.to(device)                 
        
        # Chroma
        elif image_model_card == "lodestones/Chroma":

            if not do_inpaint and not enabled_items and not do_convert:
                import torch
                from diffusers import BitsAndBytesConfig, ChromaTransformer2DModel, ChromaPipeline
                from diffusers.quantizers import PipelineQuantizationConfig

                if gfx_device == "mps" or low_vram():
                    print("Quant: 4-bit")

                    dtype = torch.bfloat16

                    repo_id = "imnotednamode/Chroma-v36-dc-diffusers"

                    pipeline_quant_config = PipelineQuantizationConfig(
                        quant_backend="bitsandbytes_4bit",
                        quant_kwargs={
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_compute_dtype": dtype,
                            "llm_int8_skip_modules": ["distilled_guidance_layer"],
                        },
                        components_to_quantize=["transformer", "text_encoder"],
                    )

                    pipe = ChromaPipeline.from_pretrained(
                        "imnotednamode/Chroma-v36-dc-diffusers",
                        quantization_config=pipeline_quant_config,
                        torch_dtype=dtype,
                    )

#                    from transformers import T5EncoderModel
#                    bfl_repo = "ChuckMcSneed/FLUX.1-dev"
#                    dtype = torch.bfloat16

#                    nf4_config = BitsAndBytesConfig(
#                        load_in_4bit=True,
#                        bnb_4bit_quant_type="nf4",
#                        bnb_4bit_compute_dtype=torch.bfloat16,
#                    )

#                    transformer = ChromaTransformer2DModel.from_single_file("https://huggingface.co/lodestones/Chroma/blob/main/chroma-unlocked-v35.safetensors", quantization_config=nf4_config, torch_dtype=dtype)

#                    text_encoder = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
#                    tokenizer = T5Tokenizer.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)

#                    pipe = ChromaPipeline.from_pretrained(bfl_repo, transformer=transformer, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=dtype)

                    if gfx_device == "mps":
                        pipe.to("mps")
                    elif low_vram():
                        pipe.enable_model_cpu_offload()
                        pipe.enable_vae_slicing()
                        pipe.vae.enable_tiling()
                else:
                    print("Quant: 8-bit")
                    dtype = torch.bfloat16
                    pipe = ChromaPipeline.from_pretrained(
                        "imnotednamode/Chroma-v36-dc-diffusers",
                        quantization_config=PipelineQuantizationConfig(
                            quant_backend="bitsandbytes_8bit",
                            quant_kwargs={"load_in_8bit": True},
                            components_to_quantize=["transformer", "text_encoder_2"]
                        ),
                        torch_dtype=dtype,
                    )
                    pipe.to("cuda")
            else:
                print("Inpaint, LoRA and img2img is not supported for Chroma!")

        # Fluently-XL
        elif image_model_card == "fluently/Fluently-XL-Final":
            from diffusers import DiffusionPipeline, DDIMScheduler

            pipe = DiffusionPipeline.from_pretrained(
                image_model_card,
                torch_dtype=torch.float16,
                scheduler=DDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                ),
            )

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # Shuttle-Jaguar # needs a quantinized version
        elif image_model_card == "shuttleai/shuttle-jaguar":
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                image_model_card,
                torch_dtype=torch.float16,
            )

            # pipe.to("cuda")

            pipe.enable_sequential_cpu_offload()
            # pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.vae.enable_tiling()
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(
                pipe.transformer, mode="max-autotune", fullgraph=True
            )

        elif image_model_card == "Alpha-VLLM/Lumina-Image-2.0":
            from diffusers import Lumina2Pipeline

            pipe = Lumina2Pipeline.from_pretrained(
                "Alpha-VLLM/Lumina-Image-2.0", torch_dtype=torch.bfloat16
            )

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
                #pipe.enable_sequential_cpu_offload()
                pipe.vae.enable_tiling()
            else:
                # pipe.enable_sequential_cpu_offload()
                # pipe.vae.enable_tiling()
                pipe.enable_model_cpu_offload()
        elif image_model_card == "THUDM/CogView4-6B":
            from diffusers import CogView4Pipeline
            pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                #pipe.enable_sequential_cpu_offload()
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_tiling()
            else:
                # pipe.enable_sequential_cpu_offload()
                # pipe.vae.enable_tiling()
                pipe.enable_model_cpu_offload()

        elif image_model_card == "Efficient-Large-Model/Sana_1600M_1024px_diffusers":
            from diffusers import (
                BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
                SanaTransformer2DModel,
                SanaPipeline,
            )
            from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder_8bit = AutoModel.from_pretrained(
                "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                subfolder="text_encoder",
                quantization_config=quant_config,
                torch_dtype=torch.float16,
            )

            quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
            transformer_8bit = SanaTransformer2DModel.from_pretrained(
                "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.float16,
            )

            pipe = SanaPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                text_encoder=text_encoder_8bit,
                transformer=transformer_8bit,
                torch_dtype=torch.float16,
                device_map="balanced",
                low_cpu_mem_usage=True,
            )

        # OmniGen
        elif image_model_card == "Shitao/OmniGen-v1-diffusers":
            from diffusers import OmniGenPipeline

            pipe = OmniGenPipeline.from_pretrained(
                "Shitao/OmniGen-v1-diffusers", torch_dtype=torch.bfloat16
            )

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_sequential_cpu_offload()
                pipe.vae.enable_tiling()
            else:
                # pipe.enable_sequential_cpu_offload()
                # pipe.vae.enable_tiling()
                pipe.enable_model_cpu_offload()
                
        # Qwen Multi-image
        elif image_model_card == "Qwen/Qwen-Image-Edit-2509":
            clear_cuda_cache() 

            print("Load: Qwen-Image-Edit-2509")

            # Import necessary classes for quantization and model components
            from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
            from transformers import Qwen2_5_VLForConditionalGeneration
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

            # Define model ID, data type, and device
            model_id = "Qwen/Qwen-Image-Edit-2509"
            torch_dtype = torch.bfloat16
            device = gfx_device

            # Configure 4-bit quantization for the transformer model
            quantization_config_diffusers = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            )

            # Load the transformer model with quantization and move to CPU initially
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quantization_config_diffusers,
                torch_dtype=torch_dtype,
            )
            transformer = transformer.to("cpu")

            # Configure 4-bit quantization for the text encoder
            quantization_config_transformers = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            # Load the text encoder with quantization and move to CPU initially
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                subfolder="text_encoder",
                quantization_config=quantization_config_transformers,
                torch_dtype=torch_dtype,
            )
            text_encoder = text_encoder.to("cpu")

            # Assemble the pipeline from the pre-loaded, quantized components
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_id,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch_dtype
            )

            print("Pipeline loaded")

            # Move the complete pipeline to the GPU for inference
            # pipeline.to(device)

            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_sequential_cpu_offload()
                pipe.vae.enable_tiling()
            else:
                #pipe.enable_sequential_cpu_offload()
                # pipe.vae.enable_tiling()
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
                    elif scene.input_strips == "input_strips" and (
                        scene.image_path or scene.movie_path
                    ):
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
                            weight_name=[
                                "ip-adapter-plus_sdxl_vit-h.safetensors",
                                "ip-adapter-plus-face_sdxl_vit-h.safetensors",
                            ],
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
                        pipe.scheduler = DDIMScheduler.from_config(
                            pipe.scheduler.config
                        )

            #                    scale = {
            #                        "down": {"block_2": [0.0, 1.0]},
            #                        "up": {"block_0": [0.0, 1.0, 0.0]},
            #                    }
            #                    pipe.set_ip_adapter_scale(scale)#[scale, scale])

            elif image_model_card == "Vargol/PixArt-Sigma_16bit":
                from diffusers import PixArtAlphaPipeline

                if scene.use_lcm:
                    pipe = PixArtAlphaPipeline.from_pretrained(
                        "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
                        torch_dtype=torch.float16,
                        local_files_only=local_files_only,
                    )
                else:
                    pipe = PixArtAlphaPipeline.from_pretrained(
                        "Vargol/PixArt-Sigma_16bit",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        local_files_only=local_files_only,
                    )

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                    pipe.vae.enable_tiling()
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
                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif image_model_card == "ByteDance/SDXL-Lightning":
                import torch
                from diffusers import (
                    StableDiffusionXLPipeline,
                    EulerAncestralDiscreteScheduler,
                    AutoencoderKL,
                )
                from huggingface_hub import hf_hub_download

                base = "stabilityai/stable-diffusion-xl-base-1.0"
                repo = "ByteDance/SDXL-Lightning"
                ckpt = "sdxl_lightning_2step_lora.safetensors"  # Use the correct ckpt for your step setting!

                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                )

                # Load model.
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    base, torch_dtype=torch.float16, vae=vae, variant="fp16"
                ).to("cuda")
                pipe.load_lora_weights(hf_hub_download(repo, ckpt))
                pipe.fuse_lora()

                # Ensure sampler uses "trailing" timesteps.
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    pipe.scheduler.config, timestep_spacing="trailing"
                )

            elif image_model_card == "Vargol/ProteusV0.4":
                from diffusers import (
                    StableDiffusionXLPipeline,
                    EulerAncestralDiscreteScheduler,
                )
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                )
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "Vargol/ProteusV0.4",
                    vae=vae,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    pipe.scheduler.config
                )
                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

            elif image_model_card == "Kwai-Kolors/Kolors-diffusers":
                import torch
                from diffusers import DPMSolverMultistepScheduler, KolorsPipeline

                pipe = KolorsPipeline.from_pretrained(
                    image_model_card,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=local_files_only,
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=True
                )
                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)
            else:
                print("Load: Auto Pipeline")
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
                elif image_model_card == "segmind/Segmind-Vega":
                    scene.movie_num_guidance = 0
                    pipe.load_lora_weights("segmind/Segmind-VegaRT")
                    pipe.fuse_lora()
            elif (
                image_model_card != "Vargol/PixArt-Sigma_16bit"
                and image_model_card != "Vargol/PixArt-Sigma_2k_16bit"
            ):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )

            if image_model_card != "Vargol/PixArt-Sigma_2k_16bit":
                pipe.watermark = NoWatermark()

                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    # torch.cuda.set_per_process_memory_fraction(0.95)  # 6 GB VRAM

                    pipe.enable_model_cpu_offload()
                    # pipe.enable_vae_slicing()
                else:
                    pipe.to(gfx_device)

        # LoRA
        if (
            (
                image_model_card == "stabilityai/stable-diffusion-xl-base-1.0"
                and ((not scene.image_path and not scene.movie_path) or do_inpaint)
            )
            or image_model_card == "stabilityai/sdxl-turbo"
            or image_model_card == "xinsir/controlnet-openpose-sdxl-1.0"
            or image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small"
            or image_model_card == "xinsir/controlnet-scribble-sdxl-1.0"
            or image_model_card == "lzyvegetable/FLUX.1-schnell"
            or image_model_card == "ChuckMcSneed/FLUX.1-dev"
            or image_model_card == "ostris/Flex.2-preview"
            or image_model_card == "Qwen/Qwen-Image-Edit-2509"
            or image_model_card == "Qwen/Qwen-Image"
#            or image_model_card == "Runware/FLUX.1-Redux-dev"
#            or image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora"
#            or image_model_card == "romanfratric234/FLUX.1-Depth-dev-lora"
        ):
            if image_model_card == "ostris/Flex.2-preview":
                image_model_card = "ostris/Flex.1-alpha"
            scene = context.scene
            if do_convert:
                pipe = converter
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

        # Refiner model - load if chosen.
        if do_refine:
            print(
                "Load Refine Model:  " + "thingthatis/stable-diffusion-xl-refiner-1.0"
            )
            from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "thingthatis/stable-diffusion-xl-refiner-1.0",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only,
            )
            refiner.watermark = NoWatermark()
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                refiner.enable_model_cpu_offload()
                # refiner.enable_vae_tiling()
                # refiner.enable_vae_slicing()
            else:
                refiner.to(gfx_device)

        # --------------------- Main Generate Loop Image -------------------------
        from PIL import Image
        import random

        for i in range(scene.movie_num_batch):
            start_time = timer()

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
                else random.randint(-2147483647, 2147483647)
            )
            print("Seed: " + str(seed))
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

            # SDXL Canny & Illusion
            if image_model_card == "diffusers/controlnet-canny-sdxl-1.0-small":
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

            # OpenPose
            elif image_model_card == "xinsir/controlnet-openpose-sdxl-1.0":
                image = None
                if scene.image_path:
                    image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    image = load_first_frame(scene.movie_path)
                if not image:
                    print("Loading strip failed!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                image = image.resize((x, y))
                # image = scale_image_within_dimensions(np.array(init_image),x,None)

                # Make OpenPose bones from normal image

                if not scene.openpose_use_bones:
                    image = np.array(image)

                    image = processor(image, hand_and_face=True)
                    # Save pose image
                    filename = clean_filename(
                        str(seed) + "_" + context.scene.generate_movie_prompt
                    )
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
                    clear_cuda_cache() 
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

            # FLUX ControlNets
            elif (image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora") or (
                image_model_card == "romanfratric234/FLUX.1-Depth-dev-lora"
            ):
                print("Process: Flux ControlNets")
                init_image = None

                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                image = init_image
                #image = scale_image_within_dimensions(np.array(init_image), x, None)

                if image_model_card == "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora":
                    image = processor(
                        image,
                        low_threshold=50,
                        high_threshold=200,
                        detect_resolution=x,
                        image_resolution=x,
                    )
                else:
                    #from image_gen_aux import DepthPreprocessor
                    #processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
                    image = processor(image)[0].convert("RGB")
                    #image = get_depth_map(image)

                image = converter(
                    prompt=prompt,
                    control_image=image,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    # controlnet_conditioning_scale=1.0,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            elif image_model_card == "Runware/FLUX.1-Redux-dev":
                init_image = None

                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                image = init_image
                pipe_prior_output = pipe_prior_redux(image)
                image = converter(
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    **pipe_prior_output,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]

            # Remove Background
            elif image_model_card == "ZhengPeng7/BiRefNet_HR":
                init_image = None

                if scene.image_path:
                    init_image = load_first_frame(scene.image_path)
                if scene.movie_path:
                    init_image = load_first_frame(scene.movie_path)
                if not init_image:
                    print("Loading strip failed!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                image = scale_image_within_dimensions(np.array(init_image), x, None)

                transform_image = transforms.Compose(
                    [
                        transforms.Resize((2048, 2048)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )

                # Load and transform the image
                image = Image.fromarray(image).convert("RGB")
                image_size = image.size
                input_image = transform_image(image).unsqueeze(0).to("cuda")

                # Generate the background mask
                with torch.no_grad():
                    preds = pipe(input_image)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()
                mask = transforms.ToPILImage()(pred)
                mask = mask.resize(image_size)

                #                # Refine the mask: Apply thresholding and feathering for smoother removal
                #                mask = mask.convert("L")

                #                threshold_value = 200
                #                mask = mask.point(lambda p: 255 if p > threshold_value else 0)

                #                feather_radius = 1
                #                mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))

                # Apply the refined mask to the image to remove the background
                image.putalpha(mask)

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
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                init_image = init_image.resize((x, y))
                style_image = init_image
                subject_strip = find_strip_by_name(scene, scene.blip_subject_image)
                if subject_strip:
                    if (
                        subject_strip.type == "MASK"
                        or subject_strip.type == "COLOR"
                        or subject_strip.type == "SCENE"
                        or subject_strip.type == "META"
                    ):
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
                inference_parameters = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": y,
                    "width": x,
                    "guidance_scale": 0.0,
                    "output_type": "pil",
                    "num_inference_steps": 2,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]
                decoder = None

            elif image_model_card == "Vargol/ProteusV0.4":
                inference_parameters = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]

            elif image_model_card == "Vargol/PixArt-Sigma_2k_16bit":
                inference_parameters = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]

            elif image_model_card == "Alpha-VLLM/Lumina-Image-2.0":
                inference_parameters = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "cfg_trunc_ratio": 0.25,
                    "cfg_normalization": True,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]
            elif (
                image_model_card == "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
            ):
                inference_parameters = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]
                
            # OmniGen    
            elif image_model_card == "Shitao/OmniGen-v1-diffusers":
                omnigen_images = []

                prompt = scene.omnigen_prompt_1
                if find_strip_by_name(scene, scene.omnigen_strip_1):
                    omnigen_images.append(
                        load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.omnigen_strip_1)
                            )
                        )
                    )
                    prompt = prompt + " <img><|image_1|></img> "

                prompt = prompt + scene.omnigen_prompt_2
                if find_strip_by_name(scene, scene.omnigen_strip_2):
                    omnigen_images.append(
                        load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.omnigen_strip_2)
                            )
                        )
                    )
                    prompt = prompt + " <img><|image_2|></img> "

                prompt = prompt + scene.omnigen_prompt_3
                if find_strip_by_name(scene, scene.omnigen_strip_3):
                    omnigen_images.append(
                        load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.omnigen_strip_3)
                            )
                        )
                    )
                    prompt = prompt + " <img><|image_3|></img> "
                print(prompt)

                if not omnigen_images:
                    omnigen_images = None
                    img_size = False
                else:
                    img_size = True
                inference_parameters = {
                    "prompt": prompt,
                    "input_images": omnigen_images,
                    "img_guidance_scale": scene.img_guidance_scale,
                    "use_input_image_size_as_output": img_size,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]

            #Qwen Multi-image
            elif image_model_card == "Qwen/Qwen-Image-Edit-2509":
                
                qwen_images = []
                init_image = None
                
                if scene.input_strips == "input_strips":
                    if scene.image_path:
                        init_image = load_first_frame(scene.image_path)
                    if scene.movie_path:
                        init_image = load_first_frame(scene.movie_path)
                    if init_image:
                        qwen_images.append(init_image)

                if find_strip_by_name(scene, scene.qwen_strip_1):
                    qwen_images.append(
                        load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.qwen_strip_1)
                            )
                        )
                    )

                if find_strip_by_name(scene, scene.qwen_strip_2):
                    qwen_images.append(
                        load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.qwen_strip_2)
                            )
                        )
                    )

                if init_image != None and find_strip_by_name(scene, scene.qwen_strip_3):
                    qwen_images.append(
                        load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.qwen_strip_3)
                            )
                        )
                    )

                if not qwen_images:
                    qwen_images = None
                    print("No input images found. Cancelled!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}

                inference_parameters = {
                    "image": qwen_images,
                    "prompt": prompt,
                    "generator": generator,
                    "true_cfg_scale": 4.0,
                    "negative_prompt": negative_prompt+" ",
                    "num_inference_steps": image_num_inference_steps,
                    #"guidance_scale": 1.0,
                    "num_images_per_prompt": 1,
#                    "height": y,
#                    "width": x,
                }

                with torch.inference_mode():
                    image = pipe(
                        **inference_parameters,
                    ).images[0]
#                    output = pipeline(**inputs)
#                    output_image = output.images[0]
#                    output_image.save("output_image_edit_plus.png")
#                    print("Image saved at", os.path.abspath("output_image_edit_plus.png"))

            # Inpaint
            elif do_inpaint:
                mask_image = None
                init_image = None
                image_reference = None
                mask_strip = find_strip_by_name(scene, scene.inpaint_selected_strip)

                if not mask_strip:
                    print("Selected mask not found!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                if (
                    mask_strip.type == "MASK"
                    or mask_strip.type == "COLOR"
                    or mask_strip.type == "SCENE"
                    or mask_strip.type == "META"
                ):
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
                    print("Loading init image failed!")
                    clear_cuda_cache() 
                    return {"CANCELLED"}
                else:
                    init_image = init_image.resize((x, y))

                if scene.kontext_strip_1:
                    if find_strip_by_name(scene, scene.kontext_strip_1):
                        input_image = load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.kontext_strip_1)
                            )
                        )
                    image_reference = input_image

                print(f"Init image loaded:      {init_image is not None}")
                print(f"Mask image loaded:      {mask_image is not None}")
                print(f"Reference image loaded: {image_reference is not None}")

                if (
                    image_model_card == "lzyvegetable/FLUX.1-schnell"
                    or image_model_card == "ChuckMcSneed/FLUX.1-dev"
                    or image_model_card == "ostris/Flex.2-preview"
                ):
                    if image_model_card == "ostris/Flex.2-preview":
                        image_model_card = "ostris/Flex.1-alpha"
                    print("Process Inpaint: " + image_model_card)
                    inference_parameters = {
                        "prompt": prompt,
                        # "prompt_2": None, # Uncomment if your pipe supports/requires it
                        "max_sequence_length": 512,
                        "image": init_image,
                        "mask_image": mask_image,
                        "num_inference_steps": image_num_inference_steps, # Ensure this has a value
                        "guidance_scale": image_num_guidance,            # Ensure this has a value
                        "height": y,
                        "width": x,
                        "generator": generator,
                        # "padding_mask_crop": 42, # Uncomment if needed
                        # "strength": 0.5,       # Uncomment if needed
                    }

                    if image_model_card == "lzyvegetable/FLUX.1-schnell":
                        # Override specific parameters for FLUX
                        inference_parameters["guidance_scale"] = 0
                        inference_parameters["num_inference_steps"] = 4

                    image = pipe(
                        **inference_parameters
                    ).images[0]
                    
                # Kontext Inpaint            
                elif (
                    image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                ):

                    print("Process Inpaint: " + image_model_card)
                    inference_parameters = {
                        "prompt": prompt,
                        # "prompt_2": None, # Uncomment if your pipe supports/requires it
                        "max_sequence_length": 512,
                        "image": init_image,
                        "mask_image": mask_image,
                        "image_reference": image_reference, 
                        "num_inference_steps": image_num_inference_steps, # Ensure this has a value
                        "guidance_scale": image_num_guidance,            # Ensure this has a value
                        "height": y,
                        "width": x,
                        "generator": generator,
                        "strength": 1.00 - scene.image_power,
                        # "padding_mask_crop": 42, # Uncomment if needed
                        # "strength": 0.5,       # Uncomment if needed
                    }

                    if image_model_card == "lzyvegetable/FLUX.1-schnell":
                        # Override specific parameters for FLUX
                        inference_parameters["guidance_scale"] = 0
                        inference_parameters["num_inference_steps"] = 4

                    image = pipe(
                        **inference_parameters
                    ).images[0]                    

                elif image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    print("Process Inpaint: " + image_model_card)
                    inference_parameters = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "image": init_image,
                        "mask_image": mask_image,
                        "num_inference_steps": image_num_inference_steps,
                        "guidance_scale": image_num_guidance,
                        "height": y,
                        "width": x,
                        "generator": generator,
                        "padding_mask_crop": 42,
                        "strength": 0.99,
                    }
                    image = pipe(
                        **inference_parameters,
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

            # Img2img
            elif do_convert:  # and not scene.aurasr:
                if enabled_items:
                    self.report(
                        {"INFO"},
                        "LoRAs are ignored for image to image processing.",
                    )
                img_path = None
                if scene.movie_path:
                    print("Process: Image to Image")
                    init_image = load_first_frame(scene.movie_path)
                    init_image = init_image.resize((x, y))
                elif scene.image_path:
                    print("Process: Image to Image")
                    init_image = load_first_frame(scene.image_path)
                    init_image = init_image.resize((x, y))
                    img_path=scene.image_path 
                # init_image = load_image(scene.image_path).convert("RGB")
                print("X: " + str(x), "Y: " + str(y))

                # MacOS
                if (image_model_card == "ChuckMcSneed/FLUX.1-dev" and os_platform == "Darwin") or (image_model_card == "lzyvegetable/FLUX.1-schnell" and os_platform == "Darwin"):
                    if not img_path:
                        print("Please, input an image!")
                        clear_cuda_cache() 
                        return {"CANCELLED"}
                    image = converter.generate_image(
                       seed=abs(int(seed)),
                       prompt=prompt,
                       config=Config(
                          num_inference_steps=image_num_inference_steps,  # "schnell" works well with 2-4 steps, "dev" works well with 20-25 steps
                          height=y,
                          width=x,
                          image_path=os.path.abspath(img_path),
                          image_strength=1.00-scene.image_power,
                       )
                    )
                    
                elif (
                    image_model_card == "stabilityai/sdxl-turbo"
                    or image_model_card == "lzyvegetable/FLUX.1-schnell"
                ):
                    image = converter(
                        prompt=prompt,
                        prompt_2=None,
                        max_sequence_length=512,
                        image=init_image,
                        strength=1.00 - scene.image_power,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=0.0,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]
                    
                    
                elif (
                    image_model_card == "ChuckMcSneed/FLUX.1-dev"
                    or image_model_card == "ostris/Flex.2-preview"
                ):
                    image = converter(
                        prompt=prompt,
                        #prompt_2=None,
                        max_sequence_length=512,
                        image=init_image,
                        strength=1.00 - scene.image_power,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=image_num_guidance,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]
                elif (
                    image_model_card == "Qwen/Qwen-Image"
                ):
                    image = converter(
                        prompt=prompt,
                        #prompt_2=None,
                        negative_prompt=negative_prompt,
                        max_sequence_length=512,
                        image=init_image,
                        strength=1.00 - scene.image_power,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=image_num_guidance,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]
                elif (
                    image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
                ):
                        
                    kontext_images = []
                    if scene.kontext_strip_1:
                        if find_strip_by_name(scene, scene.kontext_strip_1):
                            input_image = load_first_frame(
                                get_strip_path(
                                    find_strip_by_name(scene, scene.kontext_strip_1)
                                )
                            )
                        init_image = input_image

                    if not kontext_images:
                        kontext_images = None
                        img_size = False
                    else:
                        img_size = True
                     
                    image = converter(
                        prompt=prompt,
                        #prompt_2=None,
                        max_sequence_length=512,
                        #input_images=kontext_images, 
                        #image=kontext_images,
                        image=init_image,
                        #strength=1.00 - scene.image_power,
                        # negative_prompt=negative_prompt,
                        num_inference_steps=image_num_inference_steps,
                        guidance_scale=image_num_guidance,
                        height=y,
                        width=x,
                        generator=generator,
                    ).images[0]

                elif (
                    image_model_card == "kontext-community/relighting-kontext-dev-lora-v3"
                ):

                    prompt_description = ""
                    style_and_direction_parts = []

                    if prompt:
                        prompt_description = prompt
                        style_and_direction_parts.append("with custom lighting")
                    else:
                        prompt_description = ILLUMINATION_OPTIONS.get(context.scene.illumination_style, "")
                        style_and_direction_parts.append(f"with {context.scene.illumination_style} lighting")

                    if context.scene.light_direction != "auto":
                        style_and_direction_parts.append(f"coming from the {context.scene.light_direction}")

                    style_description = " ".join(style_and_direction_parts)
                    final_prompt = (
                        f"Relight the image {style_description}. "
                        f"{prompt_description} "
                        "Maintain the identity of the foreground subjects."
                    )
                    
                    print(f"AI Relight: Running inference with prompt: {final_prompt}")
                    image = converter(
                        image=init_image, prompt=final_prompt, num_inference_steps=image_num_inference_steps, guidance_scale=image_num_guidance,
                        width=x, height=y, generator=generator
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
                        # height=y,
                        # width=x,
                        generator=generator,
                    ).images[0]

            # MacOS
            elif (image_model_card == "ChuckMcSneed/FLUX.1-dev" and os_platform == "Darwin") or (image_model_card == "lzyvegetable/FLUX.1-schnell" and os_platform == "Darwin"):
                image = pipe.generate_image(
                   seed=abs(int(seed)),
                   prompt=prompt,
                   config=Config(
                      num_inference_steps=image_num_inference_steps,  # "schnell" works well with 2-4 steps, "dev" works well with 20-25 steps
                      height=y,
                      width=x,
                   )
                )

            # Flux Schnell
            elif (
                image_model_card == "lzyvegetable/FLUX.1-schnell"
            ):  # and not scene.aurasr:
                inference_parameters = {
                    "prompt": prompt,
                    "prompt_2": None,
                    "max_sequence_length": 512,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]

            # Flux Dev
            elif (
                image_model_card == "ChuckMcSneed/FLUX.1-dev"
                or image_model_card == "ostris/Flex.2-preview"
            ):
                inference_parameters = {
                    "prompt": prompt,
                    "prompt_2": None,
                    "negative_prompt": negative_prompt,
                    "max_sequence_length": 512,
                    #"image": init_image,
                    #"mask_image": mask_image,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }

                image = pipe(
                    **inference_parameters,
                ).images[0]
            elif (
                image_model_card == "yuvraj108c/FLUX.1-Kontext-dev"
            ):
                
                kontext_images = []
                init_image = None
                if scene.kontext_strip_1:
                    if find_strip_by_name(scene, scene.kontext_strip_1):
                        input_image = load_first_frame(
                            get_strip_path(
                                find_strip_by_name(scene, scene.kontext_strip_1)
                            )
                        )
                    init_image = input_image              
                image = converter(
                    prompt=prompt,
                    #prompt_2=None,
                    max_sequence_length=512,
                    image=init_image,
                    #strength=1.00 - scene.image_power,
                    # negative_prompt=negative_prompt,
                    num_inference_steps=image_num_inference_steps,
                    guidance_scale=image_num_guidance,
                    height=y,
                    width=x,
                    generator=generator,
                ).images[0]
            # Chroma
            elif (image_model_card == "lodestones/Chroma"):
                inference_parameters = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "max_sequence_length": 512,
                    #"image": init_image,
                    #"mask_image": mask_image,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "generator": generator,
                }

                image = pipe(
                    **inference_parameters,
                ).images[0]

            # Generate Stable Diffusion etc.
            elif (
                image_model_card == "stabilityai/stable-diffusion-3-medium-diffusers"
                or image_model_card == "adamo1139/stable-diffusion-3.5-large-ungated"
                or image_model_card == "adamo1139/stable-diffusion-3.5-medium-ungated"
            ):
                print("Generate: Stable Diffusion Image ")
                inference_parameters = {
                    "prompt": "",
                    "prompt_3": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": image_num_inference_steps,
                    "guidance_scale": image_num_guidance,
                    "height": y,
                    "width": x,
                    "max_sequence_length": 512,
                    "generator": generator,
                }
                image = pipe(
                    **inference_parameters,
                ).images[0]
            else:
                print("Generate: Image")
                from diffusers.utils import load_image

                # IPAdapter
                if (
                    scene.ip_adapter_face_folder or scene.ip_adapter_style_folder
                ) and image_model_card == "stabilityai/stable-diffusion-xl-base-1.0":
                    mask_image = None
                    init_image = None
                    ip_adapter_image = None

                    if scene.ip_adapter_face_folder and scene.ip_adapter_style_folder:
                        face_images = load_images_from_folder(
                            (scene.ip_adapter_face_folder).replace("\\", "/")
                        )
                        style_images = load_images_from_folder(
                            (scene.ip_adapter_style_folder).replace("\\", "/")
                        )
                        ip_adapter_image = [style_images, face_images]
                    elif scene.ip_adapter_face_folder:
                        face_images = load_images_from_folder(
                            (scene.ip_adapter_face_folder).replace("\\", "/")
                        )
                        ip_adapter_image = [face_images]
                    elif scene.ip_adapter_style_folder:
                        style_images = load_images_from_folder(
                            (scene.ip_adapter_style_folder).replace("\\", "/")
                        )
                        ip_adapter_image = [style_images]

                    # Inpaint
                    if scene.inpaint_selected_strip:
                        print("Process: Inpaint")
                        mask_strip = find_strip_by_name(
                            scene, scene.inpaint_selected_strip
                        )

                        if not mask_strip:
                            print("Selected mask not found!")
                            clear_cuda_cache() 
                            return {"CANCELLED"}
                        if (
                            mask_strip.type == "MASK"
                            or mask_strip.type == "COLOR"
                            or mask_strip.type == "SCENE"
                            or mask_strip.type == "META"
                        ):
                            mask_strip = get_render_strip(self, context, mask_strip)
                        mask_path = get_strip_path(mask_strip)
                        mask_image = load_first_frame(mask_path)

                        if not mask_image:
                            print("Loading mask failed!")
                            return
                        mask_image = mask_image.resize((x, y))
                        mask_image = pipe.mask_processor.blur(
                            mask_image, blur_factor=33
                        )

                        if scene.image_path:
                            init_image = load_first_frame(scene.image_path)
                        if scene.movie_path:
                            init_image = load_first_frame(scene.movie_path)
                        if not init_image:
                            print("Loading strip failed!")
                            clear_cuda_cache() 
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
                    elif scene.input_strips == "input_strips" and (
                        scene.image_path or scene.movie_path
                    ):
                        if scene.image_path:
                            init_image = load_first_frame(scene.image_path)
                        if scene.movie_path:
                            init_image = load_first_frame(scene.movie_path)
                        if not init_image:
                            print("Loading strip failed!")
                            clear_cuda_cache() 
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
                elif image_model_card == "stabilityai/sdxl-turbo":
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
                            prompt,
                            # negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=0.0,
                            height=y,
                            width=x,
                            generator=generator,
                            max_sequence_length=512,
                        ).images[0]
                        
                # Qwen
                elif image_model_card == "Qwen/Qwen-Image":
                    # LoRA.
                    if enabled_items:
                        image = pipe(
                            # prompt_embeds=prompt, # for compel - long prompts
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            #guidance_scale=0.0,
                            height=y,
                            width=x,
                            true_cfg_scale=4.0,
                            generator=generator,
                        ).images[0]

                    # No LoRA.
                    else:
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            true_cfg_scale=4.0,
                            height=y,
                            width=x,
                            generator=generator,
                            max_sequence_length=512,
                        ).images[0]

                # Not Turbo
                else:  # if not scene.aurasr:
                    # LoRA.
                    if enabled_items:
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            cross_attention_kwargs={"scale": 1.0},
                            generator=generator,
                            max_sequence_length=512,
                        ).images[0]
                    # No LoRA.
                    else:
                        image = pipe(
                            prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=image_num_inference_steps,
                            guidance_scale=image_num_guidance,
                            height=y,
                            width=x,
                            generator=generator,
                            max_sequence_length=512,
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
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    vae=vae,
                    variant="fp16",
                    torch_dtype=torch.float16,
                )
                if gfx_device == "mps":
                    pipe.to("mps")
                elif low_vram():
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(gfx_device)

                face_prompt = (
                    prompt + ", face, (8k, RAW photo, best quality, masterpiece:1.2)"
                )
                face_n_prompt = "nsfw, blurry, disfigured"
                face_mask_pad = 32
                mask_blur = 4
                mask_dilation = 4
                strength = 0.4
                ddim_steps = 20
                ad_images = image

                ad_components = pipe.components
                ad_pipe = AdPipelineBase(**ad_components)

                model_path = hf_hub_download(
                    "Bingsu/adetailer",
                    "face_yolov8n.pt",
                    local_dir="asdff/yolo_models",
                    local_dir_use_symlinks=False,
                )
                common = {
                    "prompt": face_prompt,
                    "n_prompt": face_n_prompt,
                    "num_inference_steps": int(image_num_inference_steps),
                    "target_size": (x, y),
                }
                inpaint_only = {"strength": strength}
                result = ad_pipe(
                    common=common,
                    inpaint_only=inpaint_only,
                    images=ad_images,
                    mask_dilation=mask_dilation,
                    mask_blur=mask_blur,
                    mask_padding=face_mask_pad,
                    model_path=model_path,
                )
                try:
                    image = result.images[0]
                except:
                    print("No images detected. ADetailer disabled.")

            # AuraSR
            if scene.aurasr:
                if do_convert:
                    if scene.movie_path:
                        print("Process: Movie Frame to Image")
                        init_image = load_first_frame(scene.movie_path)
                        init_image = init_image.resize((x, y))
                    elif scene.image_path:
                        print("Process: Image to Image")
                        init_image = load_first_frame(scene.image_path)
                        init_image = init_image.resize((x, y))
                    image = init_image

                if image:
                    from aura_sr import AuraSR

                    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
                    image = aura_sr.upscale_4x_overlapped(image)

            # Move to folder
            filename = clean_filename(
                str(seed) + "_" + context.scene.generate_movie_prompt
            )
            out_path = solve_path(filename + ".png")
            image.save(out_path)
            bpy.types.Scene.genai_out_path = out_path

            if input == "input_strips":
                old_strip = active_strip

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

                if inference_parameters != None:
                    set_ai_metadata_from_dict(
                        strip=strip,
                        params_dict=inference_parameters
                    )

                scene.sequence_editor.active_strip = strip
                if i > 0:
                    scene.frame_current = (
                        scene.sequence_editor.active_strip.frame_final_start
                    )
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
                                scene.frame_current = (
                                    scene.sequence_editor.active_strip.frame_final_start
                                )
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


def remove_duplicate_phrases(input_string: str) -> str:
    """
    Removes duplicate comma-separated phrases from a string.

    This function is designed for strings that are lists of phrases,
    like "phrase one, phrase two, phrase one". It preserves the
    first occurrence of each unique phrase.

    Args:
        input_string: The string containing comma-separated phrases.

    Returns:
        A new string with duplicate phrases removed, properly formatted.
    """
    # 1. Split the string into a list of phrases using the comma as a delimiter.
    #    We then use a list comprehension to strip leading/trailing whitespace
    #    from each resulting phrase.
    phrases = [phrase.strip() for phrase in input_string.split(',')]

    # Use a set for fast lookups to track phrases we've already seen.
    seen_phrases = set()

    # This list will hold the unique phrases in their original order.
    unique_phrases_in_order = []

    for phrase in phrases:
        # Ignore any empty phrases that might result from trailing commas, etc.
        if not phrase:
            continue

        # If we haven't seen this phrase before...
        if phrase not in seen_phrases:
            # ...add it to our list of unique phrases...
            unique_phrases_in_order.append(phrase)
            # ...and record that we have now seen it.
            seen_phrases.add(phrase)

    # 3. Join the unique phrases back together with a comma and a space.
    return ", ".join(unique_phrases_in_order)


class SEQUENCER_OT_generate_text(Operator):
    """Generate Text"""

    bl_idname = "sequencer.generate_text"
    bl_label = "Prompt"
    bl_description = "Generate texts from strips"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        input = scene.input_strips
        seq_editor = scene.sequence_editor
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        local_files_only = addon_prefs.local_files_only
        guidance = scene.movie_num_guidance
        current_frame = scene.frame_current
        #prompt = style_prompt(scene.generate_movie_prompt)[0]
        prompt = scene.generate_movie_prompt
        x = scene.generate_movie_x = closest_divisible_32(scene.generate_movie_x)
        y = scene.generate_movie_y = closest_divisible_32(scene.generate_movie_y)
        active_strip = context.scene.sequence_editor.active_strip
        old_duration = duration = active_strip.frame_final_duration
        render = bpy.context.scene.render
        fps = render.fps / render.fps_base
        show_system_console(True)
        set_system_console_topmost(True)

        if not seq_editor:
            scene.sequence_editor_create()

        if addon_prefs.text_model_card == "Salesforce/blip-image-captioning-large":
            try:
                import torch
                from PIL import Image
                from transformers import BlipProcessor, BlipForConditionalGeneration
            except ModuleNotFoundError as e:
                print("Dependencies needs to be installed in the add-on preferences. "+str(e.name))

                self.report(
                    {"INFO"},
                    "Dependencies need to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}

        elif (
            addon_prefs.text_model_card == "ZuluVision/MoviiGen1.1_Prompt_Rewriter"
        ):
            try:
                import torch
                from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
            except ModuleNotFoundError as e:
                print("Dependencies needs to be installed in the add-on preferences. "+str(e.name))

                self.report(
                    {"INFO"},
                    "Dependencies need to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}
        elif (
            addon_prefs.text_model_card == "yownas/Florence-2-large"
        ):
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
            except ModuleNotFoundError as e:
                print("Dependencies needs to be installed in the add-on preferences. "+str(e.name))

                self.report(
                    {"INFO"},
                    "Dependencies need to be installed in the add-on preferences.",
                )
                return {"CANCELLED"}

        # clear the VRAM
        clear_cuda_cache()


        if not addon_prefs.text_model_card == "ZuluVision/MoviiGen1.1_Prompt_Rewriter":
            if scene.movie_path:
                init_image = load_first_frame(bpy.path.abspath(scene.movie_path))
            else:
                init_image = load_first_frame(bpy.path.abspath(scene.image_path))
            if init_image:
                init_image = init_image.resize((x, y))
            else:
                print("No input image loaded succesfully. Cancelling.")
                return {"CANCELLED"}
            
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
            inputs = processor(init_image, text, return_tensors="pt").to(
                gfx_device, torch.float16
            )

            out = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(out[0], skip_special_tokens=True)
            text = clean_string(text)
            print("Generated text: " + text)

        elif (
            addon_prefs.text_model_card == "yownas/Florence-2-large"
        ):

            model = (
                AutoModelForCausalLM.from_pretrained(
                    "yownas/Florence-2-large", trust_remote_code=True
                )
                .to(gfx_device)
                .eval()
            )
            processor = AutoProcessor.from_pretrained(
                "yownas/Florence-2-large", trust_remote_code=True
            )

            # Ensure the image is in RGB mode
            if init_image.mode != "RGB":
                init_image = init_image.convert("RGB")

            # prompt = "<MIXED_CAPTION_PLUS>"
            prompt = "<MORE_DETAILED_CAPTION>"

            inputs = processor(text=prompt, images=init_image, return_tensors="pt").to(
                gfx_device
            )

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                repetition_penalty=1.10,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(init_image.width, init_image.height),
            )
            text = parsed_answer[prompt]
            print("Generated text: " + str(text))

        elif addon_prefs.text_model_card == "ZuluVision/MoviiGen1.1_Prompt_Rewriter":
            if input == "input_strips" and active_strip and active_strip.type != "TEXT":
                print("Unsupported strip type: "+active_strip.name)
                return {"CANCELLED"}
            print("Enhancing prompt.")
            quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

            model_name = "ZuluVision/MoviiGen1.1_Prompt_Rewriter"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                #torch_dtype="auto",
                device_map="auto",
                quantization_config=quantization_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            prompt = prompt
            messages = [
                #{"role": "system", "content": "Be creative and expand the input into a single line of comma-separated cinematic keywords, strictly ordered as: camera, camera motion, subject, distinct subject details, distinct situation, distinct location details, setting, lighting, atmosphere, style."},
                #{"role": "system", "content": "You are an advanced AI model tasked with You must respond in the language used by the user."},
                #{"role": "system", "content": "You enhance the input prompt to a 400 characters image prompt, in precise cinematic language, in comma separated nouns and adjectives. First camera angle and framing, then be creative and expand on all the input elements, don't change the order, by specifying subjects, their situation, one by one, then the settings, lighting, color, atmosphere, mood, style, motion, and camera movement. Do not repeat words or elements. Example: a cinematic wide-shot of a young woman, red hair, army clothes, dark forest, dramatic lightning. "},
                {"role": "system", "content": "As a cinematic prompt engineer, be creative, rewrite the following into a comma-separated list of visual details, starting with camera angle, camera motion and progressing through subject, setting, lighting, atmosphere, style."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #print("Generated text: " + str(text))
            text = remove_duplicate_phrases(text)
            print("Generated text: " + str(text))

        if input == "input_strips" and active_strip:
            start_frame = int(active_strip.frame_start)
            end_frame = (
                start_frame + active_strip.frame_final_duration
            )
        else:
            start_frame = int(scene.frame_current)
            end_frame = (
                start_frame + 100
            )

        empty_channel = find_first_empty_channel(
            start_frame,
            end_frame,
        )

        # Add strip
        if text:
            strip = scene.sequence_editor.sequences.new_effect(
                name=str(text),
                type="TEXT",
                frame_start=start_frame,
                frame_end=end_frame,
                channel=empty_channel,
            )
            strip.frame_final_end = end_frame
            strip.text = text
            strip.wrap_width = 0.68
            strip.font_size = 16
            strip.location[0] = 0.5
            strip.location[1] = 0.2
            strip.anchor_x = "CENTER"
            strip.anchor_y = "TOP"
            strip.alignment_x = "LEFT"
            strip.use_shadow = True
            strip.use_box = True
            strip.box_color = (0, 0, 0, 0.7)
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
                            <= (
                                active_strip.frame_final_start
                                + active_strip.frame_final_duration
                            )
                        ):
                            pass
                        else:
                            scene.frame_current = (
                                active_strip.frame_final_start
                            )
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
        if not strips == context.selected_sequences:
            active_strip.select = True
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
            if strip.type in {"MOVIE", "IMAGE", "TEXT", "SCENE", "META", "SOUND"}:
                break
        else:
            self.report(
                {"INFO"},
                "None of the selected strips are movie, image, text, meta or scene types.",
            )
            return {"CANCELLED"}

        if type == "text":
            for strip in strips:
                if strip.type in {"MOVIE", "IMAGE", "TEXT", "SCENE", "META"}:
                    print("Process: Processing to Text")
                    break
            else:
                self.report(
                    {"INFO"},
                    "None of the selected strips are possible to process to text.",
                )
                return {"CANCELLED"}

        if use_strip_data:
            print("Use file seed and prompt: Yes")
        else:
            print("Use file seed and prompt: No")

        if gfx_device == "cuda":
            total_vram = 0
            for i in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(i)
                total_vram += properties.total_memory
                print("Total VRAM: " + str(total_vram))
                print("Total GPU Cards: " + str(torch.cuda.device_count()))
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
            cudnn_version = torch.backends.cudnn.version()
            print(f"cuDNN version: {cudnn_version}")

        for count, strip in enumerate(strips):
            for dsel_strip in bpy.context.scene.sequence_editor.sequences:
                dsel_strip.select = False
            strip.select = True
            context.scene.sequence_editor.active_strip = strip

            # render intermediate mp4 file
            if (
                strip.type == "SCENE" or strip.type == "MOVIE" or strip.type == "META"
            ):  # or strip.type == "IMAGE"
                # Make the current frame overlapped frame, the temp strip.
                if type == "image" or type == "text":
                    trim_frame = find_overlapping_frame(strip, current_frame)

                    if trim_frame and len(strips) == 1:
                        bpy.ops.sequencer.duplicate_move(
                            SEQUENCER_OT_duplicate={},
                            TRANSFORM_OT_seq_slide={
                                "value": (0, 1),
                                "use_restore_handle_selection": False,
                                "snap": False,
                                "view2d_edge_pan": False,
                                "release_confirm": False,
                                "use_accurate": False,
                            },
                        )
                        intermediate_strip = bpy.context.selected_sequences[0]
                        intermediate_strip.frame_start = strip.frame_start
                        intermediate_strip.frame_offset_start = int(trim_frame)
                        intermediate_strip.frame_final_duration = 1
                        temp_strip = strip = get_render_strip(
                            self, context, intermediate_strip
                        )

                        if intermediate_strip is not None:
                            delete_strip(intermediate_strip)

                    elif type == "text":
                        bpy.ops.sequencer.copy()
                        bpy.ops.sequencer.paste(keep_offset=True)
                        intermediate_strip = bpy.context.selected_sequences[0]
                        intermediate_strip.frame_start = strip.frame_start

                        # intermediate_strip.frame_offset_start = int(trim_frame)
                        intermediate_strip.frame_final_duration = (
                            strip.frame_final_duration
                        )
                        temp_strip = strip = get_render_strip(
                            self, context, intermediate_strip
                        )

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
                    if type == "text":
                        sequencer.generate_text()
                    # context.scene.generate_movie_prompt = prompt
                    # scene.generate_movie_negative_prompt = negative_prompt

                    context.scene.movie_use_random = use_random
                    context.scene.movie_num_seed = seed
                    # scene.generate_movie_prompt = prompt

                    scene.generate_movie_negative_prompt = negative_prompt

                    if use_strip_data:
                        scene.movie_use_random = use_random
                        scene.movie_num_seed = seed

            if strip.type == "SOUND":
                if strip.sound:
                    print("\n" + str(count + 1) + "/" + str(len(strips)))
                    print("Prompt: " + prompt)
                    #print("Negative Prompt: " + negative_prompt)
                    #scene.generate_movie_prompt = strip.text + ", " + prompt
                    scene.frame_current = strip.frame_final_start
                    context.scene.sequence_editor.active_strip = strip

                    if type == "movie":
                        sequencer.generate_movie()
                    if type == "audio":
                        sequencer.generate_audio()
                    if type == "image":
                        sequencer.generate_image()

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

                        if file_seed and use_strip_data:
                            strip_prompt = strip_prompt.replace(
                                str(file_seed) + "_", ""
                            )
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed

                    if use_strip_data:
                        styled_prompt = style_prompt(strip_prompt + ", " + prompt)[0]
                        styled_negative_prompt = style_prompt(
                            strip_prompt + ", " + prompt
                        )[1]
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
                            strip_prompt = strip_prompt.replace(
                                str(file_seed) + "_", ""
                            )
                            context.scene.movie_use_random = False
                            context.scene.movie_num_seed = file_seed

                    if use_strip_data:
                        styled_prompt = style_prompt(strip_prompt + ", " + prompt)[0]
                        styled_negative_prompt = style_prompt(
                            strip_prompt + ", " + prompt
                        )[1]
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

            for strip in context.scene.sequence_editor.sequences_all:
                # Check if the strip has a transform property before accessing it
                if hasattr(strip, 'transform'):
                    scale_y = strip.transform.scale_y
                else:
                    # If not, assume a default scale of 1.0 (occupies one channel)
                    scale_y = 1.0

                # Calculate the vertical bounds of the strip in view space
                strip_y_min_view = strip.channel - 0.5 * scale_y
                strip_y_max_view = strip.channel + 0.5 * scale_y

                if (
                    strip.frame_start <= mouse_x_view < strip.frame_final_end and
                    strip_y_min_view <= mouse_y_view < strip_y_max_view
                ):
                    self.perform_action(context, strip)
                    context.window.cursor_modal_restore()
                    return {"FINISHED"}


#                # Calculate the vertical bounds of the strip in view space
#                # Assuming each channel has a nominal height of 1.0 in view space
#                strip_y_min_view = strip.channel - 0.5 * strip.transform.scale_y  # Consider the scaled height
#                strip_y_max_view = strip.channel + 0.5 * strip.transform.scale_y

#                if (
#                    strip.frame_start <= mouse_x_view < strip.frame_final_end and
#                    (strip.type == "IMAGE" or strip.type =="MOVIE")#and
#                    #strip_y_min_view <= mouse_y_view < strip_y_max_view
#                ):
#                    self.perform_action(context, strip)
#                    context.window.cursor_modal_restore()
#                    return {"FINISHED"}

            # If no strip picked, don't exit — allow continuous clicking
            return {"RUNNING_MODAL"}

        elif event.type in {"RIGHTMOUSE", "ESC"}:
            context.window.cursor_modal_restore()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def perform_action(self, context, strip):
        """Handle different actions on the picked strip"""
        scene = context.scene
        if self.action == "omni_select1":
            self.report({"INFO"}, f"Picked: {strip.name}")
            if find_strip_by_name(scene, strip.name):
                scene.omnigen_strip_1 = strip.name
        elif self.action == "omni_select2":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.omnigen_strip_2 = strip.name
        elif self.action == "omni_select3":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.omnigen_strip_3 = strip.name
                
        if self.action == "qwen_select1":
            self.report({"INFO"}, f"Picked: {strip.name}")
            if find_strip_by_name(scene, strip.name):
                scene.qwen_strip_1 = strip.name
        elif self.action == "qwen_select2":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.qwen_strip_2 = strip.name
        elif self.action == "qwen_select3":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.qwen_strip_3 = strip.name
                
        elif self.action == "minimax_select":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.minimax_subject = strip.name
                
        elif self.action == "inpaint_select":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.inpaint_selected_strip = strip.name
                
        elif self.action == "out_frame_select":
            print(f"Picked Strip Name: {strip.name}")
            self.report({"INFO"}, f"Picked '{strip.name}'")
            if find_strip_by_name(scene, strip.name):
                context.scene.out_frame = strip.name
                
        if self.action == "kontext_select1":
            self.report({"INFO"}, f"Picked: {strip.name}")
            if find_strip_by_name(scene, strip.name):
                scene.kontext_strip_1 = strip.name
#        else:
#            self.report({"WARNING"}, f"Unknown action: {self.action}")

    def invoke(self, context, event):
        if context.area.type == 'SEQUENCE_EDITOR':
            context.window_manager.modal_handler_add(self)
            context.window.cursor_modal_set("EYEDROPPER")
            return {"RUNNING_MODAL"}
        else:
            self.report({'WARNING'}, "This operator only works in the Video Sequence Editor")
            return {"CANCELLED"}


AI_METADATA_PREFIX = "ai_meta_"


def set_ai_metadata_from_dict(strip: bpy.types.Strip, params_dict: dict):
    """
    Sets AI metadata custom properties on a VSE strip from a dictionary.

    Stores parameter names (dict keys) and their string representations (dict values)
    as custom properties, prefixed with 'ai_meta_'.

    Args:
        strip: The VSE strip (Image or Movie) to add metadata to.
        params_dict: The dictionary containing the inference parameters.
    """
    if not strip or strip.type not in {'IMAGE', 'MOVIE'}:
        print(f"Error: Cannot set metadata. Invalid strip: {strip}")
        return False
    if not isinstance(params_dict, dict):
        print(f"Error: Second argument must be a dictionary.")
        return False

    print(f"Setting AI Metadata on strip: {strip.name} from dictionary")
    set_count = 0

    # Optional: Clear existing AI metadata first?
    # existing_keys = [k for k in strip.keys() if k.startswith(AI_METADATA_PREFIX)]
    # for k in existing_keys:
    #     del strip[k]
    # print(f"  Cleared {len(existing_keys)} existing AI metadata properties.")

    for key, value in params_dict.items():
        prop_key = f"{AI_METADATA_PREFIX}{key}"
        value_str = "" # Default empty string

        # Convert value to a suitable string representation
        if value is None:
            value_str = "None"
        elif isinstance(value, (str, int, float, bool)):
            value_str = str(value)
        elif isinstance(value, torch.Generator):
            try:
                # Use the documented method to get the initial seed
                seed = value.initial_seed()
                #value_str = f"torch.Generator(seed={seed})"
                value_str = f"Seed: {seed}"
            except Exception as e:
                # Fallback if initial_seed() fails for some reason
                print(f"  Warning: Could not get initial_seed for {key}: {e}")
                value_str = f"torch.Generator(object)"
        elif hasattr(value, '__dict__') or hasattr(value, '__slots__') or callable(getattr(value, '__repr__', None)):
             try:
                 repr_val = repr(value)
                 if len(repr_val) > 200: # Limit length for UI sanity
                      repr_val = repr_val[:200] + "..."
                 value_str = repr_val
             except Exception:
                 value_str = f"Object({type(value).__name__})"
        else:
             # Final fallback conversion
             try:
                 value_str = str(value)
             except Exception as e:
                 print(f"  Warning: Could not convert value for '{key}' to string: {e}. Using type name.")
                 value_str = f"<{type(value).__name__}>"

        try:
            strip[prop_key] = value_str # Store the STRING representation
            # print(f"  Set '{prop_key}' = '{value_str}'") # Can be verbose
            set_count += 1
        except Exception as e:
            print(f"  Error setting property '{prop_key}' with value '{value_str}': {e}")


    # Force UI update attempt
    if hasattr(strip, "frame_final_duration"):
        try:
           strip.frame_final_duration = strip.frame_final_duration
        except AttributeError: pass

    print(f"Finished setting {set_count} metadata properties.")
    return True


class AI_Metadata_PT_Panel(bpy.types.Panel):
    """Displays AI Generation Metadata stored as custom properties"""
    bl_label = "AI Metadata"
    bl_idname = "SEQUENCER_PT_ai_metadata"
    bl_space_type = 'SEQUENCE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Metadata"

    @classmethod
    def poll(cls, context):
        if context.space_data.view_type in {'SEQUENCER', 'PREVIEW'}:
            if context.scene and context.scene.sequence_editor:
                active_strip = context.scene.sequence_editor.active_strip
                if active_strip and active_strip.type in {'IMAGE', 'MOVIE'}:
                    return True
        return False

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        seq_editor = scene.sequence_editor
        strip = seq_editor.active_strip

        if not strip:
            return

        col = layout.column(align=True)
        displayed_anything = False

        #ai_prop_keys = sorted([k for k in strip.keys() if k.startswith(AI_METADATA_PREFIX)])
        ai_prop_keys = strip.keys()

        if not ai_prop_keys:
            col.label(text="No AI metadata found on this strip.")
            col.label(text="Use 'set_ai_metadata_from_dict'")
            col.label(text="to add data.")
            return

        col.label(text="Name:                "+strip.name)

        for prop_key in ai_prop_keys:
            param_name = prop_key[len(AI_METADATA_PREFIX):]
            label_text = param_name.replace('_', ' ').title()
            # Use prop for easy copy/paste of the string value
            col.prop(strip, f'["{prop_key}"]', text=label_text)
            displayed_anything = True


classes = (
    GeneratorAddonPreferences,
    SEQUENCER_OT_generate_movie,
    SEQUENCER_OT_generate_audio,
    SEQUENCER_OT_generate_image,
    SEQUENCER_OT_generate_text,
    SEQUENCER_OT_ai_strip_picker,
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
    AI_Metadata_PT_Panel,
)


def get_enum_items(options_dict):
    """Converts a dictionary to the format required by bpy.props.EnumProperty."""
    return [(key, key.replace("_", " ").title(), desc) for key, desc in options_dict.items()]


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
    bpy.types.Scene.generate_audio_prompt = bpy.props.StringProperty(
        name="generate_audio_prompt", default=""
    )
    bpy.types.Scene.generate_movie_x = bpy.props.IntProperty(
        name="generate_movie_x",
        default=1024,
        step=32,
        min=256,
        max=4096,
        description="Use the power of 64",
    )
    bpy.types.Scene.generate_movie_y = bpy.props.IntProperty(
        name="generate_movie_y",
        default=576,
        step=32,
        min=256,
        max=4096,
        description="Use the power of 64",
    )

    # The number of frames to be generated.
    bpy.types.Scene.generate_movie_frames = bpy.props.IntProperty(
        name="generate_movie_frames",
        default=6,
        min=-1,
        max=500,
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
        update=input_strips_updated,
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
    bpy.types.Scene.inpaint_selected_strip = bpy.props.StringProperty(
        name="inpaint_selected_strip", default=""
    )

    # Inpaint
    bpy.types.Scene.out_frame = bpy.props.StringProperty(
        name="out_frame", default=""
    )

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
    bpy.types.Scene.movie_path = bpy.props.StringProperty(
        name="movie_path",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.movie_path = ""

    # image path
    bpy.types.Scene.image_path = bpy.props.StringProperty(
        name="image_path",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
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
        min=0.00,
        max=1.00,
        description="Preserve the input image in vid/img to img/vid processes",
    )
    styles_array = load_styles(
        os.path.dirname(os.path.abspath(__file__)) + "/styles.json"
    )
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

    bpy.types.Scene.use_lcm = bpy.props.BoolProperty(
        name="use_lcm",
        default=0,
        description="Higher Speed, lower quality. Try Quality Steps: 1-10",
        update=lcm_updated,
    )
    bpy.types.Scene.remove_silence = bpy.props.BoolProperty(
        name="remove_silence",
        default=1,
        description="Remove Silence",
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
        options={"TEXTEDIT_UPDATE"},
        default="",
        update=update_folder_callback,
    )
    bpy.types.Scene.audio_path = bpy.props.StringProperty(
        name="audio_path",
        default="",
        description="Path to speaker voice",
        options={"TEXTEDIT_UPDATE"},
    )

    # The frame audio duration.
    bpy.types.Scene.audio_speed = bpy.props.IntProperty(
        name="audio_speed",
        default=13,
        min=1,
        max=20,
        description="Speech speed.",
    )
    # The frame audio duration.
    bpy.types.Scene.audio_speed_tts = bpy.props.FloatProperty(
        name="audio_speed_tts",
        default=1.0,
        min=0.3,
        max=1.4,
        description="Speech speed. 1 is normal speed.",
    )
    bpy.types.Scene.ip_adapter_face_folder = bpy.props.StringProperty(
        name="File",
        description="Select a file or folder",
        default="",
        options={"TEXTEDIT_UPDATE"},
        # update=update_ip_adapter_face_callback,
    )
    bpy.types.Scene.ip_adapter_face_files_to_import = bpy.props.CollectionProperty(
        type=IPAdapterFaceProperties
    )
    bpy.types.Scene.ip_adapter_style_folder = bpy.props.StringProperty(
        name="File",
        description="Select a file or folder",
        default="",
        options={"TEXTEDIT_UPDATE"},
        # update=update_ip_adapter_style_callback,
    )
    bpy.types.Scene.ip_adapter_style_files_to_import = bpy.props.CollectionProperty(
        type=IPAdapterStyleProperties
    )

    bpy.types.Scene.genai_out_path = bpy.props.StringProperty(
        name="genai_out_path", default=""
    )
    bpy.types.Scene.genai_out_path = ""

    bpy.types.Scene.minimax_subject = bpy.props.StringProperty(
        name="minimax_subject", default=""
    )

    bpy.types.Scene.omnigen_prompt_1 = bpy.props.StringProperty(
        name="omnigen_prompt_1",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.omnigen_prompt_2 = bpy.props.StringProperty(
        name="omnigen_prompt_2",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.omnigen_prompt_3 = bpy.props.StringProperty(
        name="omnigen_prompt_3",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    bpy.types.Scene.omnigen_strip_1 = bpy.props.StringProperty(
        name="omnigen_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.omnigen_strip_2 = bpy.props.StringProperty(
        name="omnigen_strip_2", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.omnigen_strip_3 = bpy.props.StringProperty(
        name="omnigen_strip_3", options={"TEXTEDIT_UPDATE"}, default=""
    )

    bpy.types.Scene.qwen_strip_1 = bpy.props.StringProperty(
        name="qwen_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.qwen_strip_2 = bpy.props.StringProperty(
        name="qwen_strip_2", options={"TEXTEDIT_UPDATE"}, default=""
    )
    bpy.types.Scene.qwen_strip_3 = bpy.props.StringProperty(
        name="qwen_strip_3", options={"TEXTEDIT_UPDATE"}, default=""
    )

    # The guidance number.
    bpy.types.Scene.img_guidance_scale = bpy.props.FloatProperty(
        name="img_guidance_scale",
        default=1.6,
        min=0,
        max=100,
    )
    bpy.types.Scene.chat_exaggeration = bpy.props.FloatProperty(
        name="Exaggeration",
        default=0.5,
        min=0,
        max=2,
        description="Chatterbox exaggeration",
    )
    bpy.types.Scene.chat_temperature = bpy.props.FloatProperty(
        name="Temperature",
        default=0.8,
        min=0,
        max=5,
        description="Chatterbox Temperature",
    )
    bpy.types.Scene.chat_pace = bpy.props.FloatProperty(
        name="Pace",
        default=0.5,
        min=0,
        max=1,
        description="Chatterbox Pace",
    )
    bpy.types.Scene.kontext_strip_1 = bpy.props.StringProperty(
        name="kontext_strip_1", options={"TEXTEDIT_UPDATE"}, default=""
    )

    bpy.types.Scene.illumination_style = bpy.props.EnumProperty(name="Lighting Style", items=get_enum_items(ILLUMINATION_OPTIONS), default="sunshine from window")
    bpy.types.Scene.light_direction = bpy.props.EnumProperty(name="Light Direction", items=get_enum_items(DIRECTION_OPTIONS), default="auto")
 

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
    del bpy.types.Scene.aurasr
    del bpy.types.Scene.adetailer
    del bpy.types.Scene.generatorai_styles
    del bpy.types.Scene.inpaint_selected_strip
    del bpy.types.Scene.out_frame
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
