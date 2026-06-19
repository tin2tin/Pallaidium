import platform
gfx_device = "cpu"
os_platform = platform.system()


# Set True to re-enable [render_meta_child_to_path] debug logging.
_DEBUG = False


def _dbg(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


import bpy
from bpy_extras.io_utils import ExportHelper
import ctypes
import random
import site
import sysconfig
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

ADDON_ID = __package__.rsplit(".", 1)[0]

print("Python: " + sys.version)

site_packages_dir = os.path.join(bpy.utils.user_resource("DATAFILES"), "Pallaidium", "site-packages")
os.makedirs(site_packages_dir, exist_ok=True)
print("Pallaidium site-packages:", site_packages_dir)

dir_path = os.path.join(bpy.utils.user_resource("DATAFILES"), "Pallaidium Media")

os.makedirs(dir_path, exist_ok=True)

def _prune_stale_blender_paths():
    # Remove sys.path entries from other Blender installs that have permission
    # issues. Python raises PermissionError when reading .py files from a locked
    # old install, which bypasses bare `except ImportError` guards and crashes jobs.
    blender_root = os.path.normcase(
        os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    )
    our_dir = os.path.normcase(site_packages_dir)
    pruned = []
    for p in sys.path:
        if not p:
            pruned.append(p)
            continue
        pn = os.path.normcase(p)
        if pn == our_dir:
            pruned.append(p)
            continue
        if "site-packages" in pn and blender_root not in pn:
            parent = os.path.normcase(os.path.abspath(os.path.join(p, "..", "..", "..", "..")))
            if "blender" in parent and blender_root not in parent:
                continue
        pruned.append(p)
    sys.path[:] = pruned

_prune_stale_blender_paths()

if site_packages_dir and site_packages_dir not in sys.path:
    sys.path.insert(0, site_packages_dir)

warnings.filterwarnings("ignore", category=FutureWarning, module="xformers.*")

warnings.filterwarnings(
    "ignore", category=UserWarning, message="1Torch was not compiled"
)

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")

warnings.filterwarnings("ignore", category=UserWarning, message="FutureWarning: ")

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

warnings.filterwarnings("ignore", category=UserWarning, message="The operator.*is not current")

warnings.filterwarnings("ignore", category=UserWarning, message="Converting a tensor to a Python boolean")

logging.getLogger("xformers").setLevel(logging.ERROR)

logging.getLogger("diffusers.models.modeling_utils").setLevel(logging.CRITICAL)

# Detect GPU device without importing torch (avoids slow startup).
# torch is imported lazily inside model load/generate calls instead.
try:
    if os_platform == "Windows":
        ctypes.CDLL("nvcuda.dll")
        gfx_device = "cuda"
    elif os_platform == "Linux":
        ctypes.CDLL("libcuda.so.1")
        gfx_device = "cuda"
    elif os_platform == "Darwin":
        # Apple Silicon always has MPS; check via sysctl without spawning torch.
        import subprocess as _sp
        _r = _sp.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=2)
        if _r.stdout.strip() == "1":
            gfx_device = "mps"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
except Exception:
    pass

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

os_platform = platform.system()

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
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/styles.json"
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
    for ch in range(1, len(bpy.context.scene.sequence_editor.strips_all) + 1):
        for seq in bpy.context.scene.sequence_editor.strips_all:
            if (
                seq.channel == ch
                and seq.frame_final_start < end_frame
                and (seq.frame_final_start + seq.frame_final_duration) > start_frame
            ):
                break
        else:
            return ch
    return 1

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
    addon_prefs = preferences.addons[ADDON_ID].preferences
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

    seq_editor = bpy.context.scene.sequence_editor
    if seq_editor is None:
        return

    # Stop the prefetch thread before removing the strip to avoid a race
    # condition where the prefetch thread reads strip memory after it is freed
    # by strips.remove() (Blender bug: seq_strip_free_ex frees before joining).
    was_prefetch = getattr(seq_editor, "use_prefetch", False)
    try:
        if was_prefetch:
            seq_editor.use_prefetch = False
        seq_editor.strips.remove(input_strip)
    except ReferenceError:
        pass
    except Exception as e:
        print(f"Failed to remove strip: {e}")
    finally:
        if was_prefetch:
            try:
                seq_editor.use_prefetch = True
            except Exception:
                pass

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

    scene = bpy.context.sequencer_scene
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

        import torch

        total_vram = 0
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            total_vram += properties.total_memory
        return (total_vram / (1024**3)) <= 16  # Y/N under 16 GB?
    except:
        print("Torch not found!")
        return True

def clear_cuda_cache():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError, PermissionError, OSError):
        pass

def release_model_cache(cache: dict) -> None:
    import gc
    skip = {"last_model_card"}
    for key in list(cache.keys()):
        if key in skip:
            continue
        obj = cache[key]
        if obj is not None:
            cache[key] = None
            del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError, PermissionError, OSError):
        pass

def python_exec():
    """Returns the path to the Blender internal python executable"""
    return sys.executable

def find_strip_by_name(scene, name):
    if scene.sequence_editor is None:
        return None
    for sequence in scene.sequence_editor.strips:
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
    if strip.type == "SOUND":
        sound_path = bpy.path.abspath(strip.filepath)
        return sound_path
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
        return None

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

def python_exec():
    return sys.executable

def install_requirements_binary_only(requirements_file):
    """
    Installs using --only-binary=:all: into the user site-packages.
    """
    if os.path.getsize(requirements_file) == 0:
        return True

    pybin = python_exec()
    cmd = [
        pybin, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--no-deps",
        "--upgrade",
        "--only-binary=:all:",
        "--target", site_packages_dir,
        "-r", requirements_file,
    ]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing binaries: {e}")
        return False

def install_requirements_allow_source(requirements_file):
    """
    Installs WITHOUT --only-binary into the user site-packages.
    """
    if os.path.getsize(requirements_file) == 0:
        return True

    pybin = python_exec()
    cmd = [
        pybin, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--no-deps",
        "--upgrade",
        "--target", site_packages_dir,
        "-r", requirements_file,
    ]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing source libs: {e}")
        return False

import re as _re
_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*[mK]')

def run_pip_streaming(cmd: list, on_line=None, cancel_event=None) -> bool:
    """Run a pip command with line-by-line stdout streaming. Returns True on success."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = _ANSI_RE.sub("", line).rstrip()
            if line and on_line:
                on_line(line)
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                proc.wait()
                return False
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        if on_line:
            on_line(f"Error: {e}")
        return False


def write_requirements_file(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def get_platform_specs():
    return platform.system()

class SmartSkipManager:
    @staticmethod
    def extract_package_name(line):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"): return None
        if "git+" in line or "http" in line:
            basename = line.split('/')[-1]
            clean = basename.replace('.git', '').replace('.whl', '')
            if "-" in clean: return clean.split('-')[0]
            return clean
        name = re.split(r'[=<>!~]', line)[0].strip()
        if "[" in name: name = name.split("[")[0]
        return name

    @staticmethod
    def parse_req_version(line):
        name = SmartSkipManager.extract_package_name(line)
        if not name: return None, None
        if "==" in line:
            try:
                parts = line.split("==")
                return name, parts[1].strip().split(' ')[0]
            except: return name, None
        return name, None

    @staticmethod
    def is_installed(line):
        name, req_version = SmartSkipManager.parse_req_version(line)
        if not name: return False

        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            try:
                alt_name = name.replace("-", "_") if "-" in name else name.replace("_", "-")
                installed_version = importlib.metadata.version(alt_name)
            except importlib.metadata.PackageNotFoundError:
                return False

        # Verify the package lives in our dedicated site-packages and is intact.
        # This must run before the git shortcut so git packages are also relocated
        # when found in the wrong directory (e.g. an old Blender purelib).
        try:
            dist = importlib.metadata.Distribution.from_name(name)
            dist_info_dir = getattr(dist, "_path", None)
            if dist_info_dir is not None:
                site_dir = os.path.normpath(os.path.dirname(str(dist_info_dir)))
                target   = os.path.normpath(site_packages_dir)
                if os.path.normcase(site_dir) != os.path.normcase(target):
                    print(f"  [RELOCATE] {name} found at {site_dir}, reinstalling to {target}")
                    return False

                record_text = dist.read_text("RECORD")
                if record_text:
                    for rec_line in record_text.splitlines():
                        file_rel = rec_line.split(",")[0].strip()
                        if not file_rel:
                            continue
                        if file_rel.startswith("__pycache__") or ".dist-info" in file_rel:
                            continue
                        norm = file_rel.replace("\\", "/")
                        if "/bin/" in norm or "/Scripts/" in norm or norm.startswith("../"):
                            continue
                        if not os.path.exists(os.path.normpath(os.path.join(site_dir, file_rel))):
                            print(f"  [BROKEN] {name} {installed_version}: {file_rel} missing, will reinstall")
                            return False
                        break
                    pkg_dir_name = name.replace("-", "_")
                    if "." not in pkg_dir_name:
                        pkg_dir = os.path.join(site_dir, pkg_dir_name)
                        if os.path.isdir(pkg_dir) and not os.path.exists(
                            os.path.join(pkg_dir, "__init__.py")
                        ):
                            print(f"  [BROKEN] {name} {installed_version}: {pkg_dir_name}/__init__.py missing, will reinstall")
                            return False
        except Exception:
            pass

        # Git/URL installs: location verified above, skip version comparison
        if "git+" in line or "http" in line:
            print(f"  [SKIP] {name} is already installed.")
            return True

        if req_version and installed_version != req_version:
            print(f"  [UPDATE] {name}: Installed {installed_version} != Required {req_version}")
            return False

        print(f"  [SKIP] {name} {installed_version} is already installed.")
        return True

    @staticmethod
    def filter_existing(requirements_list):
        needed = []
        for line in requirements_list:
            if not SmartSkipManager.is_installed(line):
                needed.append(line)
        return needed

class BlenderInternalManager:
    @staticmethod
    def get_protected_modules():
        return {
            "pip", "setuptools", "wheel", "ensurepip", "_distutils_hack", "distutils",
            "numpy", "requests", "cython", "zstandard", 
            "urllib3", "idna", "certifi", "charset-normalizer", 
            "openimageio", "pyopencolorio", "materialx", "oslquery", 
            "mesonbuild", "autopep8", "pycodestyle", 
            "bpy", "mathutils", "gpu", "bl_math", "bl_ui_utils"
        }

    @staticmethod
    def is_protected(package_name):
        if not package_name: return False
        clean = package_name.lower().replace("_", "-")
        return clean in BlenderInternalManager.get_protected_modules()

    @staticmethod
    def filter_list(requirements_list):
        safe_list = []
        for line in requirements_list:
            name = SmartSkipManager.extract_package_name(line)
            if not BlenderInternalManager.is_protected(name):
                safe_list.append(line)
        return safe_list

def _linux_flash_attn_compatible():
    """Return True only if system nvcc version matches the torch CUDA 12.8 target.

    flash-attn must be compiled from source on Linux. If the system CUDA toolkit
    version (reported by nvcc) differs from torch's cu128 target, the build fails.
    Skip the install rather than leaving the user with a noisy error batch.
    """
    import subprocess, re
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL, timeout=5, text=True
        )
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2))) == (12, 8)
    except Exception:
        pass
    return False


class DependencyManager:
    def __init__(self):
        self.os_platform = get_platform_specs()
        self.py_major = sys.version_info[0]
        self.py_minor = sys.version_info[1]

    def get_phase_1_5_source_libs(self):
        """
        Pure Python libs or libs needing source build (no wheels).
        """
        return [
            "antlr4-python3-runtime==4.9.3",
            "argbind==0.3.9",
            "chatterbox-tts",
            "demucs-onnx",
            "dctorch==0.1.2",
            "einx==0.3.0",
            "encodec==0.1.1",
            "imhist==0.0.4",
            "julius==0.2.7",
            #"pathtools==0.1.2",
            "progressbar==2.5",
            "pyloudnorm==0.1.1",
            "pystoi==0.4.1",
            "pyvers==0.1.0",
            "randomname==0.2.1",
            "s3tokenizer==0.2.0",
            "screenplain==0.11.1",
            "torch-stoi==0.2.3",
            "transformers-stream-generator==0.0.5",
            "wget==3.2",
            "x-transformers==2.11.23",
            "qwen-vl-utils",
        ]

    def get_phase_2_torch(self):
        if self.os_platform == "Windows":
            return [
                #"--index-url https://download.pytorch.org/whl/cu124",
                "--index-url https://download.pytorch.org/whl/cu128",
                #"torch==2.6.0+cu124",
                "torch==2.9.1+cu128", #torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
                #"torchvision==0.21.0+cu124",
                "torchvision==0.24.1+cu128",
                #"torchaudio==2.6.0+cu124",
                "torchaudio==2.9.1+cu128",
                #"xformers"
            ]
        elif self.os_platform == "Linux":
            return [
                "--index-url https://download.pytorch.org/whl/cu128",
                "torch==2.9.1+cu128",
                "torchvision==0.24.1+cu128",
                "torchaudio==2.9.1+cu128",
                # nvidia CUDA runtime wheels — fix libcusparseLt.so.0 and related .so import errors
                "nvidia-cublas-cu12==12.8.4.1",
                "nvidia-cuda-runtime-cu12==12.8.90",
                "nvidia-cusparselt-cu12==0.7.1",
                "nvidia-cudnn-cu12==9.10.2.21",
                "nvidia-nccl-cu12==2.27.5",
                "nvidia-nvtx-cu12==12.8.90",
                # xformers removed: 0.0.35 was built for Python 3.10/PyTorch 2.10 and causes
                # "cannot import name 'GroupName'" from diffusers on Python 3.13 + PyTorch 2.9
            ]
        else:
            # macOS — CPU / MPS builds from PyPI
            return ["torch", "torchvision", "torchaudio"]

    def get_phase_linux_binary_only(self):
        """Packages that must be installed as binary wheels on Linux.

        thinc and spacy have Cython extensions that fail to compile from source under
        Python 3.13 due to C API changes. Binary wheels work correctly.
        """
        if self.os_platform != "Linux":
            return []
        return [
            "thinc",
            "spacy",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
        ]

    # for installing branch: git+https://github.com/huggingface/diffusers.git@ltx2-i2v-lora-mixin-fix

    def get_phase_3_git_and_extensions(self):
        reqs = [
            "git+https://github.com/huggingface/diffusers.git",
            #"git+https://github.com/SWivid/F5-TTS.git",
            "faster-qwen3-tts",
            #"git+https://github.com/QwenLM/Qwen3-TTS.git",
            #"git+https://github.com/huggingface/parler-tts.git",
            "stable-audio-tools",
            "torcheval",
            "torchao",
            # spacy and model wheel: Windows/macOS install here; Linux uses get_phase_linux_binary_only()
        ]

        if self.py_major == 3 and self.py_minor >= 8:
             reqs.append("git+https://github.com/huggingface/image_gen_aux")

        if self.os_platform == "Windows":
            reqs.extend([
                "spacy",
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
                #"https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
                "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp313-cp313-win_amd64.whl",
                "git+https://github.com/hkchengrex/MMAudio.git",
                #"https://github.com/woct0rdho/triton-windows/releases/download/empty/triton-3.4.0-py3-none-any.whl",
                #"triton-windows<3.3",
                "nvidia-vfx",
            ])
        elif self.os_platform == "Linux":
            reqs.extend([
                "git+https://github.com/hkchengrex/MMAudio.git",
                "triton==3.5.1",
                "sageattention==1.0.6",
            ])
            # flash-attn requires source compilation on Linux; skip when the system CUDA
            # toolkit version doesn't match torch's cu128 target to avoid build failures
            if _linux_flash_attn_compatible():
                reqs.append("flash-attn")
        else:
            # macOS — CPU / MPS
            reqs.extend([
                "spacy",
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
                "flash-attn",
            ])
        return reqs

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
    addon_prefs = preferences.addons[ADDON_ID].preferences
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
            "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora",
            "romanfratric234/FLUX.1-Depth-dev-lora",
            "Runware/FLUX.1-Redux-dev",
            "kontext-community/relighting-kontext-dev-lora-v3",
            "jdopensource/JoyAI-Image-Edit-Diffusers",
        }:
            scene.input_strips = "input_strips"

        try:
            from ..models import get_plugin as _gp
            _p = _gp(image_model)
            if _p:
                scene.movie_num_inference_steps = _p.PARAMS.steps
                scene.movie_num_guidance = _p.PARAMS.guidance
                if getattr(_p, "requires_input_strip", False) and scene.input_strips != "input_strips":
                    scene.input_strips = "input_strips"
        except Exception:
            pass

    # Movie Type Handling
    elif scene_type == "movie":
        _p = None
        try:
            from ..models import get_plugin as _gp
            _p = _gp(movie_model)
            if _p:
                scene.movie_num_inference_steps = _p.PARAMS.steps
                scene.movie_num_guidance = _p.PARAMS.guidance
                scene.generate_movie_x = _p.PARAMS.width
                scene.generate_movie_y = _p.PARAMS.height
                scene.generate_movie_frames = _p.PARAMS.frames
        except Exception:
            pass
        if _p and getattr(_p, "requires_input_strip", False) and scene.input_strips != "input_strips":
            scene.input_strips = "input_strips"
        elif (
            movie_model in {
                "Hailuo/MiniMax/img2vid",
                "Hailuo/MiniMax/subject2vid"
            }
        ) and scene.input_strips != "input_strips":
            scene.input_strips = "input_strips"

    elif scene_type == "audio":
        if audio_model == "StemSplitter" and scene.input_strips != "input_strips":
            scene.input_strips = "input_strips"
        try:
            from ..models import get_plugin as _gp
            _p = _gp(audio_model)
            if _p:
                scene.movie_num_inference_steps = _p.PARAMS.steps
                if getattr(_p, "requires_input_strip", False) and scene.input_strips != "input_strips":
                    scene.input_strips = "input_strips"
        except Exception:
            pass

    # Reset style for output types that don't use image styles
    if scene_type in {"text", "audio"} and hasattr(scene, "generatorai_styles"):
        scene.generatorai_styles = "no_style"

    # Common Handling for Selected Strip
    if scene_type in {"movie", "audio"} or image_model == "xinsir/controlnet-scribble-sdxl-1.0":
        scene.inpaint_selected_strip = ""

    # LORA Handling
    if scene.lora_folder:
        bpy.ops.lora.refresh_files()

    # Clear Paths if Input is Prompt
    if scene.input_strips == "input_prompt":
        scene.movie_path = ""
        scene.image_path = ""
        scene.sound_path = ""

def output_strips_updated(self, context):
    prefs = context.preferences
    addon_prefs = prefs.addons[ADDON_ID].preferences
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
            "fuliucansheng/FLUX.1-Canny-dev-diffusers-lora",
            "romanfratric234/FLUX.1-Depth-dev-lora",
            "Runware/FLUX.1-Redux-dev",
            "kontext-community/relighting-kontext-dev-lora-v3",
        ]:
            scene.input_strips = "input_strips"
        else:
            try:
                from ..models import get_plugin as _gp
                _p = _gp(image_model)
                if _p:
                    scene.movie_num_inference_steps = _p.PARAMS.steps
                    scene.movie_num_guidance = _p.PARAMS.guidance
                    if getattr(_p, "requires_input_strip", False) and scene.input_strips != "input_strips":
                        scene.input_strips = "input_strips"
            except Exception:
                pass

    # === MOVIE TYPE === #
    elif type == "movie":
        try:
            from ..models import get_plugin as _gp
            _p = _gp(movie_model)
            if _p:
                movie_res_x = _p.PARAMS.width
                movie_res_y = _p.PARAMS.height
                movie_frames = _p.PARAMS.frames
                movie_inference = _p.PARAMS.steps
                movie_guidance = _p.PARAMS.guidance
        except Exception:
            pass
        if movie_model in [
            "Hailuo/MiniMax/img2vid",
            "Hailuo/MiniMax/subject2vid"
        ]:
            scene.input_strips = "input_strips"

    # === AUDIO TYPE === #
    elif type == "audio":
        if audio_model == "StemSplitter" and scene.input_strips != "input_strips":
            scene.input_strips = "input_strips"
        try:
            from ..models import get_plugin as _gp
            _p = _gp(audio_model)
            if _p:
                scene.movie_num_inference_steps = _p.PARAMS.steps
                if getattr(_p, "requires_input_strip", False) and scene.input_strips != "input_strips":
                    scene.input_strips = "input_strips"
        except Exception:
            pass

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

def copy_struct(source, target):
    """
    Robustly copies properties from source to target. 
    Handles cases where source has properties that target does not (e.g. Movie -> Sound).
    """
    if not source or not target:
        return

    # Properties to ignore
    ignore_props = {"rna_type", "name", "name_full", "original", "is_evaluated"}

    for name, prop in source.bl_rna.properties.items():
        if name in ignore_props:
            continue

        # 1. Safely get the source value
        try:
            src_value = getattr(source, name)
        except (AttributeError, TypeError):
            continue

        # 2. Try to set the property on the target
        try:
            setattr(target, name, src_value)
        except AttributeError:
            # 3. If setattr failed, it's either:
            #    A) A read-only nested struct (like strip.transform)
            #    B) A property that doesn't exist on the target (like use_deinterlace on Sound)
            
            # Safely check if the target actually HAS this property
            # We use default=None to prevent the crash you are seeing
            tgt_value = getattr(target, name, None)

            # Only recurse if both exist and are valid structs
            if tgt_value is not None and src_value is not None:
                if hasattr(src_value, "bl_rna"):
                    copy_struct(src_value, tgt_value)
                    
        except TypeError:
            # Handles issues like attempting to write to collection properties
            pass

def get_render_strip(self, context, strip, meta_strip=None):
    """Render selected strip to hard-disk. Returns the new strip object or None."""
    
    # Access the VSE scene properly for 5.1
    vse_scene = getattr(context, 'sequencer_scene', context.scene)
    if not vse_scene or not vse_scene.sequence_editor: 
        return None
    
    seq_editor = vse_scene.sequence_editor
    
    # 1. PRE-RENDER PREPARATION: Disable Caching/Prefetching to stop crashes
    # Note: 5.1 uses 'strips' instead of 'sequences'
    orig_prefetch = seq_editor.use_prefetch
    orig_cache = seq_editor.use_cache_raw
    orig_mute_states = {s: s.mute for s in seq_editor.strips}
    
    seq_editor.use_prefetch = False
    seq_editor.use_cache_raw = False
    
    # 2. ISOLATE STRIP
    target = meta_strip if meta_strip else strip
    render_start = int(target.frame_final_start)
    render_duration = int(target.frame_final_duration)
    render_end = render_start + render_duration - 1
    
    # Mute others (using strips collection)
    for s in seq_editor.strips:
        s.mute = (s != target)
    
    orig_f_start = vse_scene.frame_start
    orig_f_end = vse_scene.frame_end
    vse_scene.frame_start = render_start
    vse_scene.frame_end = render_end
    
    # 3. RENDER LOGIC
    addon_prefs = bpy.context.preferences.addons[ADDON_ID].preferences
    rendered_dir = os.path.join(addon_prefs.generator_ai, str(date.today()), "Rendered_Strips")
    os.makedirs(rendered_dir, exist_ok=True)
    
    safe_name = re.sub(r'[^\w]', '', strip.name)
    
    try:
        if strip.type == "SOUND":
            output_path = os.path.abspath(os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}.wav"))
            bpy.ops.sound.mixdown(filepath=output_path, container='WAV', codec='PCM')
        else:
            output_path = os.path.abspath(os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}.mp4"))
            vse_scene.render.filepath = output_path
            
            # 5.1 API: Configure video output without setting file_format to FFMPEG
            # We set the media type to VIDEO, which tells the renderer to use FFMPEG internally
            if hasattr(vse_scene.render.image_settings, "media_type"):
                vse_scene.render.image_settings.media_type = 'VIDEO'
            
            # Ensure the container is set for video
            vse_scene.render.ffmpeg.format = 'MPEG4'
            vse_scene.render.ffmpeg.codec = 'H264'
            vse_scene.render.ffmpeg.audio_codec = 'AAC'
            
            # Force sequencer rendering
            vse_scene.render.use_sequencer = True
            
            # Execute render
            bpy.ops.render.render(animation=True, write_still=False)
            
            # Handle FFMPEG automatic output pathing
            # Blender often appends frame numbers (e.g., .mp40001)
            # We look for the actual file generated
            if not os.path.exists(output_path):
                pattern = os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}*.mp4")
                files = glob.glob(pattern)
                if files:
                    files.sort(key=os.path.getmtime)
                    output_path = files[-1]
            
    finally:
        # 4. RESTORE STATE
        for s, state in orig_mute_states.items():
            if s: s.mute = state
        vse_scene.frame_start = orig_f_start
        vse_scene.frame_end = orig_f_end
        seq_editor.use_prefetch = orig_prefetch
        seq_editor.use_cache_raw = orig_cache
        
    # 5. IMPORT RESULT
    if os.path.exists(output_path):
        channel = max([s.channel for s in seq_editor.strips], default=0) + 1
        if strip.type == "SOUND":
            new_strip = seq_editor.strips.new_sound(name="rendered_sound", filepath=output_path, channel=channel, frame_start=render_start)
        else:
            new_strip = seq_editor.strips.new_movie(name="rendered_movie", filepath=output_path, channel=channel, frame_start=render_start)
        
        seq_editor.active_strip = new_strip
        return new_strip
        
    return None

# Paths created by render_strip_to_path(); cleared after each generation run.
_rendered_temp_paths: set = set()


def render_strip_to_path(context, strip, image_output=False):
    """Render a VSE strip through the pipeline and return the output file path.

    Unlike get_render_strip() this does NOT add a new strip to the VSE — it
    writes a temp file and returns its path.  The path is registered in
    _rendered_temp_paths so callers can bulk-delete after inference.

    image_output=True  → single-frame PNG (for image plugins)
    image_output=False → animation MP4  (for video plugins)
    Returns None on failure.
    """
    vse_scene = getattr(context, 'sequencer_scene', context.scene)
    if not vse_scene or not vse_scene.sequence_editor:
        return None

    seq_editor = vse_scene.sequence_editor
    orig_prefetch   = seq_editor.use_prefetch
    orig_cache      = seq_editor.use_cache_raw
    orig_mute_states = {s: s.mute for s in seq_editor.strips}
    orig_f_start    = vse_scene.frame_start
    orig_f_end      = vse_scene.frame_end
    orig_f_current  = vse_scene.frame_current
    orig_filepath   = vse_scene.render.filepath
    orig_format     = vse_scene.render.image_settings.file_format
    orig_media      = getattr(vse_scene.render.image_settings, "media_type", None)
    orig_use_seq    = vse_scene.render.use_sequencer
    orig_res_x      = vse_scene.render.resolution_x
    orig_res_y      = vse_scene.render.resolution_y
    orig_res_pct    = vse_scene.render.resolution_percentage

    seq_editor.use_prefetch = False
    seq_editor.use_cache_raw = False

    render_start = int(strip.frame_final_start)
    render_end   = int(strip.frame_final_start + strip.frame_final_duration - 1)

    for s in seq_editor.strips:
        s.mute = (s != strip)

    vse_scene.frame_start = render_start
    vse_scene.frame_end   = render_end

    # Render IMAGE/MOVIE strips at their native resolution so the output fills
    # the frame with no transparent margins. Otherwise a strip smaller than the
    # scene render size composites centered with letterbox padding, which makes
    # downstream consumers (Florence-2 Box Editor background, img2img/init
    # frames) misalign with content that was analyzed/resized to fill.
    target_res = None
    if strip.type in ("IMAGE", "MOVIE"):
        try:
            elem = strip.elements[0]
            if elem.orig_width and elem.orig_height:
                target_res = (elem.orig_width, elem.orig_height)
        except Exception:
            target_res = None
    if target_res:
        vse_scene.render.resolution_x          = target_res[0]
        vse_scene.render.resolution_y          = target_res[1]
        vse_scene.render.resolution_percentage = 100

    addon_prefs  = bpy.context.preferences.addons[ADDON_ID].preferences
    rendered_dir = os.path.join(addon_prefs.generator_ai, str(date.today()), "Rendered_Strips")
    os.makedirs(rendered_dir, exist_ok=True)
    safe_name   = re.sub(r'[^\w]', '', strip.name)
    output_path = None

    try:
        if strip.type == "SOUND":
            output_path = os.path.abspath(
                os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}.wav"))
            bpy.ops.sound.mixdown(filepath=output_path, container='WAV', codec='PCM')

        elif image_output:
            base = os.path.abspath(
                os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}_img"))
            vse_scene.render.filepath = base
            if orig_media is not None:
                vse_scene.render.image_settings.media_type = 'IMAGE'
            vse_scene.render.image_settings.file_format = 'PNG'
            vse_scene.render.use_sequencer = True
            vse_scene.frame_current = render_start
            bpy.ops.render.render(animation=False, write_still=True)
            # Blender appends frame digits; find the actual file
            for pad in (4, 5, 6):
                candidate = f"{base}{render_start:0{pad}d}.png"
                if os.path.exists(candidate):
                    output_path = candidate
                    break
            if not output_path:
                files = sorted(glob.glob(f"{base}*.png"), key=os.path.getmtime)
                if files:
                    output_path = files[-1]

        else:
            output_path = os.path.abspath(
                os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}.mp4"))
            vse_scene.render.filepath = output_path
            if orig_media is not None:
                vse_scene.render.image_settings.media_type = 'VIDEO'
            vse_scene.render.ffmpeg.format      = 'MPEG4'
            vse_scene.render.ffmpeg.codec       = 'H264'
            vse_scene.render.ffmpeg.audio_codec = 'AAC'
            vse_scene.render.use_sequencer = True
            bpy.ops.render.render(animation=True, write_still=False)
            if not os.path.exists(output_path):
                files = sorted(
                    glob.glob(os.path.join(rendered_dir,
                                           f"{safe_name}_{render_start:06d}*.mp4")),
                    key=os.path.getmtime)
                if files:
                    output_path = files[-1]

    finally:
        for s, state in orig_mute_states.items():
            if s:
                s.mute = state
        vse_scene.frame_start   = orig_f_start
        vse_scene.frame_end     = orig_f_end
        vse_scene.frame_current = orig_f_current
        seq_editor.use_prefetch  = orig_prefetch
        seq_editor.use_cache_raw = orig_cache
        vse_scene.render.filepath = orig_filepath
        if orig_media is not None:
            vse_scene.render.image_settings.media_type = orig_media
        vse_scene.render.image_settings.file_format = orig_format
        vse_scene.render.use_sequencer = orig_use_seq
        vse_scene.render.resolution_x          = orig_res_x
        vse_scene.render.resolution_y          = orig_res_y
        vse_scene.render.resolution_percentage = orig_res_pct

    if output_path and os.path.exists(output_path):
        _rendered_temp_paths.add(output_path)
        return output_path
    return None


def render_meta_child_to_path(context, meta_strip, child_strip, image_output=False):
    """Render one child strip inside a META through the VSE compositor.

    Unlike render_strip_to_path(), this keeps the META unmuted (so the child's
    content is composited correctly) while muting all other top-level strips.

    For SOUND children the frame range is set to the META strip's full range so
    that the exported WAV has exactly the META's duration and the audio sits at
    the correct relative position within it (silence fills any gap before/after
    the child strip).  This guarantees the downstream model receives audio whose
    length matches the video it will generate.

    image_output=True              → single-frame PNG at child.frame_final_start
    child.type == 'SOUND'          → PCM WAV covering the META's full duration
    otherwise                      → MP4 for the child's trimmed duration
    Returns the absolute output path, or None on failure.
    """
    vse_scene = getattr(context, 'sequencer_scene', context.scene)
    if not vse_scene or not vse_scene.sequence_editor:
        return None

    seq_editor = vse_scene.sequence_editor
    orig_prefetch    = seq_editor.use_prefetch
    orig_cache       = seq_editor.use_cache_raw
    orig_mute_states = {s: s.mute for s in seq_editor.strips}
    orig_f_start     = vse_scene.frame_start
    orig_f_end       = vse_scene.frame_end
    orig_f_current   = vse_scene.frame_current
    orig_filepath    = getattr(vse_scene.render, 'filepath', '')
    orig_format      = vse_scene.render.image_settings.file_format
    orig_media       = getattr(vse_scene.render.image_settings, 'media_type', None)
    orig_use_seq     = vse_scene.render.use_sequencer

    seq_editor.use_prefetch  = False
    seq_editor.use_cache_raw = False

    # For SOUND: render the full META range so the exported audio duration equals
    # the META duration and the child's audio lands at its correct relative offset.
    # For IMAGE/MOVIE: use the child's own range as before.
    if child_strip.type == "SOUND":
        render_start = int(meta_strip.frame_final_start)
        render_end   = int(meta_strip.frame_final_start + meta_strip.frame_final_duration - 1)
    else:
        render_start = int(child_strip.frame_final_start)
        render_end   = int(child_strip.frame_final_start + child_strip.frame_final_duration - 1)

    # Keep META unmuted so children composite correctly; mute everything else
    for s in seq_editor.strips:
        s.mute = (s != meta_strip)

    vse_scene.frame_start = render_start
    vse_scene.frame_end   = render_end

    addon_prefs  = bpy.context.preferences.addons[ADDON_ID].preferences
    rendered_dir = os.path.join(addon_prefs.generator_ai, str(date.today()), "Rendered_Strips")
    os.makedirs(rendered_dir, exist_ok=True)
    safe_name   = re.sub(r'[^\w]', '', child_strip.name) or "strip"
    output_path = None

    try:
        if child_strip.type == "SOUND":
            output_path = os.path.abspath(
                os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}_meta_audio.wav"))
            _fps_sound   = vse_scene.render.fps / max(1.0, getattr(vse_scene.render, 'fps_base', 1.0))
            _expected_s  = meta_strip.frame_final_duration / _fps_sound
            # How many seconds the META's trim clips from the child's beginning.
            # When child starts before meta_final_start, that many seconds of the
            # child are invisible; we must skip them in both the source read and
            # the effective duration so the output WAV matches what Blender plays.
            _meta_clip_s = max(0.0, (meta_strip.frame_final_start - child_strip.frame_final_start) / _fps_sound)
            _src_start_s = child_strip.frame_offset_start / _fps_sound + _meta_clip_s
            _child_dur_s = max(0.0, child_strip.frame_final_duration / _fps_sound - _meta_clip_s)
            _child_off_s = max(0.0, (child_strip.frame_final_start - meta_strip.frame_final_start) / _fps_sound)
            _src_path    = bpy.path.abspath(child_strip.sound.filepath)

            _dbg(f"[render_meta_child_to_path] ── SOUND IN ──────────────────────────")
            print(f"  source file      : {_src_path!r}")
            print(f"  fps              : {_fps_sound:.4f}")
            print(f"  meta  frames     : {meta_strip.frame_final_start} – {meta_strip.frame_final_start + meta_strip.frame_final_duration - 1}  ({meta_strip.frame_final_duration} fr = {_expected_s:.3f}s)")
            print(f"  child frames     : {child_strip.frame_final_start} – {child_strip.frame_final_start + child_strip.frame_final_duration - 1}  ({child_strip.frame_final_duration} fr = {child_strip.frame_final_duration / _fps_sound:.3f}s)")
            print(f"  meta clip into child: {_meta_clip_s:.3f}s")
            print(f"  child offset_start (source skip): {child_strip.frame_offset_start} fr + {_meta_clip_s:.3f}s meta clip = {_src_start_s:.3f}s total")
            print(f"  child offset in META : {_child_off_s:.3f}s")
            print(f"  will write       : {_child_dur_s:.3f}s of audio at +{_child_off_s:.3f}s into {_expected_s:.3f}s output")
            _dbg(f"[render_meta_child_to_path] ──────────────────────────────────────")

            # sound.mixdown ignores scene.frame_start/end and always exports the full
            # source file, so use soundfile directly for precise in/out control.
            try:
                import soundfile as _sf
                import numpy as _np
                _info   = _sf.info(_src_path)
                _sr     = _info.samplerate
                _ch     = _info.channels
                _s0     = int(_src_start_s * _sr)
                _s1     = _s0 + int(_child_dur_s * _sr)
                _dbg(f"[render_meta_child_to_path] soundfile read: sr={_sr} ch={_ch} samples [{_s0}:{_s1}]")
                _child_data, _ = _sf.read(_src_path, start=_s0, stop=_s1, always_2d=True)
                _total_smp = int(_expected_s * _sr)
                _off_smp   = int(_child_off_s * _sr)
                _out_arr   = _np.zeros((_total_smp, _ch), dtype='float32')
                _end_smp   = min(_off_smp + len(_child_data), _total_smp)
                _out_arr[_off_smp:_end_smp] = _child_data[:_end_smp - _off_smp]
                _sf.write(output_path, _out_arr, _sr, subtype='PCM_16')
                _dbg(f"[render_meta_child_to_path] soundfile write OK")
            except Exception as _sf_err:
                _dbg(f"[render_meta_child_to_path] soundfile write failed: {_sf_err}")
                import traceback as _tb; _tb.print_exc()

            if os.path.exists(output_path):
                try:
                    import soundfile as _sf_chk
                    _info_chk = _sf_chk.info(output_path)
                    _actual_s = _info_chk.frames / _info_chk.samplerate
                    _dbg(f"[render_meta_child_to_path] ── WAV OUT ───────────────────────────")
                    print(f"  output file    : {output_path!r}")
                    print(f"  actual duration: {_actual_s:.3f}s  ({_info_chk.frames} samples @ {_info_chk.samplerate} Hz)")
                    print(f"  expected       : {_expected_s:.3f}s")
                    print(f"  delta          : {_actual_s - _expected_s:+.3f}s")
                    _dbg(f"[render_meta_child_to_path] ──────────────────────────────────────")
                except Exception as _chk_err:
                    _dbg(f"[render_meta_child_to_path] WAV OUT check failed: {_chk_err}")

        elif image_output:
            base = os.path.abspath(
                os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}_meta_img"))
            vse_scene.render.filepath = base
            if orig_media is not None:
                vse_scene.render.image_settings.media_type = 'IMAGE'
            vse_scene.render.image_settings.file_format = 'PNG'
            vse_scene.render.use_sequencer = True
            vse_scene.frame_current = render_start
            bpy.ops.render.render(animation=False, write_still=True)
            for pad in (4, 5, 6):
                candidate = f"{base}{render_start:0{pad}d}.png"
                if os.path.exists(candidate):
                    output_path = candidate
                    break
            if not output_path:
                files = sorted(glob.glob(f"{base}*.png"), key=os.path.getmtime)
                if files:
                    output_path = files[-1]

        else:
            output_path = os.path.abspath(
                os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}_meta_vid.mp4"))
            vse_scene.render.filepath = output_path
            if orig_media is not None:
                vse_scene.render.image_settings.media_type = 'VIDEO'
            vse_scene.render.ffmpeg.format      = 'MPEG4'
            vse_scene.render.ffmpeg.codec       = 'H264'
            vse_scene.render.ffmpeg.audio_codec = 'AAC'
            vse_scene.render.use_sequencer = True
            bpy.ops.render.render(animation=True, write_still=False)
            if not os.path.exists(output_path):
                files = sorted(
                    glob.glob(os.path.join(rendered_dir,
                                           f"{safe_name}_{render_start:06d}_meta_vid*.mp4")),
                    key=os.path.getmtime)
                if files:
                    output_path = files[-1]

    finally:
        for s, state in orig_mute_states.items():
            if s:
                s.mute = state
        vse_scene.frame_start    = orig_f_start
        vse_scene.frame_end      = orig_f_end
        vse_scene.frame_current  = orig_f_current
        seq_editor.use_prefetch  = orig_prefetch
        seq_editor.use_cache_raw = orig_cache
        vse_scene.render.filepath = orig_filepath
        if orig_media is not None:
            vse_scene.render.image_settings.media_type = orig_media
        vse_scene.render.image_settings.file_format = orig_format
        vse_scene.render.use_sequencer = orig_use_seq

    if output_path and os.path.exists(output_path):
        _rendered_temp_paths.add(output_path)
        return output_path
    return None


def render_strip_to_wav(context, strip):
    """Render any strip (SOUND or MOVIE-with-audio) to a PCM WAV via sound.mixdown.

    Mutes all other strips and restricts the frame range to the strip's trimmed
    in/out points, so exactly the audible region is captured regardless of source
    format, trimming, volume envelopes, or speed effects.
    Returns the absolute path to the generated WAV, or None on failure.
    """
    vse_scene = getattr(context, "sequencer_scene", context.scene)
    if not vse_scene or not vse_scene.sequence_editor:
        return None

    seq_editor = vse_scene.sequence_editor
    orig_prefetch    = seq_editor.use_prefetch
    orig_cache       = seq_editor.use_cache_raw
    orig_mute_states = {s: s.mute for s in seq_editor.strips}
    orig_f_start     = vse_scene.frame_start
    orig_f_end       = vse_scene.frame_end

    seq_editor.use_prefetch  = False
    seq_editor.use_cache_raw = False

    render_start = int(strip.frame_final_start)
    render_end   = int(strip.frame_final_start + strip.frame_final_duration - 1)

    for s in seq_editor.strips:
        s.mute = (s != strip)

    vse_scene.frame_start = render_start
    vse_scene.frame_end   = render_end

    addon_prefs  = bpy.context.preferences.addons[ADDON_ID].preferences
    rendered_dir = os.path.join(addon_prefs.generator_ai, str(date.today()), "Rendered_Strips")
    os.makedirs(rendered_dir, exist_ok=True)

    safe_name   = re.sub(r"[^\w]", "", strip.name)
    output_path = os.path.abspath(
        os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}_stem_input.wav"))

    _fps_stw      = vse_scene.render.fps / max(1.0, getattr(vse_scene.render, 'fps_base', 1.0))
    _expected_stw = strip.frame_final_duration / _fps_stw
    try:
        _src_stw   = bpy.path.abspath(strip.sound.filepath)
        _off_stw   = strip.frame_offset_start / _fps_stw
    except Exception:
        _src_stw = "(unknown)"
        _off_stw = 0.0
    print(f"[render_strip_to_wav] ── SOUND IN ──────────────────────────")
    print(f"  strip            : {strip.name!r}  type={strip.type}")
    print(f"  source file      : {_src_stw!r}")
    print(f"  fps              : {_fps_stw:.4f}")
    print(f"  strip frames     : {render_start} – {render_end}  ({strip.frame_final_duration} fr = {_expected_stw:.3f}s)")
    print(f"  source skip      : frame_offset_start={strip.frame_offset_start} fr = {_off_stw:.3f}s")
    print(f"  mixdown range    : scene.frame_start={render_start}  scene.frame_end={render_end}")
    print(f"  expected WAV dur : {_expected_stw:.3f}s")
    print(f"[render_strip_to_wav] ──────────────────────────────────────")

    try:
        bpy.ops.sound.mixdown(filepath=output_path, container="WAV", codec="PCM")
    finally:
        for s, state in orig_mute_states.items():
            if s:
                s.mute = state
        vse_scene.frame_start    = orig_f_start
        vse_scene.frame_end      = orig_f_end
        seq_editor.use_prefetch  = orig_prefetch
        seq_editor.use_cache_raw = orig_cache

    if not os.path.exists(output_path):
        print("[render_strip_to_wav] mixdown produced no file")
        return None

    # Post-trim: mixdown may ignore the frame range and export the full source.
    # Cap to the strip's actual trimmed duration.
    try:
        import soundfile as _sf_stw
        _info_stw   = _sf_stw.info(output_path)
        _actual_stw = _info_stw.frames / _info_stw.samplerate
        print(f"[render_strip_to_wav] ── WAV OUT ───────────────────────────")
        print(f"  output file    : {output_path!r}")
        print(f"  actual duration: {_actual_stw:.3f}s  ({_info_stw.frames} samples @ {_info_stw.samplerate} Hz)")
        print(f"  expected       : {_expected_stw:.3f}s")
        print(f"  delta          : {_actual_stw - _expected_stw:+.3f}s")
        print(f"[render_strip_to_wav] ──────────────────────────────────────")
        if _actual_stw > _expected_stw + 0.1:
            print(f"[render_strip_to_wav] audio too long — re-trimming")
            import av as _av_stw
            _tmp_stw = output_path + ".trimtmp.wav"
            _trim_ok = False
            with _av_stw.open(output_path) as _cin:
                _astr = next((s for s in _cin.streams if s.type == 'audio'), None)
                if _astr:
                    with _av_stw.open(_tmp_stw, 'w', format='wav') as _cout:
                        _os = _cout.add_stream('pcm_s16le', rate=_astr.codec_context.sample_rate)
                        _w = 0.0
                        for _f in _cin.decode(_astr):
                            if _w >= _expected_stw:
                                break
                            _w += _f.samples / _f.sample_rate
                            _f.pts = None
                            for _p in _os.encode(_f):
                                _cout.mux(_p)
                        for _p in _os.encode(None):
                            _cout.mux(_p)
                    _trim_ok = True
            if _trim_ok and os.path.exists(_tmp_stw):
                # os.replace can fail on Windows if the destination is still
                # locked by the OS/AV scanner; unlink first as a workaround.
                try:
                    os.replace(_tmp_stw, output_path)
                except OSError:
                    try:
                        os.unlink(output_path)
                        os.rename(_tmp_stw, output_path)
                    except OSError as _e2:
                        print(f"[render_strip_to_wav] replace failed: {_e2} — keeping trim as {_tmp_stw!r}")
                        output_path = _tmp_stw
                print(f"[render_strip_to_wav] re-trimmed → {_expected_stw:.2f}s")
    except Exception as _e_stw:
        print(f"[render_strip_to_wav] post-trim failed: {_e_stw}")

    return output_path


def render_meta_audio_to_path(context, meta_strip):
    """Mix all SOUND children inside a META strip to a single WAV of the META's duration.

    Each child is positioned at its correct offset within the META and trimmed via
    frame_offset_start / frame_final_duration using soundfile — no Blender mixdown call
    needed, so the frame-range is exact regardless of source length.

    Returns the absolute path of the output WAV, or None on failure.
    """
    try:
        import soundfile as _sf
        import numpy as _np
    except ImportError as _ie:
        print(f"[render_meta_audio_to_path] soundfile/numpy unavailable: {_ie}")
        return None

    vse_scene = getattr(context, "sequencer_scene", context.scene)
    if not vse_scene or not vse_scene.sequence_editor:
        return None

    sound_children = [c for c in meta_strip.strips if c.type == "SOUND"]
    if not sound_children:
        return None

    fps          = vse_scene.render.fps / max(1.0, getattr(vse_scene.render, "fps_base", 1.0))
    meta_start_f = meta_strip.frame_final_start
    meta_dur_s   = meta_strip.frame_final_duration / fps

    addon_prefs  = bpy.context.preferences.addons[ADDON_ID].preferences
    rendered_dir = os.path.join(addon_prefs.generator_ai, str(date.today()), "Rendered_Strips")
    os.makedirs(rendered_dir, exist_ok=True)
    safe_name    = re.sub(r"[^\w]", "", meta_strip.name) or "meta"
    render_start = int(meta_start_f)
    output_path  = os.path.abspath(
        os.path.join(rendered_dir, f"{safe_name}_{render_start:06d}_meta_audio_mix.wav"))

    print(f"[render_meta_audio_to_path] ── META AUDIO MIX ────────────────────")
    print(f"  meta     : {meta_strip.name!r}  {meta_strip.frame_final_duration} fr = {meta_dur_s:.3f}s")
    print(f"  children : {len(sound_children)} SOUND strip(s)")

    sr      = None
    n_ch    = 2
    out_arr = None
    total_s = 0

    for child in sound_children:
        try:
            src_path = bpy.path.abspath(child.sound.filepath)
        except AttributeError:
            print(f"  WARN: {child.name!r} has no sound.filepath, skipped")
            continue
        if not os.path.isfile(src_path):
            print(f"  WARN: {child.name!r} file not found: {src_path!r}, skipped")
            continue
        try:
            info     = _sf.info(src_path)
            child_sr = info.samplerate

            if sr is None:
                sr      = child_sr
                n_ch    = max(1, info.channels)
                total_s = int(meta_dur_s * sr)
                out_arr = _np.zeros((total_s, n_ch), dtype="float32")
            elif child_sr != sr:
                print(f"  WARN: {child.name!r} sr={child_sr} != output sr={sr}, skipped")
                continue

            # How many seconds the META's trim clips from the child's beginning.
            meta_clip_s = max(0.0, (meta_start_f - child.frame_final_start) / fps)
            src_start_s = child.frame_offset_start / fps + meta_clip_s
            child_dur_s = max(0.0, child.frame_final_duration / fps - meta_clip_s)
            child_off_s = max(0.0, (child.frame_final_start - meta_start_f) / fps)

            s0   = int(src_start_s * sr)
            s1   = s0 + int(child_dur_s * sr)
            data, _ = _sf.read(src_path, start=s0, stop=s1, always_2d=True)

            # Normalise channel count to match output array
            if data.shape[1] < n_ch:
                data = _np.tile(data, (1, n_ch // data.shape[1] + 1))[:, :n_ch]
            elif data.shape[1] > n_ch:
                data = data[:, :n_ch]

            off_s = int(child_off_s * sr)
            end_s = min(off_s + len(data), total_s)
            out_arr[off_s:end_s] += data[: end_s - off_s]

            print(f"  mixed {child.name!r}: {child_dur_s:.3f}s at +{child_off_s:.3f}s "
                  f"(src skip {src_start_s:.3f}s, meta clip {meta_clip_s:.3f}s)")
        except Exception as _e:
            print(f"  WARN: failed to mix {child.name!r}: {_e}")
            continue

    if out_arr is None:
        print("[render_meta_audio_to_path] no SOUND children could be mixed")
        return None

    # Prevent clipping from additive mixing
    _peak = float(_np.abs(out_arr).max())
    if _peak > 1.0:
        out_arr /= _peak

    try:
        _sf.write(output_path, out_arr, sr, subtype="PCM_16")
        _rendered_temp_paths.add(output_path)
        print(f"  wrote {meta_dur_s:.3f}s mix → {output_path!r}")
        print(f"[render_meta_audio_to_path] ─────────────────────────────────────")
        return output_path
    except Exception as _e:
        print(f"[render_meta_audio_to_path] write failed: {_e}")
        return None


def decompose_meta(context, meta_strip, target_type="video"):
    """Break a META strip into typed temp files for multi-modal plugin inputs.

    target_type="video":
        Renders the whole META as a composite MP4 for video input, extracts
        each child's typed path for image/audio, and reads TEXT children.
        Returns {"images": [path,...], "videos": [path,...],
                 "audio": path|None, "text": str}

    target_type="image":
        Renders the whole META as a single composite PNG (VSE composites all
        children) for img2img input, and collects TEXT child text for prompt.
        Returns {"image": path|None, "text": str}
    """
    if target_type == "image":
        image_path = render_strip_to_path(context, meta_strip, image_output=True)
        texts = [c.text for c in meta_strip.strips if c.type == "TEXT" and c.text]
        return {"image": image_path, "text": ", ".join(texts)}

    # target_type == "video": decompose children by type
    images, videos, audio, texts = [], [], None, []
    images_with_frames = []   # (raw_path, frame_start, child_strip) triples for FLF routing
    print(f"[decompose_meta] META strip '{meta_strip.name}' has {len(list(meta_strip.strips))} children:")
    for child in meta_strip.strips:
        print(f"  child type={child.type!r} name={child.name!r}")
        if child.type == "TEXT":
            if child.text and child.text.strip():
                texts.append(child.text.strip())
                print(f"    -> TEXT accepted: {child.text.strip()!r}")
            else:
                print(f"    -> TEXT empty, skipped")
        elif child.type == "IMAGE":
            path = get_strip_path(child)
            if path and os.path.isfile(path):
                images.append(path)
                images_with_frames.append((path, child.frame_start, child))
                print(f"    -> IMAGE accepted: {path!r}")
            else:
                print(f"    -> IMAGE path missing or not found: {path!r}")
        elif child.type == "MOVIE":
            path = get_strip_path(child)
            if path and os.path.isfile(path):
                videos.append(path)
                print(f"    -> MOVIE accepted: {path!r}")
            else:
                print(f"    -> MOVIE path missing or not found: {path!r}")
        elif child.type == "SOUND":
            try:
                path = bpy.path.abspath(child.sound.filepath)
            except Exception:
                path = get_strip_path(child)
            if path and os.path.isfile(path):
                audio = path
                print(f"    -> SOUND accepted: {path!r}")
            else:
                print(f"    -> SOUND path missing or not found: {path!r}")
        else:
            print(f"    -> type {child.type!r} not handled, skipped")
    result = {"images": images, "images_with_frames": images_with_frames, "videos": videos, "audio": audio, "text": ", ".join(texts)}
    print(f"[decompose_meta] result: images={images}, videos={videos}, audio={audio!r}, text={result['text']!r}")
    return result


def load_strip_as_pil(strip, context=None):
    """Load a VSE strip as PIL.Image, respecting VSE trims and transforms.

    Drop-in upgrade for: load_first_frame(get_strip_path(strip))

    * IMAGE with no transforms  → fast path reading the source file directly
    * IMAGE with transforms / SCENE / META / MASK / COLOR → render via VSE
    * MOVIE → seek to the in-point frame (respects the strip's trim offset)
    """
    import cv2
    from PIL import Image as _PILImage

    if strip.type in ("SCENE", "META", "MASK", "COLOR"):
        if context is None:
            context = bpy.context
        path = render_strip_to_path(context, strip, image_output=True)
        if path:
            return _PILImage.open(path).convert("RGB")
        return None

    if strip.type == "IMAGE":
        try:
            tx = strip.transform
            has_transform = (
                tx.scale_x != 1.0 or tx.scale_y != 1.0
                or tx.offset_x != 0.0 or tx.offset_y != 0.0
                or getattr(strip, "use_crop", False)
            )
        except Exception:
            has_transform = False
        if not has_transform:
            return load_first_frame(get_strip_path(strip))
        if context is None:
            context = bpy.context
        path = render_strip_to_path(context, strip, image_output=True)
        if path:
            return _PILImage.open(path).convert("RGB")
        return None

    if strip.type == "MOVIE":
        raw_path = get_strip_path(strip)
        if not raw_path:
            return None
        offset = getattr(strip, "frame_offset_start", 0)
        if offset == 0:
            return load_first_frame(raw_path)
        cap = cv2.VideoCapture(raw_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(offset))
        ret, frame = cap.read()
        cap.release()
        if ret:
            return _PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    return None


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
    try:
        bpy.ops.lora.refresh_files()
    except RuntimeError:
        pass

class OBJECT_OT_FluxAddStrip(bpy.types.Operator):
    bl_idname = "object.flux_add_strip"
    bl_label = "Add Flux Image Input"
    bl_description = "Adds another input slot for Flux images"

    def execute(self, context):
        scene = context.scene
        if scene.flux_visible_strips < 9:
            scene.flux_visible_strips += 1
        else:
            self.report({'INFO'}, "Maximum 9 Flux image inputs already displayed.")
        return {'FINISHED'}

class OBJECT_OT_FluxHideStrip(bpy.types.Operator):
    bl_idname = "object.flux_hide_strip"
    bl_label = "Hide Flux Image Input"
    bl_description = "Hides the last Flux image input and clears its value"

    strip_index: bpy.props.IntProperty(default=0) # Property to know which strip's value to clear

    def execute(self, context):
        scene = context.scene

        if scene.flux_visible_strips > 1:
            # Clear the value of the strip corresponding to this button
            strip_to_clear_name = f"flux_strip_{self.strip_index}"
            if hasattr(scene, strip_to_clear_name):
                setattr(scene, strip_to_clear_name, "") # Clear its string property
            scene.flux_visible_strips -= 1
        else:
            # If only one is left, don't hide it, just clear its value
            strip_to_clear_name = f"flux_strip_{self.strip_index}"
            if hasattr(scene, strip_to_clear_name):
                setattr(scene, strip_to_clear_name, "") # Clear its string property
            self.report({'INFO'}, "Minimum one Flux image input must be visible. Value cleared.")
        return {'FINISHED'}

class NoWatermark:
    def apply_watermark(self, img):
        return img

def invoke_video_generation(prompt, api_key, image_url, movie_model_card):
    import requests
    import json
    import base64
    import os

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

_pallaidium_movie_model_cache = {
    "pipe": None,
    "refiner": None,
    "last_model_card": None
}

def resize_and_pad_image(input_image, target_width, target_height, background_color=(0, 0, 0)):
    """
    Resizes an image to fit within the target dimensions while preserving aspect ratio,
    then pads the remaining space.

    Args:
        input_image (PIL.Image.Image): The image to process.
        target_width (int): The final width of the image.
        target_height (int): The final height of the image.
        background_color (tuple): RGB tuple for the padding color.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    from PIL import Image
    # Calculate the aspect ratios
    target_aspect = target_width / target_height
    image_aspect = input_image.width / input_image.height

    # Determine the new size
    if image_aspect > target_aspect:
        # Image is wider than target, fit to target width
        new_width = target_width
        new_height = int(new_width / image_aspect)
    else:
        # Image is taller than target (or same aspect), fit to target height
        new_height = target_height
        new_width = int(new_height * image_aspect)

    # Resize the image using a high-quality filter
    resized_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the target dimensions and background color
    padded_image = Image.new("RGB", (target_width, target_height), background_color)

    # Calculate coordinates to paste the resized image in the center
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste the resized image onto the padded background
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image

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

def split_long_sentence(spacy_sentence_span: 'Any', max_len: int) -> list[str]:
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


    return [p for p in parts if p]

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

prompt_items=None

_pallaidium_audio_model_cache = {
    "pipe": None,
    "vocoder": None,          # For F5-TTS
    "model": None,            # For MMAudio / Chatterbox
    "feature_extractor": None,# For MMAudio
    "last_model_card": None
}

class IPAdapterFaceProperties(bpy.types.PropertyGroup):
    files_to_import: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)

class IPAdapterStyleProperties(bpy.types.PropertyGroup):
    files_to_import: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)

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

_pallaidium_model_cache = {
    "pipe": None,
    "converter": None,
    "refiner": None,
    "last_model_card": None
}

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

_pallaidium_text_model_cache = {
    "model": None,
    "processor": None,
    "tokenizer": None,
    "last_model_card": None
}

def delete_linked_audio(context, movie_strip):
    if movie_strip.type != 'MOVIE':
        return

    seq_editor = context.scene.sequence_editor
    if not seq_editor:
        return

    movie_path = movie_strip.filepath
    movie_start = movie_strip.frame_start

    for s in seq_editor.strips_all:
        if (
            s.type == 'SOUND' and
            getattr(s.sound, "filepath", None) == movie_path and
            s.frame_start == movie_start
        ):
            try:
                delete_strip(s)
                print(f"Deleted linked audio strip: {s.name}")
            except Exception as e:
                print(f"Warning: Could not delete linked audio {s.name}: {e}")
            break

AI_METADATA_PREFIX = "ai_meta_"

def set_ai_metadata_from_dict(strip: bpy.types.Strip, params_dict: dict):
    """
    Sets AI metadata custom properties on a VSE strip from a dictionary.

    Stores parameter names (dict keys) and their string representations (dict values)
    as custom properties, prefixed with 'ai_meta_'.

    Args:
        strip: The VSE strip (Image, Movie, Text, or Sound) to add metadata to.
        params_dict: The dictionary containing the inference parameters.
    """
    if not strip or strip.type not in {'IMAGE', 'MOVIE', 'TEXT', 'SOUND'}:
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
        elif type(value).__name__ == "Generator" and type(value).__module__ == "torch":
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

class SEQUENCER_OT_redo_from_metadata(bpy.types.Operator):
    """Reload this strip's AI generation settings into the Pallaidium panel"""

    bl_idname = "sequencer.redo_from_metadata"
    bl_label  = "Redo from Metadata"
    bl_description = "Reload this strip's AI generation settings into the Pallaidium panel"

    def execute(self, context):
        seq_scene = getattr(context, 'sequencer_scene', None) or context.scene
        if not (seq_scene and seq_scene.sequence_editor):
            self.report({'WARNING'}, "No sequence editor")
            return {'CANCELLED'}
        strip = seq_scene.sequence_editor.active_strip
        if not strip:
            self.report({'WARNING'}, "No active strip")
            return {'CANCELLED'}

        scene = context.scene
        prefs = context.preferences.addons[ADDON_ID].preferences

        def _get(key):
            return strip.get(AI_METADATA_PREFIX + key)

        # Set typeselect + model first — input_strips_updated fires here and
        # overwrites x/y/frames with model defaults.  All explicit values
        # are written afterwards so they win over those defaults.
        if strip.type == 'MOVIE':
            output_type = "movie"
        elif strip.type == 'SOUND':
            output_type = "audio"
        else:
            output_type = "image"
        scene.generatorai_typeselect = output_type

        model = _get("model")
        model_attr = {
            "movie": "movie_model_card",
            "audio": "audio_model_card",
        }.get(output_type, "image_model_card")
        if model:
            try:
                setattr(prefs, model_attr, str(model))
            except TypeError:
                pass

        # Now set all generation params — these override any callback defaults.
        v = _get("prompt")
        if v is not None:
            scene.generate_movie_prompt = str(v)
        v = _get("negative_prompt")
        if v is not None:
            scene.generate_movie_negative_prompt = str(v)
        v = _get("steps")
        if v is not None:
            scene.movie_num_inference_steps = int(v)
        v = _get("guidance")
        if v is not None:
            scene.movie_num_guidance = float(v)
        v = _get("seed")
        if v is not None:
            scene.movie_num_seed = int(v)
            scene.movie_use_random = False
        v = _get("width")
        if v is not None:
            scene.generate_movie_x = int(v)
        v = _get("height")
        if v is not None:
            scene.generate_movie_y = int(v)
        v = _get("frames")
        if v is not None:
            scene.generate_movie_frames = int(v)

        # Extra UI properties
        v = _get("img_guidance_scale")
        if v is not None and hasattr(scene, "img_guidance_scale"):
            scene.img_guidance_scale = float(v)
        v = _get("illumination_style")
        if v is not None and hasattr(scene, "illumination_style"):
            scene.illumination_style = str(v)
        v = _get("light_direction")
        if v is not None and hasattr(scene, "light_direction"):
            scene.light_direction = str(v)
        v = _get("ip_adapter_face_folder")
        if v is not None and hasattr(scene, "ip_adapter_face_folder"):
            scene.ip_adapter_face_folder = str(v)
        v = _get("ip_adapter_style_folder")
        if v is not None and hasattr(scene, "ip_adapter_style_folder"):
            scene.ip_adapter_style_folder = str(v)
        v = _get("openpose_use_bones")
        if v is not None and hasattr(scene, "openpose_use_bones"):
            scene.openpose_use_bones = str(v).lower() in ("true", "1", "yes")
        v = _get("use_scribble_image")
        if v is not None and hasattr(scene, "use_scribble_image"):
            scene.use_scribble_image = str(v).lower() in ("true", "1", "yes")
        v = _get("ideogram_prompt_upsampling")
        if v is not None and hasattr(scene, "ideogram_prompt_upsampling"):
            scene.ideogram_prompt_upsampling = str(v).lower() in ("true", "1", "yes")

        # ltx23_multi_v2 guidance params
        for _attr, _cast in [
            ("ltx23m_modality_scale",       float),
            ("ltx23m_audio_guidance",       float),
            ("ltx23m_audio_stg_scale",      float),
            ("ltx23m_audio_modality_scale", float),
            ("ltx23m_audio_noise_scale",    float),
            ("ltx23m_audio_start_time",     float),
        ]:
            _v = _get(_attr)
            if _v is not None and hasattr(scene, _attr):
                try:
                    setattr(scene, _attr, _cast(_v))
                except Exception:
                    pass

        # ltx23_multi_ic_lora params
        for _attr, _cast in [
            ("ltx23ic_control_strength",  float),
            ("ltx23ic_control_downscale", int),
            ("ltx23ic_control_audio_str", float),
            ("ltx23ic_identity_guidance", float),
        ]:
            _v = _get(_attr)
            if _v is not None and hasattr(scene, _attr):
                try:
                    setattr(scene, _attr, _cast(_v))
                except Exception:
                    pass

        # ltx23_extend params
        for _attr, _cast in [
            ("ltx23ext_extend_frames", int),
            ("ltx23ext_video_strength", float),
        ]:
            _v = _get(_attr)
            if _v is not None and hasattr(scene, _attr):
                try:
                    setattr(scene, _attr, _cast(_v))
                except Exception:
                    pass

        # Maxine VSR
        _v = _get("maxine_quality")
        if _v is not None and hasattr(scene, "maxine_quality"):
            try:
                scene.maxine_quality = str(_v)
            except Exception:
                pass

        # MOSS-TTS params (audio strips)
        for _attr, _cast in [
            ("moss_model_variant",   str),
            ("moss_language",        str),
            ("moss_duration_tokens", int),
            ("moss_max_new_tokens",  int),
            ("moss_temperature",     float),
            ("moss_top_p",           float),
            ("moss_top_k",           int),
            ("moss_ref_audio_path",  str),
        ]:
            _v = _get(_attr)
            if _v is not None and hasattr(scene, _attr):
                try:
                    setattr(scene, _attr, _cast(_v))
                except Exception:
                    pass

        # Restore LoRA: scan full folder so all files appear in the UIList,
        # then mark the saved (enabled) ones and restore their weights.
        lora_folder = _get("lora_folder")
        if lora_folder:
            scene.lora_folder = str(lora_folder)
        lora_json = _get("lora_files_json")
        try:
            lora_raw = json.loads(str(lora_json)) if lora_json else []
        except (json.JSONDecodeError, ValueError):
            lora_raw = []
        # Metadata only stores enabled LoRAs; map name -> weight.
        saved_loras = {item.get("name", ""): item.get("weight", 1.0) for item in lora_raw}
        scene.lora_files.clear()
        directory = bpy.path.abspath(scene.lora_folder) if scene.lora_folder else ""
        if directory and os.path.isdir(directory):
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".safetensors"):
                    stem  = filename.replace(".safetensors", "")
                    entry = scene.lora_files.add()
                    entry.name = stem
                    if stem in saved_loras:
                        entry.enabled      = True
                        entry.weight_value = saved_loras[stem]
                    else:
                        entry.enabled      = False
                        entry.weight_value = 1.0
        elif lora_raw:
            # Folder not accessible — fall back to just the saved entries.
            for item in lora_raw:
                entry = scene.lora_files.add()
                entry.name         = item.get("name", "")
                entry.weight_value = item.get("weight", 1.0)
                entry.enabled      = True

        self.report({'INFO'}, "Settings loaded from strip metadata")
        return {'FINISHED'}


class AI_Metadata_PT_Panel(bpy.types.Panel):
    """Displays AI Generation Metadata stored as custom properties"""
    bl_label = "AI Metadata"
    bl_idname = "SEQUENCER_PT_ai_metadata"
    bl_space_type = 'SEQUENCE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Metadata"

    @classmethod
    def poll(cls, context):
        if context.space_data.view_type in {'SEQUENCER', 'PREVIEW', 'SEQUENCER_PREVIEW'}:
            seq_scene = getattr(context, 'sequencer_scene', None) or context.scene
            if seq_scene and seq_scene.sequence_editor:
                active_strip = seq_scene.sequence_editor.active_strip
                if active_strip and active_strip.type in {'IMAGE', 'MOVIE', 'SOUND'}:
                    return True
        return False

    def draw(self, context):
        layout = self.layout
        seq_scene = getattr(context, 'sequencer_scene', None) or context.scene
        seq_editor = seq_scene.sequence_editor
        strip = seq_editor.active_strip

        if not strip:
            return

        col = layout.column(align=True)
        displayed_anything = False

        ai_prop_keys = [k for k in strip.keys() if k.startswith(AI_METADATA_PREFIX)]

        if not ai_prop_keys:
            col.label(text="No AI metadata found on this strip.")
            col.label(text="Use 'set_ai_metadata_from_dict'")
            col.label(text="to add data.")
            return

        _SKIP = {"lora_files_json"}
        _ORDER = [
            "model", "mode", "prompt", "negative_prompt",
            "seed", "width", "height", "frames", "steps", "guidance",
            "lora_folder",
        ]
        _order_map = {f"{AI_METADATA_PREFIX}{k}": i for i, k in enumerate(_ORDER)}
        ai_prop_keys = sorted(
            ai_prop_keys,
            key=lambda k: (_order_map.get(k, len(_ORDER)), k)
        )

        col.prop(strip, "name", text="Name")

        for prop_key in ai_prop_keys:
            param_name = prop_key[len(AI_METADATA_PREFIX):]
            if param_name in _SKIP:
                continue
            label_text = param_name.replace('_', ' ').title()
            col.prop(strip, f'["{prop_key}"]', text=label_text)
            displayed_anything = True

        row = col.row()
        row.alignment = 'RIGHT'
        row.operator("sequencer.redo_from_metadata", text="", icon="LOOP_BACK")

def get_enum_items(options_dict):
    """Converts a dictionary to the format required by bpy.props.EnumProperty."""
    return [(key, key.replace("_", " ").title(), desc) for key, desc in options_dict.items()]


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def bench_print(label: str, t0: float = None) -> float:
    """Print *label* with elapsed time, system RAM, and GPU VRAM usage.

    Parameters
    ----------
    label : str
        Human-readable step name shown in the print.
    t0 : float, optional
        Start timestamp from ``time.perf_counter()``.  When supplied, the
        elapsed time since ``t0`` is appended.

    Returns
    -------
    float
        ``time.perf_counter()`` captured just before printing, so the return
        value can be passed as ``t0`` for the *next* call to chain steps::

            t = bench_print("step A")
            do_work()
            t = bench_print("step B", t)   # shows time for step B
    """
    now = time.perf_counter()

    parts = [f"[bench] {label}"]

    if t0 is not None:
        parts.append(f"{now - t0:.2f}s")

    # ── RAM ──────────────────────────────────────────────────────────────
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(
            f"RAM {vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f} GB"
            f" ({vm.percent:.0f}%)"
        )
    except Exception:
        pass

    # ── VRAM ─────────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 2
            res   = torch.cuda.memory_reserved()  / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            parts.append(
                f"VRAM {alloc:.0f}/{res:.0f}/{total:.0f} MB"
                f" (alloc/reserved/total)"
            )
    except Exception:
        pass

    print("  |  ".join(parts))
    return now

if __name__ == "__main__":
    pass