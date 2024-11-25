bl_info = {
    "name": "2D Asset Generator",
    "author": "tintwotin",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "category": "3D View",
    "location": "3D Editor > Sidebar > 2D Asset",
    "description": "2D Asset Generator in the 3D View",
}


import bpy
from bpy.types import Operator, PropertyGroup, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty
import os, re
import subprocess
import sys
import math
from os.path import join
from mathutils import Vector
import venv
import importlib
from typing import Optional
import platform


def gfx_device():
    try:
        import torch
        if torch.cuda.is_available():
            gfxdevice = "cuda"
        elif torch.backends.mps.is_available():
            gfxdevice = "mps"
        else:
            gfxdevice = "cpu"
    except:
        print("2D Asset Generator dependencies needs to be installed and Blender needs to be restarted.")
        gfxdevice = "cpu"
    return gfxdevice

DEBUG = False

dir_path = os.path.join(bpy.utils.user_resource("DATAFILES"), "2D_Asset_Generator")
os.makedirs(dir_path, exist_ok=True)


def debug_print(*args, **kwargs):
    """Conditional print function based on the DEBUG variable."""
    if DEBUG:
        print(*args, **kwargs)


def addon_script_path() -> str:
    """Return the path where the add-on script is located (addon directory)."""
    addon_path = os.path.dirname(__file__)  # Use __file__ to get the script directory
    debug_print(f"Addon script path is: {addon_path}")
    return addon_path


def venv_path(env_name="virtual_dependencies") -> str:
    """Define the path for the virtual environment directory in the add-on's folder."""
    addon_path = addon_script_path()
    env_path = os.path.join(addon_path, env_name)  # Create virtual environment relative to add-on script
    debug_print(f"Virtual environment path is: {env_path}")
    return env_path


def python_exec() -> str:
    """Return the path to the Python executable in the virtual environment if it exists."""
    env_python = os.path.join(venv_path(), 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_path(), 'bin', 'python')
    debug_print(f"Python executable in the virtual environment is: {env_python}")
    return env_python if os.path.exists(env_python) else sys.executable


def create_venv(env_name="virtual_dependencies"):
    """Create a virtual environment if it doesn't exist."""
    env_dir = venv_path(env_name)
    if not os.path.exists(env_dir):
        venv.create(env_dir, with_pip=True)
        debug_print(f"Virtual environment created at {env_dir}")
        ensure_pip_installed()  # Ensure pip is available after environment creation
    else:
        debug_print("Virtual environment already exists.")


def ensure_pip_installed():
    """Ensure pip is installed in the virtual environment."""
    python_exe = python_exec()
    subprocess.run([python_exe, '-m', 'ensurepip', "--disable-pip-version-check"])
    debug_print("Ensured that pip is installed.")


def import_module(module, install_module):
    module = str(module)
    python_exe = python_exec()
    target = venv_path()

    try:
        subprocess.call([python_exe, "import ", packageName])
    except:
        print("\nInstalling: " + module + " module")
        subprocess.call([python_exe, "-m", "pip", "install", install_module, "--no-warn-script-location", "--no-dependencies", "--upgrade", '--target', target, "-q", "--use-deprecated=legacy-resolver", "--disable-pip-version-check"])

        try:
            exec("import " + module)
        except ModuleNotFoundError:
            return False
    return True


def add_virtualenv_to_syspath():
    """Add the virtual environment's directory to sys.path."""
    # Define the virtual environment path
    env_dir = venv_path()

    # Ensure the site-packages folder of the venv is in the sys.path
    site_packages_path = os.path.join(env_dir, 'lib', 'site-packages') if os.name == 'nt' else os.path.join(env_dir, 'lib', 'python3.x', 'site-packages')
    
    # Check if the site-packages directory exists
    if not os.path.exists(site_packages_path):
        debug_print(f"Virtual environment site-packages not found: {site_packages_path}")
        return False
    
    # Add the site-packages path to sys.path
    sys.path.insert(0, site_packages_path)

    # Add the virtual environment directory to sys.path for imports
    if os.path.exists(env_dir):
        sys.path.append(env_dir)
        debug_print(f"Added virtual environment directory to sys.path: {env_dir}")
    else:
        debug_print(f"Virtual environment directory not found at: {env_dir}")

    # Debug print sys.path
    print(f"Using Python from: {sys.executable}")


def set_virtualenv_python():
    """Set the Python executable from the virtual environment."""
    python_exe = os.path.join(venv_path(), 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_path(), 'bin', 'python')

    # Update sys.executable to use the virtual environment's Python
    if os.path.exists(python_exe):
        sys.executable = python_exe
        debug_print(f"Using Python executable from virtual environment: {python_exe}")
    else:
        debug_print(f"Python executable not found in virtual environment: {python_exe}")


def activate_virtualenv():
    """Activate the virtual environment for the add-on."""
    venv_path = os.path.join(bpy.utils.user_resource("SCRIPTS"), "addons", "2D_Assets-main", "virtual_dependencies")
    
    if not os.path.exists(venv_path):
        print(f"Virtual environment path not found: {venv_path}")
        return False
    
    # Define the correct paths for Windows or Unix-based systems
    if platform.system() == 'Windows':
        scripts_path = os.path.join(venv_path, "Scripts")
        python_exe = os.path.join(scripts_path, "python.exe")
    else:
        bin_path = os.path.join(venv_path, "bin")
        python_exe = os.path.join(bin_path, "python")
    
    if not os.path.exists(python_exe):
        print(f"Python executable not found at: {python_exe}")
        return False

    # Set the virtual environment's Python executable as the current Python
    sys.executable = python_exe

    # Modify the PATH and PYTHONPATH to use the virtual environment's directories
    if platform.system() == "Windows":
        os.environ["PATH"] = scripts_path + os.pathsep + os.environ["PATH"]
    else:
        os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
    
    # Update sys.path to include site-packages from the virtual environment
    site_packages_path = os.path.join(venv_path, 'lib', 'site-packages')
    sys.path.insert(0, site_packages_path)

    print(f"Virtual environment activated: {venv_path}")
    return True


def install_packages(override: Optional[bool] = False):
    """Install or update packages from the requirements.txt file."""
    create_venv()  # Ensure the virtual environment exists before installation
    # Add the virtual environment’s directory to sys.path
    add_virtualenv_to_syspath()
    activate_virtualenv()

    # Set Python executable to the virtual environment
    set_virtualenv_python()
    
    os_platform = platform.system()
    
    # Determine the name of the executables directory based on the OS
    bin_dir_name = 'Scripts' if os.name == 'nt' else 'bin'
    
    # Construct the path to the 'bin' or 'Scripts' directory
    bin_path = os.path.join(venv_path(), bin_dir_name)    
    
    python_exe = os.path.join(bin_path, "python")
    
    #os.environ["PIP_TARGET"] = venv_path()
    requirements_txt = os.path.join(addon_script_path(), "requirements.txt")
    venvpath = venv_path()
    target = os.path.join(venvpath, 'lib', 'site-packages') if os.name == 'nt' else os.path.join(venvpath, 'lib', 'python3.x', 'site-packages')
    
    # Ensure pip is installed
    ensure_pip_installed()
    
    # Upgrade pip
    #subprocess.run([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Install dependencies with or without override
    if override:
        subprocess.run([python_exe, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', '-r', requirements_txt, '--target', target, "--no-warn-script-location","--disable-pip-version-check"])
    else:
        subprocess.run([python_exe, '-m', 'pip', 'install', '--upgrade', '-r', requirements_txt, '--target', target, "--no-warn-script-location", "--disable-pip-version-check"])

#    if os_platform == "Windows":
#        subprocess.call([python_exe, "-m", "pip", "install", "--disable-pip-version-check", "https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp311-cp311-win_amd64.whl", '--target', target, "--upgrade"])
#    else:
#        import_module("triton", "triton")

#    if os_platform == "Windows":
#        subprocess.call([python_exe, "-m", "pip", "install", "--disable-pip-version-check", "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-win_amd64.whl", '--target', target, "--upgrade"])
#    elif os_platform == "Linux":
#        subprocess.call([python_exe, "-m", "pip", "install", "--disable-pip-version-check", "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl", '--target', target, "--upgrade"])
#    else:
#        subprocess.call([python_exe, "-m", "pip", "install", "--disable-pip-version-check", "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-macosx_13_1_arm64.whl", '--target', target, "--upgrade"])

    subprocess.call([python_exe, "-m", "pip", "install", "--disable-pip-version-check", "git+https://github.com/huggingface/accelerate.git", '--target', target, "--upgrade"])

    print("\nInstalling: torch module")
    if os_platform == "Windows":
        #subprocess.call([python_exe, "-m", "pip", "install", "torch==2.1.2+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.2+cu121 xformers==2.1.2+cu121", "--index-url", "https://download.pytorch.org/whl/cu121", "--user", "--upgrade"])
        subprocess.call([python_exe, "-m", "pip", "install", "torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 xformers==2.5.0+cu121", "--index-url", "https://download.pytorch.org/whl/cu121", "--user", "--upgrade"])
#        torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
#        subprocess.call(
#            [
#                python_exe,
#                "-m",
#                "pip",
#                "install",
#                '--force-reinstall',
#                "torch==2.3.1+cu121",
#                "xformers",
#                "torchvision",
#                #"torchaudio",
#                "--index-url",
#                "https://download.pytorch.org/whl/cu121",
#                "--no-warn-script-location",
#                "--disable-pip-version-check",
#                '--target', target,
#                "--upgrade",
#            ]
#        )
    else:
        import_module("torch", "torch")
        import_module("torchvision", "torchvision")
        #import_module("torchaudio", "torchaudio")
        import_module("xformers", "xformers")

    subprocess.call([python_exe, "-m", "pip", "install", "--user", '--force-reinstall', "numpy==1.26.4", "--no-warn-script-location", "--no-warn-script-location", "--disable-pip-version-check"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", '--force-reinstall', "numpy==1.26.4", "--no-warn-script-location", '--target', target, "--no-warn-script-location", "--disable-pip-version-check"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", '--force-reinstall', "protobuf==3.20.3", "--no-warn-script-location", '--target', target, "--no-warn-script-location", "--disable-pip-version-check"])

    # Check if all dependencies are installed
    check_dependencies_installed()
    print("\nDependency installation finished.")


def parse_package_name(package_line):
    """
    Parse package name by removing version constraints and replacing hyphens with underscores.
    """
    # Split the package name on any version constraint symbols
    package_name = re.split(r'[<>=!~]', package_line.strip())[0]
    # Replace hyphens with underscores to match Python import conventions
    package_name = package_name.replace('-', '_')
    return package_name

def check_dependencies_installed() -> bool:
    """Check if all the packages in the requirements.txt file are importable."""
    import os
    # Determine the name of the executables directory based on the OS
    bin_dir_name = 'Scripts' if os.name == 'nt' else 'bin'
    venvpath = venv_path()
    target = os.path.join(venvpath, 'lib', 'site-packages') if os.name == 'nt' else os.path.join(venvpath, 'lib', 'python3.x', 'site-packages')
    
    # Construct the path to the 'bin' or 'Scripts' directory
    bin_path = os.path.join(venv_path(), bin_dir_name) 
    python_exe = os.path.join(bin_path, "python")
    subprocess.call([python_exe, "-m", "pip", "list", "--disable-pip-version-check"])
    requirements_txt = os.path.join(addon_script_path(), "requirements.txt")

    if not os.path.exists(requirements_txt):
        debug_print(f"Requirements file '{requirements_txt}' not found.")
        return False

    add_virtualenv_to_syspath()
    activate_virtualenv()
    set_virtualenv_python()

    with open(requirements_txt, 'r') as file:
        packages = file.readlines()

    missing_packages = []
    
    # Check if each package is importable
    for package in packages:
        package_name_raw = package.strip()
        if package_name_raw:  # Avoid empty lines
            # Parse the package name to get the importable format
            package_name = parse_package_name(package_name_raw)
            try:
                importlib.import_module(package_name)
                print(f"Package '{package_name}' is already installed and importable.")
            except ImportError:
                missing_packages.append(package_name)  # Keep original name in case of error
                print(f"Package '{package_name_raw}' is missing or not importable.")

    if missing_packages:
        print(f"Missing or non-importable packages: {', '.join(missing_packages)}")
        return False
    return True

def uninstall_packages():
    """Uninstall all packages listed in the requirements.txt file."""
    # Determine the name of the executables directory based on the OS
    bin_dir_name = 'Scripts' if os.name == 'nt' else 'bin'
    add_virtualenv_to_syspath()
    activate_virtualenv()
    set_virtualenv_python()
    
    # Construct the path to the 'bin' or 'Scripts' directory
    bin_path = os.path.join(python_exec(), bin_dir_name)    
    
    python_exe = os.path.join(bin_path, "python")
    
    requirements_txt = os.path.join(addon_script_path(), "requirements.txt")

    if not os.path.exists(requirements_txt):
        debug_print("Requirements file not found for uninstallation.")
        return

    # Ensure pip is installed before running uninstall
    ensure_pip_installed()
    add_virtualenv_to_syspath()

    with open(requirements_txt, 'r') as file:
        packages = file.readlines()
    
    #os.environ["PIP_TARGET"] = venv_path()

    for package in packages:
        package_name = package.strip()
        if package_name:  # Avoid empty lines
            subprocess.run([python_exe, '-m', 'pip', 'uninstall', '-y', package_name])
            debug_print(f"Uninstalled package: {package_name}")
            
    print("\nDependency uninstallation finished. Manually, delete this folder: "+venv_path())


# Panel for Add-On Preferences
class AssetGeneratorPreferences(AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        # Install Dependencies Button
        row.operator("virtual_dependencies.install_dependencies", text="Install Dependencies")

        # Check Dependencies Button
        row.operator("virtual_dependencies.check_dependencies", text="Check Dependencies")

        # Uninstall Dependencies Button
        row.operator("virtual_dependencies.uninstall_dependencies", text="Uninstall Dependencies")


# Operators for install, uninstall, and check dependencies
class InstallDependenciesOperator(bpy.types.Operator):
    bl_idname = "virtual_dependencies.install_dependencies"  # Updated the bl_idname here to match the class name
    bl_label = "Install Dependencies"

    def execute(self, context):
        install_packages(override=True)  # You can change `override` to `False` as needed
        return {'FINISHED'}


class UninstallDependenciesOperator(bpy.types.Operator):
    bl_idname = "virtual_dependencies.uninstall_dependencies"  # Updated the bl_idname here to match the class name
    bl_label = "Uninstall Dependencies"

    def execute(self, context):
        uninstall_packages()
        return {'FINISHED'}


class CheckDependenciesOperator(bpy.types.Operator):
    bl_idname = "virtual_dependencies.check_dependencies"  # Updated the bl_idname here to match the class name
    bl_label = "Check Dependencies"

    def execute(self, context):
        check_dependencies_installed()
        return {'FINISHED'}


def flush():
    import torch
    import gc
    gc.collect()
    if gfx_device() == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        # torch.cuda.reset_peak_memory_stats()


def python_exec():
    return sys.executable


# Get a list of text blocks in Blender
def texts(self, context):
    return [(text.name, text.name, "") for text in bpy.data.texts]


# Property Group for storing the selected text block and toggle
class Import_Text_Props(PropertyGroup):
    def update_text_list(self, context):
        self.script = bpy.data.texts[self.scene_texts].name
        return None

    # EnumProperty to toggle between Text-Block and Prompt
    input_type: EnumProperty(
        name="Input Type",
        description="Choose between Text-Block and Prompt",
        items=[
            ("PROMPT", "Prompt", "Input: Typed in prompt"),
            ("TEXT_BLOCK", "Text-Block", "Input: Text from the Blender Text Editor"),
        ],
        default="TEXT_BLOCK",
    )

    script: StringProperty(default="", description="Browse Text to be Linked")
    scene_texts: EnumProperty(
        name="Text-Blocks",
        items=texts,
        update=update_text_list,
        description="Text-Blocks",
    )


def get_unique_asset_name(self, context):
    """Generates a unique asset name if there is a conflict, ensuring a name is always returned."""

    # Retrieve base name and check for validity
    base_name = context.scene.asset_name
    if base_name == "":
        # If base name is missing, use the asset prompt or a default
        prompt = context.scene.asset_prompt
        base_name = "_".join(prompt.split()[:2]) if prompt else "Asset"
        context.scene.asset_name = base_name

    # Collect existing names to detect conflicts
    existing_names = {obj.name for obj in bpy.data.objects if getattr(obj, "asset_data", None)}

    # If the base name is unique, return it directly
    if base_name in existing_names:

        # Attempt to extract an existing number suffix in parentheses, if present
        match = re.search(r"\((\d+)\)$", base_name)
        if match:
            base_name = base_name[: match.start()].strip()
            counter = int(match.group(1)) + 1
        else:
            counter = 1

        # Generate a unique name by incrementing the counter until no conflicts remain
        unique_name = f"{base_name} ({counter})"
        while unique_name in existing_names:
            counter += 1
            unique_name = f"{base_name} ({counter})"

        # Set the unique name in the context and return it
        #unique_name = get_unique_file_name(unique_name)
        context.scene.asset_name = unique_name
    return


def get_unique_file_name(base_path):
    """Generates a unique file name if there is a conflict in the file system."""
    base_name, extension = os.path.splitext(base_path)

    # Regular expression to detect if the file name has a number in parentheses
    match = re.search(r"\((\d+)\)$", base_name)

    if match:
        # If there's a number, increment it
        base_name = base_name[: match.start()].strip()
        counter = int(match.group(1)) + 1
    else:
        # If no number, start at 1
        counter = 1
    unique_path = f"{base_name} ({counter}){extension}"
    while os.path.exists(unique_path):
        counter += 1
        unique_path = f"{base_name} ({counter}){extension}"
    return unique_path


class FLUX_OT_GenerateAsset(bpy.types.Operator):
    """Generate asset image from description and convert to 3D object"""

    bl_idname = "object.generate_asset"
    bl_label = "Generate Asset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        create_venv() 
        add_virtualenv_to_syspath()
        activate_virtualenv()
        set_virtualenv_python()
        try:
            import diffusers
        except:
            self.report({"ERROR"}, "Error: Install dependencies in the add-on Preferences!")
            return {"CANCELLED"}
        from PIL import Image, ImageFilter
        
        try:
            pipe = self.load_model(context)
            input_type = context.scene.import_text.input_type

            if input_type == "TEXT_BLOCK":
                # text = bpy.data.texts[import_text.scene_texts]
                text = bpy.data.texts[context.scene.import_text.scene_texts]
                lines = [line.body for line in text.lines]
                lines = [line for line in lines if line.strip()]
            elif input_type == "PROMPT":
                lines = [context.scene.asset_prompt]

            for index, line in enumerate(lines):
                if line:
                    # Fetch the prompt from the scene
                    if input_type == "TEXT_BLOCK":
                        context.scene.asset_prompt = line
                        base_name = " ".join(line.split()[:3]) if line else "Asset"
                        context.scene.asset_name = base_name.title()
                    else:
                        base_name = context.scene.asset_name
                        
                    description = context.scene.asset_prompt
                    print(str(index + 1) + "/" + str(len(lines)) + ": " + base_name.title())

                    if not description:
                        self.report({"ERROR"}, "Asset prompt is empty.")
                        return {"CANCELLED"}

                    # Generate image using FLUX
                    image_path = bpy.path.abspath(self.generate_image(context, description, pipe))
                    if DEBUG:
                        print(f"Image Path: {image_path}")

                    # Remove background from the generated image
                    transparent_image_path = bpy.path.abspath(self.remove_background(context, image_path))
                    if DEBUG:
                        print(f"Transparent Path: {transparent_image_path}")

                    # separate islands
                    image_paths = self.split_by_alpha_islands(transparent_image_path, output_prefix=base_name)
                    if DEBUG:
                        print(f"Image Paths: {image_paths}")
                    if image_paths:
                        # Iterating through the saved images
                        for path in image_paths:
                            with Image.open(path) as img:
                                # Convert the transparent image to a 3D object
                                self.convert_to_3d(context, path, description)
                                # Example of additional processing could go here
                                if DEBUG:
                                    print(f"Converting to asset: {path}")
                    else:
                        if DEBUG:
                            print("No valid content generated.")
            flush()
            # Save the .blend file so that the asset is persistent
            bpy.ops.wm.save_mainfile()
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Error: {str(e)}")
            return {"CANCELLED"}

    #    #SD 3.5 Medium
    #    def generate_image(self, context, description):
    #        """Generates an image using the Stable Diffusion 3 model based on user input."""

    #        # Import dependencies inside the method to avoid potential module issues before installation
    #        from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
    #        import torch

    #        # Define model configuration and ID
    #        model_id = "stabilityai/stable-diffusion-3.5-medium"
    #        asset_name = context.scene.asset_name

    #        # Configure quantization settings for 4-bit loading
    #        nf4_config = BitsAndBytesConfig(
    #            load_in_4bit=True,
    #            bnb_4bit_quant_type="nf4",
    #            bnb_4bit_compute_dtype=torch.bfloat16
    #        )

    #        # Initialize the transformer model with quantization settings
    #        model_nf4 = SD3Transformer2DModel.from_pretrained(
    #            model_id,
    #            subfolder="transformer",
    #            quantization_config=nf4_config,
    #            torch_dtype=torch.bfloat16
    #        )

    #        # Load the Stable Diffusion pipeline with the transformer model
    #        pipeline = StableDiffusion3Pipeline.from_pretrained(
    #            model_id,
    #            transformer=model_nf4,
    #            torch_dtype=torch.bfloat16
    #        )

    #        # Enable CPU offloading for memory optimization
    #        pipeline.enable_model_cpu_offload()

    #        # Construct the prompt and generate the image
    #        prompt = "neutral background, " + description
    #        out = pipeline(
    #            prompt=prompt,
    #            guidance_scale=2.8,
    #            height=1440,
    #            width=1440,
    #            num_inference_steps=30,
    #            max_sequence_length=256,
    #        ).images[0]

    #        # Save the generated image to the specified path
    #        asset_name = re.sub(r'[<>:"/\\|?*]', '', context.scene.asset_name)
    #        image_path = bpy.path.abspath(f"//{asset_name}_generated_image.png")
    #        out.save(image_path)
    #        flush()
    #        return image_path

    # FLUX
    def load_model(self, context):
        """Generates an image using the FLUX model based on the user input."""

        # Import dependencies inside the method to avoid potential module issues before installation
        from diffusers import FluxPipeline
        import torch

        asset_name = context.scene.asset_name
         #If bitsandbytes doesn't work, use this:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

#        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

#        image_model_card = "ChuckMcSneed/FLUX.1-dev"
#        nf4_config = BitsAndBytesConfig(
#            load_in_4bit=True,
#            bnb_4bit_quant_type="nf4",
#            bnb_4bit_compute_dtype=torch.bfloat16,
#        )
#        model_nf4 = FluxTransformer2DModel.from_pretrained(
#            image_model_card,
#            subfolder="transformer",
#            quantization_config=nf4_config,
#            torch_dtype=torch.bfloat16,
#        )

 #       pipe = FluxPipeline.from_pretrained(image_model_card, transformer=model_nf4, torch_dtype=torch.bfloat16)

        if gfx_device() == "mps":
            pipe.to(gfx_device())
        else:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.vae.enable_tiling()
            #pipe.enable_model_cpu_offload()
        return pipe

    # FLUX
    def generate_image(self, context, description, pipe):
        """Generates an image using the FLUX model based on the user input."""

        asset_name = context.scene.asset_name

        # Generate the image
        prompt = "neutral background, " + description
        out = pipe(
            prompt=prompt,
            guidance_scale=2.8,
            height=1024,
            width=1024,
            num_inference_steps=25,
            max_sequence_length=256,
        ).images[0]

        # Save the generated image
        asset_name = re.sub(r'[<>:"/\\|?*]', "", context.scene.asset_name)
        debug_print("Datafiles: "+bpy.utils.user_resource("DATAFILES"))
        image_path = bpy.path.abspath(os.path.join(bpy.path.abspath(bpy.utils.user_resource("DATAFILES")), "2D_Asset_Generator", f"{asset_name}_generated_image.png"))
        out.save(image_path)
        debug_print("Save Path: "+image_path)
        return image_path

    def remove_background(self, context, image_path):
        """Removes the background from the image using the BiRefNet segmentation model."""
        # Import dependencies inside the method
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        from PIL import Image, ImageFilter
        import torch

        asset_name = context.scene.asset_name

        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        birefnet.to(gfx_device())

        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        input_image = transform_image(image).unsqueeze(0).to(gfx_device())

        # Generate the background mask
        with torch.no_grad():
            preds = birefnet(input_image)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred)
        mask = mask.resize(image_size)

        # Refine the mask: Apply thresholding and feathering for smoother removal
        refined_mask = self.refine_mask(mask)

        # Apply the refined mask to the image to remove the background
        image.putalpha(refined_mask)
        asset_name = re.sub(r'[<>:"/\\|?*]', "", context.scene.asset_name)
        transparent_image_path = bpy.path.abspath(os.path.join(bpy.path.abspath(bpy.utils.user_resource("DATAFILES")), "2D_Asset_Generator", f"{asset_name}_generated_image_transparent.png"))

        debug_print("Save Transparent Path: "+transparent_image_path)
        image.save(transparent_image_path)

        return transparent_image_path

    def refine_mask(self, mask):
        """Refines the mask by applying thresholding and feathering."""
        from PIL import Image, ImageFilter
        mask = mask.convert("L")

        # Apply thresholding
        threshold_value = 200
        mask = mask.point(lambda p: 255 if p > threshold_value else 0)

        # Apply feathering (blur)
        feather_radius = 1
        mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))

        return mask

    def process_image(self, image):
        """Process the image for background removal and crop to the non-transparent areas."""
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        from PIL import Image, ImageFilter

        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        birefnet.to(gfx_device())
        image_size = image.size
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_images = transform_image(image).unsqueeze(0).to(gfx_device())

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        # Create a mask from the prediction
        mask = pred_pil.resize(image_size)

        # Apply the mask to the original image
        image.putalpha(mask)

        # Crop the image to the non-transparent areas
        return self.crop_to_non_transparent(image)

    def crop_to_non_transparent(self, image):
        """Crops the image to the bounding box of non-transparent areas."""

        # Convert to RGBA if not already
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Get the data from the image
        data = image.getdata()

        # Create a mask for the non-transparent pixels
        non_transparent_pixels = [(r, g, b, a) for r, g, b, a in data if a > 0]

        # If there are no non-transparent pixels, return the original image
        if not non_transparent_pixels:
            return image

        # Find the bounding box of non-transparent pixels
        x_coords = [i % image.width for i in range(len(data)) if data[i][3] > 0]
        y_coords = [i // image.width for i in range(len(data)) if data[i][3] > 0]

        left = min(x_coords)
        right = max(x_coords)
        top = min(y_coords)
        bottom = max(y_coords)

        # Crop the image to the bounding box
        return image.crop((left, top, right + 1, bottom + 1))


    def split_by_alpha_islands(self, image_path, output_prefix):
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import label, find_objects
        import os

        # Load the image and convert it to RGBA
        img = Image.open(image_path).convert("RGBA")
        img_data = np.array(img)

        # Create a binary alpha mask (1 for opaque, 0 for transparent)
        alpha_mask = img_data[:, :, 3] > 0  # True where pixel is non-transparent

        # Label connected components in the alpha mask
        labeled_array, num_features = label(alpha_mask)

        # Prepare an array to store the file paths of saved images
        saved_paths = []

        # Iterate over each detected component (island of pixels)
        for i, bbox in enumerate(find_objects(labeled_array), start=1):
            if bbox is not None:
                # Extract bounding box
                character_img = img.crop((bbox[1].start, bbox[0].start, bbox[1].stop, bbox[0].stop))

                # Generate the file path and save each cropped character instance
                file_path = os.path.dirname(image_path) + "\\" + f"{output_prefix}_{i}.png"
                file_path = get_unique_file_name(file_path)
                character_img.save(file_path)
                saved_paths.append(file_path)
                if DEBUG:
                    print(f"Saved Asset part: {file_path}")

        return saved_paths


    def create_normal_map(self, image_path, output_path=None):
        """ Generates a normal map from the input image and saves it to the specified location. """
        # Import necessary libraries
        #import os
        from controlnet_aux import NormalBaeDetector
        from diffusers.utils import load_image
        from PIL import Image

        # Load the NormalBaeDetector model
        normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        normal_bae.to("cuda")  # Move model to GPU

        # Load the input image
        input_image = load_image(image_path)
        
        # Generate the normal map
        normal_map = normal_bae(input_image)

        # Define the output path
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_normal_map{ext}"

        # Convert normal_map to a PIL image and save it
        normal_map_image = Image.fromarray((normal_map * 255).astype("uint8"))
        normal_map_image.save(output_path)

        return output_path


    def convert_to_3d(self, context, transparent_image_path, prompt):
        """Converts an image with transparency into a 3D object (plane) and adds it to the asset library."""
        #import os
        #import bpy
        from PIL import Image, ImageFilter

        #get_unique_asset_name(self, context)
        asset_name = context.scene.asset_name

        # Ensure the image exists
        if not os.path.exists(transparent_image_path):
            self.report({"ERROR"}, f"Image not found at {transparent_image_path}")
            return {"CANCELLED"}

        # Load the image into Blender
        image = image = Image.open(transparent_image_path).convert("RGB")

        # Create a mask and crop the image to non-transparent areas
        processed_image = self.process_image(image)
        asset_name = re.sub(r'[<>:"/\\|?*]', "", asset_name)
        # Save the cropped image
        processed_image_path = bpy.path.abspath(os.path.join(bpy.path.abspath(bpy.utils.user_resource("DATAFILES")), "2D_Asset_Generator", f"{asset_name}_processed_image.png"))

        if DEBUG:
            print("processed_image_path: "+processed_image_path)

        processed_image.save(processed_image_path)

        normal_map_path = create_normal_map(processed_image_path)

        # Create a new material with transparency support
        material = bpy.data.materials.new(name="ImageMaterial")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs[12].default_value = 0  # Set Alpha to 0 for transparency
        bsdf.inputs["IOR"].default_value = 1.0  # Minimum effective IOR for transparency

        # Load the image into the material's base color and alpha inputs
        tex_image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        tex_image_node.image = bpy.data.images.load(processed_image_path)
        tex_image_node.interpolation = "Linear"

        # Add a ColorRamp node between the texture and BSDF
        color_ramp_node = material.node_tree.nodes.new("ShaderNodeValToRGB")

        # Add an Image Texture node for the normal map
        normal_map_texture_node = nodes.new(type='ShaderNodeTexImage')
        normal_map_texture_node.image = bpy.data.images.load(normal_map_path)  # Load the saved normal map
        normal_map_texture_node.image.colorspace_settings.name = 'Non-Color'  # Set as Non-Color Data

        # Add a Normal Map node
        normal_map_node = nodes.new(type='ShaderNodeNormalMap')

        # Link the nodes
        links.new(normal_map_texture_node.outputs["Color"], normal_map_node.inputs["Color"])  # Link image to normal map node
        links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])  # Link normal map to BSD

        # Position nodes for better visual organization in the node editor
        tex_image_node.location = (-900, 300)
        color_ramp_node.location = (-600, 300)
        normal_map_texture_node.location = (-400, 0)
        normal_map_node.location = (-200, 0)
        bsdf.location = (-300, 300)

        # Connect the texture's color output to the ColorRamp node input
        material.node_tree.links.new(color_ramp_node.inputs["Fac"], tex_image_node.outputs["Alpha"])

        # Connect the ColorRamp output to the Base Color input of the Principled BSDF
        material.node_tree.links.new(bsdf.inputs["Alpha"], color_ramp_node.outputs["Color"])

        # Connect the texture's alpha output to the BSDF's alpha input
        material.node_tree.links.new(bsdf.inputs["Base Color"], tex_image_node.outputs["Color"])

        # Adjust ColorRamp black and white stops
        color_ramp_node.color_ramp.elements[0].position = 0.75  # Move black point to 75%
        color_ramp_node.color_ramp.elements[0].color = (
            0,
            0,
            0,
            1,
        )  # Ensure black is fully black

        color_ramp_node.color_ramp.elements[1].position = 0.95  # Move white point to 95%
        color_ramp_node.color_ramp.elements[1].color = (
            1,
            1,
            1,
            1,
        )  # Ensure white is fully white

        #        # Create a new material with transparency support
        #        material = bpy.data.materials.new(name="ImageMaterial")
        #        material.use_nodes = True
        #        bsdf = material.node_tree.nodes.get("Principled BSDF")
        #        bsdf.inputs[12].default_value = 0

        #        # Load the image into the material's base color and alpha inputs
        #        tex_image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        #        tex_image_node.image = bpy.data.images.load(processed_image_path)
        #        tex_image_node.interpolation = 'Linear'

        #        # Connect the texture's color and alpha channels to the material's shader
        #        material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
        #        material.node_tree.links.new(bsdf.inputs['Alpha'], tex_image_node.outputs['Alpha'])

        # Enable transparency in the material
        material.blend_method = "HASHED"  #'BLEND'
        #material.shadow_method = "OPAQUE"
        material.surface_render_method = "DITHERED"

        # obj = bpy.ops.image.import_as_mesh_planes(filepath=transparent_image_path, files=[{"name":os.path.basename(transparent_image_path), "name":os.path.basename(transparent_image_path)}], directory=os.path.dirname(transparent_image_path))

        # Create a plane to hold the image
        bpy.ops.mesh.primitive_plane_add(size=1, location=bpy.context.scene.cursor.location)
        obj = bpy.context.object

        # Set the name for the new plane
        obj.name = asset_name

        # Assign the material to the plane
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        # obj.active_material.surface_render_method = 'BLENDED'

        # Adjust the plane's size to match the image aspect ratio
        img_width, img_height = processed_image.size
        aspect_ratio = img_width / img_height
        obj.scale = (aspect_ratio, 1, 1)

        # Apply transforms
        bpy.ops.transform.rotate(value=math.radians(-90), orient_axis="X")
        bpy.ops.transform.translate(
            value=(0, 0, 0.5),
            orient_type="GLOBAL",
            constraint_axis=(False, False, True),
        )
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Set origin point
        # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")

        bpy.context.view_layer.objects.active = obj

        # Avoid deleting the asset when deleting the object
        obj.data.use_fake_user = True

        # Mark the object as an asset
        obj.asset_mark()

        with context.temp_override(id=obj):
            bpy.ops.ed.lib_id_load_custom_preview(filepath=transparent_image_path)

        # Set asset metadata
        obj.asset_data.author = "2D Asset Generator"
        obj.asset_data.description = prompt  # f"Generated from: {os.path.basename(transparent_image_path)}"
        obj.asset_data.tags.new(name="GeneratedAsset")

        #get_unique_asset_name(self, context)

        if DEBUG:
            self.report({"INFO"}, "3D object created and added to the asset library")
            
        context.scene.asset_name = context.scene.asset_name


# UI Panel for Asset Generation and Setup
class FLUX_PT_GenerateAssetPanel(bpy.types.Panel):
    """Creates a Panel in the 3D View for the FLUX character generator"""

    bl_label = "2D Asset Generator"
    bl_idname = "VIEW3D_PT_generate_asset"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "2D Asset"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        import_text = scene.import_text

        # Button to install dependencies
        #layout = self.layout
        #layout.operator("object.setup_flux_env", text="Set-up Dependencies")

        layout = layout.box()

        # Toggle between Text-Block and Prompt as an expandable row
        row = layout.row()
        row.prop(import_text, "input_type", expand=True)

        # Show Text-Block selector if 'TEXT_BLOCK' is selected, otherwise show the prompt input
        if import_text.input_type == "TEXT_BLOCK":
            row = layout.row(align=True)
            row.prop(import_text, "scene_texts", text="", icon="TEXT", icon_only=True)
            row.prop(import_text, "script", text="")
        else:
            layout.prop(scene, "asset_prompt", text="Prompt")
            layout.prop(context.scene, "asset_name", text="Name")

        # Button to generate the character

        layout.operator("object.generate_asset", text="Generate")


# Register and Unregister classes and properties
classes = (
    Import_Text_Props,
    AssetGeneratorPreferences,
    #FLUX_OT_SetupEnvironment,
    FLUX_OT_GenerateAsset,
    FLUX_PT_GenerateAssetPanel,
    #virtual_dependencies_PreferencesPanel,
    InstallDependenciesOperator,
    UninstallDependenciesOperator,
    CheckDependenciesOperator,
)


# Registering the add-on and properties
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.import_text = bpy.props.PointerProperty(type=Import_Text_Props)

    # Register the asset_prompt as a scene property
    bpy.types.Scene.asset_prompt = bpy.props.StringProperty(
        name="Asset Description",
        description="Describe the asset to generate",
        default="",
    )
    bpy.types.Scene.asset_name = bpy.props.StringProperty(  # Add asset name property
        name="Asset Name",
        description="Name for the generated asset",
        default="",
        update=get_unique_asset_name,
    )


def unregister():
    del bpy.types.Scene.import_text
    for cls in classes:
        bpy.utils.unregister_class(cls)

    # Remove the asset_prompt property
    del bpy.types.Scene.asset_prompt
    del bpy.types.Scene.asset_name


if __name__ == "__main__":
    register()
