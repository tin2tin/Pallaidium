import sys
import os

addon_dir = r"c:\Users\peter\Downloads\Pallaidium-Blender-5.1\Pallaidium-Blender-5.1"
sys.path.insert(0, addon_dir)
import zipfile
import addon_utils
import shutil

addon_path = r"c:\Users\peter\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\user_default\pallaidium_generative_ai"
shutil.rmtree(addon_path, ignore_errors=True)
shutil.copytree(addon_dir, addon_path)

import bl_ext.user_default.pallaidium_generative_ai
print(bl_ext.user_default.pallaidium_generative_ai.classes)

