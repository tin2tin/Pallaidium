import ast
import os
import sys

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_folder("properties")
create_folder("ui")
create_folder("operators")
create_folder("utils")
create_folder("models")

with open("__init__org.py", "r", encoding="utf-8") as f:
    source = f.read()

tree = ast.parse(source)

preferences = []
scene_props = []
panels = []
operators = []
helpers = []

imports = []
bl_info_node = None
classes_list = []
register_func = None
unregister_func = None

for node in tree.body:
    src = ast.get_source_segment(source, node)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append(src)
    elif isinstance(node, ast.Assign):
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'bl_info' in targets:
            bl_info_node = src
        elif 'classes' in targets:
            classes_list.append(src)
        else:
            helpers.append(src)
    elif isinstance(node, ast.FunctionDef):
        if node.name == 'register':
            register_func = src
        elif node.name == 'unregister':
            unregister_func = src
        else:
            helpers.append(src)
    elif isinstance(node, ast.ClassDef):
        basenames = [b.id for b in node.bases if isinstance(b, ast.Name)]
        if 'AddonPreferences' in basenames:
            preferences.append(src)
        elif 'PropertyGroup' in basenames:
            scene_props.append(src)
        elif 'Panel' in basenames:
            panels.append(src)
        elif 'Operator' in basenames:
            operators.append(src)
        else:
            helpers.append(src)
    else:
        helpers.append(src)

import_block = "from __future__ import annotations\n" + "\n".join(imports) + "\n\n"

# Write utils/helpers.py
with open("utils/helpers.py", "w", encoding="utf-8") as f:
    f.write(import_block)
    f.write("\n\n".join(helpers))

# Write properties/preferences.py
with open("properties/preferences.py", "w", encoding="utf-8") as f:
    f.write(import_block)
    f.write("from ..utils.helpers import *\n\n")
    f.write("\n\n".join(preferences))

# Write properties/scene_props.py
with open("properties/scene_props.py", "w", encoding="utf-8") as f:
    f.write(import_block)
    f.write("from ..utils.helpers import *\n\n")
    f.write("\n\n".join(scene_props))

# Write ui/panels.py
with open("ui/panels.py", "w", encoding="utf-8") as f:
    f.write(import_block)
    f.write("from ..utils.helpers import *\nfrom ..properties.scene_props import *\nfrom ..properties.preferences import *\n\n")
    f.write("\n\n".join(panels))

# Write operators/main_ops.py
with open("operators/main_ops.py", "w", encoding="utf-8") as f:
    f.write(import_block)
    f.write("from ..utils.helpers import *\nfrom ..properties.scene_props import *\nfrom ..properties.preferences import *\nfrom ..ui.panels import *\n\n")
    f.write("\n\n".join(operators))

# Reconstruct classes list and register/unregister
# The classes variable in __init__org.py contained all classes in order.
# We will just write a new __init__.py that imports everything.

init_content = f"""from __future__ import annotations

if "bpy" in locals():
    import importlib
    importlib.reload(utils)
    importlib.reload(properties)
    importlib.reload(ui)
    importlib.reload(operators)
else:
    from .utils.helpers import *
    from .properties import *
    from .ui import *
    from .operators import *

{bl_info_node}

{chr(10).join(classes_list)}

{register_func}

{unregister_func}

if __name__ == "__main__":
    register()
"""

with open("__init__.py", "w", encoding="utf-8") as f:
    f.write(init_content)

# We also need empty __init__.py files in submodules to make them packages.
open("utils/__init__.py", "w").close()

with open("properties/__init__.py", "w") as f:
    f.write("from .preferences import *\nfrom .scene_props import *\n")

with open("ui/__init__.py", "w") as f:
    f.write("from .panels import *\n")

with open("operators/__init__.py", "w") as f:
    f.write("from .main_ops import *\n")

print("Refactoring complete.")
