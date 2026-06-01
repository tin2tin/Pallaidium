bl_info = {
    "name": "Pallaidium Test Kit",
    "author": "tintwotin",
    "version": (2, 8, 2),
    "blender": (5, 1, 0),
    "location": "Text Editor > Sidebar > Pallaidium Test",
    "description": "Scans and tests all available models in the 'Pallaidium - Generative AI' add-on.",
    "category": "Testing",
    "doc_url": "https://github.com/tin2tin/Pallaidium",
}

import sys
import bpy
import traceback

# --- Configuration ---

PALLAIDIUM_MODULE_NAME = "bl_ext.user_default.pallaidium_generative_ai"
PALLAIDIUM_OPERATOR_IDNAME = "SEQUENCER_OT_generate_image"

# Plugin MODEL_TYPE (lowercase) -> addon preference attribute name
_TYPE_TO_PREF = {
    "image": "image_model_card",
    "text":  "text_model_card",
    "audio": "audio_model_card",
    "video": "movie_model_card",  # plugin registry uses "video"; pref uses "movie_model_card"
}

# Plugin MODEL_TYPE -> sequencer operator callable
_TYPE_TO_OP = {
    "image": lambda: bpy.ops.sequencer.generate_image(),
    "text":  lambda: bpy.ops.sequencer.generate_text(),
    "audio": lambda: bpy.ops.sequencer.generate_audio(),
    "video": lambda: bpy.ops.sequencer.generate_movie(),
}

# Plugin MODEL_TYPE -> generatorai_typeselect enum value
# ("video" in registry maps to "movie" in the scene enum)
_TYPE_TO_SCENE_SELECT = {
    "image": "image",
    "text":  "text",
    "audio": "audio",
    "video": "movie",
}

# --- Helpers ---

def is_pallaidium_enabled():
    return hasattr(bpy.types, PALLAIDIUM_OPERATOR_IDNAME)


def _get_plugin_registry():
    """Return PLUGIN_REGISTRY from the loaded Pallaidium models module, or None."""
    mod = sys.modules.get(f"{PALLAIDIUM_MODULE_NAME}.models")
    if mod is None:
        try:
            import importlib
            mod = importlib.import_module(f"{PALLAIDIUM_MODULE_NAME}.models")
        except Exception:
            return None
    return getattr(mod, "PLUGIN_REGISTRY", None)


def get_sequencer_override(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'SEQUENCE_EDITOR':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                        }
                        if hasattr(context, 'workspace'):
                            override['workspace'] = context.workspace
                        return override
    return None


# --- Properties ---

class PallaidiumTestModel(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Model Name")
    is_tested: bpy.props.BoolProperty(name="Test This Model", default=True)
    model_type: bpy.props.StringProperty(name="Model Type")
    model_id: bpy.props.StringProperty(name="Model Identifier")


class PallaidiumTestSettings(bpy.types.PropertyGroup):
    models: bpy.props.CollectionProperty(type=PallaidiumTestModel)
    is_initialized: bpy.props.BoolProperty(name="Has Scanned for Models", default=False)


# --- Operators ---

class PALLAIDIUM_OT_ToggleAll(bpy.types.Operator):
    """Enable or disable all models at once."""
    bl_idname = "pallaidium.toggle_all"
    bl_label = "Toggle All Models"

    mode: bpy.props.EnumProperty(items=[('ON', 'On', 'Enable all'), ('OFF', 'Off', 'Disable all')])

    def execute(self, context):
        settings = context.scene.pallaidium_test_settings
        new_state = (self.mode == 'ON')
        for model in settings.models:
            model.is_tested = new_state
        return {'FINISHED'}


class PALLAIDIUM_OT_ToggleType(bpy.types.Operator):
    """Enable or disable all models of a specific type."""
    bl_idname = "pallaidium.toggle_type"
    bl_label = "Toggle Model Type"

    mode: bpy.props.EnumProperty(items=[('ON', 'On', 'Enable type'), ('OFF', 'Off', 'Disable type')])
    model_type: bpy.props.StringProperty()

    def execute(self, context):
        settings = context.scene.pallaidium_test_settings
        new_state = (self.mode == 'ON')
        for model in settings.models:
            if model.model_type == self.model_type:
                model.is_tested = new_state
        return {'FINISHED'}


class PALLAIDIUM_OT_RefreshModels(bpy.types.Operator):
    """Scans the Pallaidium plugin registry to find all available models."""
    bl_idname = "pallaidium.refresh_models"
    bl_label = "Scan for Pallaidium Models"

    def execute(self, context):
        settings = context.scene.pallaidium_test_settings
        settings.models.clear()

        if PALLAIDIUM_MODULE_NAME not in bpy.context.preferences.addons:
            self.report({'ERROR'}, f"Could not find Add-on: {PALLAIDIUM_MODULE_NAME}")
            return {'CANCELLED'}

        registry = _get_plugin_registry()
        if registry is None:
            self.report({'ERROR'}, "Could not access Pallaidium plugin registry. See System Console.")
            settings.is_initialized = False
            return {'CANCELLED'}

        try:
            total = 0
            for model_id, plugin in registry.items():
                item = settings.models.add()
                item.name = plugin.DISPLAY_NAME
                item.model_type = plugin.MODEL_TYPE  # lowercase: "image", "text", "audio", "video"
                item.model_id = model_id
                total += 1
        except Exception:
            self.report({'ERROR'}, "Error during scan. See System Console.")
            traceback.print_exc()
            settings.is_initialized = False
            return {'CANCELLED'}

        settings.is_initialized = True
        self.report({'INFO'}, f"Scan complete. Found {total} models.")
        return {'FINISHED'}


class PALLAIDIUM_OT_RunTests(bpy.types.Operator):
    """Runs inference tests on the selected Pallaidium models."""
    bl_idname = "pallaidium.run_tests"
    bl_label = "Run Selected Tests"

    @classmethod
    def poll(cls, context):
        settings = context.scene.pallaidium_test_settings
        return settings.is_initialized and len(settings.models) > 0

    def execute(self, context):
        if PALLAIDIUM_MODULE_NAME not in bpy.context.preferences.addons:
            self.report({'ERROR'}, f"Could not find Add-on: {PALLAIDIUM_MODULE_NAME}")
            return {'CANCELLED'}

        if not hasattr(context.scene, 'generatorai_typeselect'):
            self.report({'ERROR'}, "Scene properties from Pallaidium are missing. Is it loaded?")
            return {'CANCELLED'}

        settings = context.scene.pallaidium_test_settings
        prefs = bpy.context.preferences.addons[PALLAIDIUM_MODULE_NAME].preferences

        models_to_test = [m for m in settings.models if m.is_tested]
        if not models_to_test:
            self.report({'WARNING'}, "No models were selected for testing.")
            return {'CANCELLED'}

        override = get_sequencer_override(context)
        if not override:
            self.report({'ERROR'}, "No Sequence Editor area found. Please open one to run tests.")
            return {'CANCELLED'}

        input_mode = getattr(context.scene, 'input_strips', 'input_prompt')
        report_lines = ["| Model | Type | Status | Notes |", "|---|---|---|---|"]

        for model in models_to_test:
            model_type = model.model_type  # lowercase: "image", "text", "audio", "video"
            pref_attr = _TYPE_TO_PREF.get(model_type)
            if pref_attr is None:
                report_lines.append(
                    f"| {model.name} | {model_type} | ⚠️ | Unknown model type, skipped. |"
                )
                continue

            self.report({'INFO'}, f"Testing: {model.name} ({model_type})...")
            try:
                setattr(prefs, pref_attr, model.model_id)

                with context.temp_override(**override):
                    if input_mode == "input_prompt":
                        op_fn = _TYPE_TO_OP.get(model_type)
                        if op_fn:
                            op_fn()
                    else:
                        scene_select = _TYPE_TO_SCENE_SELECT.get(model_type, model_type)
                        context.scene.generatorai_typeselect = scene_select
                        bpy.ops.sequencer.text_to_generator()

                report_lines.append(f"| {model.name} | {model_type} | ✅ | Works as expected. |")

            except Exception as e:
                error_message = str(e).replace("\n", " ").strip().replace("|", r"\|")
                report_lines.append(f"| {model.name} | {model_type} | ❌ | Error: {error_message} |")

        report_text = bpy.data.texts.new("Pallaidium Test Report.md")
        report_text.write("\n".join(report_lines))
        self.report({'INFO'}, "All tests complete. Report created in Text Editor.")
        return {'FINISHED'}


# --- UI Panel ---

class PALLAIDIUM_PT_TestPanel(bpy.types.Panel):
    bl_label = "Pallaidium Test Kit"
    bl_idname = "TEXT_PT_pallaidium_test"
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Pallaidium Test'

    def draw(self, context):
        layout = self.layout
        if not is_pallaidium_enabled():
            layout.box().label(text="Pallaidium add-on is not enabled.", icon='ERROR')
            return

        settings = context.scene.pallaidium_test_settings
        layout.operator("pallaidium.refresh_models", icon='FILE_REFRESH')

        if settings.is_initialized:
            if not settings.models:
                layout.label(text="No models found. Press Scan again.", icon='INFO')
                return

            row = layout.row(align=True)
            all_enabled = all(m.is_tested for m in settings.models)
            if all_enabled:
                op = row.operator("pallaidium.toggle_all", text="Disable All", icon='CHECKBOX_HLT')
                op.mode = 'OFF'
            else:
                op = row.operator("pallaidium.toggle_all", text="Enable All", icon='CHECKBOX_DEHLT')
                op.mode = 'ON'

            model_types = sorted(set(m.model_type for m in settings.models))
            for model_type in model_types:
                box = layout.box()
                header = box.row(align=True)
                type_models = [m for m in settings.models if m.model_type == model_type]
                type_all_enabled = all(m.is_tested for m in type_models)

                header.label(text=f"{model_type.title()} Models", icon='MOD_WAVE')

                if type_all_enabled:
                    op = header.operator("pallaidium.toggle_type", text="", icon='CHECKBOX_HLT')
                    op.mode = 'OFF'
                    op.model_type = model_type
                else:
                    op = header.operator("pallaidium.toggle_type", text="", icon='CHECKBOX_DEHLT')
                    op.mode = 'ON'
                    op.model_type = model_type

                for model in type_models:
                    box.prop(model, "is_tested", text=model.name)

            layout.separator()
            layout.operator("pallaidium.run_tests", icon='PLAY')
        else:
            layout.label(text="Click 'Scan' to find available models.", icon='INFO')


# --- Registration ---

classes = (
    PallaidiumTestModel,
    PallaidiumTestSettings,
    PALLAIDIUM_OT_ToggleAll,
    PALLAIDIUM_OT_ToggleType,
    PALLAIDIUM_OT_RefreshModels,
    PALLAIDIUM_OT_RunTests,
    PALLAIDIUM_PT_TestPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pallaidium_test_settings = bpy.props.PointerProperty(type=PallaidiumTestSettings)


def unregister():
    if hasattr(bpy.types.Scene, 'pallaidium_test_settings'):
        del bpy.types.Scene.pallaidium_test_settings
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
