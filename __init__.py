# https://modelscope.cn/models/damo/text-to-video-synthesis/summary

bl_info = {
    "name": "Text to Video",
    "author": "tintwotin",
    "version": (1, 0),
    "blender": (3, 40, 0),
    "location": "Video Sequence Editor > Sidebar > Text to Video",
    "description": "Convert text to video",
    "category": "Video Sequence Editor",
}

import bpy
from bpy.types import Operator, Panel
import site
import subprocess
import sys


def import_module(self, module):
    module = str(module)
    try:
        exec("import " + module)
    except ModuleNotFoundError:
        app_path = site.USER_SITE
        if app_path not in sys.path:
            sys.path.append(app_path)
        pybin = sys.executable  # bpy.app.binary_path_python # Use for 2.83

        print("Ensuring: pip")
        try:
            subprocess.call([pybin, "-m", "ensurepip"])
        except ImportError:
            pass
        self.report({"INFO"}, "Installing: " + module + " module.")
        print("Installing: " + module + " module")
        subprocess.check_call([pybin, "-m", "pip", "install", module])
        try:
            exec("import " + module)
        except ModuleNotFoundError:
            return False
    return True


class SequencerImportMovieOperator(Operator):
    """Import a movie strip into the Sequencer at the current frame"""

    bl_idname = "sequencer.import_movie"
    bl_label = "Prompt"
    bl_description = "Import a movie strip into the Sequencer at the current frame"
    bl_options = {"REGISTER", "UNDO"}

    filename: bpy.props.StringProperty(
        name="File Name", default="", description="Name of the movie file to import"
    )

    def execute(self, context):
        scene = context.scene

        app_path = site.USER_SITE
        if app_path not in sys.path:
            sys.path.append(app_path)
        pybin = sys.executable

        try:
            subprocess.call([pybin, "-m", "ensurepip"])
        except ImportError:
            pass       
        
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                "git+https://github.com/modelscope/modelscope.git",
                "--user",
            ]
        )
        try:
            import modelscope
        except ModuleNotFoundError:
            print("Installation of the modelscope module failed")
            self.report(
                {"INFO"},
                "Installing modelscope module failed! Try to run Blender as administrator.",
            )
            return False
        import_module(self, "open_clip_torch")
        import_module(self, "pytorch-lightning")

        from modelscope.pipelines import pipeline
        from modelscope.outputs import OutputKeys

        p = pipeline("text-to-video-synthesis", "damo/text-to-video-synthesis")
        test_text = {"text": self.filename}
        output_video_path = p(
            test_text,
        )[OutputKeys.OUTPUT_VIDEO]

        filepath = bpy.path.abspath(output_video_path)
        strip = scene.sequence_editor.sequences.new_movie(
            name=self.filename,
            filepath=filepath,
            channel=1,
            frame_start=scene.frame_current,
        )
        if strip:
            strip.frame_final_duration = strip.frame_duration
        return {"FINISHED"}


class SequencerPanel(Panel):
    """Text to Video usin ModelScope"""

    bl_idname = "SEQUENCER_PT_sequencer_panel"
    bl_label = "Text to Video"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Text to Video"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        row.prop(context.scene, "my_movie_filename", text="")
        row = layout.row()
        row.operator("sequencer.import_movie", text="Generate Movie")


def register():
    bpy.utils.register_class(SequencerImportMovieOperator)
    bpy.utils.register_class(SequencerPanel)
    bpy.types.Scene.my_movie_filename = bpy.props.StringProperty(
        name="My Movie Filename", default=""
    )


def unregister():
    bpy.utils.unregister_class(SequencerImportMovieOperator)
    bpy.utils.unregister_class(SequencerPanel)
    del bpy.types.Scene.my_movie_filename


if __name__ == "__main__":
    register()
