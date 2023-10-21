import os
import sys
import scNodes.core.config as cfg
from scNodes.core.window import *
from scNodes.core.segmentation_editor import *
from imgui.integrations.glfw import GlfwRenderer
import tkinter as tk

tkroot = tk.Tk()
tkroot.withdraw()

directory = os.path.join(os.path.dirname(__file__))
directory = directory[:directory.rfind("\\")]
sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(directory)
cfg.root = os.path.join(os.path.dirname(__file__))

if __name__ == "__main__":
    if not glfw.init():
        raise Exception("Could not initialize GLFW library!")

    # Init the main window, its imgui context, and a glfw rendering impl.
    main_window = Window(cfg.window_width, cfg.window_height, settings.ne_window_title)
    main_window.set_callbacks()
    main_window_imgui_context = imgui.create_context()
    main_window_imgui_glfw_implementation = GlfwRenderer(main_window.glfw_window)
    main_window.set_mouse_callbacks()
    main_window.set_window_callbacks()

    # set up editor
    segmentation_editor = SegmentationEditor(main_window, main_window_imgui_context, main_window_imgui_glfw_implementation)
    cfg.segmentation_editor = segmentation_editor


    while not glfw.window_should_close(main_window.glfw_window):
        if not main_window.focused:
            glfw.poll_events()
        segmentation_editor.on_update()
        segmentation_editor.end_frame()




