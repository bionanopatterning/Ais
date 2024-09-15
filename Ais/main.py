import os
import sys
import Ais.core.config as cfg
from Ais.core.window import *
from Ais.core.segmentation_editor import *
import Ais.core.headless_processes as aiscli
from imgui.integrations.glfw import GlfwRenderer
import tkinter as tk
import argparse
import time

tkroot = tk.Tk()
tkroot.withdraw()

directory = os.path.join(os.path.dirname(__file__))
directory = directory[:directory.rfind("\\")]
sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(directory)
cfg.root = os.path.join(os.path.dirname(__file__))

def run_ais():
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
    segmentation_editor.force_not_embedded()
    cfg.segmentation_editor = segmentation_editor
    main_window.set_icon(segmentation_editor.ICON)

    from Ais.core import start

    while not glfw.window_should_close(main_window.glfw_window):
        if not main_window.focused:
            glfw.poll_events()
        segmentation_editor.on_update()
        segmentation_editor.end_frame()


def windowless():
    if not glfw.init():
        raise Exception("Could not initialize GLFW library for headless start!")
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(1, 1, "invisible window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create invisible window!")
    glfw.make_context_current(window)
    return window


def main():
    parser = argparse.ArgumentParser(description="Ais headless CLI parser")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    segment_parser = subparsers.add_parser('segment', help='Segment .mrc volumes more efficiently in GUI-less mode.')
    segment_parser.add_argument('-m', '--model_path', required=True, type=str, help="Path to the model file")
    segment_parser.add_argument('-d', '--data_directory', required=True, type=str, help="Directory containing the data")
    segment_parser.add_argument('-ou', '--output_directory', required=True, type=str, help="Directory to save the output")
    segment_parser.add_argument('-gpu', '--gpus', required=True, type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,3,4)")
    segment_parser.add_argument('-s', '--skip', required=False, type=int, default=1, help="Integer 1 (default) or 0: whether to skip (yes if 1, no if 0) tomograms for which a corresponding segmentation is already found.")

    args = parser.parse_args()
    if args.command is None:
        print("Launching Ais!")
        run_ais()
    elif args.command is 'segment':
        windowless()
        aiscli.dispatch_parallel_segment(args.model_path, args.data_directory, args.output_directory, args.gpus, args.skip)


if __name__ == "__main__":
    main()


