import os
import sys
import Ais.core.config as cfg
from Ais.core.window import *
from Ais.core.segmentation_editor import *
from imgui.integrations.glfw import GlfwRenderer
import tkinter as tk
import argparse
import time

tkroot = tk.Tk()
tkroot.withdraw()

# TODO: try using strategy = tf.distribute.MirroredStrategy() in SEModel train to make use of all GPUs.
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
    segment_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Integer 1 (default) or 0: whether to launch multiple parallel processes using one GPU each, or a single process using all GPUs.")
    segment_parser.add_argument('-o', '--overlap', required=False, type=float, help="Overlap to use between adjacent boxes, when cropping the input image into the shape required for the model. Default is whichever value is saved in the .scnm file.")

    train_parser = subparsers.add_parser('train', help='Train a model.')
    train_parser.add_argument('-a', '--model_architecture', required=False, type=int, help="Integer, index of which model architecture to use. Use -models for a list of available architectures.")
    train_parser.add_argument('-m', '--model_path', required=False, type=str, help="(Optional) path to a previously saved model to continue training. Overrides -a argument.")
    train_parser.add_argument('-t', '--training_data', required=True, type=str, help="Path to the training data (.scnt) file")
    train_parser.add_argument('-ou', '--output_directory', required=True, type=str, help="Directory to save the output")
    train_parser.add_argument('-gpu', '--gpus', required=True, type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,3,4)")
    train_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Integer 1 (default) or 0: whether to use TensorFlow's distribute.MirroredStrategy() for training in parallel on multiple GPUs, or a single process using all GPUs.")
    train_parser.add_argument('-e', '--epochs', required=False, type=int, default=50, help="Number of epochs to train the model for (default: 50).")
    train_parser.add_argument('-b', '--batch_size', required=False, type=int, default=32, help="Batch size to use during training (default: 32).")
    train_parser.add_argument('-n', '--negatives', required=False, type=float, default=1.3, help="Ratio of negative to positive samples to use. If the training data contains 50 positive samples and 50 negatives, a number of negatives will be sampled than once to reach this ratio.")
    train_parser.add_argument('-c', '--copies', required=False, type=int, default=10, help="Number of copies of the input images to include in the training data (all samples in different orientations).")
    train_parser.add_argument('-models', '--model_architectures', required=False, action='store_true', help='List available model architectures.')

    args, unknown = parser.parse_known_args()
    if args.command is None:
        run_ais()
    else:
        import Ais.core.headless_processes as aiscli

        if args.command == 'segment':
            gpus = [int(g) for g in args.gpus.split(',')]
            aiscli.dispatch_parallel_segment(model_path=args.model_path,
                                             data_directory=args.data_directory,
                                             output_directory=args.output_directory,
                                             gpus=gpus,
                                             skip=args.skip,
                                             parallel=args.parallel,
                                             overlap=args.overlap)
        elif args.command == 'train':
            if args.model_architectures:
                aiscli.print_available_model_architectures()
            else:
                aiscli.train_model(args.training_data, args.output_directory, args.model_architecture, args.model_path, args.gpus, args.parallel)


if __name__ == "__main__":
    main()


