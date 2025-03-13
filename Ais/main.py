from Ais.core.window import *
from Ais.core.segmentation_editor import *
from imgui.integrations.glfw import GlfwRenderer
import tkinter as tk
import argparse
import time


directory = os.path.join(os.path.dirname(__file__))
directory = directory[:directory.rfind("\\")]
sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(directory)
cfg.root = os.path.join(os.path.dirname(__file__))


def run_ais():
    tkroot = tk.Tk()
    tkroot.withdraw()

    if not glfw.init():
        raise Exception("Could not initialize GLFW library!")

    cfg.glfw_initialized = True
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


    while not glfw.window_should_close(main_window.glfw_window):
        if not main_window.focused:
            glfw.poll_events()
        segmentation_editor.on_update()
        segmentation_editor.end_frame()


def main():
    parser = argparse.ArgumentParser(description="Ais headless CLI parser")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    segment_parser = subparsers.add_parser('segment', help='Segment .mrc volumes more efficiently in GUI-less mode.')
    segment_parser.add_argument('-m', '--model_path', required=True, type=str, help="Path to the model file")
    segment_parser.add_argument('-d', '--data_directory', required=True, type=str, help="Directory containing the data") # todo allow .mrc input instead of directory.
    segment_parser.add_argument('-ou', '--output_directory', required=True, type=str, help="Directory to save the output")
    segment_parser.add_argument('-gpu', '--gpus', required=True, type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,3,4)")#
    segment_parser.add_argument('-tta', '--test-time-augmentation', required=False, type=int, default=1, help="Integer between 1 and 8. If 1, no test time augmentation applied. If 2 - 8, differently oriented copies of the input tomogram are also segmented and the results averaged; orientations are [0, 90, 180, 270, 0*, 90*, 180*, 270*] (*=horizontal flip); e.g., when -tta 4, four samples of each tomogram are segmented, sampled with 0, 90, 180, and 270 deg. rotations relative to the original.")
    segment_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Integer 1 (default) or 0: whether to launch multiple parallel processes using one GPU each, or a single process using all GPUs.")
    segment_parser.add_argument('-overwrite', '--overwrite', required=False, type=int, default=0, help="If set to 1, tomograms for which a corresponding segmentation in the output_directory already exists are skipped (default 0).")

    pick_parser = subparsers.add_parser('pick', help='Pick particles using segmented volumes.')
    pick_parser.add_argument('-d', '--data_directory', required=True, type=str, help="Path to directory containing input .mrc's.")
    pick_parser.add_argument('-t', '--target', required=True, type=str, help="Feature to pick. For example, if segmented volumes are named '<tomogram_name>__Ribosome.mrc', '-t Ribosome' will select these.")
    pick_parser.add_argument('-ou', '--output_directory', required=False, type=str, default=None, help="Directory to save output coordinate files to. If left empty, will save to the input data directory.")
    pick_parser.add_argument('-m', '--margin', required=False, type=int, default=16, help="Margin (in pixels) to avoid picking particles close to tomogram edges.")
    pick_parser.add_argument('-threshold', required=False, type=float, default=128, help="Threshold to apply to volumes prior to finding local maxima (default 128).")
    pick_parser.add_argument('-b', '--binning', required=False, type=int, default=1, help="Bining factor to apply before processing (faster, possibly less accurate). Default is 1 (no binning)")
    pick_parser.add_argument('-spacing', required=False, type=float, default=10.0, help="Minimum distance between particles in Angstrom. Use ``-spacing-px`` to specify the minimum distance in voxel units instead.")
    pick_parser.add_argument('-spacing-px', required=False, type=float, default=None, help="Minimum distance between particles in px.")
    pick_parser.add_argument('-size', required=False, type=float, default=10.0, help="Minimum particle size in cubic Angstrom. Use ``-size-px`` to specify the minimum size in cubic voxel units instead.")
    pick_parser.add_argument('-size-px', required=False, type=float, default=None, help="Minimum particle size in number of voxels.")
    pick_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Number of parallel picking processes to use (e.g. ``-p 64``, or however many threads your system can run at a time).")
    pick_parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Verbose (1 or 0)")

    train_parser = subparsers.add_parser('train', help='Train a model.')
    train_parser.add_argument('-a', '--model_architecture', required=False, type=int, help="Integer, index of which model architecture to use. Use -models for a list of available architectures.")
    train_parser.add_argument('-m', '--model_path', required=False, type=str, default='', help="(Optional) path to a previously saved model to continue training. Overrides -a argument.")
    train_parser.add_argument('-t', '--training_data', required=False, type=str, help="Path to the training data (.scnt) file")
    train_parser.add_argument('-ou', '--output_directory', required=False, type=str, help="Directory to save the output")
    train_parser.add_argument('-gpu', '--gpus', required=False, default="0", type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,4,5)")
    train_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Integer 1 (default) or 0: whether to use TensorFlow's distribute.MirroredStrategy() for training in parallel on multiple GPUs, or a single process using all GPUs.")
    train_parser.add_argument('-e', '--epochs', required=False, type=int, default=50, help="Number of epochs to train the model for (default: 50).")
    train_parser.add_argument('-b', '--batch_size', required=False, type=int, default=32, help="Batch size to use during training (default: 32).")
    train_parser.add_argument('-n', '--negatives', required=False, type=float, default=0.0, help="If 0.0 (default), all images in the input training data are weighted identically. If argument supplied, the value determines the ratio of negative to positive samples to use. For example: if the training data contains 50 positive samples and 50 negatives, and the negative to positive ratio is 1.5, a number of negatives will be sampled twice in order to reach this ratio.") # TODO: maybe remove this altogether
    train_parser.add_argument('-c', '--copies', required=False, type=int, default=10, help="Number of copies of the input images to include in the training data (all samples in different orientations).")
    train_parser.add_argument('-r', '--rate', required=False, type=float, default=1e-3,help="Learning rate (default 1e-3)")
    train_parser.add_argument('-name', '--model_name', required=False, type=str, default="Unnamed model", help="Model name. File will be saved as output_directory/<boxsize>_<apix>_<name>.scnm")
    train_parser.add_argument('-models', '--model_architectures', required=False, action='store_true', help='List available model architectures.')

    args, unknown = parser.parse_known_args()
    if args.command is None:
        run_ais()
    else:
        import Ais.core.cli_fn as aiscli

        if args.command == 'segment':
            gpus = [int(g) for g in args.gpus.split(',')]
            aiscli.dispatch_parallel_segment(model_path=args.model_path,
                                             data_directory=args.data_directory,
                                             output_directory=args.output_directory,
                                             gpus=gpus,
                                             test_time_augmentation=args.test_time_augmentation,
                                             parallel=args.parallel,
                                             overwrite=args.overwrite)
        elif args.command == 'pick':
            output_directory = args.output_directory if args.output_directory else args.data_directory
            aiscli.dispatch_parallel_pick(target=args.target,
                                          data_directory=args.data_directory,
                                          output_directory=output_directory,
                                          margin=args.margin,
                                          threshold=args.threshold,
                                          binning=args.binning,
                                          spacing=args.spacing,
                                          size=args.size,
                                          parallel=args.parallel,
                                          spacing_px=args.spacing_px,
                                          size_px=args.size_px,
                                          verbose=args.verbose==1)
        elif args.command == 'train':
            if args.model_architectures:
                aiscli.print_available_model_architectures()
            else:
                aiscli.train_model(training_data=args.training_data,
                                   output_directory=args.output_directory,
                                   architecture=args.model_architecture,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   negatives=args.negatives,
                                   copies=args.copies,
                                   model_path=args.model_path,
                                   gpus=args.gpus,
                                   parallel=args.parallel,
                                   rate=args.rate,
                                   name=args.model_name)


if __name__ == "__main__":
    main()


