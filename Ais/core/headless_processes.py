from Ais.core.se_frame import SEFrame
from Ais.core.se_model import SEModel
from Ais.core.segmentation_editor import QueuedExport
import Ais.core.config as cfg
import tensorflow as tf
import os
import time
import multiprocessing
import glob
import itertools
import glfw
from tensorflow.keras.models import MOdel


def glfw_init():
    if not glfw.init():
        raise Exception("Could not initialize GLFW library for headless start!")
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(1, 1, "invisible window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create invisible window!")
    glfw.make_context_current(window)
    return window


def _segmentation_thread(model_path, data_paths, output_dir, gpu_id, overlap):
    if isinstance(gpu_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    glfw_init()

    model = SEModel()
    model.load(model_path, compile=False)

    inference_model =
    if overlap is not None:
        model.overlap = min([0.9, max(overlap, 0.0)])

    queued_exports = list()
    for p in data_paths:
        queued_exports.append(QueuedExport(output_dir, SEFrame(p), [model], 1, False))

    for qe in queued_exports:
        qe.start()
        while qe.process.progress < 1.0:
            time.sleep(0.5)


def dispatch_parallel_segment(model_path, data_directory, output_directory, gpus, skip=1, parallel=1, overlap=None):
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    if not os.path.isabs(data_directory):
        data_directory = os.path.join(os.getcwd(), data_directory)

    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    # distribute data:
    all_data_paths = glob.glob(os.path.join(data_directory, "*.mrc"))

    if skip == 1:
        model_metadata = SEModel.load_metadata(model_path)
        model_title = model_metadata["title"]
        all_data_paths = [p for p in all_data_paths if not os.path.exists(os.path.join(data_directory, os.path.splitext(os.path.basename(p))[0]+f"__{model_title}.mrc"))]

    data_div = {gpu: list() for gpu in gpus}
    for gpu, data_path in zip(itertools.cycle(gpus), all_data_paths):
        data_div[gpu].append(data_path)

    # launch the _segmentation_threads (on different CPUs?) using different GPUs

    if parallel == 1:
        processes = []
        for gpu_id in data_div:
            p = multiprocessing.Process(target=_segmentation_thread,
                                        args=(model_path, data_div[gpu_id], output_directory, gpu_id, overlap))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        _segmentation_thread(model_path, all_data_paths, output_directory, gpu_id=",".join(str(n) for n in gpus), overlap)


def print_available_model_architectures():
    model = SEModel()
    model.load_models()
    for j, key in enumerate(SEModel.AVAILABLE_MODELS):
        print(f"index: {j}\tarchitecture name: {key}\t|\t <-a {j}>")


def train_model(training_data, output_directory, epochs, batch_size, negatives, copies, architecture=None, model_path=None, gpus="0", parallel=1):
    if not os.path.isabs(training_data):
        model_path = os.path.join(os.getcwd(), training_data)
    if not os.path.isabs(output_directory):
        model_path = os.path.join(os.getcwd(), output_directory)
    if model_path is not None and not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    model = SEModel()
    model.load_models()
    if architecture is None and model_path is None:
        print(f"using default model architecture: {SEModel.AVAILABLE_MODELS[SEModel.DEFAULT_MODEL_ENUM]}")
    elif architecture is None:
        print(f"loading model {model_path}")
        model.load(model_path)
    elif model_path is None:
        model.model_enum = architecture
        print(f"using model architecture {architecture}: {SEModel.AVAILABLE_MODELS[architecture]}")

    model.train_data_path = training_data
    model.epochs = epochs
    model.batch_size = batch_size
    model.excess_negative = int((100 * negatives) - 100)
    model.n_copies = copies

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if parallel:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model.train()
    else:
        model.train()

    while model.background_process_train.progress < 1.0:
        time.sleep(0.2)

    model.save(os.path.join(output_directory, f"{model.apix:.2f}_{model.box_size}_{model.loss:.4f}_{model.title}.{cfg.filetype_semodel}"))
