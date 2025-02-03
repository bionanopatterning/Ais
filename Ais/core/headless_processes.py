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
import mrcfile
import numpy as np


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


def _segment_tomo(tomo_path, model):
    volume = np.array(mrcfile.read(tomo_path).data)
    volume -= np.mean(volume)
    volume /= np.std(volume)
    segmented_volume = np.zeros_like(volume)

    for j in range(volume.shape[0]):
        segmented_volume[j, :, :] = np.squeeze(model.predict(volume[j, :, :][np.newaxis, :, :]))

    return segmented_volume


def _segmentation_thread(model_path, data_paths, output_dir, gpu_id, overwrite=False):
    from keras.models import clone_model
    from keras.layers import Input

    if isinstance(gpu_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    glfw_init()

    se_model = SEModel()
    se_model.load(model_path, compile=False)
    model = se_model.model
    new_input = Input(shape=(None, None, 1))
    new_model = clone_model(model, input_tensors=new_input)
    new_model.set_weights(model.get_weights())


    for j, p in enumerate(data_paths):
        tomo_name = os.path.basename(os.path.splitext(p)[0])
        out_path = os.path.join(output_dir, tomo_name+"__"+se_model.title+".mrc")
        print(f"{j+1}/{len(data_paths)}({gpu_id}) - {p}")
        if os.path.exists(out_path) and not overwrite:
            continue

        in_voxel_size = mrcfile.open(p, header_only=True).voxel_size
        segmented_volume = _segment_tomo(p, new_model)
        segmented_volume = (segmented_volume * 255).astype(np.uint8)
        with mrcfile.new(out_path, overwrite=True) as mrc:
            mrc.set_data(segmented_volume)
            mrc.voxel_size = in_voxel_size


def dispatch_parallel_segment(model_path, data_directory, output_directory, gpus, parallel=1, overwrite=0):
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    if not os.path.isabs(data_directory):
        data_directory = os.path.join(os.getcwd(), data_directory)

    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    # distribute data:
    all_data_paths = glob.glob(os.path.join(data_directory, "*.mrc"))

    data_div = {gpu: list() for gpu in gpus}
    for gpu, data_path in zip(itertools.cycle(gpus), all_data_paths):
        data_div[gpu].append(data_path)

    # launch the _segmentation_threads (on different CPUs?) using different GPUs

    if parallel == 1:
        processes = []
        for gpu_id in data_div:
            p = multiprocessing.Process(target=_segmentation_thread,
                                        args=(model_path, data_div[gpu_id], output_directory, gpu_id, overwrite))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        _segmentation_thread(model_path, all_data_paths, output_directory, gpu_id=",".join(str(n) for n in gpus), overwrite=bool(overwrite))


def print_available_model_architectures():
    model = SEModel(no_glfw=True)
    model.load_models()
    for j, key in enumerate(SEModel.AVAILABLE_MODELS):
        print(f"index: {j} (-a {j})\tarchitecture name: {key}")


def train_model(training_data, output_directory, architecture=None, epochs=50, batch_size=32, negatives=1.0, copies=4, model_path='', gpus="0", parallel=1, name="Unnamed model"):
    if not os.path.isabs(training_data):
        training_data = os.path.join(os.getcwd(), training_data)
    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    print(f"training data: {training_data}")
    print(f"output directory: {output_directory}")
    model = SEModel(no_glfw=True)
    model.load_models()
    model.title = name
    if architecture is None and model_path is '':
        print(f"using default model architecture: {SEModel.AVAILABLE_MODELS[SEModel.DEFAULT_MODEL_ENUM]}")
    elif architecture is None:
        print(f"loading model {model_path}")
        model.load(model_path)
    elif model_path is '':
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

    model.save(os.path.join(output_directory, f"{model.apix:.2f}_{model.box_size}_{model.loss:.4f}_{model.title}{cfg.filetype_semodel}"))


def _pick_tomo(tomo_path, output_path, margin, threshold, spacing, size, spacing_px, size_px):
    from Ais.core.util import get_maxima_3d_watershed

    # find right values for spacing and size.
    voxel_size = mrcfile.open(tomo_path, header_only=True).voxel_size.x

    if spacing_px is None:
        min_spacing = spacing / 10.0
    else:
        min_spacing = spacing_px * voxel_size / 10.0

    if size_px is None:
        min_size = size / 1000.0
    else:
        min_size = size_px * (voxel_size / 10.0)**3

    n_particles = get_maxima_3d_watershed(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, min_spacing=min_spacing, min_size=min_size, pixel_size=voxel_size / 10.0, sort_by_weight=True, verbose=False)
    return n_particles


def _picking_thread(data_paths, output_directory, margin, threshold, spacing, size, spacing_px, size_px, process_id):
    for j, p in enumerate(data_paths):
        out_path = os.path.join(output_directory, os.path.splitext(os.path.basename(p))[0]+"_coords.tsv")
        n_particles = _pick_tomo(p, out_path, margin, threshold, spacing, size, spacing_px, size_px)
        print(f"{j+1}/{len(data_paths)} (process {process_id}) - {n_particles} particles in {p}")


def dispatch_parallel_pick(target, data_directory, output_directory, margin, threshold, spacing, size, parallel=1, spacing_px=None, size_px=None):
    data_directory = os.path.abspath(data_directory)
    output_directory = os.path.abspath(output_directory)

    os.makedirs(output_directory, exist_ok=True)
    all_data_paths = glob.glob(os.path.join(data_directory, f"*__{target}.mrc"))
    data_div = {p_id: list() for p_id in range(parallel)}
    for p_id, data_path in zip(itertools.cycle(range(parallel)), all_data_paths):
        data_div[p_id].append(data_path)

    processes = []
    for p_id in data_div:
        p = multiprocessing.Process(target=_picking_thread,
                                    args=(data_div[p_id], output_directory, margin, threshold, spacing, size, spacing_px, size_px, p_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
