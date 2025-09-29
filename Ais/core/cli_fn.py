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
import json
import random


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


def _segment_tomo(tomo_path, model, tta=1, binning=1):
    volume = np.array(mrcfile.read(tomo_path).data).astype(np.float32)
    if binning != 1:
        volume = _bin_volume(volume, binning)
    volume -= np.mean(volume)
    volume /= np.std(volume)
    segmented_volume = np.zeros_like(volume)


    w = 32 * (volume.shape[1] // 32)
    w_pad = (volume.shape[1] % 32) // 2
    h = 32 * (volume.shape[2] // 32)
    h_pad = (volume.shape[2] % 32) // 2

    if tta == 1:
        for j in range(volume.shape[0]):
            segmented_volume[j, w_pad:w_pad+w, h_pad:h_pad+h] = np.squeeze(model.predict(volume[j, w_pad:w_pad+w, h_pad:h_pad+h][np.newaxis, :, :]))
    else:
        r = [0, 1, 2, 3, 0, 1, 2, 3]
        f = [0, 0, 0, 0, 1, 1, 1, 1]
        for k in range(tta):
            volume_instance = np.rot90(volume, k=r[k], axes=(1, 2))
            segmented_instance = np.zeros_like(volume_instance)
            w = 32 * (volume_instance.shape[1] // 32)
            w_pad = (volume_instance.shape[1] % 32) // 2
            h = 32 * (volume_instance.shape[2] // 32)
            h_pad = (volume_instance.shape[2] % 32) // 2

            if f[k]:
                volume_instance = np.flip(volume_instance, axis=2)

            for j in range(volume_instance.shape[0]):
                segmented_instance[j, w_pad:w_pad+w, h_pad:h_pad+h] = np.squeeze(model.predict(volume_instance[j, w_pad:w_pad+w, h_pad:h_pad+h][np.newaxis, :, :]))

            if f[k]:
                segmented_instance = np.flip(segmented_instance, axis=2)
            segmented_instance = np.rot90(segmented_instance, k=-r[k], axes=(1,2))

            segmented_volume += segmented_instance
        segmented_volume /= tta

    segmented_volume = np.clip(segmented_volume, 0.0, 1.0)
    segmented_volume[:32, :, :] = 0
    segmented_volume[-32:, :, :] = 0
    segmented_volume[:, :32, :] = 0
    segmented_volume[:, -32:, :] = 0
    segmented_volume[:, :, :32] = 0
    segmented_volume[:, :, -32:] = 0
    return segmented_volume


def _bin_volume(vol, b=1):
    if b == 1:
        return vol
    else:
        j, k, l = vol.shape
        vol = vol[:j // b * b, :k // b * b, :l // b * b]
        vol = vol.reshape((j // b, b, k // b, b, l // b, b)).mean(5).mean(3).mean(1)
        return vol


def _segmentation_thread(model_path, data_paths, output_dir, gpu_id, test_time_augmentation=1, overwrite=False, binning=1):
    from keras.models import clone_model
    from keras.layers import Input

    if isinstance(gpu_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


    se_model = SEModel(no_glfw=True)
    se_model.load(model_path, compile=False)
    model = se_model.model
    new_input = Input(shape=(None, None, 1))
    new_model = clone_model(model, input_tensors=new_input)
    new_model.set_weights(model.get_weights())


    for j, p in enumerate(data_paths):
        try:
            tomo_name = os.path.basename(os.path.splitext(p)[0])
            out_path = os.path.join(output_dir, tomo_name+"__"+se_model.title+".mrc")
            if os.path.exists(out_path) and not overwrite:
                print(f"{j + 1}/{len(data_paths)} (GPU {gpu_id}) - {os.path.basename(model_path)} - {p}")
                continue
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(np.zeros((10, 10, 10), dtype=np.float32))
                mrc.voxel_size = 1.0

            in_voxel_size = mrcfile.open(p, header_only=True).voxel_size
            segmented_volume = _segment_tomo(p, new_model, tta=test_time_augmentation, binning=binning)
            segmented_volume = (segmented_volume * 255).astype(np.uint8)
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(segmented_volume)
                mrc.voxel_size = in_voxel_size
            print(f"{j + 1}/{len(data_paths)} (GPU {gpu_id}) - {os.path.basename(model_path)} - {p}")
        except Exception as e:
            print(f"Error segmenting {p}:\n{e}")


def dispatch_parallel_segment(model_path, data_directory, output_directory, gpus, test_time_augmentation=1, parallel=1, overwrite=0, binning=1):
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    if not os.path.isabs(data_directory):
        data_directory = os.path.join(os.getcwd(), data_directory)

    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    all_data_paths = glob.glob(os.path.join(data_directory, "*.mrc"))

    data_div = {gpu: list() for gpu in gpus}
    for gpu, data_path in zip(itertools.cycle(gpus), all_data_paths):
        data_div[gpu].append(data_path)

    if parallel == 1:
        processes = []
        for gpu_id in data_div:
            p = multiprocessing.Process(target=_segmentation_thread,
                                        args=(model_path, data_div[gpu_id], output_directory, gpu_id, test_time_augmentation, overwrite, binning))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        _segmentation_thread(model_path, all_data_paths, output_directory, gpu_id=",".join(str(n) for n in gpus), test_time_augmentation=test_time_augmentation, overwrite=bool(overwrite), binning=binning)


def print_available_model_architectures():
    model = SEModel(no_glfw=True)
    model.load_models()

    first_col = [f"index: {j} (-a {j})" for j in range(len(SEModel.AVAILABLE_MODELS))]
    col_width = max(len(text) for text in first_col) + 2

    for j, key in enumerate(SEModel.AVAILABLE_MODELS):
        print(f"{first_col[j]:<{col_width}}architecture name: {key}")


def train_model(training_data, output_directory, architecture=None, epochs=50, batch_size=32, negatives=0.0, copies=4, model_path='', gpus="0", parallel=1, rate=1e-3, name="Unnamed model"):
    import keras.callbacks
    ## TODO: -m input currently ignorned it seems.
    class CheckpointCallback(keras.callbacks.Callback):
        def __init__(self, se_model, path):
            super().__init__()
            self.se_model = se_model
            self.path = path
            self.best_loss = 1e9

        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get('loss', None)
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
                self.se_model.save(self.path)

    if not os.path.isabs(training_data):
        training_data = os.path.join(os.getcwd(), training_data)
    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    print(f"training data: {training_data}")
    print(f"output directory: {output_directory}")
    model = SEModel(no_glfw=True)
    model.load_models()
    model.title = name
    if architecture is None and model_path == '':
        print(f"using default model architecture: {SEModel.AVAILABLE_MODELS[SEModel.DEFAULT_MODEL_ENUM]}")
    elif model_path:
        print(f"loading model {model_path}")
        model.load(model_path)
    elif model_path == '':
        model.model_enum = architecture
        print(f"using model architecture {architecture}: {SEModel.AVAILABLE_MODELS[architecture]}")

    model.train_data_path = training_data
    model.epochs = epochs
    model.batch_size = batch_size
    model.excess_negative = int((100 * negatives) - 100)
    model.n_copies = copies

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    checkpoint_callback = CheckpointCallback(model, os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}"))
    if parallel:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model.train(rate=rate, external_callbacks=[checkpoint_callback])
    else:
        model.train(rate=rate, external_callbacks=[checkpoint_callback])

    while model.background_process_train.progress < 1.0:
        time.sleep(0.2)

    file_path = os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}")
    model.load(os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}"))
    model.toggle_inference()
    model.save(file_path)
    print(f"\nDone training {os.path.join(output_directory, f'{model.title}{cfg.filetype_semodel}')}")


def _pick_tomo(tomo_path, output_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose, filament=False):
    # find right values for spacing and size.
    voxel_size = mrcfile.open(tomo_path, header_only=True).voxel_size.x
    if voxel_size == 0.0:
        print(f"warning: {tomo_path} has voxel size 0.0")
        voxel_size = 10.0
    if spacing_px is None:
        min_spacing = spacing / 10.0
    else:
        min_spacing = spacing_px * voxel_size / 10.0

    if size_px is None:
        min_size = size / 1000.0
    else:
        min_size = size_px * (voxel_size / 10.0)**3

    if filament:
        from Ais.core.filaments import pick_filament
        return pick_filament(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, spacing_nm=min_spacing, size_nm=min_size, pixel_size=voxel_size / 10.0)
    else:
        from Ais.core.util import pick_particles
        return pick_particles(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, min_spacing=min_spacing, min_size=min_size, pixel_size=voxel_size / 10.0, verbose=verbose, output_star=True)


def _clr_print(txt, clr):
    colors = {
        "none": "\033[37m",
        "few": "\033[33m",
        "mid": "\033[36m",
        "many": "\033[32m",
    }
    print(f"{colors[clr]}{txt}\033[0m")

def _picking_thread(data_paths, output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, process_id, verbose, filament=False):
    try:
        for j, p in enumerate(data_paths):
            out_path = os.path.join(output_directory, os.path.splitext(os.path.basename(p))[0]+"_coords.star")
            n_particles, n_filaments = _pick_tomo(p, out_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose, filament)

            clr = 'none'
            if 0 < n_particles < 10:
                clr = 'few'
            elif 10 <= n_particles < 100:
                clr = 'mid'
            elif n_particles >= 100:
                clr = 'many'
            _clr_print(f"{j+1}/{len(data_paths)} (process {process_id}) - {n_particles} {'particles' if not filament else f'coordinates in {n_filaments} filaments'} in {p}", clr)
    except KeyboardInterrupt:
        pass


def dispatch_parallel_pick(target, data_directory, output_directory, margin, threshold, binning, spacing, size, parallel=1, spacing_px=None, size_px=None, verbose=False, pom_capp_config="", filament=False):
    data_directory = os.path.abspath(data_directory)
    output_directory = os.path.abspath(output_directory)

    os.makedirs(output_directory, exist_ok=True)
    all_data_paths = glob.glob(os.path.join(data_directory, f"*__{target}.mrc"))
    pom_capp_info_str = ""
    if pom_capp_config:
        with open(pom_capp_config, 'r') as f:
            config = json.load(f)
            subsets = config.get('subsets', ['all'])
            subsets = ['all'] if not subsets else subsets
            if not 'all' in subsets:
                all_data_paths = []
                for s in subsets:
                    subset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pom_capp_config))), 'subsets', f'{s}.json')
                    with open(subset_path, 'r') as sf:
                        subset = json.load(sf)['tomos']
                        print(subset)
                        for t in subset:
                            all_data_paths.append(os.path.join(data_directory, f"{t}__{target}.mrc"))


        pom_capp_info_str = f"in selected Pom subsets: {', '.join(subsets)}"
    print(f'Found {len(all_data_paths)} files with pattern {os.path.join(data_directory, f"*__{target}.mrc")} {pom_capp_info_str}. Picking in {"blob" if not filament else "filament"} mode.')
    data_div = {p_id: list() for p_id in range(parallel)}
    for p_id, data_path in zip(itertools.cycle(range(parallel)), all_data_paths):
        data_div[p_id].append(data_path)

    processes = []
    try:
        for p_id in data_div:
            p = multiprocessing.Process(target=_picking_thread,
                                        args=(data_div[p_id], output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, p_id, verbose, filament))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
