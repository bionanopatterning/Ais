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

#
# class PoolDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, volume_pairs, box_size=64, batch_size=32, steps=1024, threshold=128, augment=True):
#         self.pairs = volume_pairs
#         self.box = box_size
#         self.bs = batch_size
#         self.steps = steps
#         self.threshold = threshold
#         self.augment = augment
#
#     def __len__(self):
#         return self.steps
#
#     def __getitem__(self, index):
#         batch_x = np.empty((self.bs, self.box, self.box, 1), dtype=np.float32)
#         batch_y = np.empty_like(batch_x)
#
#         for i in range(self.bs):
#             mmap_t, mmap_s = random.choice(self.pairs)
#             z_max, y_max, x_max = mmap_t.shape
#             half = self.box // 2
#             z = np.random.randint(0, z_max)
#             y = np.random.randint(half, y_max - half)
#             x = np.random.randint(half, x_max - half)
#
#             tomo_crop = mmap_t[z, y - half:y + half, x - half:x + half]
#             seg_crop = (mmap_s[z, y - half:y + half, x - half:x + half] > self.threshold).astype(np.float32)
#
#             if self.augment:
#                 k = np.random.randint(0, 4)
#                 tomo_crop = np.rot90(tomo_crop, k=k, axes=(0, 1))
#                 seg_crop = np.rot90(seg_crop, k=k, axes=(0, 1))
#
#                 if np.random.rand() > 0.5:
#                     tomo_crop = np.flip(tomo_crop, axis=1)
#                     seg_crop = np.flip(seg_crop, axis=1)
#
#             mu = tomo_crop.mean()
#             std = tomo_crop.std() + 1e-7
#             tomo_crop = (tomo_crop - mu) / std
#
#             batch_x[i, ..., 0] = tomo_crop
#             batch_y[i, ..., 0] = seg_crop
#
#         return batch_x, batch_y
#
#
# class StopAtLoss(tf.keras.callbacks.Callback):
#     def __init__(self, stop_loss, monitor='val_loss'):
#         super().__init__()
#         self.stop_loss = stop_loss
#         self.monitor = monitor
#
#     def on_epoch_end(self, epoch, logs=None):
#         current_loss = logs.get(self.monitor)
#         if current_loss is not None and current_loss <= self.stop_loss:
#             print(f"Stopping training at epoch {epoch} with loss {current_loss:.4f} (threshold: {self.stop_loss})")
#             self.model.stop_training = True
#
#
# def pool_train_model(tomo_dir, segmentation_dir, feature, architecture, boxsize, threshold, output_directory, epochs, epoch_size, stop_loss, validation_split, batch_size, model_path, gpus, parallel, rate, name):
#     import keras.callbacks
#
#     class CheckpointCallback(keras.callbacks.Callback):
#         def __init__(self, se_model, path):
#             super().__init__()
#             self.se_model = se_model
#             self.path = path
#             self.best_score = 1e9
#
#         def on_epoch_end(self, epoch, logs=None):
#             score = logs.get("val_loss", logs.get("loss"))
#             if score is not None and score < self.best_score:
#                 self.best_score = score
#                 self.se_model.save(self.path)
#
#     # List tomograms and segmentations, split them into training and validation
#     segmentations = glob.glob(os.path.join(segmentation_dir, f"*__{feature}.mrc"))
#     volume_pairs = list()
#     for s in segmentations:
#         tomo_name = os.path.basename(s).split('__')[0]
#         tomo_path = os.path.join(tomo_dir, tomo_name + ".mrc")
#
#         mmap_s = mrcfile.mmap(s).data
#         mmap_t = mrcfile.mmap(tomo_path).data
#         volume_pairs.append((mmap_t, mmap_s))
#
#     np.random.shuffle(volume_pairs)
#     n_validation = int(validation_split * len(volume_pairs))
#     print(f'Found {len(volume_pairs)} pairs of tomograms and segmentations. Using {len(volume_pairs) - n_validation} for training, {n_validation} for validation.')
#     validation_pairs = volume_pairs[:n_validation]
#     training_pairs = volume_pairs[n_validation:]
#
#     # Training generator
#     train_generator = PoolDataGenerator(training_pairs, box_size=boxsize, batch_size=batch_size, steps=epoch_size, threshold=threshold)
#     if validation_split > 0.0:
#         val_generator = PoolDataGenerator(validation_pairs, box_size=boxsize, batch_size=batch_size, steps=epoch_size // 5, threshold=threshold, augment=False)  # 20% of the training steps should be fine if samples are large enough to be unbiased; lowering val loss is always as solution.
#         monitor = 'val_loss'
#     else:
#         val_generator = None
#         monitor = 'loss'
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpus
#     tf_strategy = tf.distribute.get_strategy() if not parallel else tf.distribute.MirroredStrategy()
#     with tf_strategy.scope():
#         if model_path == '':
#             se_model = SEModel(no_glfw=True)
#             se_model.load_models()
#             se_model.model_enum = architecture
#             se_model.compile(boxsize)
#         else:
#             se_model = SEModel(no_glfw=True)
#             se_model.load(model_path)
#             se_model.toggle_training()
#
#         se_model.title = name
#         se_model.model.optimizer.learning_rate.assign(rate)
#
#         callbacks = [
#             StopAtLoss(stop_loss, monitor),
#             tf.keras.callbacks.ModelCheckpoint(os.path.join(output_directory, f"{se_model.title}{cfg.filetype_semodel}"), monitor=monitor, save_best_only=True, mode='min'),
#         ]
#
#         se_model.model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks, workers=16, use_multiprocessing=True, max_queue_size=32)



def _pick_tomo(tomo_path, output_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose):
    from Ais.core.util import pick_particles

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

    n_particles = pick_particles(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, min_spacing=min_spacing, min_size=min_size, pixel_size=voxel_size / 10.0, verbose=verbose, output_star=True)
    return n_particles

def _clr_print(txt, clr):
    colors = {
        "none": "\033[37m",
        "few": "\033[33m",
        "mid": "\033[36m",
        "many": "\033[32m",
    }
    print(f"{colors[clr]}{txt}\033[0m")

def _picking_thread(data_paths, output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, process_id, verbose):
    try:
        for j, p in enumerate(data_paths):
            out_path = os.path.join(output_directory, os.path.splitext(os.path.basename(p))[0]+"_coords.tsv")
            n_particles = _pick_tomo(p, out_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose)

            clr = 'none'
            if 0 < n_particles < 10:
                clr = 'few'
            elif 10 <= n_particles < 100:
                clr = 'mid'
            elif n_particles >= 100:
                clr = 'many'
            _clr_print(f"{j+1}/{len(data_paths)} (process {process_id}) - {n_particles} particles in {p}", clr)
    except KeyboardInterrupt:
        pass


def dispatch_parallel_pick(target, data_directory, output_directory, margin, threshold, binning, spacing, size, parallel=1, spacing_px=None, size_px=None, verbose=False, pom_capp_config=""):
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
    print(f'Found {len(all_data_paths)} files with pattern {os.path.join(data_directory, f"*__{target}.mrc")} {pom_capp_info_str}')
    data_div = {p_id: list() for p_id in range(parallel)}
    for p_id, data_path in zip(itertools.cycle(range(parallel)), all_data_paths):
        data_div[p_id].append(data_path)

    processes = []
    try:
        for p_id in data_div:
            p = multiprocessing.Process(target=_picking_thread,
                                        args=(data_div[p_id], output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, p_id, verbose))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
