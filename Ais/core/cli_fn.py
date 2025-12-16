from Ais.core.se_frame import SEFrame
from Ais.core.se_model import SEModel
from Ais.core.segmentation_editor import QueuedExport
import Ais.core.config as cfg
import tensorflow as tf
import os, time, multiprocessing, glob, itertools, glfw, mrcfile, json
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


def _pad_volume(volume):
    _, k, l = volume.shape
    pad_k = ((32 - (k % 32)) % 32) + 64
    pad_l = ((32 - (l % 32)) % 32) + 64

    padding_dim_k = (pad_k // 2, pad_k - pad_k // 2)
    padding_dim_l = (pad_l // 2, pad_l - pad_l // 2)
    volume = np.pad(volume, ((0, 0), padding_dim_k, padding_dim_l), mode='reflect')
    return volume, (*padding_dim_k, *padding_dim_l)


def _remove_padding(volume, padding):
    pt, pb, pl, pr = padding
    return volume[:, pt:None if pb == 0 else -pb, pl:None if pr == 0 else -pr]


def scale_volume_xy(volume, scale_factor):
    from scipy.ndimage import zoom
    if np.isclose(scale_factor, 1.0):
        return volume
    if scale_factor > 1.0:
        return zoom(volume, (1.0, scale_factor, scale_factor), order=1)
    else:
        b = int(np.floor(1.0 / scale_factor))  # bin factor
        j, k, l = volume.shape
        volume = volume[:, :k//b*b, :l//b*b].reshape(j, k//b, b, l//b, b).mean(4).mean(2)
        residual_scale_factor = b * scale_factor
        volume = zoom(volume, (1.0, residual_scale_factor, residual_scale_factor), order = 1)
        return volume


def _segment_tomo(tomo_path, model, tta=1, model_apix=None, model_depth=1, model_dimensionality=2):
    def parse_input_for_slice(input_volume, j, model_depth, model_dimensionality):
        n_slices, _, _ = input_volume.shape

        if model_dimensionality == 2 or model_depth <= 1:
            sl = input_volume[np.clip(j, 0, n_slices - 1)]
            return sl[..., np.newaxis]

        half = model_depth // 2
        idx = np.clip(np.arange(j - half, j + half + 1), 0, n_slices - 1)
        slab = input_volume[idx, :, :]
        if model_dimensionality == 3:
            return np.transpose(slab, (1, 2, 0))[..., np.newaxis]
        else:
            return np.transpose(slab, (1, 2, 0))

    with mrcfile.open(tomo_path) as m:
        volume = m.data.astype(np.float32)
        volume_apix = float(m.voxel_size.x)
        original_tomogram_shape = volume.shape
        if model_apix is not None and volume_apix == 1.0:
            print(f'warning: {tomo_path} header lists voxel size as 1.0 A/px, which might be incorrect.')

    if model_apix is not None:
        from scipy.ndimage import zoom
        scale = volume_apix / float(model_apix)
        volume = scale_volume_xy(volume, scale)

    for k in range(volume.shape[0]):
        sl = volume[k]
        sl -= sl.mean()
        sl /= sl.std() + 1e-6
        volume[k] = sl

    volume, padding = _pad_volume(volume)
    segmented_volume = np.zeros_like(volume, dtype=np.float32)

    r = [0, 1, 2, 3, 0, 1, 2, 3]
    f = [0, 0, 0, 0, 1, 1, 1, 1]
    for k in range(tta):
        volume_instance = np.rot90(volume, k=r[k], axes=(1, 2))
        if f[k]: volume_instance = np.flip(volume_instance, axis=2)
        segmented_instance = np.zeros_like(volume_instance, dtype=np.float32)

        for j in range(volume_instance.shape[0]):
            model_input = parse_input_for_slice(volume_instance, j, model_depth, model_dimensionality)[np.newaxis, ...]
            segmented_instance[j] = np.squeeze(model.predict(model_input))

        if f[k]:
            segmented_instance = np.flip(segmented_instance, axis=2)
        segmented_instance = np.rot90(segmented_instance, k=-r[k], axes=(1, 2))
        segmented_volume += segmented_instance

    segmented_volume /= float(tta)
    segmented_volume = np.clip(segmented_volume, 0.0, 1.0)
    segmented_volume = _remove_padding(segmented_volume, padding)

    if model_apix is not None:
        from scipy.ndimage import zoom
        zy, zx = original_tomogram_shape[1], original_tomogram_shape[2]
        cy, cx = segmented_volume.shape[1], segmented_volume.shape[2]
        fy, fx = zy / cy, zx / cx
        segmented_volume = zoom(segmented_volume, (1.0, fy, fx), order=1)
        segmented_volume = segmented_volume[:, :zy, :zx]

    return segmented_volume


def _bin_volume_xy(vol, b=1):
    if b == 1:
        return vol
    else:
        j, k, l = vol.shape
        vol = vol[:, :k // b * b, :l // b * b]
        vol = vol.reshape((j, k // b, b, l // b, b)).mean(4).mean(2)
        return vol


def _bin_volume(vol, b=1):
    if b == 1:
        return vol
    else:
        j, k, l = vol.shape
        vol = vol[:j // b * b, :k // b * b, :l // b * b]
        vol = vol.reshape((j // b, b, k // b, b, l // b, b)).mean(5).mean(3).mean(1)
        return vol


def _segmentation_thread(model_path, data_paths, output_dir, gpu_id, test_time_augmentation=1, overwrite=False, model_apix=None):
    from keras.models import clone_model
    from keras.layers import Input

    if isinstance(gpu_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    se_model = SEModel(no_glfw=True)
    se_model.load(model_path, compile=False)
    model_depth = se_model.model_depth
    model_dimensionality = 2.5 if model_depth > 1 else 2    # or it can be 3, see elif rank == 5 below
    model = se_model.model

    input_shape = model.input_shape
    rank = len(input_shape)

    if rank == 4:
        new_input = Input(shape=(None, None, model_depth))
    elif rank == 5:
        model_dimensionality = 3
        new_input = Input(shape=(None, None, model_depth, 1))
    else:
        raise ValueError(f"Unsupported model input rank: {input_shape}")

    new_model = clone_model(model, input_tensors=new_input)
    new_model.set_weights(model.get_weights())

    print(f"GPU {gpu_id} - starting inference with {model_dimensionality}D model '{se_model.title}'")
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
            segmented_volume = _segment_tomo(p, new_model, tta=test_time_augmentation, model_apix=model_apix, model_depth=model_depth, model_dimensionality=model_dimensionality)
            segmented_volume = (segmented_volume * 255).astype(np.uint8)
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(segmented_volume)
                mrc.voxel_size = in_voxel_size
            print(f"{j + 1}/{len(data_paths)} (GPU {gpu_id}) - {os.path.basename(model_path)} - {p}")
        except Exception as e:
            print(f"Error segmenting {p}:\n{e}")


def dispatch_parallel_segment(model_path, data_patterns, output_directory, gpus, test_time_augmentation=1, parallel=1, overwrite=0, processing_apix=1):
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    # Normalise data_directory to a list of patterns/paths
    if isinstance(data_patterns, (list, tuple)):
        patterns = list(data_patterns)
    else:
        patterns = [data_patterns]

    # Make patterns absolute where appropriate
    abs_patterns = []
    for p in patterns:
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        abs_patterns.append(p)
    patterns = abs_patterns

    # Make output directory absolute
    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    # Collect all .mrc inputs from all patterns
    all_data_paths = []
    for p in patterns:
        if os.path.isdir(p):
            # Treat as directory: pick up all .mrc files
            matches = glob.glob(os.path.join(p, "*.mrc"))
        else:
            # Treat as glob pattern or explicit file
            matches = glob.glob(p)
        all_data_paths.extend(matches)

    # Deduplicate and sort for determinism
    all_data_paths = [f for f in sorted(set(all_data_paths)) if os.path.splitext(f)[-1] == ".mrc"]

    if len(all_data_paths) == 0:
        print(f"No .mrc files found for data_directory={patterns}. Nothing to do.")
        return

    # Divide work over GPUs
    data_div = {gpu: [] for gpu in gpus}
    for gpu, data_path in zip(itertools.cycle(gpus), all_data_paths):
        data_div[gpu].append(data_path)

    if parallel == 1:
        # One process per GPU
        processes = []
        for gpu_id in data_div:
            p = multiprocessing.Process(
                target=_segmentation_thread,
                args=(
                    model_path,
                    data_div[gpu_id],
                    output_directory,
                    gpu_id,
                    test_time_augmentation,
                    overwrite,
                    processing_apix,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        # Single process using all GPUs (gpu_id is a comma-separated list)
        gpu_id_str = ",".join(str(n) for n in gpus)
        _segmentation_thread(
            model_path,
            all_data_paths,
            output_directory,
            gpu_id_str,
            test_time_augmentation,
            overwrite,
            processing_apix,
        )

def print_available_model_architectures():
    model = SEModel(no_glfw=True)
    model.load_models()

    first_col = [f"index: {j} (-a {j})" for j in range(len(SEModel.AVAILABLE_MODELS))]
    col_width = max(len(text) for text in first_col) + 2

    for j, key in enumerate(SEModel.AVAILABLE_MODELS):
        print(f"{first_col[j]:<{col_width}}architecture name: {key}")


def train_model(training_data, output_directory, architecture=None, epochs=50, batch_size=32, negatives=0.0, copies=4, model_path='', gpus="0", parallel=1, rate=1e-3, name="Unnamed model", extra_augmentations=False):
    import keras.callbacks
    ## TODO: -m input currently ignored it seems.
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
            model.train(rate=rate, external_callbacks=[checkpoint_callback], extra_augmentations=extra_augmentations)
    else:
        model.train(rate=rate, external_callbacks=[checkpoint_callback], extra_augmentations=extra_augmentations)

    while model.background_process_train.progress < 1.0:
        time.sleep(0.2)

    file_path = os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}")
    model.load(os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}"))
    model.toggle_inference()
    model.save(file_path)
    print(f"\nDone training {os.path.join(output_directory, f'{model.title}{cfg.filetype_semodel}')}")


def _pick_tomo(tomo_path, output_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose, filament=False, filament_length=500.0, filament_length_px=None, centroid=False, min_particles=0):
    # find right values for spacing and size.
    voxel_size = mrcfile.open(tomo_path, permissive=True, header_only=True).voxel_size.x
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

    if filament_length_px is None:
        filament_length = filament_length / 10.0
    else:
        filament_length = filament_length_px * (voxel_size / 10.0)

    if filament:
        from Ais.core.filaments import pick_filament
        return pick_filament(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, spacing_nm=min_spacing, size_nm=min_size, pixel_size=voxel_size / 10.0, min_length=filament_length)
    else:
        from Ais.core.util import pick_particles
        return pick_particles(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, min_spacing=min_spacing, min_size=min_size, pixel_size=voxel_size / 10.0, verbose=verbose, centroid=centroid, min_particles=min_particles)


def _clr_print(txt, clr):
    colors = {
        "none": "\033[37m",
        "few": "\033[33m",
        "mid": "\033[36m",
        "many": "\033[32m",
        "red": "\033[31m"
    }
    print(f"{colors[clr]}{txt}\033[0m")


def _picking_thread(data_paths, output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, process_id, verbose, filament=False, filament_length=500.0, filament_length_px=None, centroid=False, min_particles=0):
    try:
        for j, p in enumerate(data_paths):
            out_path = os.path.join(output_directory, os.path.splitext(os.path.basename(p))[0]+"_coords.star")
            n_particles, n_filaments = _pick_tomo(p, out_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose, filament, filament_length, filament_length_px, centroid, min_particles)

            if n_particles < min_particles:
                _clr_print(
                    f"{j + 1}/{len(data_paths)} (process {process_id}) - {n_particles} {'particles' if not filament else f'coordinates in {n_filaments} filaments'} in {os.path.basename(p)}", 'red')
                continue

            clr = 'none'
            if 0 < n_particles < 10:
                clr = 'few'
            elif 10 <= n_particles < 50:
                clr = 'mid'
            elif n_particles >= 50:
                clr = 'many'
            _clr_print(f"{j+1}/{len(data_paths)} (process {process_id}) - {n_particles} {'particles' if not filament else f'coordinates in {n_filaments} filaments'} in {os.path.basename(p)}", clr)
    except KeyboardInterrupt:
        pass


def dispatch_parallel_pick(target, data_directory, output_directory, margin, threshold, binning, spacing, size, parallel=1, spacing_px=None, size_px=None, verbose=False, pom_capp_config="", filament=False, filament_length=500.0, centroid=False, min_particles=0):
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
                                        args=(data_div[p_id], output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, p_id, verbose, filament, filament_length, None, centroid, min_particles))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)


def extract_training_data(features, data_directory, output_directory, box_size, box_depth, binning=1, margin=0):
    import pickle, tifffile
    # TODO: replace scipy.ndimage.zoom with fourier cropping!

    def _get_box(j, k, l, path, tomo_n_slices):
        j_indices = np.clip(np.arange(j - box_depth // 2, j + box_depth // 2 + 1), 0, tomo_n_slices - 1)

        vol = mrcfile.mmap(path).data  # shape: (Z, Y, X)
        zdim, ydim, xdim = vol.shape

        depth = len(j_indices)
        half = box_size // 2
        y0, y1 = l - half, l + half
        x0, x1 = k - half, k + half

        box = np.zeros((depth, box_size, box_size), dtype=np.float32)
        validity = np.zeros((box_size, box_size), dtype=np.float32)

        iy0, iy1 = max(y0, 0), min(y1, ydim)
        ix0, ix1 = max(x0, 0), min(x1, xdim)

        if iy0 < iy1 and ix0 < ix1:
            for zi, z in enumerate(j_indices):
                box[zi, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = vol[z, iy0:iy1, ix0:ix1]
            validity[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = 1

        return box, validity

    def _get_annotation(j, k, l, feature):
        sl = feature.slices[j]  # 2D slice, shape (Y, X)
        h, w = sl.shape

        half = box_size // 2
        y0, y1 = l - half, l + half
        x0, x1 = k - half, k + half  # note k↔x, l↔y like before

        labels = np.zeros((box_size, box_size), dtype=sl.dtype)

        iy0, iy1 = max(y0, 0), min(y1, h)
        ix0, ix1 = max(x0, 0), min(x1, w)

        if iy0 < iy1 and ix0 < ix1:
            labels[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = sl[iy0:iy1, ix0:ix1]

        return labels

    def _bin(x):
        if binning == 1:
            return x
        else:
            j, k, l = x.shape
            x = x[:, :k // binning * binning, :l // binning * binning]
            x = x.reshape((j, k // binning, binning, l // binning, binning)).mean(4).mean(2)
            return x

    annotated_tomograms = glob.glob(os.path.join(data_directory, "*.scns"))

    print(f'scanning {len(annotated_tomograms)} annotated tomograms for {len(features)} features...')

    training_boxes = dict()
    for f in features:
        training_boxes[f] = []

    apix = -1.0
    for j, annotation in enumerate(annotated_tomograms, start=1):
        print('\033[93m' + f'{j}/{len(annotated_tomograms)} - {os.path.basename(annotation)}' + '\033[0m')

        try:
            with open(annotation, 'rb') as pf:
                se_frame = pickle.load(pf)
        except Exception as e:
            print(f"\terror loading {j}\n\t{e}")
            continue

        tomo = se_frame.path
        if not os.path.exists(tomo):
            tomo = os.path.join(os.path.dirname(annotation), os.path.basename(se_frame.path.replace('\\','/')))

        if j == 1:
            apix = mrcfile.open(tomo, header_only=True).voxel_size.x
            print(f'Pixel size for the first tomogram is {apix} Å - writing this value to the training data metadata.')

        for f in se_frame.features:
            if f.title not in features:
                print(f"\tskipping feature '{f.title}'")
                continue

            box_coordinates = [(z, box[0], box[1]) for z in f.boxes for box in f.boxes[z]]
            print('\033[96m' + f"\tparsing feature '{f.title}' ({len(box_coordinates)} boxes)" + '\033[0m')

            for j, k, l in box_coordinates:
                box_pixel_data, validity = _get_box(j, k, l, tomo, se_frame.n_slices)
                box_label = _get_annotation(j, k, l, f)
                box_label[validity == 0] = 2

                box_pixel_data = _bin(box_pixel_data)
                box_label = _bin(box_label[None, :, :])
                box_data = np.concatenate([box_pixel_data, box_label], axis=0)
                training_boxes[f.title].append(box_data)


    os.makedirs(output_directory, exist_ok=True)

    for f in features:
        data = np.array(training_boxes[f], dtype=np.float32)
        _m = margin // binning
        if _m > 0:
            data[:, -1, :_m, :] = 2
            data[:, -1, -_m:, :] = 2
            data[:, -1, :, :_m] = 2
            data[:, -1, :, -_m:] = 2

        _binning_tag = "" if binning == 1 else f"_bin{binning}"
        out_path = f"{box_size}x{box_size}x{box_depth}"+_binning_tag+ f"_{f}.scnt"

        if len(training_boxes[f]) == 0:
            print('\033[96m' + f'{f}: 0 training boxes. Skipping export.' + '\033[0m')
        else:
            print('\033[96m' + f'{f}: {len(training_boxes[f])} training boxes. Saving as {out_path}' + '\033[0m')

        tifffile.imwrite(os.path.join(output_directory, out_path), data, description=f"apix={apix*binning}")




