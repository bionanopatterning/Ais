from Ais.core.se_frame import SEFrame
from Ais.core.se_model import SEModel
import tensorflow as tf
from Ais.core.segmentation_editor import QueuedExport
from Ais.main import windowless
import os
import time
import multiprocessing
import glob
import itertools
import argparse


def _segmentation_thread(model_path, data_paths, output_dir, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    windowless()

    model = SEModel()
    model.load(model_path)

    queued_exports = list()
    for p in data_paths:
        queued_exports.append(QueuedExport(output_dir, SEFrame(p), [model], 1, False))

    for qe in queued_exports:
        qe.start()
        while qe.process.progress < 1.0:
            time.sleep(0.5)


def dispatch_parallel_segment(model_path, data_directory, output_directory, gpus, skip=1):
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    if not os.path.isabs(data_directory):
        data_directory = os.path.join(os.getcwd(), data_directory)

    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    # distribute data:
    all_data_paths = glob.glob(os.path.join(data_directory, "*.mrc"))

    if skip==1:
        # read model title
        model_metadata = SEModel.load_metadata(model_path)
        model_title = model_metadata.title
        all_data_paths = [p for p in all_data_paths if not os.path.exists(os.path.join(data_directory, os.path.splitext(os.path.basename(p))[0]+f"__{model_title}.mrc"))]

    data_div = {gpu: list() for gpu in gpus}
    for gpu, data_path in zip(itertools.cycle(gpus), all_data_paths):
        data_div[gpu].append(data_path)

    # launch the _segmentation_threads (on different CPUs?) using different GPUs
    processes = []

    for gpu_id in data_div:
        p = multiprocessing.Process(target=_segmentation_thread,
                                    args=(model_path, data_div[gpu_id], output_directory, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
