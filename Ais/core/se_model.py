from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model, clone_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.backend import clear_session
import tifffile
import numpy as np
from itertools import count
import glob
import os
import sys
import Ais.core.config as cfg
import importlib
import threading
import json
from Ais.core.opengl_classes import Texture
from Ais.core.util import generate_thumbnail
from scipy.ndimage import rotate, zoom, binary_dilation
import datetime
import time
import tarfile
import tempfile

# Note 230522: getting tensorflow to use the GPU is a pain. Eventually it worked with:
# Python 3.9, CUDA D11.8, cuDNN 8.6, tensorflow 2.8.0, protobuf 3.20.0, and adding
# LIBRARY_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64 to the PyCharm run configuration environment variables.


class SEModel:
    idgen = count(0)
    AVAILABLE_MODELS = []
    MODELS = dict()
    MODELS_LOADED = False
    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]
    DEFAULT_MODEL_ENUM = 1

    def __init__(self, no_glfw=False):
        if not SEModel.MODELS_LOADED:
            SEModel.load_models()

        uid_counter = next(SEModel.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.title = "Unnamed model"
        self.colour = SEModel.DEFAULT_COLOURS[(uid_counter) % len(SEModel.DEFAULT_COLOURS)]
        self.apix = -1.0
        self.compiled = False
        self.compilation_mode = None
        self.inference_model = None
        self.box_size = -1
        self.model = None
        self.model_enum = SEModel.DEFAULT_MODEL_ENUM
        self.epochs = 25
        self.batch_size = 32
        self.train_data_path = "(path to training dataset)"
        self.active = True
        self.export = True
        self.blend = False
        self.show = True
        self.alpha = 0.75
        self.threshold = 0.5
        self.overlap = 0.2
        self.active_tab = 0
        self.background_process_train = None
        self.background_process_apply = None
        self.n_parameters = 0
        self.n_copies = 4
        self.excess_negative = 30
        self.info = ""
        self.info_short = ""
        self.loss = 0.0
        self.data = None
        if not no_glfw:
            self.texture = Texture(format="r32f")
            self.texture.set_linear_mipmap_interpolation()
        self.bcprms = dict()  # backward compatibility params dict.
        self.emit = False
        self.absorb = False
        self.interactions = list()  # list of ModelInteraction objects.

    def delete(self):
        for interaction in self.interactions:
            ModelInteraction.all.remove(interaction)

    def save(self, file_path, validation_slice=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            model_path = os.path.join(temp_dir, base_name + '_weights.h5')
            metadata_path = os.path.join(temp_dir, base_name + "_metadata.json")
            slice_path = None
            thumbnail_path = None

            self.model.save(model_path)
            if validation_slice is not None:
                slice_path = os.path.join(temp_dir, base_name + "_slice.tiff")
                tifffile.imwrite(slice_path, validation_slice)

                thumbnail_path = os.path.join(temp_dir, base_name + "_preview.png")
                segmentation = self.apply_to_slice(validation_slice, 1.0)
                thumbnail = generate_thumbnail(validation_slice, segmentation, self.colour)
                thumbnail.save(thumbnail_path)

            # Save metadata
            metadata = {
                'title': self.title,
                'colour': self.colour,
                'apix': self.apix,
                'compiled': self.compiled,
                'box_size': self.box_size,
                'model_enum': self.model_enum,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'active': self.active,
                'blend': self.blend,
                'show': self.show,
                'alpha': self.alpha,
                'threshold': self.threshold,
                'overlap': self.overlap,
                'active_tab': self.active_tab,
                'n_parameters': self.n_parameters,
                'n_copies': self.n_copies,
                'info': self.info,
                'info_short': self.info_short,
                'excess_negative': self.excess_negative,
                'emit': self.emit,
                'absorb': self.absorb,
                'loss': self.loss
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            with tarfile.open(file_path, 'w') as archive:
                archive.add(model_path, arcname=os.path.basename(model_path))
                archive.add(metadata_path, arcname=os.path.basename(metadata_path))
                if slice_path:
                    archive.add(slice_path, arcname=os.path.basename(slice_path))
                    archive.add(thumbnail_path, arcname=os.path.basename(thumbnail_path))

    @staticmethod
    def load_metadata(file_path):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(file_path, 'r') as archive:
                    archive.extractall(path=temp_dir)

                metadata_file = glob.glob(os.path.join(temp_dir, "*_metadata.json"))[0]

                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)


                return metadata

        except Exception as e:
            print("Error loading model - see details below\n", e)

    def load(self, file_path, compile=False):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(file_path, 'r') as archive:
                    archive.extractall(path=temp_dir)

                model_file = glob.glob(os.path.join(temp_dir, "*_weights.h5"))[0]
                metadata_file = glob.glob(os.path.join(temp_dir, "*_metadata.json"))[0]

                # Load the Keras model
                self.model = load_model(model_file, compile=compile)
                self.toggle_inference()
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.title = metadata['title']
                self.colour = metadata['colour']
                self.apix = metadata['apix']
                self.compiled = metadata['compiled']
                self.box_size = metadata['box_size']
                self.model_enum = metadata['model_enum']
                self.epochs = metadata['epochs']
                self.batch_size = metadata['batch_size']
                self.active = metadata['active']
                self.blend = metadata['blend']
                self.show = metadata['show']
                self.alpha = metadata['alpha']
                self.threshold = metadata['threshold']
                self.overlap = metadata['overlap']
                self.active_tab = metadata['active_tab']
                self.n_parameters = metadata['n_parameters']
                self.n_copies = metadata['n_copies']
                self.info = metadata['info']
                self.info_short = metadata['info_short']
                self.excess_negative = metadata['excess_negative']
                self.emit = metadata['emit']
                self.absorb = metadata['absorb']
                self.loss = metadata['loss']

        except Exception as e:
            print("Error loading model - see details below\n", e)

    def train(self, rate=None, external_callbacks=None):
        if self.train_data_path:
            if self.compilation_mode == 'inference':
                self.toggle_training()
            process = BackgroundProcess(self._train, (rate, external_callbacks), name=f"{self.title} training")
            self.background_process_train = process
            self.inference_model = None
            process.start()

    def load_training_data(self):
        with tifffile.TiffFile(self.train_data_path) as train_data:
            train_data_apix = float(train_data.pages[0].description.split("=")[1])
            if self.apix == -1.0:
                self.apix = train_data_apix
            elif self.apix != train_data_apix:
                print(f"Note: the selected training data has a different pixel size {train_data_apix:.3f} than what the model was previously trained with ({self.apix:.3f}).")
        train_data = tifffile.imread(self.train_data_path)
        train_x = train_data[:, 0, :, :, None]
        train_y = train_data[:, 1, :, :, None]
        n_samples = train_x.shape[0]

        # split up the positive and negative indices
        positive_indices = list()
        negative_indices = list()
        for i in range(n_samples):
            if np.any(train_y[i]):
                positive_indices.append(i)
            else:
                negative_indices.append(i)

        n_pos = len(positive_indices)
        n_neg = len(negative_indices)
        positive_x = list()
        positive_y = list()

        for i in positive_indices:
            for _ in range(self.n_copies):
                if self.n_copies == 1:
                    norm_train_x = train_x[i] - np.mean(train_x[i])
                    denom = np.std(norm_train_x)
                    if denom != 0.0:
                        norm_train_x /= denom
                    positive_x.append(norm_train_x)
                    positive_y.append(train_y[i])
                else:
                    angle = [0, 90, 180, 270][_] if _ < 4 else np.random.uniform(0, 360)
                    x_rotated = rotate(train_x[i], angle, reshape=False, cval=np.mean(train_x[i]))
                    y_rotated = rotate(train_y[i], angle, reshape=False, cval=0.0)
                    y_rotated = np.clip(y_rotated, 0, 1)
                    x_rotated = (x_rotated - np.mean(x_rotated))
                    denom = np.std(x_rotated)
                    if denom != 0.0:
                        x_rotated /= np.std(x_rotated)

                    positive_x.append(x_rotated)
                    positive_y.append(y_rotated)

        if n_neg == 0:
            return np.array(positive_x), np.array(positive_y)

        if self.excess_negative != -100:
            extra_negative_factor = 1 + self.excess_negative / 100.0
            negative_sample_indices = negative_indices * int(extra_negative_factor * self.n_copies * n_pos // n_neg) + negative_indices[:int(extra_negative_factor * self.n_copies * n_pos) % n_neg]
        else:
            negative_sample_indices = negative_indices * self.n_copies

        negative_x = list()
        negative_y = list()
        n_neg_copied = 0
        for i in negative_sample_indices:
            _ = (n_neg_copied // n_neg)
            angle = [0, 90, 180, 270][_] if _ < 4 else np.random.uniform(0, 360)

            x_rotated = rotate(train_x[i], angle, reshape=False, cval=np.mean(train_x[i]))
            x_rotated = (x_rotated - np.mean(x_rotated))
            denom = np.std(x_rotated)
            if denom != 0.0:
                x_rotated /= denom

            negative_x.append(x_rotated)
            negative_y.append(train_y[i])
            n_neg_copied += 1
        print(f"Loaded a training dataset with {len(positive_x)} positive and {len(negative_x)} negative samples.")
        return np.array(positive_x + negative_x), np.array(positive_y + negative_y)

    def _train(self, rate=None, external_callbacks=None, process=None):
        try:
            start_time = time.time()
            train_x, train_y = self.load_training_data()
            n_samples = train_x.shape[0]
            box_size = train_x.shape[1]
            # compile, if not compiled yet
            if not self.compiled:
                self.compile(box_size)
            # if training data box size is not compatible with the compiled model, abort.
            if box_size != self.box_size:
                self.train_data_path = f"DATA HAS WRONG BOX SIZE ({box_size[0]} x {box_size[1]})"
                process.set_progress(1.0)
                return

            # train
            validation_split = 0.0 if "VALIDATION_SPLIT" not in self.bcprms else self.bcprms["VALIDATION_SPLIT"]
            learning_rate = rate if rate is not None else cfg.settings["LEARNING_RATE"]
            self.model.optimizer.learning_rate.assign(learning_rate)
            callbacks = [TrainingProgressCallback(process, n_samples, self.batch_size, self), StopTrainingCallback(process.stop_request)]
            if not external_callbacks is None:
                callbacks += external_callbacks
            self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, validation_split=validation_split, callbacks=callbacks)
            process.set_progress(1.0)
            print(f"{self.title} " + self.info + f" {time.time() - start_time:.1f} seconds of training.")
        except Exception as e:
            cfg.set_error(e, "Could not train model - see details below.")
            process.stop()

    def reset_textures(self):
        pass

    def compile(self, box_size):
        model_module_name = SEModel.AVAILABLE_MODELS[self.model_enum]
        self.model = SEModel.MODELS[model_module_name]((box_size, box_size, 1))
        self.compiled = True
        self.compilation_mode = 'training'
        self.box_size = box_size
        self.n_parameters = self.model.count_params()
        self.update_info()

    def toggle_inference(self):
        config = self.model.get_config()
        weights = self.model.get_weights()

        del self.model
        clear_session()

        self.model = None
        config["layers"][0]["config"]["batch_input_shape"] = (1, None, None, 1)
        self.model = Model.from_config(config)
        self.model.set_weights(weights)

        self.update_info()
        self.compilation_mode = 'inference'

    def toggle_training(self):
        weights = self.model.get_weights()

        del self.model
        clear_session()

        self.model = None
        self.compile(self.box_size)
        self.model.set_weights(weights)

        self.update_info()
        self.compilation_mode = 'training'

    def update_info(self):
        validation_split_tag = "" if ("VALIDATION_SPLIT" not in self.bcprms or self.bcprms["VALIDATION_SPLIT"] == 0.0) else f"|{int(self.bcprms['VALIDATION_SPLIT']*100.0)}%"
        if self.compilation_mode == 'training':
            self.info = SEModel.AVAILABLE_MODELS[self.model_enum] + f" ({self.n_parameters}, {self.box_size}, {self.apix:.3f}, {self.loss:.4f}{validation_split_tag})"
            self.info_short = "(" + SEModel.AVAILABLE_MODELS[self.model_enum] + f", {self.box_size}, {self.apix:.3f}, {self.loss:.4f}{validation_split_tag})"
        elif self.compilation_mode == 'inference':
            self.info = SEModel.AVAILABLE_MODELS[self.model_enum] + f" ({self.n_parameters}, {self.box_size}, {self.apix:.3f}, {self.loss:.4f}{validation_split_tag})"
            self.info_short = "(" + SEModel.AVAILABLE_MODELS[self.model_enum] + f", {self.box_size}, {self.apix:.3f}, {self.loss:.4f}{validation_split_tag})"

    def get_model_title(self):
        return SEModel.AVAILABLE_MODELS[self.model_enum]

    def set_slice(self, slice_data, slice_pixel_size, roi, original_size):
        try:
            self.data = np.zeros(original_size)
            if not self.compiled:
                return False
            if not self.active:
                return False
            rx, ry = roi
            self.data[rx[0]:rx[1], ry[0]:ry[1]] = self.apply_to_slice(slice_data[rx[0]:rx[1], ry[0]:ry[1]], slice_pixel_size)
            return True
        except Exception as e:
            print(e)
            return False

    def update_texture(self):
        if not cfg.glfw_initialized:
            return
        if not self.compiled or not self.active:
            return
        self.texture.update(self.data)

    def slice_to_boxes(self, image, pixel_size, as_array=True):
        scale_fac = pixel_size * 10.0 / self.apix
        # if abs(scale_fac - 1.0) > 0.1:
        #     image = zoom(image, scale_fac)
        w, h = image.shape
        self.overlap = min([0.9, self.overlap])
        pad_w = self.box_size - (w % self.box_size)
        pad_h = self.box_size - (h % self.box_size)
        # tile
        stride = int(self.box_size * (1.0 - self.overlap))
        boxes = list()
        image = np.pad(image, ((0, pad_w), (0, pad_h)), mode='reflect')
        for x in range(0, w + pad_w - self.box_size + 1, stride):
            for y in range(0, h + pad_h - self.box_size + 1, stride):
                box = image[x:x + self.box_size, y:y + self.box_size]
                mu = np.mean(box, axis=(0, 1), keepdims=True)
                std = np.std(box, axis=(0, 1), keepdims=True)
                std[std == 0] = 1.0
                box = (box - mu) / std
                boxes.append(box)
        if as_array:
            boxes = np.array(boxes)
        return boxes, (w, h), (pad_w, pad_h), stride

    def boxes_to_slice(self, boxes, size, original_pixel_size, padding, stride):
        pad_w, pad_h = padding
        w, h = size
        out_image = np.zeros((w + pad_w, h + pad_h))
        count = np.zeros((w + pad_w, h + pad_h), dtype=int)
        # 240809: apply a mask to all the segmented boxes so that only center (best) bit us used
        box_size = boxes[0].shape[0]
        mask = np.ones((box_size, box_size), dtype=int)
        overlap = 1 - (stride / box_size)
        margin = int(overlap * box_size / 2)
        if margin > 0:
            mask[-margin:, :] = 0.0
            mask[:margin, :] = 0.0
            mask[:, -margin:] = 0.0
            mask[:, :margin] = 0.0

        if cfg.settings["OVERLAP_MODE"] == 0:
            mask[:, :] = 1
        i = 0
        for x in range(0, w + pad_w - self.box_size + 1, stride):
            for y in range(0, h + pad_h - self.box_size + 1, stride):
                out_image[x:x + self.box_size, y:y + self.box_size] += boxes[i] * mask
                count[x:x + self.box_size, y:y + self.box_size] += mask
                i += 1
        c_mask = count == 0
        count[c_mask] = 1
        out_image[c_mask] = 0
        out_image = out_image / count
        out_image = out_image[:w, :h]
        out_image = out_image[:w, :h]
        return out_image

    def apply_to_slice(self, image, pixel_size):
        if self.compilation_mode == 'training' and self.background_process_train is None and cfg.settings["TILED_MODE"]==0:
            self.toggle_inference()

        start_time = time.time()

        if self.compilation_mode == 'inference' and cfg.settings["TILED_MODE"]==0:  # improved way of doing segmentation (250318)
            j, k = image.shape
            image -= np.mean(image)
            image /= np.std(image)
            image = np.pad(image, ((0, 32 - (image.shape[0] % 32)), (0, 32 - (image.shape[1] % 32))))
            segmentation = np.squeeze(self.model.predict(image[np.newaxis, :, :]))[:j, :k]
            print(self.info + f" cost for {segmentation.shape[0]}x{segmentation.shape[1]} slice: {time.time() - start_time:.3f} s.")
        else:  # original Ais gui segmentation way
            boxes, image_size, padding, stride = self.slice_to_boxes(image, pixel_size)
            seg_boxes = np.squeeze(self.model.predict(boxes))
            segmentation = self.boxes_to_slice(seg_boxes, image_size, pixel_size, padding, stride)
            print(self.info + f" cost for {segmentation.shape[0]}x{segmentation.shape[1]} slice ({len(boxes.shape[0])} boxes): {time.time()-start_time:.3f} s.")

        if cfg.settings["TRIM_EDGES"] == 1:
            margin = 16
            segmentation[:margin, :] = 0
            segmentation[-margin:, :] = 0
            segmentation[:, :margin] = 0
            segmentation[:, -margin:] = 0
        return segmentation


    @staticmethod
    def load_models():
        model_files = glob.glob(os.path.join(cfg.root, "models", "*.py"))
        model_files_user = glob.glob(os.path.join(os.path.dirname(cfg.settings_path), "models", "*.py"))
        model_files = sorted(model_files + model_files_user, key=lambda x: os.path.basename(x))
        for file in model_files:
            module_name = os.path.basename(file)[:-3]
            try:
                # if in default library:
                if os.path.dirname(file) == os.path.join(cfg.root, "models"):
                    mod = importlib.import_module(("Ais." if not cfg.frozen else "")+"models."+module_name)
                else:
                    spec = importlib.util.spec_from_file_location(module_name, file)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                if mod.include:
                    SEModel.MODELS[mod.title] = mod.create
            except Exception as e:
                cfg.set_error(e, "Could not load SegmentationEditor model at path: "+file)
        SEModel.MODELS_LOADED = True
        SEModel.AVAILABLE_MODELS = list(SEModel.MODELS.keys())
        if 'VGGNet M' in SEModel.AVAILABLE_MODELS:
            SEModel.DEFAULT_MODEL_ENUM = SEModel.AVAILABLE_MODELS.index('VGGNet M')

    def __eq__(self, other):
        if isinstance(other, SEModel):
            return self.uid == other.uid
        return False


class ModelInteraction:
    idgen = count(0)
    TYPES = ["Colocalize", "Avoid"]

    all = list()

    def __init__(self, parent, partner):
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + next(ModelInteraction.idgen)
        self.parent = parent
        self.partner = partner
        self.type = 0
        self.radius = 10.0  # nanometer
        self.threshold = 0.5
        self.kernel = np.zeros((1, 1))
        self.kernel_info = "none"
        ModelInteraction.all.append(self)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.uid == other.uid
        return False

    def get_kernel(self, pixel_size):
        info_str = f"{self.radius}_{pixel_size}"
        if self.kernel_info == info_str:  # check if the previously generated kernel was the same (i.e. same radius, pixel_size). If so, return it, else, compute the new kernel and return that.
            return self.kernel
        else:
            radius_pixels = int(self.radius // pixel_size)
            self.kernel = np.zeros((radius_pixels * 2 + 1, radius_pixels * 2 + 1), dtype=np.float32)
            r2 = radius_pixels**2
            for x in range(0, 2*radius_pixels+1):
                for y in range(0, 2*radius_pixels+1):
                    self.kernel[x, y] = 1.0 if ((x-radius_pixels)**2 + (y-radius_pixels)**2) < r2 else 0.0
            self.kernel_info = f"{self.radius}_{pixel_size}"
            return self.kernel

    def apply(self, pixel_size):
        print(f"Applying model interaction {self.uid}")
        if self.parent.active:
            self.parent.data = self.apply_to_images(pixel_size, self.partner.data, self.parent.data)

    def apply_to_images(self, pixel_size, partner_image, parent_image):
        int_mask = False
        if partner_image.dtype == np.uint8:
            partner_mask = np.where(partner_image > self.threshold * 255, 1, 0)
            int_mask = True
        else:
            partner_mask = np.where(partner_image > self.threshold, 1, 0)
        kernel = self.get_kernel(pixel_size)
        if self.type == 0:
            mask = binary_dilation(partner_mask, structure=kernel).astype(np.float32 if not int_mask else np.uint8)
            parent_image = parent_image * mask  # this might be sped up by in place multiplication; [] *= []
        elif self.type == 1:
            mask = 1.0 - binary_dilation(partner_mask, structure=kernel).astype(np.float32 if not int_mask else np.uint8)
            parent_image = parent_image * mask
        return parent_image

    def as_dict(self):
        mdict = dict()
        mdict['parent_title'] = self.parent.title
        mdict['partner_title'] = self.partner.title
        mdict['type'] = self.type
        mdict['radius'] = self.radius
        mdict['threshold'] = self.threshold
        return mdict

    @staticmethod
    def from_dict(mdict):
        parent_title = mdict['parent_title']
        partner_title = mdict['partner_title']
        partner_model = None
        parent_model = None
        for m in cfg.se_models:
            if m.title == parent_title:
                parent_model = m
            elif m.title == partner_title:
                partner_model = m
        if partner_model is None or parent_model is None:
            return
        interaction = ModelInteraction(parent_model, partner_model)
        interaction.type = mdict['type']
        interaction.radius = mdict['radius']
        interaction.threshold = mdict['threshold']
        parent_model.interactions.append(interaction)


class TrainingProgressCallback(Callback):
    def __init__(self, process, n_samples, batch_size, model):
        self.params = dict()
        self.params['epochs'] = 1
        super().__init__()
        self.process = process
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.batches_in_epoch = 0
        self.current_epoch = 0
        self.se_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.batches_in_epoch = self.n_samples // self.batch_size
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if "VALIDATION_LOSS" in self.se_model.bcprms and not self.se_model.bcprms["VALIDATION_LOSS"] == 0.0:
            return
        if logs is not None:
            val_loss = logs.get('val_loss')
            if val_loss is not None:
                self.se_model.loss = val_loss
                self.se_model.update_info()

    def on_batch_end(self, batch, logs=None):
        progress_in_current_epoch = (batch + 1) / self.batches_in_epoch
        total_progress = (self.current_epoch + progress_in_current_epoch) / self.params['epochs']
        self.process.set_progress(total_progress)
        if "VALIDATION_LOSS" not in self.se_model.bcprms or self.se_model.bcprms["VALIDATION_LOSS"] == 0.0:
            self.se_model.loss = logs['loss']
        self.se_model.update_info()


class StopTrainingCallback(Callback):
    def __init__(self, stop_request):
        self.params = dict()
        self.params['epochs'] = 1
        super().__init__()
        self.stop_request = stop_request

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        if self.stop_request.is_set():
            self.model.stop_training = True


class BackgroundProcess:
    idgen = count(0)

    def __init__(self, function, args, name=None):
        self.uid = next(BackgroundProcess.idgen)
        self.function = function
        self.args = args
        self.name = name
        self.thread = None
        self.progress = 0.0
        self.stop_request = threading.Event()

    def start(self):
        _name = f"BackgroundProcess {self.uid} - "+(self.name if self.name is not None else "")
        self.thread = threading.Thread(daemon=True, target=self._run, name=_name)
        self.thread.start()

    def _run(self):
        self.function(*self.args, self)

    def set_progress(self, progress):
        self.progress = progress

    def stop(self):
        self.stop_request.set()
        self.progress = 1.0

    def __str__(self):
        return f"BackgroundProcess {self.uid} with function {self.function} and args {self.args}"