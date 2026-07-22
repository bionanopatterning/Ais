Ais python module
__________

Ais was originally created as a part of our cryoCLEM processing suite `scNodes <https://github.com/bionanopatterning/scNodes>`_. As such, the naming system of Ais as a standalone python package can be a little bit confusing. Here we outline some of the classes and functions of Ais that you may be useful in case you want to strip certain functionality from Ais; for example, to create a script for batch processing that can be called from the command line (see below).

Ais imports two filetypes for segmentation: .mrc files and .scns files. When importing a .mrc and saving the dataset, a .scns file is created that links to that dataset. When the original .mrc file is moved, the link in the .scns becomes incorrect. By right-clicking a dataset in the Datasets panel of the main menu and selecting the 'Relink dataset' this link can be repaired.

Filetypes
^^^^^^^^^^^^^^

The following filetypes are created and used by Ais:

**.scns**, or 'scNodes segmentables', are pickled objects of the class SEFrame, which is defined in scNodes/core/se_frame.py and is the base class that implements .mrc reading, that holds objects for OpenGL rendering, and that has a list (.features) of Segmentation type objects. These are objects that contain the manually drawn annotations and various related parameters.

**.scnt**, or 'scNodes training data', is essentially just a .tif file (they can be opened in ImageJ/FIJI) but containing some additional metadata that is read by Ais when training a neural network.

**.scnm**, or 'scNodes model', are .tar archives that contain one file each of the following types: .h5, the model weights, .json, the metadata (colour, name, pixel size, etc.), and .tif and .png files that are used for validation when uploading a model to aiscryoet.org.

**.scnmgroup**, or 'scNodes model group' files are .tar archives that are created when saving model groups. The .scnmgroup archive contains multiple .scnm files, as well as a .json that describes the interactions between the models in the group.

Models
^^^^^^^^^^^^^^

When booting Ais, a library of neural network architectures is loaded from the Ais/models directory. Adding new models to the software is a matter of properly defining a tensorflow.keras model in a .py file and saving this in the models directory. A template, scNodes/models/model_template.py is available for further instructions. A custom model object can also be used, as long as the .fit, .predict. and various other builtin keras models, are implemented.

All models in Ais are contained in an object of the class SEModel, defined in Ais/core/se_model.py. Models and model interactions are defined in this file.

We are aware that the documentation here is currently lacking - Ais was more built for usage than to be a repurposable Python package. We are happy to help with any questions though, so if you're looking for some specific information please contact us via m.g.f.last@lumc.nl or `the GitHub page <https://www.github.com/bionanopatterning/Ais/issues>`_

Scripting with Ais
^^^^^^^^^^^^

Below is an example of a script to import a tomogram into Ais, load neural network on a previously generated training dataset, and then to segment a tomogram.

::

   from Ais.main import windowless
   from Ais.core.se_frame import SEFrame
   from Ais.core.se_model import SEModel
   import time
   import numpy as np

   tomo = SEFrame("C:/Users/mgflast/Desktop/tomo.mrc")

   tomo.crop_roi = [100, 100, 500, 500]  # specify some region of interest (optional, default is the full tomogram)
   tomo.export_bottom = 50  # first slice to process; i.e., skip slices 0-49
   tomo.export_top = 150  # last slice to process; i.e., skip slices 150-end.

   model = SEModel()

   print(SEModel.AVAILABLE_MODELS)  # print a list of available models; would print, e.g. ['Eman2', 'InceptionNet', 'Pix2pix', 'ResNet', 'UNet deep', 'UNet dropout', 'UNet lite', 'VGGNet', 'VGGNet double']
   model.model_enum = 7  # select UNet lite (VGGNet is the default)

   model.epochs = 25
   model.batch_size = 32
   model.excess_negative = 100  # +100% negative samples s.t. 2 negative images per 1 positive.
   # model.box_size will be determined by the box size of the training data, as will model.apix
   model.train_data_path = "C:/Users/mgflast/Desktop/64_1.000_Ribosomes.scnt"
   model.title = "Ribosomes"

   model.train()  # this will start a background process. We need to manually wait for it to complete.

   while model.background_process_train.progress != 1.0:
       time.sleep(0.1)

   ## Option 1: directly using methods from SEModel

   n = tomo.n_slices
   pxs = tomo.pixel_size
   volume = np.zeros((tomo.height, tomo.width, n))
   s_volume = np.zeros_like(volume)

   for i in range(n):
       s_volume[:, :, i] = model.apply_to_slice(volume[:, :, i], pxs)

   ## Option 2: using QueuedExports - easier when you want to schedule many exports.

   from Ais.core.segmentation_editor import QueuedExport

   out_dir = "C:/Users/mgflast/Desktop/segmentations"

   job = QueuedExport(out_dir, tomo, [model], 1, False)
   job.start()  # create one QueuedExport object per tomogram you want to segment, then start them sequentially; running multiple QueuedExport jobs at the same time is inefficient.

   while job.process.progress != 1.0:
       print(f"Processing tomogram - progress: {job.process.progress * 100.0}%")
       time.sleep(0.1)


Implementing custom model architectures
^^^^^^^^^^^^

Adding a Keras model
#########

Most models in the standard Ais library are Keras models (tensorflow.keras). Adding an extra keras model with a new architecture is relatively straightforward and can be achieved by adding a .py file to Ais/models directory. The .py file requires three components: a title for the model, a boolean that specifies whether the model should be available in the software, and a function 'create' that returns a keras model. The implementation of the VGGNet model (vggnet.py) is copied below as an example.



::

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
    from tensorflow.keras.optimizers import Adam

    title = "VGGNet"
    include = True

    def create(input_shape):
        inputs = Input(input_shape)

        # Block 1
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Block 2
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Block 3
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

        # Upsampling and Decoding
        up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(pool3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

        up3 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(conv8)
        output = Conv2D(1, (1, 1), activation='sigmoid')(up3)

        # create the model
        model = Model(inputs=[inputs], outputs=[output])
        model.compile(optimizer=Adam(), loss='binary_crossentropy')

        return model

Adding a non-Keras model
#########

Adding a non-Keras model is also possible but requires a little bit of extra work. Only a small number of methods of the Keras model object type are directly accessed by Ais. These are: count_params, fit, predict, save, and load. Adding a custom model thus requires adding a .py file to the Ais/models that contains four components: a title, a boolean that specifies whether the model is available in the software, and a function 'create' that returns model object (these are as before, with adding a keras model), and additionally a definition of a class that implements the required methods. The return types of these methods should be the same as those returned by the corresponding Keras methods. The contents of the model_template.py template file are copied below as an example.

::

    title = "Template_model"
    include = False


    def create(input_shape):
        return TemplateModel(input_shape)


    class TemplateModel:
        def __init__(self, input_shape):
            self.img_shape = input_shape
            self.generator, self.discriminator = self.compile_custom_model()

        def compile_custom_model(self):
            # e.g.: compile generator, compile discriminator, return.
            return 0, 0

        def count_params(self):
            # e.g. return self.generator.count_params()
            # for the default models, the number of parameters that is returned is the amount that are involved in processing, not in training. So for e.g. a GAN, the discriminator params are not included.
            return 0

        def fit(self, train_x, train_y, epochs, batch_size=1, shuffle=True, callbacks=[]):
            for c in callbacks:
                c.params['epochs'] = epochs

            # fit model, e.g.:
            for e in range(epochs):
                for i in range(len(train_x) // batch_size):
                    # fit batch
                    pass

                    logs = {'loss': 0.0}
                    for c in callbacks:
                        c.on_batch_end(i, logs)

        def predict(self, images):
            # e.g.: return self.generator.predict(images)
            return None

        def save(self, path):
            pass

        def load(self, path):
            pass

A more concrete example of the implementation of a custom model can be found in `Ais/models/pix2pix.py <https://github.com/bionanopatterning/Ais/blob/master/Ais/models/pix2pix.py>`_. The pix2pix model is implemented in Keras, but since it internally requires the use of two separate Keras model objects (the generator and the discriminator), implementing it in Ais was a matter of wrapping the pix2pix models in a custom class.




