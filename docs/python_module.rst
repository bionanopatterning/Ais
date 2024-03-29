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

Important note: the first function that must always be called to enable Ais functionality in a script is scNodes.main.windowless(). This will set up an invisible OpenGL context which you don't need to think about but that is required for some behind the scenes stuff.

::

   from Ais.main import windowless
   from Ais.core.se_frame import SEFrame
   from Ais.core.se_model import SEModel
   import time
   import numpy as np

   windowless()

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



















