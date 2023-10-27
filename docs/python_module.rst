Pom python module
__________

Pom was originally created as a part of our cryoCLEM processing suite `scNodes <https://github.com/bionanopatterning/scNodes>`_. As such, the naming system of Pom as a standalone python package can be a little bit confusing. Here we outline some of the classes and functions of Pom that you may be useful in case you want to strip certain functionality from Pom; for example, to create script for batch processing that can be called from the command line.

Pom imports two filetypes for segmentation (as of october 27th, 2023; we will include .tiff in the future): .mrc files and .scns files. When importing a .mrc and saving the dataset, a .scns file is created that links to that dataset. When the original .mrc file is moved, the link in the .scns becomes incorrect. By right-clicking a dataset in the Datasets panel of the main menu and selecting the 'Relink dataset' this link can be repaired.

Filetypes
^^^^^^^^^^^^^^

The following filetypes are created and used by Pom:

| *.scns*, or 'scNodes segmentables', are pickled objects of the class SEFrame, which is defined in scNodes/core/se_frame.py and is the base class that implements .mrc reading, that holds objects for OpenGL rendering, and that has a list (.features) of Segmentation type objects. These are objects that contain the manually drawn annotations and various related parameters.
| *.scnt*, or 'scNodes training data', is essentially just a .tif file (they can be opened in ImageJ/FIJI) but containing some additional metadata that is read by Pom when training a neural network.
| *.scnm*, or 'scNodes model', are .json files containing the parameters of segmentation models; such as their colour, the A/pix they were trained on, etc. A '[model_name]_weights.h5' file also exists for every .scnm file and these files are expectedto be in the same folder when loading a model into Pom.
| *.scnmgroup*, or 'scNodes model group' files are created when saving model groups. The .scnmgroup file links to multiple .scnm and .h5 files, as well as saved the model interactions between these groups.

Models
^^^^^^^^^^^^^^

When booting Pom, a library of neural network architectures is loaded from the scNodes/models directory. Adding new models to the software is a matter of properly defining a tensorflow.keras file in a .py and saving this in the models directory. A template, scNodes/models/model_template.py is available for further instructions.

All models in Pom are ultimately contained in an object of the class SEModel, defined in scNodes/core/se_model.py. Models and model interactions are defined in this file.

We are aware that the documentation here is currently lacking - Pom was more built for use than as a repurposable Python package. We are happy to help with any questions though, so if you're looking for some specific information please contact us via m.g.f.last@lumc.nl or `the GitHub page <https://www.github.com/bionanopatterning/Pom/issues>`_

Scripting with Pom
^^^^^^^^^^^^

Below is an example of a script to import a tomogram into Pom, load neural network on a previously generated training dataset, and then to segment a tomogram.

Important note: the first function that must always be called to enable Pom functionality in a script is scNodes.main.windowless(). This will set up an invisible OpenGL context which you don't need to think about but that is required for some behind the scenes stuff.

::

    from scNodes.core import se_frame, se_model





