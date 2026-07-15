![Ais](docs/res/ais_banner.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/bionanopatterning/Ais/blob/master/Ais/LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/Ais-cryoET)](https://pypi.org/project/Ais-cryoET/)
[![Documentation Status](https://readthedocs.org/projects/ais-cryoet/badge/?version=latest)](https://ais-cryoet.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/bionanopatterning/Ais)

# Segmentation with Ais #
## Fast and user-friendly annotation and segmentation of cryo-electron tomography data using convolutional neural networks ##

This repository comprises a standalone version of Ais, the segmentation editor for cryoET. For the version integrated into the correlative microscopy data processing suite _scNodes_, see the [scNodes](https://www.github.com/bionanopatterning/scNodes) repository.

A timelapse video of the full workflow, from reconstructed tomograms to segmented volumes showing membranes, ribosomes, mitochondrial granules, and microtubuli, is available on our [YouTube channel](https://www.youtube.com/watch?v=2JIBVJf3kYQ&ab_channel=scNodes).

Contact: mlast@mrc-lmb.cam.ac.uk

### Installation ###
Ais works on Windows and Linux machines but not on MacOS. Install as follows: 
```
conda create --name ais
conda activate ais
conda install python==3.9
conda install pip
pip install git+https://github.com/bionanopatterning/Ais
```

Then run using either of the following commands:
```
ais
ais-cryoet
```
#### Tensorflow & CUDA compatibility ####
Compatibility between Python, tensorflow, and CUDA versions can be an issue. The following combination was used during development and is known to work:

Python 3.9<br/>
Tensorflow 2.8.0<br/>
CUDA 11.8<br/>
cuDNN 8.6<br/>
protobuf 3.20.0<br/>

The software will work without CUDA, but only on the CPU. This is much slower but still reasonably interactive if the tomograms aren't too big (in XY). We do recommend installing CUDA and cuDNN in order for tensorflow to be able to use the GPU. See: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html for help installing CUDA and cuDNN. 

## Features ##
### Annotate any number of features ###

<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/annotation.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/annotation.mp4">Watch: annotation.mp4</a></video>

### Train a network & watch it learn ###

<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/training_run.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/training_run.mp4">Watch: training_run.mp4</a></video>

### Model-assisted annotation ###
When networks still need a bit of improvement before you're ready to segment your data, *model assisted annotation* helps you quickly polish the training data. Screen the output of a model, copy it to the training annotations, and edit any mistakes that need fixing. 

<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/model_assisted_annotation.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/model_assisted_annotation.mp4">Watch: model_assisted_annotation.mp4</a></video>

### Segment with multiple models ###
<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/models.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/models.mp4">Watch: models.mp4</a></video>

### Inspect results ###

<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/rendering.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/rendering.mp4">Watch: rendering.mp4</a></video>

### Connect Ais to a Pom database ###
Ais integrates with [Pom](https://github.com/bionanopatterning/Pom), a tool to present large cryoET datasets as searchable databases. Use Pom to organise the data, and Ais to mine it.
<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/pom_database.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/pom_database.mp4">Watch: pom_database.mp4</a></video>

### Organise your features ###

<video src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/feature_library.mp4" controls muted width="100%"><a href="https://github.com/bionanopatterning/Ais/raw/master/docs/res/feature_library.mp4">Watch: feature_library.mp4</a></video>

## See our other tools ##

<p align="center">
  <a href="https://github.com/bionanopatterning/Pom"><img src="https://github.com/bionanopatterning/Pom/raw/main/docs/res/pom_banner.png" width="49%"></a>
  <a href="https://github.com/mgflast/easymode"><img src="https://github.com/mgflast/easymode/raw/master/assets/easymode_banner.png" width="49%"></a>
</p>