Using Ais via the Command Line Interface
========================================

Since version 1.0.35, Ais offers expanded functionality via the command line interface (CLI). Below are some processing tasks that can be done without the graphical user interface.

**Batch Segmentation (`ais segment`)**

Usage:
::

   ais segment -m <model_path> -d <data_directory> -ou <output_directory> -gpu <gpu_ids> [-s <skip>] [-p <parallel>]

Options:
  ``-m``:
    Path to the segmentation model file (required).

  ``-d``:
    Directory containing the input data (required).

  ``-ou``:
    Directory to save the output results (required).

  ``-gpu``:
    Comma-separated list of GPU IDs (required).

  ``-s``:
    Optional. Integer, 1 (default) or 0. Skip tomograms for which a segmentation, generated with the selected model, is already found in the output directory.

  ``-p``:
    Optional. Integer, 1 (default) or 0. Specify whether to launch multiple parallel processes, each using one GPU, or a single process using all GPUs.

Examples:
::

   ais segment -m models/15.68_64_0.0261_Membrane.scnm -d volumes -ou segmentations -gpu 0,1,2,3,4,5,6,7
   ais segment -m models/15.68_64_0.0261_Membrane.scnm -d volumes -ou segmentations -gpu 0 -s 0,1 -p 0

Use ais segment to segment all tomograms in a directory. On systems with multipe GPUs, segmenting multiple volumes simultaneously using parallel processes that eacch use only 1 GPU is often considerably faster than running a single process that uses all GPUs at once. Unless tomograms or te used model architectures are very large (i.e. unbinned), a single GPU per volume is usually sufficient: when tested on a system with 8 NVIDIA GeForce RTX 3080's and a 96-core CPU, parallel processing reduced the average computation time per tomogram 8-fold.
