Using Ais via the Command Line Interface
========================================

Since version 1.0.35, Ais offers expanded functionality via the command line interface (CLI). Below are some processing tasks that can be done without the graphical user interface.

**Training models (`ais train`)**

::

   ais train -t <training_data_path> -ou <output_directory> -gpu <gpu_ids> [-a <model_architecture_enum] [-m <model_path>] [-p <parallel>] [-e <epochs>] [-b <batch_size>] [-n <negatives_ratio>] [-c <n_copies>] [-models <print_available_model_architecture_enums>]

Options:
  ``-t``:
    Path to the .scnt training data (required).

  ``-ou``:
    Directory to save the model in (required).

  ``-gpu``:
    GPUs to use, e.g. "0,1,2,3" (required).

  ``-a``:
    Integer enumerator, index of the model architecture to use (default VGGNet M). Use argument '-models' to print a list of available architectures and corresponding -a values.

  ``-m``:
    Path to a previously used Ais model file (.scnm), to continue training for. Overrides -a.

  ``-p``:
    Whether to use TensorFlow's distribute.MirroredStrategy() during training (1 for yes, 0 for no).

  ``-e``:
    Number of epochs to train the model for (default 50).

  ``-b``:
    Batch size (default 32).

  ``-n``:
    If 0.0 (default), all images in the input training data are weighted identically. If argument supplied, the value determines the ratio of negative to positive samples to use. For example: if the training data contains 50 positive samples and 50 negatives, and the negative to positive ratio is 1.5, a number of negatives will be sampled twice in order to reach this ratio

  ``-c``:
    Number of copies of the training data to use (default 10). Note that this copies the positive (i.e. with annotations that are not all zeroes) training images, and that the number of negative training images (where no annotations were drawn) is n_copies * (n_positive_images) * negative_ratio.

  ``-r``:
    Learning rate. Default is 1e-3. All default Ais networks use Adam as the optimizer.

  ``-models``:
    Override all other arguments; print a list of available model architectures and corresponding values for arg -a.

Examples:
::

   ais train -models
   ais train -t 64_15.680_Ribosome.scnt -ou training_output -gpu 0,1,2,3 -a 5 -p 1 -n 1.5 -c 10


**Batch Segmentation (`ais segment`)**

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

  ``-tta``:
    Test time augmentation. (optional, default 1). Integer between 1 and 8. If 1, no test time augmentation applied. If 2 - 8, the tomogram is processed multiple times in different orientations and the results averaged. Orientations are [0, 90, 180, 270, 0', 90', 180', 270'] (': horizontal flip); e.g., when -tta 4, each input tomogram is segmented four times, sampled with 0, 90, 180, and 270 deg. rotations relative to the original.

  ``-overwrite``:
    If set to 1, tomograms for which a corresponding segmentation in the output_directory already exists are skipped (default 0).

  ``-p``:
    Integer, 1 (default) or 0. Specify whether to launch multiple parallel processes, each using one GPU, or a single process using all GPUs.

Examples:
::

   ais segment -m models/15.68_64_0.0261_Membrane.scnm -d volumes -ou segmentations -gpu 0,1,2,3,4,5,6,7
   ais segment -m models/Proteasome.scnm -d volumes -ou segmentations -gpu 0,1 -o 1 -p 0

Use ais segment to segment all tomograms in a directory. On systems with multipe GPUs, segmenting multiple volumes simultaneously using parallel processes that eacch use only 1 GPU is often considerably faster than running a single process that uses all GPUs at once. Unless tomograms or te used model architectures are very large (i.e. unbinned), a single GPU per volume is usually sufficient: when tested on a system with 8 NVIDIA GeForce RTX 3080's and a 96-core CPU, parallel processing reduced the average computation time per tomogram 8-fold.


**Batch particle picking (`ais pick`)**

::

   ais pick -d <input_directory> -t <target_particle> -ou <output_directory> -m <margin> -threshold <> -spacing <> -size <> -p <>

Options:
  ``-d``:
    Directory containing the input data (required).

  ``-t``:
    Target to pick. For example, ``-t Ribosome`` will pick from all files ``*__Ribosome.mrc`` in the chosen directory and output ``*__Ribosome_coords.tsv`` (required).

  ``-ou``:
    Output directory. If left empty, data will be saved to the input directory.

  ``-m``:
    Margin (in pixels) to avoid picking close to tomogram edges. Default 16.

  ``-threshold``:
    Threshold to apply to volumes during picking (default 128).

  ``-spacing`` OR ``-spacing-px``:
    Minimum distance between particles in Angstrom. Use ``-spacing-px`` to specify the minimum distance in voxel units instead.

  ``-size`` OR ``-size-px``:
    Minimum particle size in cubic Angstrom. Use ``-size-px`` to specify the minimum size in voxel units instead.

  ``-p``:
    Number of parallel picking processes to use (e.g. ``-p 64``, or however many threads your system can run at a time).

Examples:
::

   ais pick -d segmentations -t Ribosome -ou coordinates -threshold 128 -spacing 250 -size 1000000 -p 64
   ais pick -d segmentations -t TRiC -ou coordinates -threshold 128 -spacing-px 15 -size-px 1500 -p 32

When picking coordinates using volumes generated with Ais the voxel values are between 0 - 255 and 128 is a good default threshold value. You can use the Ais 3D renderer to test which threshold, spacing, and size values work well for your target particle.



