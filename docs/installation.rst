Installation
__________

Installing Ais via pip
^^^^^^^^^^^^
Ais works on Windows and Linux systems, but not on iOS. To ensure Ais's requirements do not clash with other packages, we recommend creating a separate environment to install Ais in. Using anaconda prompt, for example, run:

::

    conda create --name ais
    conda activate ais
    conda install python==3.9
    conda install pip
    pip install ais-cryoet

Run Ais with either of these commands:

::

    ais
    ais-cryoet

Alternatively, clone the project from https://www.github.com/bionanopatterning/Ais into an IDE of your choice.

CUDA & Tensorflow
^^^^^^^^^^^^

To enable processing on the GPU, tensorflow must be set up to use CUDA. This can be a bit of a pain, as only particular combination of versions of tensorflow, CUDA, cuDNN, and protobuf (a Python package) tend to work. When installing Ais via pip, the versioning should be handled, but CUDA must still be manually installed. For instructions, see:

Installing CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Installing cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

When running Ais from within an IDE some paths may need to be manually specified. In PyCharm, add the path to the zlib .dll to the run configuration environment variables as follows to enable tensorflow using the GPU:
LIBRARY_PATH=C:\Program Files\zlib123dllx64\dll_x64


Settings
^^^^^^^^^^^^

To be able to directly port 3D scenes into Blender or ChimeraX, the paths to the Blender and ChimeraX executables must be specified. These can be set via the main menu bar -> Settings -> 3rd party applications.
