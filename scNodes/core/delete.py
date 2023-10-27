from scNodes.main import windowless
from scNodes.core.se_frame import SEFrame
from scNodes.core.se_model import SEModel
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

from scNodes.core.segmentation_editor import QueuedExport

out_dir = "C:/Users/mgflast/Desktop/segmentations"

job = QueuedExport(out_dir, tomo, [model], 1, False)
job.start()  # create one QueuedExport object per tomogram you want to segment, then start them sequentially; running multiple QueuedExport jobs at the same time is inefficient.

while job.process.progress != 1.0:
    print(f"Processing tomogram - progress: {job.process.progress * 100.0}%")
    time.sleep(0.1)





