from scNodes.main import windowless
from scNodes.core.se_frame import SEFrame
from scNodes.core.se_model import SEModel
import time
import numpy as np

windowless()

tomo = SEFrame("C:/Users/mgflast/Desktop/tomo.mrc")

tomo.crop_roi = [100, 100, 500, 500]  # specify some region of interest (optional, default is the full tomogram)

model = SEModel()
print(SEModel.AVAILABLE_MODELS)  # print a list of available models; would print, e.g. ["UNet", "VGGNet", "Pix2Pix"]
model.model_enum = 0  # select the UNet

model.epochs = 50
model.batch_size = 32
model.excess_negative = 100  # +100% negative samples s.t. 2 negative images per 1 positive.
# model.box_size will be determined by the box size of the training data, as will model.apix
model.train_data_path = "C:/Users/mgflast/Desktop/64_1.000_Ribosomes.scnt"

model.train()  # this will start a background process. We need to manually wait for it to complete.

while model.background_process_train.progress != 1.0:
    time.sleep(0.1)

n = tomo.n_slices
pxs = tomo.pixel_size
volume = np.zeros((tomo.height, tomo.width, n))
s_volume = np.zeros_like(volume)

for i in range(n):
    s_volume[:, :, i] = model.apply_to_slice(volume[:, :, i], pxs)








