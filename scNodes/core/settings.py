import numpy as np

ne_window_title = "scNodes"

autocontrast_saturation = 0.03  # Autocontrast will set contrast lims such that this % of pixels is over/under saturated.
autocontrast_subsample = 2  # Autocontrast works on sub-sampled images to avoid costly computations. When this value is e.g. 2, every 2nd pixel in X/Y is used.