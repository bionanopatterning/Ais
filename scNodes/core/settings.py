import numpy as np

ne_window_title = "Ais"

autocontrast_saturation = 0.06  # Autocontrast will set contrast lims such that this % of pixels is over/under saturated.
autocontrast_subsample = 4  # Autocontrast works on sub-sampled images to avoid costly computations. When this value is e.g. 2, every 2nd pixel in X/Y is used.