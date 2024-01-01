import Ais.core.config as cfg
from Ais.core.segmentation_editor import SegmentationEditor

cfg.segmentation_editor.import_dataset("C:/Users/mart_/Desktop/test/g11001_volb4_rotx.mrc")
SegmentationEditor.PICKING_FRAME_ALPHA = 0.0
