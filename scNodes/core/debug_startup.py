import scNodes.core.config as cfg
from scNodes.core.segmentation_editor import Segmentation, SegmentationEditor

cfg.segmentation_editor.import_dataset("C:/Users/mgflast/Desktop/DELETE.scns")

ds = cfg.se_active_frame
SegmentationEditor.OVERLAY_ALPHA = 0.0
ds.features.append(Segmentation(cfg.se_active_frame, "Debug feature"))