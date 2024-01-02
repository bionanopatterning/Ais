from Ais.core.segmentation_editor import QueuedExtract

q = QueuedExtract('W:/mgflast/14. scSegmentation/Fig5/Montse/g70502_volb4_rotx_Ribosomes.mrc', threshold=128, min_size=1.0, min_spacing=1.0, save_dir='C:/Users/mgflast/Desktop/test', binning=4)

q.start()

while q.process.progress < 1.0:
    pass


# import mrcfile
#
# from scipy.ndimage import label, center_of_mass, distance_transform_edt
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed
# import matplotlib.pyplot as plt
# import numpy as np
#
# p = "W:/mgflast/14. scSegmentation/Fig5/Montse/delete.tif"
#
# data = mrcfile.read('W:/mgflast/14. scSegmentation/Fig5/Montse/g70502_volb4_rotx_Ribosomes.mrc')
#
# data_b = data > 128
# min_distance = 10
#
# distance = distance_transform_edt(data_b)
#
# coords = peak_local_max(distance, min_distance=min_distance)
#
# mask = np.zeros(distance.shape, dtype=bool)
# mask[tuple(coords.T)] = True
#
# markers, _ = label(mask)
#
# labels = watershed(-distance, markers, mask=data_b)
# plt.imshow(labels[100, :, :])
# plt.show()
# print(np.sum(labels>0))
# Z, Y, X = np.nonzero(labels)
# print(Z)
# print(Y)
# print(X)