import tifffile
import mrcfile
import os
import glob

paths = glob.glob("C:/Users/mart_/Desktop/*_cropped_b2.tif")

for p in paths:
    print(p)
    name = os.path.splitext(p)[0]
    data = tifffile.imread(p)
    with mrcfile.new(name+".mrc", overwrite=True) as f:
        f.set_data(data)
        f.voxel_size = 10.0
