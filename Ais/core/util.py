import os
from PIL import Image
import numpy as np
import tifffile
import glob
from Ais.core import config as cfg
import time
import mrcfile
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

timer = 0.0


def coords_from_tsv(coords_path):
    coords = []
    with open(coords_path, 'r') as file:
        for line in file:
            x, y, z = map(int, line.strip().split('\t'))
            coords.append((x, y, z))
    return coords


def extract_particles(vol_path, coords_path, boxsize, unbin=1, two_dimensional=False, normalize=True):
    coords = coords_from_tsv(coords_path)
    coords = np.array(coords) * unbin
    data = mrcfile.mmap(vol_path, mode='r').data
    imgs = list()

    d = boxsize // 2
    for p in coords:
        x, y, z = p
        if two_dimensional:
            imgs.append(data[z, y - d:y + d, x - d:x + d])
        else:
            imgs.append(data[z - d:z + d, y - d:y + d, x - d:x + d])

    if normalize:
        for i in range(len(imgs)):
            img = np.array(imgs[i]).astype(np.float32)
            img -= np.mean(img)
            img /= np.std(img)
            imgs[i] = img
    return imgs


def get_maxima_3d_watershed(mrcpath="", threshold=128, min_spacing=10.0, min_weight=None, save_txt=True, sort_by_weight=True, save_dir=None, process=None, array=None, array_pixel_size=None, return_coords=False, binning=1):
    if array is None:
        data = mrcfile.read(mrcpath)
        if process:
            process.set_progress(0.2)
        pixel_size = mrcfile.open(mrcpath, header_only=True).voxel_size.x / 10.0
    else:
        data = array
        pixel_size = array_pixel_size
    if binning > 1:
        z, y, x = data.shape
        b = int(binning)
        data = data[:z // b * b, :y // b * b, :x // b * b]
        _type = data.dtype
        data = data.reshape((z // b, b, y // b, b, x // b, b)).mean(5, dtype=_type).mean(3, dtype=_type).mean(1, dtype=_type)

    binary_vol = data > threshold
    distance = distance_transform_edt(binary_vol)
    min_distance = int(min_spacing / pixel_size)
    if process:
        process.set_progress(0.3)

    local_max = peak_local_max(distance, footprint=np.ones((min_distance, min_distance, min_distance)), labels=binary_vol)
    markers, _ = label(local_max)
    if process:
        process.set_progress(0.4)

    labels = watershed(-distance, markers, mask=binary_vol)
    Z, Y, X = np.nonzero(labels)
    if process:
        process.set_progress(0.5)
    # parse blobs
    blobs = dict()
    for i in range(len(X)):
        z = Z[i]
        y = Y[i]
        x = X[i]

        l = labels[z, y, x]
        if l not in blobs:
            blobs[l] = Blob()

        blobs[l].x.append(x)
        blobs[l].y.append(y)
        blobs[l].z.append(z)
        blobs[l].v.append(data[z, y, x])
    if process:
        process.set_progress(0.6)
    if min_weight:
        to_pop = list()
        for key in blobs:
            weight = blobs[key].get_weight()
            if weight < min_weight:
                to_pop.append(key)
        for key in to_pop:
            blobs.pop(key)
    if process:
        process.set_progress(0.7)
    blobs = list(blobs.values())
    metrics = list()
    for blob in blobs:
        if sort_by_weight:
            metrics.append(blob.get_weight())
        else:
            metrics.append(blob.get_volume())

    indices = np.argsort(metrics)[::-1]
    coordinates = list()
    for i in indices:
        coordinates.append(blobs[i].get_centroid() * binning)
    if process:
        process.set_progress(0.8)
    # remove points that are too close to others.
    remove = list()
    i = 0
    while i < len(coordinates):
        for j in range(0, i):
            if i in remove:
                continue
            p = np.array(coordinates[i])
            q = np.array(coordinates[j])
            d = np.sum((p - q) ** 2) ** 0.5 * pixel_size
            if d < min_spacing:
                remove.append(j)
        i += 1
    if process:
        process.set_progress(0.9)

    remove.sort()
    for i in reversed(remove):
        coordinates.pop(i)

    if not return_coords:
        if not save_txt:
            return len(coordinates)

        out_path = mrcpath[:-4] + "_coords.txt"
        if save_dir is not None:
            out_path = save_dir + os.path.basename(mrcpath)[:-4] + "_coords.txt"
        with open(out_path, 'w') as out_file:
            for i in range(len(coordinates)):
                x = int(coordinates[i][0])
                y = int(coordinates[i][1])
                z = int(coordinates[i][2])
                out_file.write(f"{x}\t{y}\t{z}\n")
        if process:
            process.set_progress(0.99)
        return len(coordinates)
    else:
        return coordinates


class Blob:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.z = list()
        self.v = list()

    def get_centroid(self):
        return np.mean(self.x), np.mean(self.y), np.mean(self.z)

    def get_center_of_mass(self):
        mx = np.sum(np.array(self.x) * np.array(self.v))
        my = np.sum(np.array(self.y) * np.array(self.v))
        mz = np.sum(np.array(self.z) * np.array(self.v))
        m = self.get_weight()
        return mx / m, my / m, mz / m

    def get_volume(self):
        return len(self.x)

    def get_weight(self):
        return np.sum(self.v)


def bin_2d_array(a, b):
    y, x = a.shape
    a = a[:y//b*b, :x//b*b]
    a = a.reshape((y//b, b, x//b, b)).mean(3).mean(1)
    return a


def bin_mrc(path, bin_factor):
    print(f"Loading '{path}'")
    data = mrcfile.read(path)
    pxs = mrcfile.open(path, header_only=True, permissive=True).voxel_size.x
    z, y, x = data.shape
    b = int(bin_factor)
    print(f"Binning dataset by factor {b} (dtype = {data.dtype})")
    data = data[:z // b * b, :y // b * b, :x // b * b]
    _type = data.dtype
    data = data.reshape((z // b, b, y // b, b, x // b, b)).mean(5, dtype=_type).mean(3, dtype=_type).mean(1, dtype=_type)
    out_path = path[:path.rfind('.mrc')]+f"_bin{b}.mrc"
    print(f"Saving dataset as: '{out_path}'")
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = pxs * b
    return out_path


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    # from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)


def save_tiff(array, path, pixel_size_nm=100, axes="ZXY"):
    if not (path[-5:] == ".tiff" or path[-4:] == ".tif"):
        path += ".tif"
    if "/" in path:
        root = path.rfind("/")
        root = path[:root]
        if not os.path.exists(root):
            os.makedirs(root)
    tifffile.imwrite(path, array.astype(np.float32), resolution=(
        1. / (1e-7 * pixel_size_nm), 1. / (1e-7 * pixel_size_nm),
        'CENTIMETER'))  # Changed to astype npfloat32 on 230105 to fix importing tifffile tiff with PIL Image open. Default for tifffile export unspecified-float np array appears to be 64bit which PIL doesnt support.


def save_png(array, path, alpha=True):
    try:
        if not path[-4:] == '.png':
            path += '.png'
        if "/" in path:
            root = path.rfind("/")
            root = path[:root]
            if not os.path.exists(root):
                os.makedirs(root)
        if array.dtype != np.dtype(np.uint8):
            array = array.astype(np.uint8)
        if alpha:
            Image.fromarray(array, mode="RGBA").save(path)
        else:
            Image.fromarray(array, mode="RGB").save(path)
    except Exception as e:
        cfg.set_error(e, "Error exporting image as .png. Is the path valid?")

#
# def plot_histogram(data, bins='auto', title=None):
#     plt.hist(data, bins=bins)
#     if title:
#         plt.title(title)
#     plt.show()


def get_filetype(path):
    return path[path.rfind("."):]


def apply_lut_to_float_image(image, lut, contrast_lims=None):
    if len(image.shape) != 2:
        print("Image input in apply_lut_to_float_image is not 2D.")
        return False
    if isinstance(lut, list):
        _lut = np.asarray(lut)
    L, n = np.shape(_lut)
    if contrast_lims is None:
        contrast_lims = (np.amin(image), np.amax(image))
    image = (L - 1) * (image - contrast_lims[0]) / (contrast_lims[1] - contrast_lims[0])
    image = image.astype(int)
    w, h = image.shape
    out_img = np.zeros((w, h, n))
    for x in range(w):
        out_img[x, :, :] = _lut[image[x, :], :]
    return out_img


def is_path_tiff(path):
    if path[-4:] == ".tif" or path[-5:] == ".tiff":
        return True
    return False


def load(path, tag=None):
    if is_path_tiff(path):
        return loadtiff(path)
    else:
        return loadfolder(path, tag)


def loadfolder(path, tag=None):
    if path[-1] != "/":
        path += "/"
    pre_paths = glob.glob(path + "*.tiff") + glob.glob(path + "*.tif")
    paths = list()
    if tag is not None:
        for path in pre_paths:
            if tag in path:
                paths.append(path)
    else:
        paths = pre_paths
    _data = Image.open(paths[0])
    _frame = np.asarray(_data)

    width = _frame.shape[0]
    height = _frame.shape[1]
    depth = len(paths)
    data = np.zeros((depth, width, height), dtype=np.float32)
    _n = len(paths)
    for f in range(0, _n):
        printProgressBar(f, _n, prefix="Loading tiff file\t", length=30, printEnd="\r")
        _data = Image.open(paths[f])
        data[f, :, :] = np.asarray(_data)
    return data


def loadtiff(path, dtype=np.int16):
    "Loads stack as 3d array with dimensions (frames, width, height)"
    _data = Image.open(path)
    _frame = np.asarray(_data)

    width = _frame.shape[0]
    height = _frame.shape[1]
    depth = _data.n_frames
    data = np.zeros((depth, width, height), dtype=dtype)
    for f in range(0, depth):
        printProgressBar(f, depth - 1, prefix="Loading tiff file\t", length=30, printEnd="\r")
        _data.seek(f)
        data[f, :, :] = np.asarray(_data)
    data = np.squeeze(data)
    return data


def tiff_to_mrc(path_in, path_out, apix=1.0):
    data = loadtiff(path_in, dtype=np.float32)
    if path_out[-4:] != ".mrc":
        path_out += ".mrc"
    with mrcfile.new(path_out, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = apix


def tic():
    global timer
    timer = time.time_ns()


def toc(msg):
    print(msg + f" {(time.time_ns() - timer) * 1e-9:.3} seconds")


def clamp(a, _min, _max):
    return min(max(a, _min), _max)

