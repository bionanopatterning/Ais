import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import label
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from collections import deque
import pandas as pd
from scipy.ndimage import gaussian_filter
import json
from copy import copy
import starfile


def prune_skeleton(skel):
    coords = np.column_stack(np.where(skel))
    if coords.size == 0:
        return np.zeros_like(skel, dtype=bool)

    voxel_to_index = {}
    for i, c in enumerate(coords):
        voxel_to_index[tuple(c)] = i

    neighbor_shifts = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    adjacency = [[] for _ in range(len(coords))]
    voxel_set = set(voxel_to_index.keys())

    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in neighbor_shifts:
            nbr = (z + dz, y + dy, x + dx)
            if nbr in voxel_set:
                adjacency[i].append(voxel_to_index[nbr])

    def bfs_farthest(start_idx):
        visited = set([start_idx])
        queue = deque([(start_idx, 0)])
        farthest_node = start_idx

        while queue:
            node, dist = queue.popleft()
            farthest_node = node
            for nbr in adjacency[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return farthest_node

    def bfs_path(start, end):
        visited = set([start])
        parent = {start: None}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node == end:
                break
            for nbr in adjacency[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = node
                    queue.append(nbr)

        # Reconstruct path from end -> start
        path_indices = []
        cur = end
        while cur is not None:
            path_indices.append(cur)
            cur = parent.get(cur, None)
        path_indices.reverse()
        return path_indices

    e2 = bfs_farthest(0)
    e3 = bfs_farthest(e2)
    path_indices = bfs_path(e2, e3)

    out_vol = np.zeros_like(skel, dtype=bool)
    for idx in path_indices:
        z, y, x = coords[idx]
        out_vol[z, y, x] = True

    return out_vol


def parameterize_instances(labels):
    filaments = list()
    for j in range(1, int(labels.max()) + 1):
        print(f'tracing filament {j}/{int(labels.max())}')
        mask = (labels == j)
        if not np.any(mask):
            continue

        print(f'\tskeletonize')
        skel = skeletonize(mask)
        print(f'\tprune')
        skel = prune_skeleton(skel)

        coords = np.column_stack(np.where(skel))
        filaments.append(Filament(coords))

    return filaments


class Filament:
    SHIFTS = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]
    SMOOTHING = 500.0

    def __init__(self, skeleton_coordinates):
        self.coords = skeleton_coordinates
        self.valid = False
        self.tck = None
        self.length = None
        self.gen_spline()
        self.linearize_spline()

    def gen_spline(self):
        if len(self.coords) < 4:
            return

        skel_voxels = set(map(tuple, self.coords))
        endpoints = []
        for c in self.coords:
            c_tup = tuple(c)
            nbr_count = sum(
                ((c_tup[0] + dz, c_tup[1] + dy, c_tup[2] + dx) in skel_voxels) for dz, dy, dx in Filament.SHIFTS)
            if nbr_count == 1:
                endpoints.append(c_tup)
        if not endpoints:
            return

        start = endpoints[0]

        visited = set()
        path = []
        queue = deque([start])
        while queue:
            v = queue.pop()
            if v not in visited:
                visited.add(v)
                path.append(v)
                for dz, dy, dx in Filament.SHIFTS:
                    nbr = (v[0] + dz, v[1] + dy, v[2] + dx)
                    if nbr in skel_voxels and nbr not in visited:
                        queue.append(nbr)

        path = np.array(path)
        if path.shape[0] < 4:
            return
        self.coords = path
        self.tck, u = splprep(path.T, s=Filament.SMOOTHING)
        self.valid = True

    def linearize_spline(self, n_samples=100):
        if not self.valid:
            return
        u_test = np.linspace(0, 1, n_samples)
        coords = np.array(splev(u_test, self.tck))
        diffs = np.diff(coords, axis=1)
        dists = np.sqrt(np.sum(diffs**2, axis=0))
        cumdist = np.concatenate(([0], np.cumsum(dists)))
        total_length = cumdist[-1]
        self.length = total_length
        arc_positions = np.linspace(0, total_length, n_samples)
        u_new = np.interp(arc_positions, cumdist, u_test)
        coords_uniform = np.array(splev(u_new, self.tck))
        tck_new, _ = splprep(coords_uniform, u=np.linspace(0, 1, n_samples), s=0, k=self.tck[2])
        self.tck = tck_new

    def to_dict(self):
        return {"skeleton_coordinates": self.coords.tolist()}

    @staticmethod
    def read(p):
        with open(p, 'r') as f:
            return Filament.from_dict(json.load(f))

    @staticmethod
    def from_dict(d):
        skeleton_coordinates = d['skeleton_coordinates']
        return Filament(skeleton_coordinates)

    @classmethod
    def from_path(cls, path):
        with open(path, 'r') as f:
            filaments = json.load(f)
        out_filaments = list()
        for f in filaments:
            out_filaments.append(cls.from_dict(f))
        return out_filaments

    def __str__(self):
        return f"Filament with length {self.length:.1f}, {len(self.coords)}-voxel skeleton."

    def plot(self):
        filament_coords = np.array(splev(np.linspace(0, 1, 100), self.tck))
        plt.plot(filament_coords[2], filament_coords[1], linewidth=10, alpha=0.2)

    def sample_coordinates(self, spacing, offset=0):
        """
        spacing: spacing (in px) along filament between coordinates.
        offset: offset (in px <TODO: check>) to start sampling.
        """
        x = np.arange(offset, 1.0, spacing / self.length)
        if x.any():
            return np.array(splev(x, self.tck)).T
        return np.array([])

    def sample_coordinates_normalized_indices(self, indices):
        """
        indices: list of normalized coordinates to sample along the spline (0.0 would be the filament start position, 1.0 the end; values < 0.0 and > 1.0 are also ok).

        """
        if isinstance(indices, int) or isinstance(indices, float):
            indices = [indices]
        return np.array(splev(np.array(indices), self.tck)).T


def bin_volume(vol, b):
    j, k, l = vol.shape
    vol = vol[:j // b * b, :k // b * b, :l // b * b]
    vol = vol.reshape((j // b, b, k // b, b, l // b, b)).mean(axis=(1, 3, 5))
    return vol


#
# def detect_filaments(filament_segmentation_path, out_path=None, save_labels=False):
#     """
#     input: path to a semantic segmentation volume.
#     output: saves a json file with filament paths.
#     """
#     volume = mrcfile.read(filament_segmentation_path)
#     print('Labelling volume')
#     labels, _ = label(volume > 128)
#
#     print('Beginning parametrization')
#     filaments = parameterize_instances(labels)
#
#     if save_labels:
#         with mrcfile.new(filament_segmentation_path.replace('.mrc', '_labels.mrc'), overwrite=True) as f:
#             f.set_data(labels.astype(np.float32))
#
#     filaments_out = dict()
#     for j, f in enumerate(filaments):
#         if f.valid:
#             filaments_out[j] = f.to_dict()
#
#     out_path = os.path.splitext(filament_segmentation_path)[0]+"__filaments.json" if out_path is None else out_path
#     with open(out_path, 'w') as f:
#         json.dump(filaments_out, f, indent=1)
#     return out_path

def pick_filament(mrcpath, out_path, threshold, spacing_nm, size_nm, binning, margin, pixel_size=10.0):
    """
    volume_path: path to a semantic segmentation volume.
    threshold: threshold to binarize the volume (0-255)
    spacing: spacing between picked points along the filament (in nm)
    size: minimum blob size (in nm3)
    binning: binning factor to apply to the volume before processing
    margin: margin (in nm) to exclude points too close to the border of the volume
    """
    volume = mrcfile.read(mrcpath)

    if volume.dtype == np.float32:
        threshold /= 255
    elif volume.dtype == np.int8:
        threshold /= 2

    volume = volume.astype(np.float32)
    volume = bin_volume(volume, binning)
    pixel_size *= binning
    margin = int(margin / pixel_size)

    labels, _ = label(volume > threshold)
    group_sizes = np.bincount(labels.ravel())
    for n in np.where(group_sizes < (size_nm / (pixel_size**3)))[0].tolist():
        labels[labels == n] = 0

    filaments = parameterize_instances(labels)

    df = pd.DataFrame(columns=['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX', 'aisFilamentID'])
    for j, f in enumerate(filaments):
        c = f.sample_coordinates(spacing=spacing_nm / pixel_size, offset=spacing_nm / pixel_size / 2.0)
        for z, y, x in c:
            df.loc[len(df)] = [z, y, x, j]

    j, k, l = volume.shape
    df = df[(df['rlnCoordinateX'] > margin) & (df['rlnCoordinateX'] < j - margin) &
            (df['rlnCoordinateY'] > margin) & (df['rlnCoordinateY'] < k - margin) &
            (df['rlnCoordinateZ'] > margin) & (df['rlnCoordinateZ'] < l - margin)]

    df['rlnCoordinateX'] *= binning
    df['rlnCoordinateY'] *= binning
    df['rlnCoordinateZ'] *= binning
    df['rlnMicrographName'] = os.path.basename(mrcpath).split("__")[0]+".mrc"
    starfile.write({'particles': df}, out_path, overwrite=True)

    return len(df), len(filaments)



