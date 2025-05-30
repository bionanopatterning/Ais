import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import label
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from collections import deque
from scipy.ndimage import gaussian_filter
import json
from copy import copy
import starfile

INSTANCE_TEMPLATE_N = 32


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

    def unbin(self, b=2):
        self.coords *= b
        self.gen_spline()
        self.linearize_spline()
        return self

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



def detect_filaments(filament_segmentation_path, out_path=None, save_labels=False):
    """
    input: path to a semantic segmentation volume.
    output: saves a json file with filament paths.
    """
    volume = mrcfile.read(filament_segmentation_path)
    print('Labelling volume')
    labels, _ = label(volume > 128)

    print('Beginning parametrization')
    filaments = parameterize_instances(labels)

    if save_labels:
        with mrcfile.new(filament_segmentation_path.replace('.mrc', '_labels.mrc'), overwrite=True) as f:
            f.set_data(labels.astype(np.float32))

    filaments_out = dict()
    for j, f in enumerate(filaments):
        if f.valid:
            filaments_out[j] = f.to_dict()

    out_path = os.path.splitext(filament_segmentation_path)[0]+"__filaments.json" if out_path is None else out_path
    with open(out_path, 'w') as f:
        json.dump(filaments_out, f, indent=1)
    return out_path

import glob
import pandas as pd

volumes = glob.glob('/cephfs/mlast/compu_projects/mt_tip_pick/segmented/*.mrc')
for v in volumes:
    print(v)
    if not os.path.exists(v.replace('.mrc', '__filaments.json')):
        detect_filaments(v)

files = glob.glob('/cephfs/mlast/dev/ais_filament/segmentations/*filaments.json')

all_coords = list()
for file in files:
    filaments = dict()
    coordinates = list()
    tomo = os.path.basename(file).split('__')[0]
    with open(file, 'r') as _:
        data = json.load(_)
        for j in data:
            filaments[j] = Filament.from_dict(data[j])
            if filaments[j].length < 50:
                continue
            tip_coordinates = filaments[j].sample_coordinates_normalized_indices([0.0, 1.0])

            for k, l, m in tip_coordinates:
                all_coords.append((m, l, k, tomo))
                print(all_coords[-1])

# Try making a fake Pom project.
root = '/cephfs/mlast/dev/ais_filament/'
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, 'capp'), exist_ok=True)
os.makedirs(os.path.join(root, 'capp', 'mt_forson'), exist_ok=True)
os.system(f'cd {root}')
os.system(f'pom')

with open(os.path.join(root, 'capp', 'mt_forson', 'config.json'), 'w') as f:
    json.dump({'target': '', 'job_name': 'mt_forson', 'context_elements': []}, f)

with open(os.path.join(root, 'project_configuration.json'), 'r') as f:
    cfg = json.load(f)
    cfg['root'] = root
    cfg['tomogram_dir'] = 'tomos'

with open(os.path.join(root, 'project_configuration.json'), 'w') as f:
    json.dump(cfg, f, indent=4)

with open(os.path.join(root, 'capp', 'mt_forson', 'all_particles.tsv'), 'w') as f:
    f.write('X\tY\tZ\ttomo\n')
    for _ in all_coords:
        f.write(f'{int(_[0])}\t{int(_[1])}\t{int(_[2])}\t{_[3]}\n')




