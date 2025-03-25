# Ais filament

import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import Pommie
import Pommie.compute as compute
import os
from scipy.ndimage import label
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from collections import deque
from scipy.ndimage import gaussian_filter
import json

volume_path = 'C:/Users/mart_/Downloads/transfer (2)/20230908_lmla011_ts_003__MT.mrc'
volume = mrcfile.open(volume_path)

def semantic_to_instance(volume, volume_voxel_size, filament_diameter_angstrom=250, n_transforms=500, score_threshold=0.75, blur_sigma=2.0):
    compute.initialize()

    volume = gaussian_filter(volume, sigma=blur_sigma)
    filament_width_px = int(filament_diameter_angstrom // volume_voxel_size)
    box_size = 2 * filament_width_px
    binning = 1

    while box_size > 32:
        volume_voxel_size *= 2.0
        binning *= 2
        filament_width_px = int(filament_diameter_angstrom // volume_voxel_size)
        box_size = 2 * filament_width_px

    template = Pommie.Mask.new(box_size)
    template.cylindrical(radius_px=filament_width_px/2)
    compute.set_tm2d_n(box_size)

    template = Pommie.Particle(template.data)
    template_mask = Pommie.Mask(template)
    template_mask.cylindrical(radius_px=template.n//2)

    transforms = Pommie.Transform.sample_unit_sphere(n_transforms, polar_lims=(-np.pi / 2, 0.0))

    volume = Pommie.Volume.from_array(volume.data, volume.voxel_size.x)
    volume = volume.bin(binning)
    volume_mask = volume.copy()

    scores, indices = compute.find_template_in_volume(volume=volume,
                                                      volume_mask=volume_mask,
                                                      template=template,
                                                      template_mask=template_mask,
                                                      transforms=transforms)

    labels, n_labels = label(scores > score_threshold)
    return labels.astype(np.float32)


def longest_path_in_skeleton(skel):
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


def parameterize_instances(labels, smoothing=500.0):
    filaments = []
    max_label = labels.max()

    neighbor_shifts = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    for j in range(1, max_label + 1):
        mask = (labels == j)
        if not np.any(mask):
            continue

        skel = skeletonize(mask)
        skel = longest_path_in_skeleton(skel)

        coords = np.column_stack(np.where(skel))
        if len(coords) < 4:
            continue

        skel_voxels = set(map(tuple, coords))
        endpoints = []
        for c in coords:
            c_tup = tuple(c)
            nbr_count = sum(((c_tup[0] + dz, c_tup[1] + dy, c_tup[2] + dx) in skel_voxels) for dz, dy, dx in neighbor_shifts)
            if nbr_count == 1:
                endpoints.append(c_tup)
        if not endpoints:
            continue

        start = endpoints[0]

        visited = set()
        path = []
        queue = deque([start])
        while queue:
            v = queue.pop()
            if v not in visited:
                visited.add(v)
                path.append(v)
                for dz, dy, dx in neighbor_shifts:
                    nbr = (v[0] + dz, v[1] + dy, v[2] + dx)
                    if nbr in skel_voxels and nbr not in visited:
                        queue.append(nbr)

        path = np.array(path)
        if path.shape[0] < 4:
            continue

        tck, u = splprep(path.T, s=smoothing)

        filaments.append(tck)

    return filaments

class Filament:
    def __init__(self, tck):
        self.tck = tck  # (t, c, k)
        self.length = None

    def __str__(self):
        return self.to_str()

    def inspect_linearity(self):
        dD = []
        p = splev(0.0, self.tck)
        for u in np.linspace(0.0, 1.0, 20)[1:]:
            q = splev(u, self.tck)
            dD.append(np.linalg.norm(np.array(q) - np.array(p)) / self.length)
            p = q
        plt.plot(dD)

    def reparameterize(self, n_samples=100):
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

    def to_str(self):
        t, c, k = self.tck
        return json.dumps({
            "t": t.tolist(),
            "c": [ci.tolist() for ci in c],
            "k": k
        })

    @staticmethod
    def from_str(s):
        data = json.loads(s)
        t = np.array(data["t"], dtype=float)
        c = [np.array(ci, dtype=float) for ci in data["c"]]
        k = int(data["k"])
        # We directly call the Filament constructor here
        return Filament((t, c, k))


def splines_to_filaments(splines):
    filaments = []
    for s in splines:
        filaments.append(Filament(s))
        filaments[-1].reparameterize()
    return filaments


labels = mrcfile.read(os.path.splitext(volume_path)[0]+"__labels.mrc").astype(np.int32)
splines = parameterize_instances(labels, smoothing=500)
filaments = splines_to_filaments(splines)

l = [f.length for f in filaments]



