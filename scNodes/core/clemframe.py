from scNodes.core import settings
from scNodes.core import config as cfg
from scNodes.core.opengl_classes import *
import tifffile
import mrcfile
from copy import copy, deepcopy
import datetime

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # NONE OF THIS CODE IS ACTUALLY USED IN STANDALONE POM, BUT IT IS INCLUDED FOR COMPATIBILITTY WITH SCNODES+POM # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class CLEMFrame:
    idgen = count(0)
    HISTOGRAM_BINS = 40

    def __init__(self, img_array):
        """Grayscale images only - img_array must be a 2D np.ndarray"""
        uid_counter = next(CLEMFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+"000") + uid_counter
        self.data = img_array
        self.is_rgb = False
        if len(self.data.shape) == 2:
            self.height, self.width = self.data.shape
        elif len(self.data.shape) == 3:
            self.height = self.data.shape[0]
            self.width = self.data.shape[1]
            self.is_rgb = True
            self.data = self.data[:, :, 0:3]
        else:
            raise Exception("CLEMFrame not able to import image data with dimensions other than (XY) or (XYC). How did you manage..?")
        self.children = list()
        self.parent = None
        self.title = "Frame "+str(uid_counter)
        self.path = None
        self.has_slices = False
        self.n_slices = 1
        self.sum_slices = 1
        self.current_sum_slices = 1
        self.current_slice = 0
        self.extension = ""
        self.particle_groups = list()
        self.particle_groups.append(ParticleGroup())
        self.current_particle_group = self.particle_groups[0]
        # transform parameters
        self.locked = False
        self.pixel_size = cfg.ce_default_pixel_size  # pixel size in nm
        self.pixel_size_z = cfg.ce_default_pixel_size
        self.pivot_point = np.zeros(2)  # pivot point for rotation and scaling of this particular image. can be moved by the user. In _local coordinates_, i.e. relative to where the frame itself is positioned.
        self.transform = Transform()
        self.flip_h = False
        self.flip_v = False
        # visuals
        self.binning = 1
        self.blend_mode = 0
        self.lut = 1
        self.lut_clamp_mode = 0
        self.colour = (1.0, 1.0, 1.0, 1.0)
        self.alpha = 1.0
        self.contrast_lims = [0.0, 65535.0]
        self.compute_autocontrast()
        self.hist_bins = list()
        self.hist_vals = list()
        self.rgb_hist_vals = list()
        self.rgb_contrast_lims = [0, 255, 0, 255, 0, 255]
        self.compute_histogram()
        self.hide = False
        self.interpolate = False  # False for pixelated, True for interpolated

        # aux
        self.corner_positions_local = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.corner_positions = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # opengl
        self.texture = None
        self.lut_texture = None
        self.quad_va = None
        self.vertex_positions = None
        self.border_va = None
        self.setup_opengl_objects()

    def update(self):
        self.update_image_texture()

    def duplicate(self):
        my_copy = deepcopy(self)
        my_copy.children = list()
        my_copy.parent = None
        uid_counter = next(CLEMFrame.idgen)
        my_copy.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        my_copy.title += " (duplicate)"
        my_copy.setup_opengl_objects()
        return my_copy

    def adjust_position_on_flip(self, axis):
        P = np.array(self.transform.translation[0:2])
        A = np.array(axis[0:2])
        U = np.array(axis[2:4])
        AP = np.subtract(P, A)
        print(P, A, AP, U)
        P_new = np.add(A, AP - 2 * np.dot(AP, U) * U)
        self.transform.translation[0] = P_new[0]
        self.transform.translation[1] = P_new[1]

    def flip(self, horizontally=True, include_children=False, axis=None):
        if horizontally:
            self.flip_h = not self.flip_h
            self.data = np.flip(self.data, axis=1)
            for group in self.particle_groups:
                for i in range(len(group.coordinates)):
                    group.coordinates[i][0] = self.width - group.coordinates[i][0]
            if axis is not None:
                self.adjust_position_on_flip(axis)
            else:
                theta = -self.transform.rotation / 180.0 * np.pi
                u = (np.cos(theta), -np.sin(theta))
                a = tuple(self.transform.translation[0:2])
                axis = (*a, *u)
                self.adjust_position_on_flip(axis)
        else:
            self.flip_v = not self.flip_v
            self.data = np.flip(self.data, axis=0)
            for group in self.particle_groups:
                for i in range(len(group.coordinates)):
                    group.coordinates[i][1] = -(self.height + group.coordinates[i][1])
            if axis is not None:
                self.adjust_position_on_flip(axis)
            else:
                theta = -self.transform.rotation / 180.0 * np.pi
                u = (np.sin(theta), np.cos(theta))
                a = tuple(self.transform.translation[0:2])
                axis = (*a, *u)
                self.adjust_position_on_flip(axis)
        if include_children:
            for child in self.children:
                child.flip(horizontally=horizontally, include_children=True, axis=axis)
        self.update_image_texture(compute_histogram=False)

    def setup_opengl_objects(self):
        if self.is_rgb:
            self.texture = Texture(format="rgb32f")
            self.texture.update(self.data.astype(np.float32))
        else:
            self.texture = Texture(format="r32f")
            self.texture.update(self.data.astype(np.float32))
        self.lut_texture = Texture(format="rgba32f")
        self.quad_va = VertexArray()
        self.vertex_positions = list()
        self.border_va = VertexArray(attribute_format="xy")
        self.update_lut()
        self.generate_va()
        self.interpolate = not self.interpolate
        self.toggle_interpolation()

    def update_image_texture(self, compute_histogram=True):
        self.texture.update(self.data.astype(np.float32))
        if not self.is_rgb and compute_histogram:
            self.compute_histogram()

    def set_slice(self, requested_slice):
        if not self.has_slices:
            return
        requested_slice = min([max([requested_slice, 0]), self.n_slices - 1])
        if requested_slice == self.current_slice and self.current_sum_slices == self.sum_slices:
            return
        self.current_sum_slices = self.sum_slices
        slices = list(range(requested_slice, requested_slice+self.sum_slices))
        for i in reversed(range(len(slices))):
            if slices[i] >= self.n_slices:
                slices.pop(i)
        actual_sum_slices = len(slices)
        try:
            # else, load slice and update texture.
            self.current_slice = requested_slice
            if self.extension == ".tiff" or self.extension == ".tif":
                self.data = tifffile.imread(self.path, key=slices.pop(0)).astype(float)
                for idx in slices:
                    self.data += tifffile.imread(self.path, key=idx).astype(float)
                self.data /= actual_sum_slices
                if self.flip_h:
                    self.data = np.flip(self.data, axis=1)
                if self.flip_v:
                    self.data = np.flip(self.data, axis=0)
                self.update_image_texture()
            elif self.extension == ".mrc":
                mrc = mrcfile.mmap(self.path, mode="r")
                self.n_slices = mrc.data.shape[0]
                self.data = mrc.data[slices.pop(0), :, :]
                target_type_dict = {np.float32: float, float: float, np.dtype('int8'): np.dtype('uint8'), np.dtype('int16'): np.dtype('float32')}
                if self.data.dtype not in target_type_dict:
                    target_type = float
                else:
                    target_type = target_type_dict[self.data.dtype]
                self.data = np.array(self.data.astype(target_type, copy=False), dtype=float)
                for idx in slices:
                    self.data += np.array(mrc.data[idx, :, :].astype(target_type, copy=False), dtype=float)
                self.data /= actual_sum_slices
                if self.flip_h:
                    self.data = np.flip(self.data, axis=1)
                if self.flip_v:
                    self.data = np.flip(self.data, axis=0)
                self.update_image_texture()
        except Exception as e:
            cfg.set_error(e, f"Could not load slice from '{self.path}'.\nHas the file been moved to a different location?")

    def translate(self, translation):
        if self.locked:
            return
        for frame in self.list_all_children(include_self=True):
            frame.pivot_point[0] += translation[0]
            frame.pivot_point[1] += translation[1]
            frame.transform.translation[0] += translation[0]
            frame.transform.translation[1] += translation[1]

    def pivoted_rotation(self, pivot, angle, ignore_children=False, ignore_lock=False):
        if self.locked and not ignore_lock:
            return
        affected_frames = [self] if ignore_children else self.list_all_children(include_self=True)
        for frame in affected_frames:
            frame.transform.rotation += angle
            p = np.matrix([frame.transform.translation[0] - pivot[0], frame.transform.translation[1] - pivot[1]]).T
            rotation_mat = np.identity(2)
            _cos = np.cos(angle / 180.0 * np.pi)
            _sin = np.sin(angle / 180.0 * np.pi)
            rotation_mat[0, 0] = _cos
            rotation_mat[1, 0] = _sin
            rotation_mat[0, 1] = -_sin
            rotation_mat[1, 1] = _cos
            delta_translation = rotation_mat * p - p
            delta_translation = [float(delta_translation[0]), float(delta_translation[1])]
            frame.transform.translation[0] += delta_translation[0]
            frame.transform.translation[1] += delta_translation[1]

    def pivoted_scale(self, pivot, scale):
        if self.locked:
            return
        for frame in self.list_all_children(include_self=True):
            offset = [pivot[0] - frame.transform.translation[0], pivot[1] - frame.transform.translation[1]]
            frame.transform.translation[0] += offset[0] * (1.0 - scale)
            frame.transform.translation[1] += offset[1] * (1.0 - scale)
            frame.pixel_size *= scale
            if not frame == self:
                frame.pivot_point[0] += offset[0] * (1.0 - scale)
                frame.pivot_point[1] += offset[1] * (1.0 - scale)

    def update_model_matrix(self):
        self.transform.scale = self.pixel_size
        self.transform.compute_matrix()
        for i in range(4):
            vec = np.matrix([*self.corner_positions_local[i], 0.0, 1.0]).T
            corner_pos = (self.transform.matrix * vec)[0:2]
            self.corner_positions[i] = [float(corner_pos[0]), float(corner_pos[1])]

    def world_to_pixel_coordinate(self, world_coordinate):
        vec = np.matrix([world_coordinate[0], world_coordinate[1], 0.0, 1.0]).T
        invmat = np.linalg.inv(self.transform.matrix)
        out_vec = invmat * vec
        va_coordinate = [out_vec[0, 0], out_vec[1, 0]]
        pixel_coordinate = [int(out_vec[0,0] + self.width / 2), int(out_vec[1,0] + self.height / 2)]
        return pixel_coordinate, va_coordinate

    def unparent(self):
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = None

    def parent_to(self, parent):
        if parent is None:
            self.unparent()
            return
        # check that parent isn't any child of self
        _parent = parent
        inheritance_loop = False
        while _parent is not None:
            if _parent == self:
                inheritance_loop = True
                break
            _parent = _parent.parent
        if inheritance_loop:
            return

        # remove self from parent's children list, set parent, and edit parent's children
        self.unparent()
        self.parent = parent
        parent.children.append(self)

    def list_all_children(self, include_self=False):
        """returns a list with all of this frame's children, + children's children, + etc."""
        def get_children(frame):
            children = list()
            for child in frame.children:
                children.append(child)
                children += get_children(child)
            return children
        all_children = get_children(self)
        if include_self:
            all_children = [self] + all_children
        return all_children

    def toggle_interpolation(self):
        self.interpolate = not self.interpolate
        if self.interpolate:
            self.texture.set_linear_mipmap_interpolation()
        else:
            self.texture.set_no_interpolation()

    def force_lut_grayscale(self):
        original_lut = copy(self.lut)
        self.lut = 1
        self.update_lut()
        self.lut = original_lut

    def update_lut(self):
        if self.lut > 0:
            lut_array = np.asarray(settings.luts[settings.lut_names[self.lut - 1]])
            self.colour = (lut_array[-1, 0], lut_array[-1, 1], lut_array[-1, 2], 1.0)
        else:
            lut_array = np.asarray(settings.luts[settings.lut_names[0]]) * np.asarray(self.colour[0:3])
        if lut_array.shape[1] == 3:
            lut_array = np.reshape(lut_array, (1, lut_array.shape[0], 3))
        # Add alpha
        lut_array_rgba = np.ones((1, lut_array.shape[1], 4))
        lut_array_rgba[:, :, 0:3] = lut_array
        if self.lut_clamp_mode == 1:
            lut_array_rgba[0, 0, 3] = 0.0
            lut_array_rgba[0, -1, 3] = 0.0
        elif self.lut_clamp_mode == 2:
            lut_array_rgba[0, 0, 3] = 0.0
        self.lut_texture.update(lut_array_rgba)

    def compute_autocontrast(self, saturation=None):
        saturation_pct = settings.autocontrast_saturation
        if saturation:
            saturation_pct = saturation
        subsample = self.data[::settings.autocontrast_subsample, ::settings.autocontrast_subsample]
        n = subsample.shape[0] * subsample.shape[1]
        sorted_pixelvals = np.sort(subsample.flatten())

        min_idx = min([int(saturation_pct / 100.0 * n), n - 1])
        max_idx = max([int((1.0 - saturation_pct / 100.0) * n), 0])
        self.contrast_lims[0] = sorted_pixelvals[min_idx]
        self.contrast_lims[1] = sorted_pixelvals[max_idx]

    def compute_histogram(self):
        if not self.is_rgb:
            # ignore very bright pixels
            mean = np.mean(self.data)
            std = np.std(self.data)
            self.hist_vals, self.hist_bins = np.histogram(self.data[self.data < (mean + 20 * std)], bins=CLEMFrame.HISTOGRAM_BINS)

            self.hist_vals = self.hist_vals.astype('float32')
            self.hist_bins = self.hist_bins.astype('float32')
            self.hist_vals = np.log(self.hist_vals + 1)
        else:
            for i in range(3):
                vals, _ = np.histogram(self.data[:, :, i], bins=CLEMFrame.HISTOGRAM_BINS)
                vals = vals.astype('float32')
                self.rgb_hist_vals.append(vals)

    def generate_va(self):
        # set up the quad vertex array
        w, h = self.width * 0.5, self.height * 0.5
        vertex_attributes = list()
        indices = list()
        n = cfg.ce_va_subdivision
        for i in range(n):
            for j in range(n):
                x = ((2 * i / (n - 1)) - 1) * w
                y = ((2 * j / (n - 1)) - 1) * h
                u = 0.5 + x / w / 2
                v = 0.5 + y / h / 2
                vertex_attributes += [x, y, 0.0, u, v]
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                indices += [idx, idx + 1, idx + n, idx + n, idx + 1, idx + n + 1]

        self.corner_positions_local = [[-w, h], [-w, -h], [w, -h], [w, h]]
        self.quad_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))
        # set up the border vertex array
        vertex_attributes = [-w, h,
                             w, h,
                             w, -h,
                             -w, -h]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.border_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def move_to_front(self):
        cfg.ce_frames.insert(0, cfg.ce_frames.pop(cfg.ce_frames.index(self)))

    def move_to_back(self):
        cfg.ce_frames.append(cfg.ce_frames.pop(cfg.ce_frames.index(self)))

    def move_backwards(self):
        idx = cfg.ce_frames.index(self)
        if idx < (len(cfg.ce_frames) - 1):
            cfg.ce_frames[idx], cfg.ce_frames[idx + 1] = cfg.ce_frames[idx + 1], cfg.ce_frames[idx]

    def move_forwards(self):
        idx = cfg.ce_frames.index(self)
        if idx > 0:
            cfg.ce_frames[idx], cfg.ce_frames[idx - 1] = cfg.ce_frames[idx - 1], cfg.ce_frames[idx]

    def __eq__(self, other):
        if isinstance(other, CLEMFrame):
            return self.uid == other.uid
        return False

    def __str__(self):
        return f"CLEMFrame with id {self.uid}."


class Transform:
    def __init__(self):
        self.translation = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.scale = 1.0
        self.matrix = np.identity(4)
        self.matrix_no_scale = np.identity(4)

    def compute_matrix(self):
        scale_mat = np.identity(4) * self.scale
        scale_mat[3, 3] = 1

        rotation_mat = np.identity(4)
        _cos = np.cos(self.rotation / 180.0 * np.pi)
        _sin = np.sin(self.rotation / 180.0 * np.pi)
        rotation_mat[0, 0] = _cos
        rotation_mat[1, 0] = _sin
        rotation_mat[0, 1] = -_sin
        rotation_mat[1, 1] = _cos

        translation_mat = np.identity(4)
        translation_mat[0, 3] = self.translation[0]
        translation_mat[1, 3] = self.translation[1]

        self.matrix = np.matmul(translation_mat, np.matmul(rotation_mat, scale_mat))
        self.matrix_no_scale = np.matmul(translation_mat, rotation_mat)

    def __add__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] + other.translation[0]
        out.translation[1] = self.translation[1] + other.translation[1]
        out.rotation = self.rotation + other.rotation
        return out

    def __str__(self):
        return f"Transform with translation = {self.translation[0], self.translation[1]}, scale = {self.scale}, rotation = {self.rotation}"

    def __sub__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] - other.translation[0]
        out.translation[1] = self.translation[1] - other.translation[1]
        out.rotation = self.rotation - other.rotation
        return out


class ParticleGroup:
    idgen = count(0)

    def __init__(self, name="New group"):
        self.id = next(ParticleGroup.idgen)
        self.name = name
        self.colour = (0.1, 0.2, 1.0, 1.0)
        self.diameter = 200.0  # Angstrom
        self.coordinates = list()

    def __eq__(self, other):
        if isinstance(other, ParticleGroup):
            return self.id == other.id
        return False


