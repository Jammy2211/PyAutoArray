import numpy as np

from autoarray.inversion.mappers.abstract import AbstractMapper

from autoarray.structures.grids.two_d.grid_2d_pixelization import PixelNeighbors


class MockMask:
    def __init__(self, native_index_for_slim_index=None):

        self.native_index_for_slim_index = native_index_for_slim_index


class MockDataset:
    def __init__(self, grid_inversion=None, psf=None, mask=None):

        self.grid_inversion = grid_inversion
        self.psf = psf
        self.mask = mask


class MockFit:
    def __init__(self,
                 dataset=MockDataset(),
                 inversion=None,
                 noise_map=None,
                 regularization_term=None,
                 log_det_curvature_reg_matrix_term=None,
                 log_det_regularization_matrix_term=None,
                 ):

        self.dataset = dataset
        self.inversion = inversion
        self.noise_map = noise_map
        self.signal_to_noise_map = noise_map

        self.regularization_term = regularization_term
        self.log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term


### Inversion ###


class MockConvolver:
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def convolve_mapping_matrix(self, mapping_matrix):
        return np.ones(self.shape)


class MockPixelization:
    def __init__(self, value, grid=None):
        self.value = value
        self.grid = grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_from_grid_and_sparse_grid(
        self,
        grid,
        sparse_grid,
        sparse_image_plane_grid=None,
        hyper_image=None,
        settings=None,
        preloads=None,
        profiling_dict=None,
    ):
        return self.value

    def sparse_grid_from_grid(self, grid, hyper_image, settings=None):
        if hyper_image is None:
            return self.grid
        else:
            return self.grid * hyper_image


class MockRegularization:
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def regularization_matrix_from_pixel_neighbors(
        self, pixel_neighbors, pixel_neighbors_sizes
    ):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def regularization_matrix_from_mapper(self, mapper):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class MockPixelizationGrid:
    def __init__(self, pixel_neighbors=None, pixel_neighbors_sizes=None):

        self.pixel_neighbors = PixelNeighbors(
            arr=pixel_neighbors, sizes=pixel_neighbors_sizes
        )
        self.shape = (len(self.pixel_neighbors.sizes),)


class MockMapper(AbstractMapper):
    def __init__(
        self,
        source_grid_slim=None,
        source_pixelization_grid=None,
        hyper_image=None,
        pixelization_index_for_sub_slim_index=None,
        mapping_matrix=None,
        pixel_signals=None,
        pixels=None,
    ):

        super().__init__(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            hyper_image=hyper_image,
        )

        self._pixelization_index_for_sub_slim_index = (
            pixelization_index_for_sub_slim_index
        )
        self._mapping_matrix = mapping_matrix

        self._pixels = pixels

        self._pixel_signals = pixel_signals

    def pixel_signals_from(self, signal_scale):
        if self._pixel_signals is None:
            return super().pixel_signals_from(signal_scale=signal_scale)
        return self._pixel_signals

    @property
    def pixels(self):
        if self._pixels is None:
            return super().pixels
        return self._pixels

    @property
    def pixelization_index_for_sub_slim_index(self):
        return self._pixelization_index_for_sub_slim_index

    @property
    def mapping_matrix(self):
        return self._mapping_matrix


class MockInversion:
    def __init__(
        self,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix_sparse_preload=None,
        curvature_matrix_preload_counts=None,
        log_det_regularization_matrix_term=None,
    ):

        self.mapper = mapper
        self.curvature_matrix_sparse_preload = curvature_matrix_sparse_preload
        self.curvature_matrix_preload_counts = curvature_matrix_preload_counts
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term

        if blurred_mapping_matrix is None:
            self.blurred_mapping_matrix = np.zeros((1, 1))
        else:
            self.blurred_mapping_matrix = blurred_mapping_matrix

        self.regularization_matrix = np.zeros((1, 1))
        self.curvature_matrix = np.zeros((1, 1))
        self.curvature_reg_matrix = np.zeros((1, 1))
        self.solution_vector = np.zeros((1))

    @property
    def reconstructed_image(self):
        return np.zeros((1, 1))


