import numpy as np


class MockFitInversion(object):
    def __init__(
        self,
        regularization_term,
        log_det_curvature_reg_matrix_term,
        log_det_regularization_matrix_term,
    ):

        self.regularization_term = regularization_term
        self.log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term


class MockPixelization(object):
    def __init__(self, value, grid=None):
        self.value = value
        self.grid = grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_from_grid_and_sparse_grid(
        self, grid, sparse_grid, inversion_uses_border, hyper_image=None
    ):
        return self.value

    def sparse_grid_from_grid(self, grid, hyper_image):
        if hyper_image is None:
            return self.grid
        else:
            return self.grid * hyper_image


class MockRegularization(object):
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def regularization_matrix_from_pixel_neighbors(
        self, pixel_neighbors, pixel_neighbors_size
    ):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def regularization_matrix_from_mapper(self, mapper):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class MockPixelizationGrid(object):
    def __init__(self, pixel_neighbors=None, pixel_neighbors_size=None):

        self.pixel_neighbors = pixel_neighbors
        self.pixel_neighbors_size = pixel_neighbors_size


class MockRegMapper(object):
    def __init__(self, pixelization_grid=None, pixel_signals=None):

        self.pixelization_grid = pixelization_grid
        self.pixel_signals = pixel_signals

    def pixel_signals_from_signal_scale(self, signal_scale):
        return self.pixel_signals


class MockMapper(object):
    def __init__(self, matrix_shape, grid=None, pixelization_grid=None):

        self.grid = grid
        self.pixelization_grid = pixelization_grid
        self.mapping_matrix = np.ones(matrix_shape)
        if pixelization_grid is not None:
            self.pixels = pixelization_grid.shape[0]
        else:
            self.pixels = None


class MockConvolver(object):
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def convolve_mapping_matrix(self, mapping_matrix):
        return np.ones(self.shape)


class MockInversion(object):
    def __init__(self):
        self.blurred_mapping_matrix = np.zeros((1, 1))
        self.regularization_matrix = np.zeros((1, 1))
        self.curvature_matrix = np.zeros((1, 1))
        self.curvature_reg_matrix = np.zeros((1, 1))
        self.solution_vector = np.zeros((1))

    @property
    def reconstructed_image(self):
        return np.zeros((1, 1))
