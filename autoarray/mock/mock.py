import numpy as np

from autoarray.inversion.mappers.abstract import AbstractMapper

from autoarray.structures.grids import grid_decorators
from autoarray.structures.grids.two_d.grid_2d_pixelization import PixelNeighbors


### Grids ###


def grid_to_grid_radii(grid):
    return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))


def ndarray_1d_from_grid(profile, grid):

    sersic_constant = (
        (2 * 2.0)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * 2.0))
        + (46.0 / (25515.0 * 2.0 ** 2))
        + (131.0 / (1148175.0 * 2.0 ** 3))
        - (2194697.0 / (30690717750.0 * 2.0 ** 4))
    )

    grid_radii = grid_to_grid_radii(grid=grid)

    return np.exp(
        np.multiply(
            -sersic_constant,
            np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
        )
    )


def grid_angle_to_profile(grid_thetas):
    """The angle between each (y,x) coordinate on the grid and the profile, in radians.

    Parameters
    -----------
    grid_thetas
        The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
    """
    return np.cos(grid_thetas), np.sin(grid_thetas)


def grid_to_grid_cartesian(grid, radius):
    """
    Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian
    coordinates.

    Parameters
    ----------
    grid
        The (y, x) coordinates in the reference frame of the profile.
    radius
        The circular radius of each coordinate from the profile center.
    """
    grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
    cos_theta, sin_theta = grid_angle_to_profile(grid_thetas=grid_thetas)
    return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)


def ndarray_2d_from_grid(profile, grid):
    return grid_to_grid_cartesian(grid=grid, radius=np.full(grid.shape[0], 2.0))


class MockGridLikeIteratorObj:
    def __init__(self):
        pass

    @property
    def sersic_constant(self):
        return (
            (2 * 2.0)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * 2.0))
            + (46.0 / (25515.0 * 2.0 ** 2))
            + (131.0 / (1148175.0 * 2.0 ** 3))
            - (2194697.0 / (30690717750.0 * 2.0 ** 4))
        )

    def grid_to_grid_radii(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each (y,x) coordinate on the grid and the profile, in radians.

        Parameters
        -----------
        grid_thetas
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return np.cos(grid_thetas), np.sin(grid_thetas)

    def grid_to_grid_cartesian(self, grid, radius):
        """
        Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian
        coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        radius
            The circular radius of each coordinate from the profile center.
        """
        grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
        cos_theta, sin_theta = self.grid_angle_to_profile(grid_thetas=grid_thetas)
        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    @grid_decorators.grid_2d_to_structure
    def ndarray_1d_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return np.exp(
            np.multiply(
                -self.sersic_constant,
                np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
            )
        )

    @grid_decorators.grid_2d_to_structure
    def ndarray_2d_from_grid(self, grid):
        return self.grid_to_grid_cartesian(
            grid=grid, radius=np.full(grid.shape[0], 2.0)
        )

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_1d_list_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return [
            np.exp(
                np.multiply(
                    -self.sersic_constant,
                    np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
                )
            )
        ]

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_2d_list_from_grid(self, grid):
        return [
            self.grid_to_grid_cartesian(grid=grid, radius=np.full(grid.shape[0], 2.0))
        ]


class MockGrid1DLikeObj:
    def __init__(self, centre=(0.0, 0.0), angle=0.0):

        self.centre = centre
        self.angle = angle

    @grid_decorators.grid_1d_to_structure
    def ndarray_1d_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    # @grid_decorators.grid_1d_to_structure
    # def ndarray_2d_from_grid(self, grid):
    #     return np.multiply(2.0, grid)

    # @grid_decorators.grid_1d_to_structure_list
    # def ndarray_1d_list_from_grid(self, grid):
    #     return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]
    #
    # @grid_decorators.grid_1d_to_structure_list
    # def ndarray_2d_list_from_grid(self, grid):
    #     return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGrid2DLikeObj:
    def __init__(self):
        pass

    @grid_decorators.grid_2d_to_structure
    def ndarray_1d_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    @grid_decorators.grid_2d_to_structure
    def ndarray_2d_from_grid(self, grid):
        return np.multiply(2.0, grid)

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_1d_list_from_grid(self, grid):
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_2d_list_from_grid(self, grid):
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGridRadialMinimum:
    def __init__(self):
        pass

    def grid_to_grid_radii(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @grid_decorators.relocate_to_radial_minimum
    def deflections_2d_from_grid(self, grid):
        return grid


class MockMask:
    def __init__(self, native_index_for_slim_index=None):

        self.native_index_for_slim_index = native_index_for_slim_index


class MockDataset:
    def __init__(self, grid_inversion=None, psf=None, mask=None):

        self.grid_inversion = grid_inversion
        self.psf = psf
        self.mask = mask


class MockFit:
    def __init__(self, dataset=MockDataset(), inversion=None, noise_map=None):

        self.dataset = dataset
        self.inversion = inversion
        self.noise_map = noise_map
        self.signal_to_noise_map = noise_map


### Inversion ###


class MockFitInversion:
    def __init__(
        self,
        regularization_term,
        log_det_curvature_reg_matrix_term,
        log_det_regularization_matrix_term,
    ):

        self.regularization_term = regularization_term
        self.log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term


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


class MockConvolver:
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def convolve_mapping_matrix(self, mapping_matrix):
        return np.ones(self.shape)


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
