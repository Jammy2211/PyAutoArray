import numpy as np

from autoarray.inversion.mappers.abstract import AbstractMapper

from autoarray.structures.grids import grid_decorators
from autoarray.structures.grids.two_d.grid_2d_pixelization import PixelNeighbors


### Grids ###


def grid_to_grid_radii(grid):
    return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))


def ndarray_1d_from(profile, grid):

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


def ndarray_2d_from(profile, grid):
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
    def ndarray_1d_from(self, grid) -> np.ndarray:
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a 1D ndarray
        of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return np.exp(
            np.multiply(
                -self.sersic_constant,
                np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
            )
        )

    @grid_decorators.grid_2d_to_structure
    def ndarray_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return self.grid_to_grid_cartesian(
            grid=grid, radius=np.full(grid.shape[0], 2.0)
        )

    @grid_decorators.grid_2d_to_vector_yx
    @grid_decorators.grid_2d_to_structure
    def ndarray_yx_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return self.grid_to_grid_cartesian(
            grid=grid, radius=np.full(grid.shape[0], 2.0)
        )

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_1d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a list of 1D
        ndarrays of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
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
    def ndarray_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D list of
        ndarrays of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [
            self.grid_to_grid_cartesian(grid=grid, radius=np.full(grid.shape[0], 2.0))
        ]

    @grid_decorators.grid_2d_to_vector_yx
    @grid_decorators.grid_2d_to_structure_list
    def ndarray_yx_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a list of 2D
        ndarrays of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [
            self.grid_to_grid_cartesian(grid=grid, radius=np.full(grid.shape[0], 2.0))
        ]


class MockGrid1DLikeObj:
    def __init__(self, centre=(0.0, 0.0), angle=0.0):

        self.centre = centre
        self.angle = angle

    @grid_decorators.grid_1d_to_structure
    def ndarray_1d_from(self, grid):
        return np.ones(shape=grid.shape[0])

    # @grid_decorators.grid_1d_to_structure
    # def ndarray_2d_from(self, grid):
    #     return np.multiply(2.0, grid)

    # @grid_decorators.grid_1d_to_structure_list
    # def ndarray_1d_list_from(self, grid):
    #     return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]
    #
    # @grid_decorators.grid_1d_to_structure_list
    # def ndarray_2d_list_from(self, grid):
    #     return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGrid2DLikeObj:
    def __init__(self):
        pass

    @grid_decorators.grid_2d_to_structure
    def ndarray_1d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a 1D ndarray
        of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return np.ones(shape=grid.shape[0])

    @grid_decorators.grid_2d_to_structure
    def ndarray_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return np.multiply(2.0, grid)

    @grid_decorators.grid_2d_to_vector_yx
    @grid_decorators.grid_2d_to_structure
    def ndarray_yx_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return 2.0 * grid

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_1d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a list of 1D
        ndarrays of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @grid_decorators.grid_2d_to_structure_list
    def ndarray_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D list of
        ndarrays of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]

    @grid_decorators.grid_2d_to_vector_yx
    @grid_decorators.grid_2d_to_structure_list
    def ndarray_yx_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a list of 2D
        ndarrays of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGridRadialMinimum:
    def __init__(self):
        pass

    def grid_to_grid_radii(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @grid_decorators.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid):
        return grid
