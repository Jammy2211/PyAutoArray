import numpy as np

from autoarray.structures import grids


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
    grid_thetas : ndarray
        The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
    """
    return np.cos(grid_thetas), np.sin(grid_thetas)


def grid_to_grid_cartesian(grid, radius):
    """
    Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian
    coordinates.

    Parameters
    ----------
    grid : grid_like
        The (y, x) coordinates in the reference frame of the profile.
    radius : ndarray
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
        grid_thetas : ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return np.cos(grid_thetas), np.sin(grid_thetas)

    def grid_to_grid_cartesian(self, grid, radius):
        """
        Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian
        coordinates.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the profile.
        radius : ndarray
            The circular radius of each coordinate from the profile center.
        """
        grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
        cos_theta, sin_theta = self.grid_angle_to_profile(grid_thetas=grid_thetas)
        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    @grids.grid_like_to_structure
    def ndarray_1d_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return np.exp(
            np.multiply(
                -self.sersic_constant,
                np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
            )
        )

    @grids.grid_like_to_structure
    def ndarray_2d_from_grid(self, grid):
        return self.grid_to_grid_cartesian(
            grid=grid, radius=np.full(grid.shape[0], 2.0)
        )

    @grids.grid_like_to_structure_list
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

    @grids.grid_like_to_structure_list
    def ndarray_2d_list_from_grid(self, grid):
        return [
            self.grid_to_grid_cartesian(grid=grid, radius=np.full(grid.shape[0], 2.0))
        ]


class MockGridLikeObj:
    def __init__(self):
        pass

    @grids.grid_like_to_structure
    def ndarray_1d_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    @grids.grid_like_to_structure
    def ndarray_2d_from_grid(self, grid):
        return np.multiply(2.0, grid)

    @grids.grid_like_to_structure_list
    def ndarray_1d_list_from_grid(self, grid):
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @grids.grid_like_to_structure_list
    def ndarray_2d_list_from_grid(self, grid):
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGridRadialMinimum:
    def __init__(self):
        pass

    def grid_to_grid_radii(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        return grid
