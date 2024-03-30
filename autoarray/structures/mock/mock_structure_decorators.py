import numpy as np

from autoarray.structures import structure_decorators


### Grids ###


def radial_grid_from(grid):
    return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))


def ndarray_1d_from(profile, grid):
    sersic_constant = (
        (2 * 2.0)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * 2.0))
        + (46.0 / (25515.0 * 2.0**2))
        + (131.0 / (1148175.0 * 2.0**3))
        - (2194697.0 / (30690717750.0 * 2.0**4))
    )

    grid_radii = radial_grid_from(grid=grid)

    return np.exp(
        np.multiply(
            -sersic_constant,
            np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
        )
    )


def angle_to_profile_grid_from(grid_angles):
    """The angle between each (y,x) coordinate on the grid and the profile, in radians.

    Parameters
    ----------
    grid_angles
        The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
    """
    return np.cos(grid_angles), np.sin(grid_angles)


def _cartesian_grid_via_radial_from(grid, radius):
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
    grid_angles = np.arctan2(grid[:, 0], grid[:, 1])
    cos_theta, sin_theta = angle_to_profile_grid_from(grid_angles=grid_angles)
    return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)


def ndarray_2d_from(profile, grid):
    return _cartesian_grid_via_radial_from(
        grid=grid, radius=np.full(grid.shape[0], 2.0)
    )


class MockGridLikeIteratorObj:
    def __init__(self):
        pass

    @property
    def sersic_constant(self):
        return (
            (2 * 2.0)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * 2.0))
            + (46.0 / (25515.0 * 2.0**2))
            + (131.0 / (1148175.0 * 2.0**3))
            - (2194697.0 / (30690717750.0 * 2.0**4))
        )

    def radial_grid_from(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    def angle_to_profile_grid_from(self, grid_angles):
        """The angle between each (y,x) coordinate on the grid and the profile, in radians.

        Parameters
        ----------
        grid_angles
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return np.cos(grid_angles), np.sin(grid_angles)

    def _cartesian_grid_via_radial_from(self, grid, radius):
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
        grid_angles = np.arctan2(grid[:, 0], grid[:, 1])
        cos_theta, sin_theta = self.angle_to_profile_grid_from(grid_angles=grid_angles)
        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    @structure_decorators.grid_2d_to_structure
    def ndarray_1d_from(self, grid) -> np.ndarray:
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a 1D ndarray
        of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        grid_radii = self.radial_grid_from(grid=grid)
        return np.exp(
            np.multiply(
                -self.sersic_constant,
                np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
            )
        )

    @structure_decorators.grid_2d_to_structure
    def ndarray_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=np.full(grid.shape[0], 2.0)
        )

    @structure_decorators.grid_2d_to_vector_yx
    @structure_decorators.grid_2d_to_structure
    def ndarray_yx_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=np.full(grid.shape[0], 2.0)
        )

    @structure_decorators.grid_2d_to_structure_list
    def ndarray_1d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a list of 1D
        ndarrays of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        grid_radii = self.radial_grid_from(grid=grid)
        return [
            np.exp(
                np.multiply(
                    -self.sersic_constant,
                    np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
                )
            )
        ]

    @structure_decorators.grid_2d_to_structure_list
    def ndarray_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D list of
        ndarrays of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [
            self._cartesian_grid_via_radial_from(
                grid=grid, radius=np.full(grid.shape[0], 2.0)
            )
        ]

    @structure_decorators.grid_2d_to_vector_yx_list
    @structure_decorators.grid_2d_to_structure_list
    def ndarray_yx_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a list of 2D
        ndarrays of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [
            self._cartesian_grid_via_radial_from(
                grid=grid, radius=np.full(grid.shape[0], 2.0)
            )
        ]

    @structure_decorators.grid_2d_to_structure_over_sample
    def ndarray_1d_over_sample_from(self, grid) -> np.ndarray:
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a 1D ndarray
        of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        grid_radii = self.radial_grid_from(grid=grid)
        return np.exp(
            np.multiply(
                -self.sersic_constant,
                np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
            )
        )

    @structure_decorators.grid_2d_to_structure_over_sample_list
    def ndarray_1d_over_sample_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a list of 1D
        ndarrays of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        grid_radii = self.radial_grid_from(grid=grid)
        return [
            np.exp(
                np.multiply(
                    -self.sersic_constant,
                    np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
                )
            )
        ]


class MockGrid1DLikeObj:
    def __init__(self, centre=(0.0, 0.0), angle=0.0):
        self.centre = centre
        self.angle = angle

    @structure_decorators.grid_1d_to_structure
    def ndarray_1d_from(self, grid):
        return np.ones(shape=grid.shape[0])

    # @structure_decorators.grid_1d_to_structure
    # def ndarray_2d_from(self, grid):
    #     return np.multiply(2.0, grid)

    # @structure_decorators.grid_1d_to_structure_list
    # def ndarray_1d_list_from(self, grid):
    #     return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]
    #
    # @structure_decorators.grid_1d_to_structure_list
    # def ndarray_2d_list_from(self, grid):
    #     return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGrid2DLikeObj:
    def __init__(self):
        pass

    @structure_decorators.grid_2d_to_structure
    def ndarray_1d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a 1D ndarray
        of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return np.ones(shape=grid.shape[0])

    @structure_decorators.grid_2d_to_structure
    def ndarray_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return np.multiply(2.0, grid)

    @structure_decorators.grid_2d_to_vector_yx
    @structure_decorators.grid_2d_to_structure
    def ndarray_yx_2d_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return 2.0 * grid

    @structure_decorators.grid_2d_to_structure_list
    def ndarray_1d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a list of 1D
        ndarrays of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @structure_decorators.grid_2d_to_structure_list
    def ndarray_2d_list_from(self, grid):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D list of
        ndarrays of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]

    @structure_decorators.grid_2d_to_vector_yx_list
    @structure_decorators.grid_2d_to_structure_list
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

    def radial_grid_from(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @structure_decorators.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid):
        return grid
