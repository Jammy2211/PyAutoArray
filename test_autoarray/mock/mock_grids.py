import numpy as np

from autoarray.structures import grids


class MockGridIteratorObj:
    def __init__(self):
        pass

    @property
    def sersic_constant(self):
        """ A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (
            (2 * 2.0)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * 2.0))
            + (46.0 / (25515.0 * 2.0 ** 2))
            + (131.0 / (1148175.0 * 2.0 ** 3))
            - (2194697.0 / (30690717750.0 * 2.0 ** 4))
        )

    def grid_to_grid_radii(self, grid):
        """Convert a grid of (y, x) coordinates to a grid of their circular radii.

        If the coordinates have not been transformed to the profile's centre, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the profile.
        """
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @grids.grid_like_to_structure
    def float_values_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return np.exp(
            np.multiply(
                -self.sersic_constant,
                np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
            )
        )

    # @grids.grid_like_to_structure
    # def tuple_values_from_grid(self, grid):
    #     return np.multiply(2.0, grid)
    #
    # @grids.grid_like_to_structure
    # def float_values_from_grid_returns_list(self, grid):
    #     return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]
    #
    # @grids.grid_like_to_structure
    # def tuple_values_from_grid_returns_list(self, grid):
    #     return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGridCoordinateInput:
    def __init__(self):
        pass

    @grids.grid_like_to_structure
    def float_values_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    @grids.grid_like_to_structure
    def tuple_values_from_grid(self, grid):
        return np.multiply(2.0, grid)

    @grids.grid_like_to_structure
    def float_values_from_grid_returns_list(self, grid):
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @grids.grid_like_to_structure
    def tuple_values_from_grid_returns_list(self, grid):
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGridRadialMinimum:
    def __init__(self):
        pass

    def grid_to_grid_radii(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        return grid
