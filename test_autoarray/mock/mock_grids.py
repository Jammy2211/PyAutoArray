import numpy as np

from autoarray.structures import grids


class MockGridIteratorObj:
    def __init__(self):
        pass

    @grids.grid_like_to_numpy
    def float_values_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    # @grids.grid_like_to_numpy
    # def tuple_values_from_grid(self, grid):
    #     return np.multiply(2.0, grid)
    #
    # @grids.grid_like_to_numpy
    # def float_values_from_grid_returns_list(self, grid):
    #     return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]
    #
    # @grids.grid_like_to_numpy
    # def tuple_values_from_grid_returns_list(self, grid):
    #     return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class MockGridCoordinateInput:
    def __init__(self):
        pass

    @grids.grid_like_to_numpy
    def float_values_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    @grids.grid_like_to_numpy
    def tuple_values_from_grid(self, grid):
        return np.multiply(2.0, grid)

    @grids.grid_like_to_numpy
    def float_values_from_grid_returns_list(self, grid):
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @grids.grid_like_to_numpy
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
