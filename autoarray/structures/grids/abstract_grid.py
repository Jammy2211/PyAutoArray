import numpy as np


def convert_grid(grid):

    if type(grid) is list:
        grid = np.asarray(grid)

    return grid
