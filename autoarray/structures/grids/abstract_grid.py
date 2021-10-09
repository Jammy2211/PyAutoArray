import numpy as np
from typing import Union, List


def convert_grid(grid: Union[np.ndarray, List]) -> np.ndarray:

    if type(grid) is list:
        grid = np.asarray(grid)

    return grid
