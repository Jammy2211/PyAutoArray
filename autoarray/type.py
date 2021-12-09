import numpy as np
from typing import Tuple, Union

PixelScales = Union[Tuple[float, float], float]

# from autoarray.structures.grids.one_d import grid_1d as g1d
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_iterate import Grid2DIterate
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular

Grid1D2DLike = Union[np.ndarray, "Grid1D", Grid2D, Grid2DIterate, Grid2DIrregular]
Grid2DLike = Union[np.ndarray, Grid2D, Grid2DIterate, Grid2DIrregular]

from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT

Transformer = Union[TransformerDFT, TransformerNUFFT]