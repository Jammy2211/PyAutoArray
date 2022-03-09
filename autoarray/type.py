import numpy as np
from typing import List, Tuple, Union

PixelScales = Union[Tuple[float, float], float]


from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D

Mask1D2DLike = Union[Mask1D, Mask2D]


from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D

Array1D2DLike = Union[Array1D, Array2D]


# from autoarray.structures.grids.one_d import grid_1d as g1d
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_iterate import Grid2DIterate
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular

Grid1D2DLike = Union[np.ndarray, "Grid1D", Grid2D, Grid2DIterate, Grid2DIrregular]
Grid2DLike = Union[np.ndarray, Grid2D, Grid2DIterate, Grid2DIrregular]

from autoarray.structures.arrays.values import ValuesIrregular
from autoarray.dataset.interferometer import Visibilities
from autoarray.dataset.interferometer import VisibilitiesNoiseMap

DataLike = Union[Array1D, Array2D, ValuesIrregular, Visibilities]
NoiseMapLike = Union[Array1D, Array2D, ValuesIrregular, VisibilitiesNoiseMap]

from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT

Transformer = Union[TransformerDFT, TransformerNUFFT]


from autoarray.layout.region import Region1D
from autoarray.layout.region import Region2D

Region1DLike = Union[Region1D, Tuple[int, int]]
Region1DList = Union[List[Region1D], List[Tuple[int, int]]]
Region2DLike = Union[Region2D, Tuple[int, int, int, int]]
Region2DList = Union[List[Region2D], List[Tuple[int, int, int, int]]]
