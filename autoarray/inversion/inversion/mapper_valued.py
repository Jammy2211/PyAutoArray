import numpy as np
from typing import List, Optional, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class MapperValued:
    def __init__(self, mapper, values, mesh_pixel_mask: Optional[np.ndarray] = None):
        """
        Pairs a `Mapper` object with an array of values (e.g. the `reconstruction` values of each value of each
        mapper pixel) in order to perform calculations which use both the `Mapper` and these values.

        For example, a common use case is to interpolate the reconstruction of values on a mapper from the
        mesh of the mapper (e.g. a Delaunay mesh) to a uniform Cartesian grid of values, because the irregular mesh
        is difficult to plot and analyze.

        This class also provides functionality to compute the magnification of the reconstruction, by comparing the
        sum of the values on the mapper in both the image and source planes, which is a specific quantity
        used in gravitational lensing.

        Parameters
        ----------
        mapper
            The `Mapper` object which pairs with the values, for example a `MapperDelaunay` object.
        values
            The values of each pixel of the mapper, which could be the `reconstruction` values of an `Inversion`,
            but alternatively could be other quantities such as the noise-map of these values.
        mesh_pixel_mask
            The mask of pixels that are omitted from the reconstruction when computing the image, for example to
            remove pixels with low signal-to-noise so they do not impact the magnification calculation.

        """
        self.mapper = mapper
        self.values = values
        self.mesh_pixel_mask = mesh_pixel_mask


