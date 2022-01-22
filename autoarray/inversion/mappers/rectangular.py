import numpy as np
from typing import Dict, Optional, Tuple

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.inversion.mappers.abstract import PixSubWeights
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D

from autoarray.numba_util import profile_func
from autoarray.structures.grids.two_d import grid_2d_util


class MapperRectangularNoInterp(AbstractMapper):
    def __init__(
        self,
        source_grid_slim: Grid2D,
        source_pixelization_grid,
        data_pixelization_grid: Grid2D = None,
        hyper_image: Array2D = None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        To understand a `Mapper` one must be familiar `Pixelization` objects and the `pixelization` package, where
        the following four grids are explained: `data_grid_slim`, `source_grid_slim`, `data_pixelization_grid` and
        `source_pixelization_grid`. If you are not familiar with these grids, read the docstrings of the
        `pixelization` package first.

        A `Mapper` determines the mappings between the masked data grid's pixels (`data_grid_slim` and
        `source_grid_slim`) and the pxelization's pixels (`data_pixelization_grid` and `source_pixelization_grid`).

        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_grid_slim[0]` corresponds to the transformed value
        of `data_grid_slim[0]` and so on).

        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `pixelization_grid`,
        noting that associations are made by pairing `source_pixelization_grid` with `source_grid_slim`.

        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `pixelization_grid` maps to the index of a pixel on the `grid_slim` as follows:

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the pixelization's 1st pixel.
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel maps to the pixelization's 4th pixel.
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel maps to the pixelization's 2nd pixel.

        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `pixelization_grid`. For example, using a
        `Delaunay` pixelization, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles):

        For a `Rectangular` pixelization every pixel in the masked data maps to only one pixel, thus the second
        dimension of `pix_indexes_for_sub_slim_index` is always of size 1.

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a pixelization. This matrix is the basis of performing an `Inversion`,
        which reconstructs the data using the `source_pixelization_grid`.

        Parameters
        ----------
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_pixelization_grid
            The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
        data_pixelization_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_pixelization_grid`.
        hyper_image
            An image which is used to determine the `data_pixelization_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        super().__init__(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @property
    def shape_native(self) -> Tuple[int, int]:
        return self.source_pixelization_grid.shape_native

    @cached_property
    @profile_func
    def pix_sub_weights(self) -> "PixSubWeights":
        """
        Returns arrays describing the mappings between of every sub-pixel in the masked data and pixel in
        the `Rectangular` pixelization.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Rectangular
        pixelization's third (index 2) pixel.
        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the Rectangular
        pixelization's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one pixelization pixel (for example a `Delaunay` pixelization
        where each data pixel maps to 3 Delaunay triangles with interpolation). For a Rectangular pixelizaiton each
        data sub-pixel maps to a single pixelization pixel, thus this dimension is of size 1.

        For the Rectangular pixelization these mappings are calculated using the uniform properties of the grid
        (see `grid_2d_util.grid_pixel_indexes_2d_slim_from`).

        Returns an arrays describing the weights of the mappings between of every sub-pixel in the masked data and
        pixel in the pixelization. Weights are a result of the mappings between data sub-pixels and pixelization
        pixels using interpolation.

        The `Rectangular` pixelization does not use interpolation therefore all weights are 1.0.

        The weights are used when creating the `mapping_matrix` and `pixel_signals_from`.
        """
        mappings = grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=self.source_grid_slim,
            shape_native=self.source_pixelization_grid.shape_native,
            pixel_scales=self.source_pixelization_grid.pixel_scales,
            origin=self.source_pixelization_grid.origin,
        ).astype("int")

        mappings = mappings.reshape((len(mappings), 1))

        return PixSubWeights(
            mappings=mappings.reshape((len(mappings), 1)),
            sizes=np.ones(len(mappings), dtype="int"),
            weights=np.ones((len(self.source_grid_slim), 1), dtype="int"),
        )
