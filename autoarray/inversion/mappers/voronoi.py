import numpy as np
from typing import Dict, Optional, Tuple

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.inversion.mappers.abstract import PixSubWeights
from autoarray.structures.two_d.array_2d import Array2D
from autoarray.structures.two_d.grids.grid_2d import Grid2D
from autoarray.structures.two_d.grids.sparse import Grid2DSparse

from autoarray.numba_util import profile_func
from autoarray.inversion.mappers import mapper_util


class AbstractMapperVoronoi(AbstractMapper):
    def __init__(
        self,
        source_grid_slim: Grid2D,
        source_pixelization_grid,
        data_pixelization_grid: Grid2DSparse = None,
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
        `.dDelaunay` pixelization, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles):

        For a `Voronoi` pixelization every pixel in the masked data maps to only one Voronoi pixel, thus the second
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
    def voronoi(self):
        return self.source_pixelization_grid.voronoi

    @property
    def pix_sub_weights_split_cross(self) -> PixSubWeights:
        (mappings, sizes, weights) = mapper_util.pix_size_weights_voronoi_nn_from(
            grid=self.source_pixelization_grid.split_cross,
            pixelization_grid=self.source_pixelization_grid,
        )

        return PixSubWeights(mappings=mappings, sizes=sizes, weights=weights)


class MapperVoronoi(AbstractMapperVoronoi):
    @cached_property
    @profile_func
    def pix_sub_weights(self) -> "PixSubWeights":
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Voronoi` pixelization.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to pixelization pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to pixelization pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's pixelization
        pixel mapping

        These are packaged into the class `PixSubWeights` with attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Rectangular
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the Rectangular
        pixelization's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one pixelization pixel.

        For a this Voronoi pixelization a natural neighbor interpolation scheme is used to map each data pixel many
        Voronoi pixels, for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the natural
        neighbor of the Voronoi pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[0, 1] = 5`: The data's first (index 0) sub-pixel also maps to the natural
        neighbor of the Voronoi pixelization's sixth (index 5) pixel.

        - `pix_indexes_for_sub_slim_index[0, 2] = 8`: The data's first (index 0) sub-pixel also maps to the natural
        neighbor of the Voronoi pixelization's ninth (index 8) pixel.

        The interpolation weights of these multiple mappings are stored in the array `pix_weights_for_sub_slim_index`.
        """

        mappings, sizes, weights = mapper_util.pix_size_weights_voronoi_nn_from(
            grid=self.source_grid_slim, pixelization_grid=self.source_pixelization_grid
        )

        mappings = mappings.astype("int")
        sizes = sizes.astype("int")

        return PixSubWeights(mappings=mappings, sizes=sizes, weights=weights)

    def interpolated_array_from(
        self, values: np.ndarray, shape_native: Tuple[int, int] = (401, 401)
    ) -> Array2D:
        return self.source_pixelization_grid.interpolated_array_from(
            values=values, shape_native=shape_native, use_nn=True
        )


class MapperVoronoiNoInterp(AbstractMapperVoronoi):
    @cached_property
    @profile_func
    def pix_sub_weights(self) -> "PixSubWeights":
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Voronoi` pixelization.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to pixelization pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to pixelization pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's pixelization
        pixel mapping

        These are packaged into the class `PixSubWeights` with attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Rectangular
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the Rectangular
        pixelization's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one pixelization pixel (for example a `Delaunay` triangulation
        where each data pixel maps to 3 Delaunay triangles with interpolation weights). The weights of multiple mappings
        are stored in the array `pix_weights_for_sub_slim_index`.

        For this Voronoi pixelizaiton each data sub-pixel maps to a single pixelization pixel, thus the second
        dimension of the array `pix_indexes_for_sub_slim_index` 1 and all entries in `pix_weights_for_sub_slim_index`
        are equal to 1.0.

        The weights are used when creating the `mapping_matrix` and `pixel_signals_from`.
        """
        mappings = mapper_util.pix_indexes_for_sub_slim_index_voronoi_from(
            grid=self.source_grid_slim,
            nearest_pix_index_for_slim_index=self.source_pixelization_grid.nearest_pixelization_index_for_slim_index,
            slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
            pixelization_grid=self.source_pixelization_grid,
            pixel_neighbors=self.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=self.source_pixelization_grid.pixel_neighbors.sizes,
        ).astype("int")

        return PixSubWeights(
            mappings=mappings,
            sizes=np.ones(self.source_grid_slim.shape[0], dtype="int"),
            weights=np.ones((self.source_grid_slim.shape[0], 1), dtype="int"),
        )
