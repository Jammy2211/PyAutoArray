import numpy as np
from typing import Dict, Optional, Tuple

from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.numba_util import profile_func
from autoarray.structures.grids import grid_2d_util


class MapperRectangularNoInterp(AbstractMapper):
    def __init__(
        self,
        mapper_grids: MapperGrids,
        regularization: Optional[AbstractRegularization],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` package, where
        the following four grids are explained: `data_grid_slim`, `source_grid_slim`, `data_mesh_grid` and
        `source_mesh_grid`. If you are not familiar with these grids, read the docstrings of the
        `mesh` package first.

        A `Mapper` determines the mappings between the masked data grid's pixels (`data_grid_slim` and
        `source_grid_slim`) and the mesh's pixels (`data_mesh_grid` and `source_mesh_grid`).

        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_grid_slim[0]` corresponds to the transformed value
        of `data_grid_slim[0]` and so on).

        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
        noting that associations are made by pairing `source_mesh_grid` with `source_grid_slim`.

        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `mesh_grid` maps to the index of a pixel on the `grid_slim` as follows:

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the mesh's 1st pixel.
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel maps to the mesh's 4th pixel.
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel maps to the mesh's 2nd pixel.

        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, a
        `Delaunay` triangulation, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles) with varying interpolation weights .

        For a `Rectangular` mesh every pixel in the masked data maps to only one pixel, thus the second
        dimension of `pix_indexes_for_sub_slim_index` is always of size 1.

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a mesh. This matrix is the basis of performing an `Inversion`,
        which reconstructs the data using the `source_mesh_grid`.

        Parameters
        ----------
        pixelization
            The pixelization object containing this mapper's mesh and regularization.
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_mesh_grid
            The 2D grid of (y,x) centres of every mesh pixel in the `source` frame.
        data_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_mesh_grid`.
        hyper_data
            An image which is used to determine the `data_mesh_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        super().__init__(
            mapper_grids=mapper_grids,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )

    @property
    def shape_native(self) -> Tuple[int, int]:
        return self.source_mesh_grid.shape_native

    @cached_property
    @profile_func
    def pix_sub_weights(self) -> "PixSubWeights":
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Rectangular` mesh.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to mesh pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to mesh pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's mesh
        pixel mapping

        These are packaged into the class `PixSubWeights` with attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the mesh pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Rectangular
        mesh's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the Rectangular
        mesh's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one mesh pixel (for example a `Delaunay` triangulation
        where each data pixel maps to 3 Delaunay triangles with interpolation weights). The weights of multiple mappings
        are stored in the array `pix_weights_for_sub_slim_index`.

        For a Rectangular pixelizaiton each data sub-pixel maps to a single mesh pixel, thus the second
        dimension of the array `pix_indexes_for_sub_slim_index` 1 and all entries in `pix_weights_for_sub_slim_index`
        are equal to 1.0.
        """
        mappings = grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=self.source_grid_slim,
            shape_native=self.source_mesh_grid.shape_native,
            pixel_scales=self.source_mesh_grid.pixel_scales,
            origin=self.source_mesh_grid.origin,
        ).astype("int")

        mappings = mappings.reshape((len(mappings), 1))

        return PixSubWeights(
            mappings=mappings.reshape((len(mappings), 1)),
            sizes=np.ones(len(mappings), dtype="int"),
            weights=np.ones((len(self.source_grid_slim), 1), dtype="int"),
        )
