import numpy as np
from typing import Dict, Optional, Tuple

from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.numba_util import profile_func
from autoarray.inversion.pixelization.mappers import mapper_util


class AbstractMapperVoronoi(AbstractMapper):
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
        `source_grid_slim`) and the pxelization's pixels (`data_mesh_grid` and `source_mesh_grid`).

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
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, using a
        `Delaunay` mesh, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles):

        For a `Voronoi` mesh every pixel in the masked data maps to only one Voronoi pixel, thus the second
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
    def voronoi(self):
        return self.source_mesh_grid.voronoi

    @property
    def pix_sub_weights_split_cross(self) -> PixSubWeights:
        (mappings, sizes, weights) = mapper_util.pix_size_weights_voronoi_nn_from(
            grid=self.source_mesh_grid.split_cross, mesh_grid=self.source_mesh_grid
        )

        return PixSubWeights(mappings=mappings, sizes=sizes, weights=weights)


class MapperVoronoi(AbstractMapperVoronoi):
    @cached_property
    @profile_func
    def pix_sub_weights(self) -> "PixSubWeights":
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Voronoi` mesh.

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
        for cases where a data pixel maps to more than one mesh pixel.

        For a this Voronoi mesh a natural neighbor interpolation scheme is used to map each data pixel many
        Voronoi pixels, for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the natural
        neighbor of the Voronoi mesh's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[0, 1] = 5`: The data's first (index 0) sub-pixel also maps to the natural
        neighbor of the Voronoi pixelization's sixth (index 5) pixel.

        - `pix_indexes_for_sub_slim_index[0, 2] = 8`: The data's first (index 0) sub-pixel also maps to the natural
        neighbor of the Voronoi pixelization's ninth (index 8) pixel.

        The interpolation weights of these multiple mappings are stored in the array `pix_weights_for_sub_slim_index`.
        """

        mappings, sizes, weights = mapper_util.pix_size_weights_voronoi_nn_from(
            grid=self.source_grid_slim, mesh_grid=self.source_mesh_grid
        )

        mappings = mappings.astype("int")
        sizes = sizes.astype("int")

        return PixSubWeights(mappings=mappings, sizes=sizes, weights=weights)

    def interpolated_array_from(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Array2D:
        """
        The reconstructed values of a mapper (e.g. the `reconstruction` of an `Inversion` may be on an irregular
        pixelization (e.g. a Delaunay triangulation, Voronoi mesh).

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to using
        this irregular grid.

        This function offers a simple alternative is therefore to interpolate the irregular reconstruction on to a
        regular grid of square pixels. The routine that performs the interpolation is specific to each pixelization
        and contained `Grid2DMesh` object, which are called by this function.

        The output interpolated reconstruction is by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        values
            The value corresponding to the reconstructed value of every pixelization pixel (e.g. Delaunay triangle
            vertexes, Voronoi mesh cells).
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        return self.source_mesh_grid.interpolated_array_from(
            values=values, shape_native=shape_native, extent=extent, use_nn=True
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
            nearest_pixelization_index_for_slim_index=self.source_mesh_grid.nearest_pixelization_index_for_slim_index,
            slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
            mesh_grid=self.source_mesh_grid,
            neighbors=self.source_mesh_grid.neighbors,
            neighbors_sizes=self.source_mesh_grid.neighbors.sizes,
        ).astype("int")

        return PixSubWeights(
            mappings=mappings,
            sizes=np.ones(self.source_grid_slim.shape[0], dtype="int"),
            weights=np.ones((self.source_grid_slim.shape[0], 1), dtype="int"),
        )
