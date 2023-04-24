import itertools
import numpy as np
from typing import Dict, List, Optional, Tuple

from autoconf import conf
from autoconf import cached_property

from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.func_list import UniqueMappings
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh

from autoarray.numba_util import profile_func
from autoarray.inversion.pixelization.mappers import mapper_util


class AbstractMapper(LinearObj):
    def __init__(
        self,
        mapper_grids: MapperGrids,
        regularization: Optional[AbstractRegularization],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` and `pixelization` packages, where
        the four grids grouped in a `MapperGrids` object are explained (`image_plane_data_grid`, `source_plane_data_grid`,
        `image_plane_mesh_grid`,`source_plane_mesh_grid`)

        If you are unfamliar withe above objects, read through the docstrings of the `pixelization`, `mesh` and
        `mapper_grids` packages.

        A `Mapper` determines the mappings between the masked data grid's pixels (`image_plane_data_grid` and
        `source_plane_data_grid`) and the pxelization's pixels (`image_plane_mesh_grid` and `source_plane_mesh_grid`).

        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_plane_data_grid[0]` corresponds to the transformed value
        of `image_plane_data_grid[0]` and so on).

        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
        noting that associations are made by pairing `source_plane_mesh_grid` with `source_plane_data_grid`.

        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `mesh_grid` maps to the index of a pixel on the `grid_slim` as follows:

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel (index 0) maps to the
          pixelization's 1st pixel (index 0).
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel (index 1) maps to the
          pixelization's 4th pixel (index 3).
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel (index 2) maps to the
          pixelization's 2nd pixel (index 1).

        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, using a
        `Delaunay` pixelization, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles):

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel (index 0) maps to the
        pixelization's 1st pixel (index 0).
        - pix_indexes_for_sub_slim_index[0, 1] = 3: the data's 1st sub-pixel (index 0) also maps to the
        pixelization's 2nd pixel (index 3).
        - pix_indexes_for_sub_slim_index[0, 2] = 5: the data's 1st sub-pixel (index 0) also maps to the
        pixelization's 6th pixel (index 5).

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a pixelization. This matrix is the basis of performing an `Inversion`,
        which reconstructs the data using the `source_plane_mesh_grid`.

        Parameters
        ----------
        mapper_grids
            An object containing the data grid and mesh grid in both the data-frame and source-frame used by the
            mapper to map data-points to linear object parameters.
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution,
            which for a mapper smooths neighboring pixels on the mesh.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        super().__init__(regularization=regularization, profiling_dict=profiling_dict)

        self.mapper_grids = mapper_grids

    @property
    def params(self) -> int:
        return self.source_plane_mesh_grid.pixels

    @property
    def pixels(self) -> int:
        return self.params

    @property
    def source_plane_data_grid(self) -> Grid2D:
        return self.mapper_grids.source_plane_data_grid

    @property
    def source_plane_mesh_grid(self) -> Abstract2DMesh:
        return self.mapper_grids.source_plane_mesh_grid

    @property
    def image_plane_mesh_grid(self) -> Grid2D:
        return self.mapper_grids.image_plane_mesh_grid

    @property
    def edge_pixel_list(self) -> List[int]:
        return self.source_plane_mesh_grid.edge_pixel_list

    @property
    def hyper_data(self) -> np.ndarray:
        return self.mapper_grids.hyper_data

    @property
    def neighbors(self) -> Neighbors:
        return self.source_plane_mesh_grid.neighbors

    @property
    def pix_sub_weights(self) -> "PixSubWeights":
        raise NotImplementedError

    @cached_property
    def pix_indexes_for_sub_slim_index(self) -> np.ndarray:
        """
        The mapping of every data pixel (given its `sub_slim_index`) to pixelization pixels (given their `pix_indexes`).

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the pixelization's
        third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the pixelization's
        fifth (index 4) pixel.
        """
        return self.pix_sub_weights.mappings

    @cached_property
    def pix_sizes_for_sub_slim_index(self) -> np.ndarray:
        """
        The number of mappings of every data pixel to pixelization pixels.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_sizes_for_sub_slim_index[0] = 2`: The data's first (index 0) sub-pixel maps to 2 pixels in the
        pixelization.

        - `pix_sizes_for_sub_slim_index[2] = 4`: The data's third (index 2) sub-pixel maps to 4 pixels in the
        pixelization.
        """
        return self.pix_sub_weights.sizes

    @cached_property
    def pix_weights_for_sub_slim_index(self) -> np.ndarray:
        """
        The interoplation weights of the mapping of every data pixel (given its `sub_slim_index`) to pixelization
        pixels (given their `pix_indexes`).

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_weights_for_sub_slim_index[0, 0] = 0.1`: The data's first (index 0) sub-pixel mapping to the
        pixelization's third (index 2) pixel has an interpolation weight of 0.1.

        - `pix_weights_for_sub_slim_index[2, 0] = 0.2`: The data's third (index 2) sub-pixel mapping to the
        pixelization's fifth (index 4) pixel has an interpolation weight of 0.2.
        """
        return self.pix_sub_weights.weights

    @property
    def slim_index_for_sub_slim_index(self) -> np.ndarray:
        """
        The mappings between every sub-pixel data point on the sub-gridded data and each data point for a grid which
        does not use sub gridding (e.g. `sub_size=1`).
        """
        return self.source_plane_data_grid.mask.derive_indexes.slim_for_sub_slim

    @property
    def sub_slim_indexes_for_pix_index(self) -> List[List]:
        """
        Returns the index mappings between each of the pixelization's pixels and the masked data's sub-pixels.

        Given that even pixelization pixel maps to multiple data sub-pixels, index mappings are returned as a list of
        lists where the first entries are the pixelization index and second entries store the data sub-pixel indexes.

        For example, if `sub_slim_indexes_for_pix_index[2][4] = 10`, the pixelization pixel with index 2
        (e.g. `mesh_grid[2,:]`) has a mapping to a data sub-pixel with index 10 (e.g. `grid_slim[10, :]).

        This is effectively a reversal of the array `pix_indexes_for_sub_slim_index`.
        """
        sub_slim_indexes_for_pix_index = [[] for _ in range(self.pixels)]

        pix_indexes_for_sub_slim_index = self.pix_indexes_for_sub_slim_index

        for slim_index, pix_indexes in enumerate(pix_indexes_for_sub_slim_index):
            for pix_index in pix_indexes:
                sub_slim_indexes_for_pix_index[int(pix_index)].append(slim_index)

        return sub_slim_indexes_for_pix_index

    @property
    @profile_func
    def sub_slim_indexes_for_pix_index_arr(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the index mappings between each of the pixelization's pixels and the masked data's sub-pixels.

        Given that even pixelization pixel maps to multiple data sub-pixels, index mappings are returned as a list of
        lists where the first entries are the pixelization index and second entries store the data sub-pixel indexes.

        For example, if `sub_slim_indexes_for_pix_index[2][4] = 10`, the pixelization pixel with index 2
        (e.g. `mesh_grid[2,:]`) has a mapping to a data sub-pixel with index 10 (e.g. `grid_slim[10, :]).

        This is effectively a reversal of the array `pix_indexes_for_sub_slim_index`.
        """

        return mapper_util.sub_slim_indexes_for_pix_index(
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=self.pix_weights_for_sub_slim_index,
            pix_pixels=self.pixels,
        )

    @cached_property
    @profile_func
    def unique_mappings(self) -> UniqueMappings:
        """
        Returns the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding pixelization pixels (e.g. `mesh_grid`).

        To perform an `Inversion` efficiently the linear algebra can bypass the calculation of a `mapping_matrix` and
        instead use the w-tilde formalism, which requires these unique mappings for efficient computation. For
        convenience, these mappings and associated metadata are packaged into the class `UniqueMappings`.

        A full description of these mappings is given in the
        function `mapper_util.data_slim_to_pixelization_unique_from()`.
        """

        (
            data_to_pix_unique,
            data_weights,
            pix_lengths,
        ) = mapper_util.data_slim_to_pixelization_unique_from(
            data_pixels=self.source_plane_data_grid.shape_slim,
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index=self.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=self.pix_weights_for_sub_slim_index,
            pix_pixels=self.params,
            sub_size=self.source_plane_data_grid.sub_size,
        )

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )

    @cached_property
    @profile_func
    def mapping_matrix(self) -> np.ndarray:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's data-points / pixels
        and the linear object parameters. It is used to construct the simultaneous linear equations which reconstruct
        the data.

        The matrix has shape [total_data_points, data_linear_object_parameters], whereby all non-zero entries
        indicate that a data point maps to a linear object parameter.

        It is described in the following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf and in more
        detail in the function  `mapper_util.mapping_matrix_from()`.
        """
        return mapper_util.mapping_matrix_from(
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=self.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=self.pix_weights_for_sub_slim_index,
            pixels=self.pixels,
            total_mask_pixels=self.source_plane_data_grid.mask.pixels_in_mask,
            slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
            sub_fraction=self.source_plane_data_grid.mask.sub_fraction,
        )

    def pixel_signals_from(self, signal_scale: float) -> np.ndarray:
        """
        Returns the (hyper) signal in each pixelization pixel, where this signal is an estimate of the expected signal
        each pixelization pixel contains given the data pixels it maps too.

        A full description of this is given in the function `mapper_util.adaptive_pixel_signals_from().

        Parameters
        ----------
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
            low signal regions.
        """
        return mapper_util.adaptive_pixel_signals_from(
            pixels=self.pixels,
            signal_scale=signal_scale,
            pixel_weights=self.pix_weights_for_sub_slim_index,
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=self.pix_sizes_for_sub_slim_index,
            slim_index_for_sub_slim_index=self.source_plane_data_grid.mask.derive_indexes.slim_for_sub_slim,
            hyper_data=self.hyper_data,
        )

    def pix_indexes_for_slim_indexes(self, pix_indexes: List) -> List[List]:
        """
        Returns the index mappings between every masked data-point (not subgridded) on the data and the mapper
        pixels / parameters that it maps too.

        The `slim_index` refers to the masked data pixels (without subgridding) and `pix_indexes` the pixelization
        pixel indexes, for example:

        - `pix_indexes_for_slim_indexes[0] = [2, 3]`: The data's first (index 0) pixel maps to the
        pixelization's third (index 2) and fourth (index 3) pixels.

        Parameters
        ----------
        pix_indexes
            A list of all pixelization indexes for which the data-points that map to these pixelization pixels are
            computed.
        """
        image_for_source = self.sub_slim_indexes_for_pix_index

        if not any(isinstance(i, list) for i in pix_indexes):
            return list(
                itertools.chain.from_iterable(
                    [image_for_source[index] for index in pix_indexes]
                )
            )
        else:
            indexes = []
            for source_pixel_index_list in pix_indexes:
                indexes.append(
                    list(
                        itertools.chain.from_iterable(
                            [
                                image_for_source[index]
                                for index in source_pixel_index_list
                            ]
                        )
                    )
                )
            return indexes

    def mapped_to_source_from(self, array: Array2D) -> np.ndarray:
        """
        Map a masked 2d image in the image domain to the source domain and sum up all mappings on the source-pixels.

        For example, suppose we have an image and a mapper. We can map every image-pixel to its corresponding mapper's
        source pixel and sum the values based on these mappings.

        This will produce something similar to a `reconstruction`, albeit it bypasses the linear algebra / inversion.

        Parameters
        ----------
        array_slim
            The masked 2D array of values in its slim representation (e.g. the image data) which are mapped to the
            source domain in order to compute their average values.
        """
        return mapper_util.mapped_to_source_via_mapping_matrix_from(
            mapping_matrix=self.mapping_matrix, array_slim=array.binned.slim
        )

    def extent_from(
        self, values: np.ndarray = None, zoom_to_brightest: bool = True
    ) -> Tuple[float, float, float, float]:

        if zoom_to_brightest and values is not None:

            zoom_percent = conf.instance["visualize"]["general"]["zoom"][
                "inversion_percent"
            ]

            fractional_value = np.max(values) * zoom_percent
            fractional_bool = values > fractional_value
            true_indices = np.argwhere(fractional_bool)
            true_grid = self.source_plane_mesh_grid[true_indices]

            from autoarray.geometry import geometry_util

            try:
                return geometry_util.extent_symmetric_from(
                    extent=(
                        np.min(true_grid[:, 0, 1]),
                        np.max(true_grid[:, 0, 1]),
                        np.min(true_grid[:, 0, 0]),
                        np.max(true_grid[:, 0, 0]),
                    )
                )
            except ValueError:
                return self.source_plane_mesh_grid.geometry.extent

        return self.source_plane_mesh_grid.geometry.extent

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
        return self.source_plane_mesh_grid.interpolated_array_from(
            values=values, shape_native=shape_native, extent=extent
        )


class PixSubWeights:
    def __init__(self, mappings: np.ndarray, sizes: np.ndarray, weights: np.ndarray):
        """
        Packages the mappings, sizes and weights of every data pixel to pixelization pixels, which are computed
        from associated ``Mapper`` properties..

        The need to store separately the mappings and sizes is so that the `sizes` can be easy iterated over when
        perform calculations for efficiency.

        Parameters
        ----------
        mappings
            The mapping of every data pixel, given its `sub_slim_index`, to its corresponding pixelization mesh
            pixels, given their `pix_indexes` (corresponds to the ``Mapper``
            property ``pix_indexes_for_sub_slim_index``)
        sizes
            The number of mappings of every data pixel to pixelization mesh pixels (corresponds to the ``Mapper``
            property ``pix_sizes_for_sub_slim_index``).
        weights
            The interpolation weights of every data pixel's pixelization pixel mapping.
        """
        self.mappings = mappings
        self.sizes = sizes
        self.weights = weights
