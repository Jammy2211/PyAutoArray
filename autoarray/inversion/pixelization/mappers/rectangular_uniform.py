import jax.numpy as jnp

from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.rectangular import MapperRectangular
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights

from autoarray.inversion.pixelization.mappers import mapper_util


class MapperRectangularUniform(MapperRectangular):
    """
    To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` and `pixelization` packages, where
    the four grids grouped in a `MapperGrids` object are explained (`image_plane_data_grid`, `source_plane_data_grid`,
    `image_plane_mesh_grid`,`source_plane_mesh_grid`)

    If you are unfamliar withe above objects, read through the docstrings of the `pixelization`, `mesh` and
    `mapper_grids` packages.

    A `Mapper` determines the mappings between the masked data grid's pixels (`image_plane_data_grid` and
    `source_plane_data_grid`) and the mesh's pixels (`image_plane_mesh_grid` and `source_plane_mesh_grid`).

    The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
    change the indexing, such that `source_plane_data_grid[0]` corresponds to the transformed value
    of `image_plane_data_grid[0]` and so on).

    A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
    noting that associations are made by pairing `source_plane_mesh_grid` with `source_plane_data_grid`.

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
    which reconstructs the data using the `source_plane_mesh_grid`.

    Parameters
    ----------
    mapper_grids
        An object containing the data grid and mesh grid in both the data-frame and source-frame used by the
        mapper to map data-points to linear object parameters.
    regularization
        The regularization scheme which may be applied to this linear object in order to smooth its solution,
        which for a mapper smooths neighboring pixels on the mesh.
    """

    @cached_property
    def pix_sub_weights(self) -> PixSubWeights:
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

        For a Rectangular pixelization each data sub-pixel maps to a single mesh pixel, thus the second
        dimension of the array `pix_indexes_for_sub_slim_index` 1 and all entries in `pix_weights_for_sub_slim_index`
        are equal to 1.0.
        """

        mappings, weights = (
            mapper_util.rectangular_mappings_weights_via_interpolation_from(
                shape_native=self.shape_native,
                source_plane_mesh_grid=self.source_plane_mesh_grid.array,
                source_plane_data_grid=self.source_plane_data_grid.over_sampled,
            )
        )

        return PixSubWeights(
            mappings=mappings,
            sizes=4 * jnp.ones(len(mappings), dtype="int"),
            weights=weights,
        )