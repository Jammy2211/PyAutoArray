import numpy as np

from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights

from autoarray.numba_util import profile_func
from autoarray.inversion.pixelization.mappers import mapper_util


class MapperDelaunay(AbstractMapper):
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

    - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the pixelization's 1st pixel.
    - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel maps to the pixelization's 4th pixel.
    - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel maps to the pixelization's 2nd pixel.

    The second dimension of this array (where all three examples above are 0) is used for cases where a
    single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, using a
    `Delaunay` pixelization, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
    triangles):

    - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the pixelization's 1st pixel.
    - pix_indexes_for_sub_slim_index[0, 1] = 3: the data's 1st sub-pixel also maps to the pixelization's 4th pixel.
    - pix_indexes_for_sub_slim_index[0, 2] = 5: the data's 1st sub-pixel also maps to the pixelization's 6th pixel.

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
    run_time_dict
        A dictionary which contains timing of certain functions calls which is used for profiling.
    """

    @property
    def delaunay(self):
        return self.source_plane_mesh_grid.delaunay

    @cached_property
    @profile_func
    def pix_sub_weights(self) -> PixSubWeights:
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Delaunay` pixelization.

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

        For a `Delaunay` pixelization each data pixel maps to 3 Delaunay triangles with interpolation, for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Delaunay
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[0, 1] = 5`: The data's first (index 0) sub-pixel also maps to the Delaunay
        pixelization's sixth (index 5) pixel.

        - `pix_indexes_for_sub_slim_index[0, 2] = 8`: The data's first (index 0) sub-pixel also maps to the Delaunay
        pixelization's ninth (index 8) pixel.

        The interpolation weights of these multiple mappings are stored in the array `pix_weights_for_sub_slim_index`.

        For the Delaunay pixelization these mappings are calculated using the Scipy spatial library
        (see `mapper_util.pix_indexes_for_sub_slim_index_delaunay_from`).
        """
        delaunay = self.delaunay

        simplex_index_for_sub_slim_index = delaunay.find_simplex(
            self.source_plane_data_grid
        )
        pix_indexes_for_simplex_index = delaunay.simplices

        mappings, sizes = mapper_util.pix_indexes_for_sub_slim_index_delaunay_from(
            source_plane_data_grid=np.array(self.source_plane_data_grid),
            simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
            pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
            delaunay_points=delaunay.points,
        )

        mappings = mappings.astype("int")
        sizes = sizes.astype("int")

        weights = mapper_util.pixel_weights_delaunay_from(
            source_plane_data_grid=np.array(self.source_plane_data_grid),
            source_plane_mesh_grid=np.array(self.source_plane_mesh_grid),
            slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
            pix_indexes_for_sub_slim_index=mappings,
        )

        return PixSubWeights(mappings=mappings, sizes=sizes, weights=weights)

    @property
    def pix_sub_weights_split_cross(self) -> PixSubWeights:
        """
        The property `pix_sub_weights` property describes the calculation of the `PixSubWeights` object, which contains
        numpy arrays describing how data-points and mapper pixels map to one another and the weights of these mappings.

        For certain regularization schemes (e.g. `ConstantSplit`, `AdaptiveBrightnessSplit`) regularization uses
        mappings which are split in a cross configuration in order to factor in the derivative of the mapper
        reconstruction.

        This property returns a unique set of `PixSubWeights` used for these regularization schemes which compute
        mappings and weights at each point on the split cross.
        """
        delaunay = self.delaunay

        splitted_simplex_index_for_sub_slim_index = delaunay.find_simplex(
            self.source_plane_mesh_grid.split_cross
        )
        pix_indexes_for_simplex_index = delaunay.simplices

        (
            splitted_mappings,
            splitted_sizes,
        ) = mapper_util.pix_indexes_for_sub_slim_index_delaunay_from(
            source_plane_data_grid=self.source_plane_mesh_grid.split_cross,
            simplex_index_for_sub_slim_index=splitted_simplex_index_for_sub_slim_index,
            pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
            delaunay_points=delaunay.points,
        )

        splitted_weights = mapper_util.pixel_weights_delaunay_from(
            source_plane_data_grid=self.source_plane_mesh_grid.split_cross,
            source_plane_mesh_grid=np.array(self.source_plane_mesh_grid),
            slim_index_for_sub_slim_index=self.source_plane_mesh_grid.split_cross,
            pix_indexes_for_sub_slim_index=splitted_mappings.astype("int"),
        )

        append_line_int = np.zeros((len(splitted_weights), 1), dtype="int") - 1
        append_line_float = np.zeros((len(splitted_weights), 1), dtype="float")

        return PixSubWeights(
            mappings=np.hstack((splitted_mappings.astype("int"), append_line_int)),
            sizes=splitted_sizes.astype("int"),
            weights=np.hstack((splitted_weights, append_line_float)),
        )
