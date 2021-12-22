from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.inversion.mappers.abstract import PixForSub

from autoarray.numba_util import profile_func
from autoarray.inversion.mappers import mapper_util


class MapperDelaunay(AbstractMapper):
    def __init__(
        self,
        source_grid_slim,
        source_pixelization_grid,
        data_pixelization_grid=None,
        hyper_image=None,
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

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the pixelization's 1st pixel.
        - pix_indexes_for_sub_slim_index[0, 1] = 3: the data's 1st sub-pixel also maps to the pixelization's 4th pixel.
        - pix_indexes_for_sub_slim_index[0, 2] = 5: the data's 1st sub-pixel also maps to the pixelization's 6th pixel.

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

    @cached_property
    @profile_func
    def pix_indexes_for_sub_slim_index(self):
        """
        The indexes mappings between the sub pixels and Voronoi pixelization pixels.
        For Delaunay tessellation, most sub pixels should have contribution of 3 pixelization pixels. However,
        for those ones not belonging to any triangle, we link its value to its closest point.

        The returning result is a matrix of (len(sub_pixels, 3)) where the entries mark the relevant source pixel indexes.
        A row like [A, -1, -1] means that sub pixel only links to source pixel A.
        """
        delaunay = self.delaunay

        simplex_index_for_sub_slim_index = delaunay.find_simplex(self.source_grid_slim)
        pix_indexes_for_simplex_index = delaunay.simplices

        mappings, sizes = mapper_util.pix_indexes_for_sub_slim_index_delaunay_from(
            source_grid_slim=self.source_grid_slim,
            simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
            pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
            delaunay_points=delaunay.points,
        )

        return PixForSub(mappings=mappings.astype("int"), sizes=sizes.astype("int"))

    @cached_property
    @profile_func
    def pix_weights_for_sub_slim_index(self):
        """
        Weights for source pixels to sub pixels. Used for creating the mapping matrix and 'pixel_signals_from'
        It has the same shape as the 'pix_indexes_for_sub_slim_index'.
        """
        return mapper_util.pixel_weights_from(
            source_grid_slim=self.source_grid_slim,
            source_pixelization_grid=self.source_pixelization_grid,
            slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index.mappings,
        )

    @property
    def delaunay(self):
        return self.source_pixelization_grid.Delaunay

    def reconstruction_from(self, solution_vector):
        return solution_vector
