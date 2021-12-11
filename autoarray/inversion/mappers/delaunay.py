from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper

from autoarray.numba_util import profile_func
from autoarray.inversion.mappers import mapper_util

import numpy as np

import time

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
        Class representing a Delaunay mapper, which maps unmasked pixels on a masked 2D array in the form of
        a grid, see the *hyper_galaxies.array.grid* module to pixels discretized on a Delaunay grid.

        The irand non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge
        of how different grid map to one another.

        Parameters
        ----------
        pixels
            The number of pixels in the Delaunay pixelization.
        source_grid_slim : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        delaunay : scipy.spatial.Delaunay
            Class storing the Delaunay grid's
        geometry : pixelization.Delaunay.Geometry
        hyper_image
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
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
    def pixelization_indexes_for_sub_slim_index(self):
        """
        The indexes mappings between the sub pixels and Voronoi pixelization pixels.
        For Delaunay tessellation, most sub pixels should have contribution of 3 pixelization pixels. However,
        for those ones not belonging to any triangle, we link its value to its closest point. 

        The returning result is a matrix of (len(sub_pixels, 3)) where the entries mark the relevant source pixel indexes.
        A row like [A, -1, -1] means that sub pixel only links to source pixel A.
        """

        simplex_index_for_sub_slim_index = self.delaunay.find_simplex(self.source_grid_slim)
        pixelization_indexes_for_simplex_index = self.delaunay.simplices

        tem_list = -1 * np.ones((len(self.source_grid_slim), 3), dtype='int')

        for i in range(len(self.source_grid_slim)):
            simplex_index = simplex_index_for_sub_slim_index[i]
            if simplex_index != -1:
                tem_list[i] = pixelization_indexes_for_simplex_index[simplex_index_for_sub_slim_index[i]]
            else:
                tem_list[i][0] = np.argmin(np.sum((self.delaunay.points - self.source_grid_slim[i])**2.0, axis=1))
                #print(tem_list[i])

        return tem_list


    @cached_property
    @profile_func
    def pixelization_weights_for_sub_slim_index(self):
        """
        Weights for source pixels to sub pixels. Used for creating the mapping matrix and 'pixel_signals_from'
        It has the same shape as the 'pixelization_indexes_for_sub_slim_index'.
        """
        return mapper_util.pixel_weights_from(
                source_grid_slim=self.source_grid_slim,
                source_pixelization_grid=self.source_pixelization_grid,
                slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
                pixelization_indexes_for_sub_slim_index=self.pixelization_indexes_for_sub_slim_index
                )

    @cached_property
    @profile_func
    def mapping_matrix(self):

        return mapper_util.mapping_matrix_Delaunay_baricentric_interpolation_from(
                pixel_weights=self.pixelization_weights_for_sub_slim_index,
                pixels=self.pixels,
                total_mask_pixels=self.source_grid_slim.mask.pixels_in_mask,
                slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
                pixelization_indexes_for_sub_slim_index=self.pixelization_indexes_for_sub_slim_index,
                sub_fraction=self.source_grid_slim.mask.sub_fraction
                )


    @property
    def delaunay(self):
        return self.source_pixelization_grid.Delaunay

    def reconstruction_from(self, solution_vector):
        return solution_vector

    def pixel_signals_from(self, signal_scale):

        return mapper_util.adaptive_pixel_signals_Delaunay_version_from(
                pixels=self.pixels,
                pixel_weights=self.pixelization_weights_for_sub_slim_index,
                signal_scale=signal_scale,
                pixelization_indexes_for_sub_slim_index=self.pixelization_indexes_for_sub_slim_index,
                slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
                hyper_image=self.hyper_image
                )
