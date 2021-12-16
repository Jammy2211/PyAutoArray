import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper, PixForSub

from autoarray.numba_util import profile_func
from autoarray.inversion.mappers import mapper_util


class MapperVoronoi(AbstractMapper):
    def __init__(
        self,
        source_grid_slim,
        source_pixelization_grid,
        data_pixelization_grid=None,
        hyper_image=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Class representing a Voronoi mapper, which maps unmasked pixels on a masked 2D array in the form of
        a grid, see the *hyper_galaxies.array.grid* module to pixels discretized on a Voronoi grid.

        The irand non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge
        of how different grid map to one another.

        Parameters
        ----------
        pixels
            The number of pixels in the Voronoi pixelization.
        source_grid_slim : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        voronoi : scipy.spatial.Voronoi
            Class storing the Voronoi grid's
        geometry : pixelization.Voronoi.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the Vornoi pixelization.
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
    def pix_indexes_for_sub_slim_index(self) -> PixForSub:
        """
        An array describing the pairing of every image-pixel coordinate to every source-pixel.

        A `pixelization_index` refers to the index of each source pixel index and a `sub_slim_index` refers to the
        index of each sub-pixel in the masked data.

        For example:

        - If the data's first sub-pixel maps to the source pixelization's third pixel then
        pix_index_for_sub_slim_index[0] = 2
        - If the data's second sub-pixel maps to the source pixelization's fifth pixel then
        pix_index_for_sub_slim_index[1] = 4

        For a Voronoi pixelization, we perform a graph search to map each coordinate of the mappers traced grid
        of (y,x) coordinates (`source_grid_slim`) to each Voronoi pixel based on its centre (`source_pixelization_grid`).
        """
        mappings = mapper_util.pix_indexes_for_sub_slim_index_voronoi_from(
            grid=self.source_grid_slim,
            nearest_pix_index_for_slim_index=self.source_pixelization_grid.nearest_pixelization_index_for_slim_index,
            slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
            pixelization_grid=self.source_pixelization_grid,
            pixel_neighbors=self.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=self.source_pixelization_grid.pixel_neighbors.sizes,
        ).astype("int")

        return PixForSub(
            mappings=mappings, sizes=np.ones(self.source_grid_slim.shape[0], dtype="int")
        )

    @cached_property
    @profile_func
    def pix_weights_for_sub_slim_index(self):
        """
        Weights for source pixels to sub pixels. Used for creating the mapping matrix and 'pixel_signals_from'
        It has the same shape as the 'pix_indexes_for_sub_slim_index'.
        """
        return np.ones((len(self.source_pixelization_grid), 1), dtype="int")

    @property
    def voronoi(self):
        return self.source_pixelization_grid.voronoi

    def reconstruction_from(self, solution_vector):
        return solution_vector
