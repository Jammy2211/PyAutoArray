from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper

from autoarray.numba_util import profile_func
from autoarray.inversion import mapper_util


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
    def pixelization_index_for_sub_slim_index(self):
        """
        The 1D index mappings between the sub pixels and Voronoi pixelization pixels.
        """

        return mapper_util.pixelization_index_for_voronoi_sub_slim_index_from(
            grid=self.source_grid_slim,
            nearest_pixelization_index_for_slim_index=self.source_pixelization_grid.nearest_pixelization_index_for_slim_index,
            slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
            pixelization_grid=self.source_pixelization_grid,
            pixel_neighbors=self.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=self.source_pixelization_grid.pixel_neighbors.sizes,
        ).astype("int")

    @property
    def voronoi(self):
        return self.source_pixelization_grid.voronoi

    def reconstruction_from(self, solution_vector):
        return solution_vector
