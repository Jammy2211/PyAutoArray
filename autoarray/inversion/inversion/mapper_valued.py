
import numpy as np
from typing import List, Optional, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.inversion.inversion import inversion_util

class MapperValued:

    def __init__(self, mapper, values):

        self.mapper = mapper
        self.values = values

    def interpolated_array_from(
        self,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Array2D]:
        """
        The reconstruction of an `Inversion` can be on an irregular pixelization (e.g. a Delaunay triangulation,
        Voronoi mesh).

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to using
        this irregular grid.

        This function therefore interpolates the irregular reconstruction on to a regular grid of square pixels.
        The routine that performs the interpolation is specific to each pixelization and contained in its
        corresponding `Mapper`` objects, which are called by this function.

        The output interpolated reconstruction cis by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        return self.mapper.interpolated_array_from(
                values=self.values,
                shape_native=shape_native,
                extent=extent,
            )

    def max_pixel_list_from(
        self, total_pixels: int = 1, filter_neighbors: bool = False
    ) -> List[List[int]]:

        max_pixel_list = []

        pixel_list = []

        pixels_ascending_list = list(
            reversed(np.argsort(self.values))
        )

        for pixel in range(total_pixels):
            pixel_index = pixels_ascending_list[pixel]

            add_pixel = True

            if filter_neighbors:
                pixel_neighbors = self.mapper.neighbors[pixel_index]
                pixel_neighbors = pixel_neighbors[pixel_neighbors >= 0]

                brightness = self.values[pixel_index]
                brightness_neighbors = self.values[
                    pixel_neighbors
                ]

                if brightness < np.max(brightness_neighbors):
                    add_pixel = False

            if add_pixel:
                pixel_list.append(pixel_index)

        max_pixel_list.append(pixel_list)

        return max_pixel_list

    @property
    def max_pixel_centre(self):

        max_pixel = np.argmax(self.values)

        max_pixel_centre = Grid2DIrregular(
            values=[self.mapper.source_plane_mesh_grid[max_pixel]]
        )

        return max_pixel_centre

    def magnification_via_interpolation_from(self) -> List[float]:

        magnification_list = []

        interpolated_reconstruction = self.interpolated_array_from(
            shape_native=(401, 401)
        )

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
                    mapping_matrix=self.mapper.mapping_matrix,
                    reconstruction=self.values,
                )
        
        dsgfgfd
        
        magnification_list.append(
            np.sum(mapped_reconstructed_image) / np.sum(interpolated_reconstruction)
        )

        return magnification_list