import numpy as np
from typing import List, Optional, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class MapperValued:
    def __init__(self, mapper, values, mesh_pixel_mask: Optional[np.ndarray] = None):
        """
        Pairs a `Mapper` object with an array of values (e.g. the `reconstruction` values of each value of each
        mapper pixel) in order to perform calculations which use both the `Mapper` and these values.

        For example, a common use case is to interpolate the reconstruction of values on a mapper from the
        mesh of the mapper (e.g. a Delaunay mesh) to a uniform Cartesian grid of values, because the irregular mesh
        is difficult to plot and analyze.

        This class also provides functionality to compute the magnification of the reconstruction, by comparing the
        sum of the values on the mapper in both the image and source planes, which is a specific quantity
        used in gravitational lensing.

        Parameters
        ----------
        mapper
            The `Mapper` object which pairs with the values, for example a `MapperDelaunay` object.
        values
            The values of each pixel of the mapper, which could be the `reconstruction` values of an `Inversion`,
            but alternatively could be other quantities such as the noise-map of these values.
        mesh_pixel_mask
            The mask of pixels that are omitted from the reconstruction when computing the image, for example to
            remove pixels with low signal-to-noise so they do not impact the magnification calculation.

        """
        self.mapper = mapper
        self.values = values
        self.mesh_pixel_mask = mesh_pixel_mask

    @property
    def values_masked(self):
        """
        Applies the `mesh_pixel_mask` to the values, setting all masked pixels to 0.0, so that mesh pixels
        can be removed to prevent them from impacting calculations.

        For example, the magnification calculation uses the values of the reconstruction to compute the magnification,
        but this can be impacted by pixels with low signal-to-noise. The `mesh_pixel_mask` can be used to remove
        these pixels from the calculation.

        Returns
        -------
        The values of the mapper with the `mesh_pixel_mask` applied.
        """

        values = self.values

        if self.mesh_pixel_mask is not None:
            if self.mapper._xp.__name__.startswith("jax"):
                values = values.at[self.mesh_pixel_mask].set(0.0)
            else:
                values = values.copy()
                values[self.mesh_pixel_mask] = 0.0

        return values

    def max_pixel_list_from(
        self, total_pixels: int = 1, filter_neighbors: bool = False
    ) -> List[List[int]]:
        """
        Returns a list of lists of the maximum cell or pixel values in the mapper.

        Neighbors can be filtered such that each maximum value in a pixel is higher than all surrounding pixels,
        thus forming a `peak` in the mapper values.

        For example, if a `reconstruction` is the mapper values and neighbor filtering is on, this would return the
        brightest pixels in the mapper reconstruction which are brighter than all pixels around them.

        In gravitational lensing, these peaks are the brightest regions of the source reconstruction and correspond
        to features like the centre of the source galaxy and knots of star formation in a galaxy.

        Parameters
        ----------
        total_pixels
            The total number of pixels to return in the list of peak pixels.
        filter_neighbors
            If True, the peak pixels are filtered such that they are the brightest pixel in the mapper and all
            of its neighbors.

        Returns
        -------

        """
        max_pixel_list = []

        pixel_list = []

        pixels_ascending_list = list(reversed(np.argsort(self.values_masked)))

        for pixel in range(total_pixels):
            pixel_index = pixels_ascending_list[pixel]

            add_pixel = True

            if filter_neighbors:
                pixel_neighbors = self.mapper.neighbors[pixel_index]
                pixel_neighbors = pixel_neighbors[pixel_neighbors >= 0]

                max_value = self.values_masked[pixel_index]
                max_value_neighbors = self.values_masked[pixel_neighbors]

                if max_value < np.max(max_value_neighbors):
                    add_pixel = False

            if add_pixel:
                pixel_list.append(pixel_index)

        max_pixel_list.append(pixel_list)

        return max_pixel_list

    @property
    def max_pixel_centre(self) -> Grid2DIrregular:
        """
        Returns the centre of the brightest pixel in the mapper values.

        Returns
        -------
        The centre of the brightest pixel in the mapper values.
        """
        max_pixel = np.argmax(self.values_masked)

        max_pixel_centre = Grid2DIrregular(
            values=[self.mapper.source_plane_mesh_grid.array[max_pixel]]
        )

        return max_pixel_centre

    def mapped_reconstructed_image_from(
        self,
    ) -> Array2D:
        """
        Returns the image of the reconstruction computed via the mapping matrix, where the image is the reconstruction
        of the source-plane image in the image-plane without accounting for the PSF convolution.

        This image is computed by mapping the reconstruction to the image, using the mapping matrix of the inversion.

        The image is used to compute magnification, where the magnification is the ratio of the surface brightness of
        image in the image-plane over the surface brightness of the source in the source-plane.

        Returns
        -------
        The image of the reconstruction computed via the mapping matrix, with the PSF convolution not accounted for.
        """

        mapping_matrix = self.mapper.mapping_matrix

        if self.mesh_pixel_mask is not None:
            if self.mapper._xp.__name__.startswith("jax"):
                mapping_matrix = mapping_matrix.at[:, self.mesh_pixel_mask].set(0.0)
            else:
                mapping_matrix[:, self.mesh_pixel_mask] = 0.0

        return Array2D(
            values=inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
                mapping_matrix=mapping_matrix,
                reconstruction=self.values_masked,
                xp=self.mapper._xp,
            ),
            mask=self.mapper.mapper_grids.mask,
        )