import numpy as np
from typing import List, Optional, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class MapperValued:
    def __init__(self, mapper, values):
        """
        Pairs a `Mapper` object with an array of values (e.g. the `reconstruction` values of each value of each
        mapper pixel) in order to perform calculations which use both the `Mapper` and these values.

        For example, a common use case is to interpolate the reconstruction of values on a mapper from the
        mesh of the mapper (e.g. a Voronoi mesh) to a uniform Cartesian grid of values, because the irregular mesh
        is difficult to plot and analyze.

        This class also provides functionality to compute the magnification of the reconstruction, by comparing the
        sum of the values on the mapper in both the image and source planes, which is a specific quantity
        used in gravitational lensing.

        Parameters
        ----------
        mapper
            The `Mapper` object which pairs with the values, for example a `MapperVoronoi` object.
        values
            The values of each pixel of the mapper, which could be the `reconstruction` values of an `Inversion`,
            but alternatively could be other quantities such as the errors on these values.
        """
        self.mapper = mapper
        self.values = values

    def interpolated_array_from(
        self,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Array2D:
        """
        The values of a mapper can be on an irregular pixelization (e.g. a Delaunay triangulation, Voronoi mesh).

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

        pixels_ascending_list = list(reversed(np.argsort(self.values)))

        for pixel in range(total_pixels):
            pixel_index = pixels_ascending_list[pixel]

            add_pixel = True

            if filter_neighbors:
                pixel_neighbors = self.mapper.neighbors[pixel_index]
                pixel_neighbors = pixel_neighbors[pixel_neighbors >= 0]

                max_value = self.values[pixel_index]
                max_value_neighbors = self.values[pixel_neighbors]

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
        max_pixel = np.argmax(self.values)

        max_pixel_centre = Grid2DIrregular(
            values=[self.mapper.source_plane_mesh_grid[max_pixel]]
        )

        return max_pixel_centre

    def mapped_reconstructed_image_from(self) -> Array2D:
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
        return Array2D(
            values=inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
                mapping_matrix=self.mapper.mapping_matrix,
                reconstruction=self.values,
            ),
            mask=self.mapper.mapper_grids.mask,
        )

    def magnification_via_mesh_from(self, pixel_mask: np.ndarray = None) -> float:
        """
        Returns the magnification of the reconstruction computed via the mesh, where the magnification is the ratio
        of the surface brightness of image in the image-plane over the surface brightness of the source in
        the source-plane.

        In the image-plane, this is computed by mapping the reconstruction to the image, summing all reconstructed
        values and multiplying by the area of each image pixel. This image-plane image is not convolved with the
        PSF, as the source plane reconstruction is a non-convolved image.

        In the source-plane, this is computed by summing the reconstruction values multiplied by the area of each
        mesh pixel, for example if the source-plane is a `Voronoi` mesh this is the area of each Voronoi pixel.

        This calculatiion is generally more robust that using an interpolated
        image (see `magnification_via_interpolation_from`), because it uses the exact the source-plane reconstruction
        values. However, certain meshes have irregular pixels, especially at the edge, which can produce large
        areas that can artificially decrease the magnification. Including cuts on which source-plane pixels are used,
        for example based on brightness, is recommended to ensure the magnification is robust.

        Returns
        -------
        The magnification of the reconstruction computed via the mesh.
        """

        if isinstance(self.mapper, MapperDelaunay):
            raise exc.MeshException(
                """
                The method `magnification_via_mesh_from` does not currently support `Delaunay` mesh objects.
                
                To compute the magnification of a `Delaunay` mesh, use the method `magnification_via_interpolation_from`.
                
                This method only supports a `Rectangular` or `Voronoi` mesh.
                """
            )

        mapped_reconstructed_image = self.mapped_reconstructed_image_from()

        mesh_areas = self.mapper.source_plane_mesh_grid.areas_for_magnification

        if np.all(mesh_areas == 0.0):
            raise exc.MeshException(
                """
                The magnification cannot be computed because the areas of the source-plane mesh pixels are all zero.
                
                This probably means you have specified an invalid source-plane mesh, for example a `Voronoi` mesh
                where all pixels are on the edge of the source-plane and therefore have an infinite border.
                """
            )

        return np.sum(
            mapped_reconstructed_image * mapped_reconstructed_image.pixel_area
        ) / np.sum(self.values * mesh_areas)

    def magnification_via_interpolation_from(self) -> float:
        """
        Returns the magnification of the reconstruction computed via interpolation, where the magnification is the ratio
        of the surface brightness of image in the image-plane over the surface brightness of the source in
        the source-plane.

        In the image-plane, this is computed by mapping the reconstruction to the image, summing all reconstructed
        values and multiplying by the area of each image pixel. This image-plane image is not convolved with the
        PSF, as the source plane reconstruction is a non-convolved image. This image therefore does not use
        interpolation.

        In the source-plane, this is computed by interpolating the reconstruction to a regular grid of pixels, for
        example a 2D grid of 401 x 401 pixels, and summing the reconstruction values multiplied by the area of each
        pixel. This calculation uses interpolation to compute the source-plane image.

        This calculation is generally less robust than using the mesh to compute the magnification,
        (see `magnification_via_mesh_from`), as the interpolation may not perfectly represent the source-plane
        reconstruction. However, it is computationally faster and can be used when the source-plane mesh has
        irregular pixels that are not suitable for computing the magnification.

        Returns
        -------
        The magnification of the reconstruction computed via interpolation.
        """
        mapped_reconstructed_image = self.mapped_reconstructed_image_from()

        interpolated_reconstruction = self.interpolated_array_from(
            shape_native=(401, 401)
        )

        return np.sum(
            mapped_reconstructed_image * mapped_reconstructed_image.pixel_area
        ) / np.sum(interpolated_reconstruction * interpolated_reconstruction.pixel_area)
