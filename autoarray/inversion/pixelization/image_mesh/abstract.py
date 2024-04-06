from typing import Optional

import numpy as np
import os

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.structures.grids import grid_2d_util

from autoarray import exc


class AbstractImageMesh:
    def __init__(self):
        """
        An abstract image mesh, which is used by pixelizations to determine the (y,x) mesh coordinates from image
        data.
        """
        pass

    @property
    def uses_adapt_images(self) -> bool:
        raise NotImplementedError

    def image_plane_mesh_grid_from(
        self, grid: Grid2D, adapt_data: Optional[np.ndarray] = None, settings=None
    ) -> Grid2DIrregular:
        raise NotImplementedError

    def mesh_pixels_per_image_pixels_from(
        self, grid: Grid2D, mesh_grid: Grid2DIrregular
    ) -> Array2D:
        """
        Returns an array containing the number of mesh pixels in every pixel of the data's mask.

        For example, image-mesh adaption may be performed on a 3.0" circular mask of data. The high weight pixels
        may have 3 or more mesh pixels per image pixel, whereas low weight regions may have zero pixels. The array
        returned by this function gives the integer number of pixels in each data pixel.

        Parameters
        ----------
        grid
            The masked (y,x) grid of the data coordinates, corresponding to the mask applied to the data. The number of
            mesh pixels mapped inside each of this grid's image-pixels is returned.
        mesh_grid
            The image mesh-grid computed by the class which adapts to the data's mask. The number of image mesh pixels
            that fall within each of the data's mask pixels is returned.

        Returns
        -------
        An array containing the integer number of image-mesh pixels that fall without each of the data's mask.
        """

        mesh_pixels_per_image_pixels = grid_2d_util.grid_pixels_in_mask_pixels_from(
            grid=np.array(mesh_grid),
            shape_native=grid.shape_native,
            pixel_scales=grid.pixel_scales,
            origin=grid.origin,
        )

        return Array2D(values=mesh_pixels_per_image_pixels, mask=grid.mask)

    def check_mesh_pixels_per_image_pixels(self, grid, mesh_grid, settings):
        """
        Checks the number of mesh pixels in every image pixel and raises an `InversionException` if there are fewer
        mesh pixels inside a certain number of image-pixels than the input settings.

        This allows a user to force a model-fit to use image-mesh's which cluster a large number of mesh pixels to
        the brightest regions of the image data (E.g. the highst weighted regions).

        The check works as follows:

        1) Compute the 2D array of the number of mesh pixels in every masked data image pixel.
        2) Find the number of mesh pixels in the N data pixels with the larger number of mesh pixels, where N is
           given by `settings.image_mesh_min_mesh_number`. For example, if `settings.image_mesh_min_mesh_number=5` then
           the number of mesh pixels in the 5 data pixels with the most data pixels is computed.
        3) Compare the lowest value above to the value `settings.image_mesh_min_mesh_pixels_per_pixel`. If the value is
           below this value, raise an `InversionException`.

        Therefore, by settings `settings.image_mesh_min_mesh_pixels_per_pixel` to a value above 1 the code is forced
        to adapt the image mesh enough to put many mesh pixels in the brightest image pixels.

        Parameters
        ----------
        grid
            The masked (y,x) grid of the data coordinates, corresponding to the mask applied to the data. The number of
            mesh pixels mapped inside each of this grid's image-pixels is returned.
        mesh_grid
            The image mesh-grid computed by the class which adapts to the data's mask. The number of image mesh pixels
            that fall within each of the data's mask pixels is returned.
        settings
            The inversion settings, which have the criteria dictating if the image-mesh has clustered enough or if
            an exception is raised.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        if settings is not None:
            if settings.image_mesh_min_mesh_pixels_per_pixel is not None:
                mesh_pixels_per_image_pixels = self.mesh_pixels_per_image_pixels_from(
                    grid=grid, mesh_grid=mesh_grid
                )

                indices_of_highest_values = np.argsort(mesh_pixels_per_image_pixels)[
                    -settings.image_mesh_min_mesh_number :
                ]
                lowest_mesh_pixels = np.min(
                    mesh_pixels_per_image_pixels[indices_of_highest_values]
                )

                if lowest_mesh_pixels < settings.image_mesh_min_mesh_pixels_per_pixel:
                    raise exc.InversionException()

        return mesh_grid

    def check_adapt_background_pixels(self, grid, mesh_grid, adapt_data, settings):
        """
        Checks the number of mesh pixels in the background of the image-mesh and raises an `InversionException` if
        there are fewer mesh pixels in the background than the input settings.

        This allows a user to force a model-fit to use image-mesh's which cluster a minimum number of mesh pixels to
        the faintest regions of the image data (E.g. the lowest weighted regions). This prevents too few image-mesh
        pixels being allocated to the background of the data.

        The check works as follows:

        1) Find all pixels in the background of the `adapt_data`, which are N pixels with the lowest values, where N is
           a percentage given by `settings.image_mesh_adapt_background_percent_check`. If N is 50%, then the half of
            pixels in `adapt_data` with the lowest values will be checked.
        2) Sum the total number of mesh pixels in these background pixels, thereby estimating the number of mesh pixels
            assigned to background pixels.
        3) Compare this value to the total number of mesh pixels multiplied
           by `settings.image_mesh_adapt_background_percent_threshold` and raise an `InversionException` if the number
           of mesh pixels is below this value, meaning the background did not have sufficient mesh pixels in it.

        Therefore, by setting `settings.image_mesh_adapt_background_percent_threshold` the code is forced
        to adapt the image mesh in a way that places many mesh pixels in the background regions.

        Parameters
        ----------
        grid
            The masked (y,x) grid of the data coordinates, corresponding to the mask applied to the data. The number of
            mesh pixels mapped inside each of this grid's image-pixels is returned.
        mesh_grid
            The image mesh-grid computed by the class which adapts to the data's mask. The number of image mesh pixels
            that fall within each of the data's mask pixels is returned.
        adapt_data
            A image which represents one or more components in the masked 2D data in the image-plane.
        settings
            The inversion settings, which have the criteria dictating if the image-mesh has clustered enough or if
            an exception is raised.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        if settings is not None:
            if settings.image_mesh_adapt_background_percent_threshold is not None:
                pixels = mesh_grid.shape[0]

                pixels_in_background = int(
                    grid.shape[0] * settings.image_mesh_adapt_background_percent_check
                )

                indices_of_lowest_values = np.argsort(adapt_data)[:pixels_in_background]
                mask_background = np.zeros_like(adapt_data, dtype=bool)
                mask_background[indices_of_lowest_values] = True

                mesh_pixels_per_image_pixels = self.mesh_pixels_per_image_pixels_from(
                    grid=grid, mesh_grid=mesh_grid
                )

                mesh_pixels_in_background = sum(
                    mesh_pixels_per_image_pixels[mask_background]
                )

                if mesh_pixels_in_background < (
                    pixels * settings.image_mesh_adapt_background_percent_threshold
                ):
                    raise exc.InversionException()
