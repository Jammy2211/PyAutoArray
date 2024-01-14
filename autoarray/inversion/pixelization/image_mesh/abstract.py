from typing import Optional

import numpy as np

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray import numba_util
from autoarray.geometry import geometry_util

@numba_util.jit()
def mesh_pixels_per_image_pixels_from(
        grid_pixel_centres,
        shape_native
) -> np.ndarray:
    """
    Returns an array containing the number of mesh pixels in every pixel of the data's mask.

    For example, image-mesh adaption may be performed on a 3.0" circular mask of data. The high weight pixels
    may have 3 or more mesh pixels per image pixel, whereas low weight regions may have zero pixels. The array
    returned by this function gives the integer number of pixels in each data pixel.

    Parameters
    ----------
    grid_pixel_centres
        The 2D integer index of every image pixel that each image-mesh pixel falls within.
    shape_native
        The 2D shape of the data's mask, which the number of image-mesh pixels that fall within eac pixel is counted.

    Returns
    -------
    An array containing the integer number of image-mesh pixels that fall without each of the data's mask.
    """
    mesh_pixels_per_image_pixel = np.zeros(shape=shape_native)

    for i in range(grid_pixel_centres.shape[0]):
        y = grid_pixel_centres[i, 0]
        x = grid_pixel_centres[i, 1]

        mesh_pixels_per_image_pixel[y, x] += 1

    return mesh_pixels_per_image_pixel


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

    def weight_map_from(self, adapt_data: np.ndarray):
        """
        Returns the weight-map used by the image-mesh to compute the mesh pixel centres.

        This is computed from an input adapt data, which is an image representing the data which the KMeans
        clustering algorithm is applied too. This could be the image data itself, or a model fit which
        only has certain features.

        The ``weight_floor`` and ``weight_power`` attributes of the class are used to scale the weight map, which
        gives the model flexibility in how it adapts the pixelization to the image data.

        Parameters
        ----------
        adapt_data
            A image which represents one or more components in the masked 2D data in the image-plane.

        Returns
        -------
        The weight map which is used to adapt the Delaunay pixels in the image-plane to components in the data.
        """

        weight_map = (np.abs(adapt_data) + self.weight_floor) ** self.weight_power
        weight_map /= np.sum(weight_map)

        return weight_map

    def image_plane_mesh_grid_from(
        self, grid: Grid2D, adapt_data: Optional[np.ndarray] = None
    ) -> Grid2DIrregular:
        raise NotImplementedError

    def mesh_pixels_per_image_pixels_from(self, grid: Grid2D, mesh_grid : Grid2DIrregular) -> Array2D:
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

        grid_pixel_centres = geometry_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=mesh_grid,
            shape_native=grid.shape_native,
            pixel_scales=grid.pixel_scales,
            origin=grid.origin,
        ).astype("int")

        mesh_pixels_per_image_pixels = mesh_pixels_per_image_pixels_from(
            grid=grid,
            grid_pixel_centres=grid_pixel_centres,
            shape_native=grid.shape_native
        )

        return Array2D(values=mesh_pixels_per_image_pixels, mask=grid.mask)
