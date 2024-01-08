from typing import Optional

import numpy as np

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


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
