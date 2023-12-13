from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans as ScipyKMeans
from typing import TYPE_CHECKING, Optional
import warnings

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.inversion.pixelization.image_mesh.abstract import AbstractImageMesh
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray import exc


class KMeans(AbstractImageMesh):
    def __init__(
        self,
        pixels=10,
        weight_floor=0.0,
        weight_power=0.0,
        n_iter: int = 1,
        max_iter: int = 5,
        seed: Optional[int] = None,
        stochastic: bool = False,
    ):
        """
        Computes an image-mesh by running a weighted KMeans clustering algorithm.

        This requires an adapt-image, which is the image that the KMeans algorithm adapts to in order to compute the
        image mesh. This could simply be the image itself, or a model fit to the image which removes certain
        features or noise.

        For example, using the adapt image, the image mesh is computed as follows:

        1) Convert the adapt image to a weight map, which is a 2D array of weight values.

        2) Run the KMeans algorithm on the weight map, such that the image mesh pixels cluster around the weight map
        values with higher values.

        Parameters
        ----------
        total_pixels
            The total number of pixels in the image mesh and input into the KMeans algortihm.
        n_iter
            The number of times the KMeans algorithm is repeated.
        max_iter
            The maximum number of iterations in one run of the KMeans algorithm.
        seed
            The random number seed, which can be used to reproduce the same image mesh via the kmeans for the same inputs.
        stochastic
            If True, the random number seed is randommly chosen every time the function is called, ensuring every
            pixel-grid is randomly determined and thus stochastic.
        """

        super().__init__()

        self.pixels = pixels
        self.weight_floor = weight_floor
        self.weight_power = weight_power
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.seed = seed
        self.stochastic = stochastic

    def weight_map_from(self, adapt_data: np.ndarray):
        """
        Returns the weight-map used by the KMeans clustering algorithm to compute the Delaunay pixel corners.

        This is computed from an input adapt_data, which is an image representing the data which the KMeans
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
        weight_map = (adapt_data - np.min(adapt_data)) / (
            np.max(adapt_data) - np.min(adapt_data)
        ) + self.weight_floor * np.max(adapt_data)

        return np.power(weight_map, self.weight_power)

    def image_mesh_from(
        self, grid: Grid2D, weight_map: Optional[np.ndarray]
    ) -> Grid2DIrregular:
        """
        Returns an image mesh by running a KMeans clustering algorithm on the weight map.

        See the `__init__` docstring for a full description of how this is performed.

        Parameters
        ----------
        grid
            The grid of (y,x) coordinates of the image data the pixelization fits, which the KMeans algorithm
            adapts to.
        weight_map
            The weights defining the regions of the image the KMeans algorithm adapts to.

        Returns
        -------

        """

        warnings.filterwarnings("ignore")

        if self.stochastic:
            seed = np.random.randint(low=1, high=2**31)
        else:
            seed = self.seed

        if self.pixels > grid.shape[0]:
            raise exc.GridException

        kmeans = ScipyKMeans(
            n_clusters=int(self.pixels),
            random_state=seed,
            n_init=self.n_iter,
            max_iter=self.max_iter,
        )

        try:
            kmeans = kmeans.fit(X=grid.binned, sample_weight=weight_map)
        except ValueError or OverflowError:
            raise exc.InversionException()

        return Grid2DIrregular(
            values=kmeans.cluster_centers_,
        )
