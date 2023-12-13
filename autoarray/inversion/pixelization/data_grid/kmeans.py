from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from typing import TYPE_CHECKING, Optional
import warnings

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray import exc
from autoarray.structures.grids import sparse_2d_util


def via_kmeans_from(
        total_pixels: int,
        grid: Grid2D,
        weight_map: np.ndarray,
        n_iter: int = 1,
        max_iter: int = 5,
        seed: Optional[int] = None,
        stochastic: bool = False,
    ) -> Grid2DIrregular:
        """
        Calculate a grid from a Grid2D and weight map.

        This is performed by running a KMeans clustering algorithm on the weight map, such that Grid2DSparse (y,x)
        coordinates cluster around the weight map values with higher values.

        This function is used in the `Inversion` package to set up the VoronoiMagnification Mesh.

        Parameters
        ----------
        total_pixels
            The total number of pixels in the Grid2DSparse and input into the KMeans algortihm.
        grid : Grid2D
            The grid of (y,x) coordinates corresponding to the weight map.
        weight_map
            The 2D array of weight values that the KMeans clustering algorithm adapts to determine the Grid2DSparse.
        n_iter
            The number of times the KMeans algorithm is repeated.
        max_iter
            The maximum number of iterations in one run of the KMeans algorithm.
        seed or None
            The random number seed, which can be used to reproduce Grid2DSparse's for the same inputs.
        stochastic
            If True, the random number seed is randommly chosen every time the function is called, ensuring every
            pixel-grid is randomly determined and thus stochastic.
        """

        warnings.filterwarnings("ignore")

        if stochastic:
            seed = np.random.randint(low=1, high=2**31)

        if total_pixels > grid.shape[0]:
            raise exc.GridException

        kmeans = KMeans(
            n_clusters=int(total_pixels),
            random_state=seed,
            n_init=n_iter,
            max_iter=max_iter,
        )

        try:
            kmeans = kmeans.fit(X=grid.binned, sample_weight=weight_map)
        except ValueError or OverflowError:
            raise exc.InversionException()

        return Grid2DIrregular(
            values=kmeans.cluster_centers_,
        )