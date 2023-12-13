from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from typing import TYPE_CHECKING, Optional
import warnings

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.abstract_structure import Structure

from autoarray import exc
from autoarray.structures.grids import sparse_2d_util


class Grid2DSparse(Structure):
    def __new__(cls, values: np.ndarray):
        """
        A sparse grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of a
        pixel on the sparse grid. To setup the sparse-grid, it is laid over a grid of unmasked pixels, such
        that all sparse-grid pixels which map inside of an unmasked grid pixel are included on the sparse grid.

        To setup this sparse grid, we thus have two sparse grid:

        - The unmasked sparse-grid, which corresponds to a uniform 2D array of pixels. The edges of this grid
          correspond to the 4 edges of the mask (e.g. the higher and lowest (y,x) scaled unmasked pixels) and the
          grid's shape is speciifed by the unmasked_sparse_grid_shape parameter.

        - The (masked) sparse-grid, which is all pixels on the unmasked sparse-grid above which fall within unmasked
          grid pixels. These are the pixels which are actually used for other modules in PyAutoArray.

        The origin of the unmasked sparse grid can be changed to allow off-center pairings with sparse-grid pixels,
        which is necessary when a mask has a centre offset from (0.0", 0.0"). However, the sparse grid itself
        retains an origin of (0.0", 0.0"), ensuring its scaled grid uses the same coordinate system as the
        other grid.

        The sparse grid is used to determine the pixel centers of an adaptive mesh.

        Parameters
        ----------
        sparse_grid or Grid2D
            The (y,x) grid of sparse coordinates.
        """
        return values.view(cls)

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

    @classmethod
    def from_total_pixels_grid_and_weight_map(
        cls,
        total_pixels: int,
        grid: Grid2D,
        weight_map: np.ndarray,
        n_iter: int = 1,
        max_iter: int = 5,
        seed: Optional[int] = None,
        stochastic: bool = False,
    ) -> "Grid2DSparse":
        """
        Calculate a Grid2DSparse from a Grid2D and weight map.

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

        return Grid2DSparse(
            values=kmeans.cluster_centers_,
        )

    @classmethod
    def from_hilbert_curve(
            cls,
            total_pixels: int,
            weight_map: np.ndarray,
            grid_hb: np.ndarray,
    ):
        drawn_id, drawn_x, drawn_y = sparse_2d_util.inverse_transform_sampling_interpolated(
            probabilities=weight_map,
            n_samples=total_pixels,
            gridx=grid_hb[:, 1],
            gridy=grid_hb[:, 0],
        )

        spix_centers = np.stack((drawn_y, drawn_x), axis=-1)

        return Grid2DSparse(values=spix_centers)

    @classmethod
    def from_snr_split(
        cls,
        pixels: int,
        fraction_high_snr: float,
        snr_cut: float,
        grid: Grid2D,
        snr_map: np.ndarray,
        n_iter: int = 1,
        max_iter: int = 5,
        seed: Optional[int] = None,
        stochastic: bool = False,
    ):
        warnings.filterwarnings("ignore")

        if stochastic:
            seed = np.random.randint(low=1, high=2**31)

        if pixels > grid.shape[0]:
            raise exc.GridException

        high_snr_pixels = int(pixels * fraction_high_snr)

        grid_high_snr = grid.binned[snr_map > snr_cut]

        kmeans = KMeans(
            n_clusters=high_snr_pixels,
            random_state=seed,
            n_init=n_iter,
            max_iter=max_iter,
        )

        try:
            kmeans_high_snr = kmeans.fit(X=grid_high_snr)
        except ValueError or OverflowError:
            raise exc.InversionException()

        low_snr_pixels = pixels - high_snr_pixels

        grid_low_snr = grid.binned[snr_map < snr_cut]

        kmeans = KMeans(
            n_clusters=low_snr_pixels,
            random_state=seed,
            n_init=n_iter,
            max_iter=max_iter,
        )

        try:
            kmeans_low_snr = kmeans.fit(X=grid_low_snr)
        except ValueError or OverflowError:
            raise exc.InversionException()

        sparse_image_plane_grid = np.concatenate(
            (kmeans_high_snr.cluster_centers_, kmeans_low_snr.cluster_centers_), axis=0
        )
        sparse_image_plane_grid = np.asarray(sparse_image_plane_grid)

        return Grid2DSparse(
            values=sparse_image_plane_grid,
        )

    @property
    def total_sparse_pixels(self) -> int:
        return len(self)
