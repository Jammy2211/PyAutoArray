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