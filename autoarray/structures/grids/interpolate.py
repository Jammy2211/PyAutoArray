import numpy as np
import scipy.spatial.qhull as qhull
from autoarray.structures import grids
from autoarray.util import grid_util


class GridInterpolate:
    def __init__(self, grid, interp_grid, interpolation_pixel_scale):
        self.grid = grid
        self.interp_grid = interp_grid
        self.interpolation_pixel_scale = interpolation_pixel_scale
        self.vtx, self.wts = self.interp_weights

    @property
    def interp_weights(self):
        tri = qhull.Delaunay(self.interp_grid)
        simplex = tri.find_simplex(self.grid)
        # noinspection PyUnresolvedReferences
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.grid - temp[:, 2]
        bary = np.einsum("njk,nk->nj", temp[:, :2, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    @classmethod
    def from_mask_grid_and_interpolation_pixel_scales(
        cls, mask, grid, interpolation_pixel_scale
    ):

        rescale_factor = mask.pixel_scale / interpolation_pixel_scale

        mask = mask.mask_sub_1

        rescaled_mask = mask.rescaled_mask_from_rescale_factor(
            rescale_factor=rescale_factor
        )

        interp_mask = rescaled_mask.edge_buffed_mask

        interp_grid = grid_util.grid_1d_via_mask_from(
            mask=interp_mask,
            pixel_scales=(interpolation_pixel_scale, interpolation_pixel_scale),
            sub_size=1,
            origin=mask.origin,
        )

        return GridInterpolate(
            grid=grid,
            interp_grid=grids.Grid(grid=interp_grid, mask=interp_mask.mask_sub_1),
            interpolation_pixel_scale=interpolation_pixel_scale,
        )

    def interpolated_values_from_values(self, values) -> np.ndarray:
        """This function uses the precomputed vertexes and weights of a Delaunay gridding to interpolate a set of
        values computed on the interpolation grid to the GridInterpolate's full grid.

        This function is taken from the SciPy interpolation method griddata
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html). It is adapted here
        to reuse pre-computed interpolation vertexes and weights for efficiency. """
        return np.einsum("nj,nj->n", np.take(values, self.vtx), self.wts)
