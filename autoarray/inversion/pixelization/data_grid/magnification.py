from __future__ import annotations
import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask.mask_2d import mask_2d_util
from autoarray import numba_util



@numba_util.jit()
def unmasked_sparse_for_sparse_from(
    total_sparse_pixels: int,
    mask: np.ndarray,
    unmasked_sparse_grid_pixel_centres: np.ndarray,
) -> np.ndarray:
    """
    Returns the mapping between every masked pixel on a grid and a set of pixel centres corresponding to an unmasked
    sparse grid. This is performed by checking whether each pixel is within the masks and then mapping their indexes.

    Parameters
    ----------
    total_sparse_pixels
        The total number of pixels in the sparse grid which fall within the masks.
    mask
        The masks within which spare pixels must be inside
    unmasked_sparse_grid_pixel_centres
        The centres of the unmasked sparse grid pixels.
    """

    unmasked_sparse_for_sparse = np.zeros(total_sparse_pixels)

    pixel_index = 0

    for full_pixel_index in range(unmasked_sparse_grid_pixel_centres.shape[0]):
        y = unmasked_sparse_grid_pixel_centres[full_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[full_pixel_index, 1]

        if not mask[y, x]:
            unmasked_sparse_for_sparse[pixel_index] = full_pixel_index
            pixel_index += 1

    return unmasked_sparse_for_sparse


@numba_util.jit()
def sparse_for_unmasked_sparse_from(
    mask: np.ndarray,
    unmasked_sparse_grid_pixel_centres: np.ndarray,
    total_sparse_pixels: int,
) -> np.ndarray:
    """
    Returns the util between every sparse-rid pixel and masked pixel on a grid. This is performed by checking whether
    each pixel is within the masks and then mapping their indexes.

    Spare pixels are paired with the next masked pixel index. This may mean that a pixel is not paired with a
    pixel near it, if the next pixel is on the next row of the grid. This is not a problem, as it is only
    unmasked pixels that are referenced when perform certain mapping where this information is not required.

    Parameters
    ----------
    mask
        The masks within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres
        The centres of the unmasked pixelization grid pixels.
    total_sparse_pixels
        The total number of pixels in the pixelization grid which fall within the masks.
    """

    total_unmasked_sparse_pixels = unmasked_sparse_grid_pixel_centres.shape[0]

    sparse_for_unmasked_sparse = np.zeros(total_unmasked_sparse_pixels)
    pixel_index = 0

    for unmasked_sparse_pixel_index in range(total_unmasked_sparse_pixels):
        y = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 1]

        sparse_for_unmasked_sparse[unmasked_sparse_pixel_index] = pixel_index

        if not mask[y, x]:
            if pixel_index < total_sparse_pixels - 1:
                pixel_index += 1

    return sparse_for_unmasked_sparse


@numba_util.jit()
def sparse_slim_index_for_mask_slim_index_from(
    regular_to_unmasked_sparse: np.ndarray, sparse_for_unmasked_sparse: np.ndarray
) -> np.ndarray:
    """
    Using the mapping between a grid and unmasked sparse grid, compute the mapping of the slimmed indexes between the
    sparse grid and the unmasked pixels in a mask.

    Parameters
    ----------
    regular_to_unmasked_sparse
        The index mapping between every unmasked pixel in the mask and masked sparse-grid pixel.
    sparse_for_unmasked_sparse
        The index mapping between every masked sparse-grid pixel and unmasked sparse-grid pixel.

    Returns
    -------
    np.ndarray
        The mapping of every 1D index on the unmasked sparse grid to the unmasked 1D index of pixels in the mask it is
        compared to.
    """
    total_regular_pixels = regular_to_unmasked_sparse.shape[0]

    sparse_slim_index_for_mask_slim_index = np.zeros(total_regular_pixels)

    for regular_index in range(total_regular_pixels):
        sparse_slim_index_for_mask_slim_index[
            regular_index
        ] = sparse_for_unmasked_sparse[regular_to_unmasked_sparse[regular_index]]

    return sparse_slim_index_for_mask_slim_index


@numba_util.jit()
def sparse_grid_via_unmasked_from(
    unmasked_sparse_grid: np.ndarray, unmasked_sparse_for_sparse: np.ndarray
) -> np.ndarray:
    """
    Use the unmasked sparse grid of (y,x) coordinates and the mapping between these grid pixels to the 1D sparse grid
    inddexes to compute the masked spaase grid of (y,x) coordinates.

    Parameters
    ----------
    unmasked_sparse_grid
        The (y,x) coordinate grid of every unmasked sparse grid pixel.
    unmasked_sparse_for_sparse
        The index mapping between every unmasked sparse 1D index and masked sparse 1D index.

    Returns
    -------
    np.ndarray
        The masked sparse grid of (y,x) Cartesian coordinates.
    """
    total_pix_pixels = unmasked_sparse_for_sparse.shape[0]

    pix_grid = np.zeros((total_pix_pixels, 2))

    for pixel_index in range(total_pix_pixels):
        pix_grid[pixel_index, 0] = unmasked_sparse_grid[
            unmasked_sparse_for_sparse[pixel_index], 0
        ]
        pix_grid[pixel_index, 1] = unmasked_sparse_grid[
            unmasked_sparse_for_sparse[pixel_index], 1
        ]

    return pix_grid


def via_magnification_from(grid: Grid2D, unmasked_sparse_shape: Tuple[int, int]):
    """
    Calculate `Grid2D` from the unmasked 2D shape of the sparse grid.

    This is performed by overlaying the 2D sparse grid (computed from the unmaksed sparse shape) over the edge
    values of the Grid2D.

    This function is used in the `Inversion` package to set up the VoronoiMagnification Mesh.

    Parameters
    ----------
    grid : Grid2D
        The grid of (y,x) scaled coordinates at the centre of every image value (e.g. image-pixels).
    unmasked_sparse_shape
        The 2D shape of the sparse grid which is overlaid over the grid.
    """

    pixel_scales = grid.mask.pixel_scales

    pixel_scales = (
        (grid.shape_native_scaled_interior[0] + pixel_scales[0])
        / (unmasked_sparse_shape[0]),
        (grid.shape_native_scaled_interior[1] + pixel_scales[1])
        / (unmasked_sparse_shape[1]),
    )

    origin = grid.mask.mask_centre

    unmasked_sparse_grid_1d = grid_2d_util.grid_2d_slim_via_shape_native_from(
        shape_native=unmasked_sparse_shape,
        pixel_scales=pixel_scales,
        sub_size=1,
        origin=origin,
    )

    unmasked_sparse_grid_pixel_centres = (
        geometry_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=unmasked_sparse_grid_1d,
            shape_native=grid.mask.shape_native,
            pixel_scales=grid.mask.pixel_scales,
        ).astype("int")
    )

    total_sparse_pixels = mask_2d_util.total_sparse_pixels_2d_from(
        mask_2d=grid.mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    unmasked_sparse_for_sparse = unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_sparse_pixels,
        mask=grid.mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    ).astype("int")

    sparse_grid = sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid_1d,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
    )

    return Grid2DIrregular(values=sparse_grid)