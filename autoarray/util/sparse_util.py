import logging

import numpy as np

from autoarray import decorator_util

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


@decorator_util.jit()
def unmasked_sparse_for_sparse_from_mask_2d_and_pixel_centres(
    total_sparse_pixels, mask_2d, unmasked_sparse_grid_pixel_centres
):
    """Determine the util between every masked pixelization-grid pixel and pixelization-grid pixel. This is \
    performed by checking whether each pixelization-grid pixel is within the masks, and util the indexes.

    Parameters
    -----------
    total_sparse_pixels : int
        The total number of pixels in the pixelization grid which fall within the masks.
    mask_2d : imaging.masks.Mask
        The masks within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    unmasked_sparse_for_sparse = np.zeros(total_sparse_pixels)

    pixel_index = 0

    for full_pixel_index in range(unmasked_sparse_grid_pixel_centres.shape[0]):

        y = unmasked_sparse_grid_pixel_centres[full_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[full_pixel_index, 1]

        if not mask_2d[y, x]:
            unmasked_sparse_for_sparse[pixel_index] = full_pixel_index
            pixel_index += 1

    return unmasked_sparse_for_sparse


@decorator_util.jit()
def sparse_for_unmasked_sparse_from_mask_2d_and_pixel_centres(
    mask_2d, unmasked_sparse_grid_pixel_centres, total_sparse_pixels
):
    """Determine the util between every pixelization-grid pixel and masked pixelization-grid pixel. This is \
    performed by checking whether each pixelization-grid pixel is within the masks, and util the indexes.

    Pixelization pixels are paired with the next masked pixel index. This may mean that a pixel is not paired with a \
    pixel near it, if the next pixel is on the next row of the grid. This is not a problem, as it is only \
    unmasked pixels that are referenced when computing image_to_pix, which is what this array is used for.

    Parameters
    -----------
    total_sparse_pixels : int
        The total number of pixels in the pixelization grid which fall within the masks.
    mask_2d : imaging.masks.Mask
        The masks within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_unmasked_sparse_pixels = unmasked_sparse_grid_pixel_centres.shape[0]

    sparse_for_unmasked_sparse = np.zeros(total_unmasked_sparse_pixels)
    pixel_index = 0

    for unmasked_sparse_pixel_index in range(total_unmasked_sparse_pixels):

        y = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 1]

        sparse_for_unmasked_sparse[unmasked_sparse_pixel_index] = pixel_index

        if not mask_2d[y, x]:
            if pixel_index < total_sparse_pixels - 1:
                pixel_index += 1

    return sparse_for_unmasked_sparse


@decorator_util.jit()
def sparse_1d_index_for_mask_1d_index_from_sparse_mappings(
    regular_to_unmasked_sparse, sparse_for_unmasked_sparse
):
    """Using the util between the grid and unmasked pixelization grid, compute the util between each \
    pixel and the masked pixelization grid.

    Parameters
    -----------
    regular_to_unmasked_sparse : ndarray
        The index util between every pixel and masked pixelization pixel.
    sparse_for_unmasked_sparse : ndarray
        The index util between every masked pixelization pixel and unmasked pixelization pixel.
    """
    total_regular_pixels = regular_to_unmasked_sparse.shape[0]

    sparse_1d_index_for_mask_1d_index = np.zeros(total_regular_pixels)

    for regular_index in range(total_regular_pixels):
        sparse_1d_index_for_mask_1d_index[regular_index] = sparse_for_unmasked_sparse[
            regular_to_unmasked_sparse[regular_index]
        ]

    return sparse_1d_index_for_mask_1d_index


@decorator_util.jit()
def sparse_grid_from_unmasked_sparse_grid(
    unmasked_sparse_grid, unmasked_sparse_for_sparse
):
    """Use the central arc-second coordinate of every unmasked pixelization grid's pixels and util between each \
    pixelization pixel and unmasked pixelization pixel to compute the central arc-second coordinate of every masked \
    pixelization grid pixel.

    Parameters
    -----------
    unmasked_sparse_grid : ndarray
        The (y,x) arc-second centre of every unmasked pixelization grid pixel.
    unmasked_sparse_for_sparse : ndarray
        The index util between every pixelization pixel and masked pixelization pixel.
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


@decorator_util.jit()
def sparse_1d_index_for_mask_1d_index_from_binned_grid(
    sparse_labels,
    binned_mask_1d_index_to_mask_1d_indexes,
    binned_mask_1d_index_to_mask_1d_sizes,
    total_unbinned_pixels,
):
    sparse_1d_index_for_mask_1d_index = np.zeros(total_unbinned_pixels)

    for cluster_index in range(binned_mask_1d_index_to_mask_1d_indexes.shape[0]):
        for cluster_count in range(
            binned_mask_1d_index_to_mask_1d_sizes[cluster_index]
        ):
            regular_index = binned_mask_1d_index_to_mask_1d_indexes[
                cluster_index, cluster_count
            ]
            sparse_1d_index_for_mask_1d_index[regular_index] = sparse_labels[
                cluster_index
            ]

    return sparse_1d_index_for_mask_1d_index
