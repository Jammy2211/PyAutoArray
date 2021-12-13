import numpy as np
from typing import Tuple

from autoarray import numba_util


@numba_util.jit()
def rectangular_neighbors_from(
    shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the 4 (or less) adjacent neighbors of every pixel on a rectangular pixelization as an ndarray of shape
    [total_pixels, 4], called `pixel_neighbors`. This uses the uniformity of the rectangular grid's geometry to speed 
    up the computation.

    Entries with values of `-1` signify edge pixels which do not have neighbors. This function therefore also returns
    an ndarray with the number of neighbors of every pixel, `pixel_neighbors_sizes`, which is iterated over when using 
    the `pixel_neighbors` ndarray.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For example, on a 3x3 grid:

    - Pixel 0 is at the top-left and has two neighbors, the pixel to its right  (with index 1) and the pixel below
    it (with index 3). Therefore, the pixel_neighbors[0,:] = [1, 3, -1, -1] and pixel_neighbors_sizes[0] = 2.

    - Pixel 1 is at the top-middle and has three neighbors, to its left (index 0, right (index 2) and below it
    (index 4). Therefore, pixel_neighbors[1,:] = [0, 2, 4, -1] and pixel_neighbors_sizes[1] = 3.

    - For pixel 4, the central pixel, pixel_neighbors[4,:] = [1, 3, 5, 7] and pixel_neighbors_sizes[4] = 4.

    Parameters
    ----------
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The ndarrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixels = int(shape_native[0] * shape_native[1])

    pixel_neighbors = -1 * np.ones(shape=(pixels, 4))
    pixel_neighbors_sizes = np.zeros(pixels)

    pixel_neighbors, pixel_neighbors_sizes = rectangular_corner_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
        shape_native=shape_native,
    )
    pixel_neighbors, pixel_neighbors_sizes = rectangular_top_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
        shape_native=shape_native,
    )
    pixel_neighbors, pixel_neighbors_sizes = rectangular_left_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
        shape_native=shape_native,
    )
    pixel_neighbors, pixel_neighbors_sizes = rectangular_right_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
        shape_native=shape_native,
    )
    pixel_neighbors, pixel_neighbors_sizes = rectangular_bottom_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
        shape_native=shape_native,
    )
    pixel_neighbors, pixel_neighbors_sizes = rectangular_central_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
        shape_native=shape_native,
    )

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def rectangular_corner_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `pixel_neighbors` and `pixel_neighbors_sizes` arrays described in the function 
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the corners.

    Parameters
    ----------
    pixel_neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular 
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.
        
    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixels = int(shape_native[0] * shape_native[1])

    pixel_neighbors[0, 0:2] = np.array([1, shape_native[1]])
    pixel_neighbors_sizes[0] = 2

    pixel_neighbors[shape_native[1] - 1, 0:2] = np.array(
        [shape_native[1] - 2, shape_native[1] + shape_native[1] - 1]
    )
    pixel_neighbors_sizes[shape_native[1] - 1] = 2

    pixel_neighbors[pixels - shape_native[1], 0:2] = np.array(
        [pixels - shape_native[1] * 2, pixels - shape_native[1] + 1]
    )
    pixel_neighbors_sizes[pixels - shape_native[1]] = 2

    pixel_neighbors[pixels - 1, 0:2] = np.array(
        [pixels - shape_native[1] - 1, pixels - 2]
    )
    pixel_neighbors_sizes[pixels - 1] = 2

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def rectangular_top_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `pixel_neighbors` and `pixel_neighbors_sizes` arrays described in the function 
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the top edge.

    Parameters
    ----------
    pixel_neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular 
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.
        
    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_native[1] - 1):
        pixel_index = pix
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [pixel_index - 1, pixel_index + 1, pixel_index + shape_native[1]]
        )
        pixel_neighbors_sizes[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def rectangular_left_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `pixel_neighbors` and `pixel_neighbors_sizes` arrays described in the function 
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the left edge.

    Parameters
    ----------
    pixel_neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular 
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.
        
    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_native[0] - 1):
        pixel_index = pix * shape_native[1]
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [
                pixel_index - shape_native[1],
                pixel_index + 1,
                pixel_index + shape_native[1],
            ]
        )
        pixel_neighbors_sizes[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def rectangular_right_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `pixel_neighbors` and `pixel_neighbors_sizes` arrays described in the function 
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the right edge.

    Parameters
    ----------
    pixel_neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular 
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.
        
    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_native[0] - 1):
        pixel_index = pix * shape_native[1] + shape_native[1] - 1
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [
                pixel_index - shape_native[1],
                pixel_index - 1,
                pixel_index + shape_native[1],
            ]
        )
        pixel_neighbors_sizes[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def rectangular_bottom_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `pixel_neighbors` and `pixel_neighbors_sizes` arrays described in the function 
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the bottom edge.

    Parameters
    ----------
    pixel_neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular 
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.
        
    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    pixels = int(shape_native[0] * shape_native[1])

    for pix in range(1, shape_native[1] - 1):
        pixel_index = pixels - pix - 1
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [pixel_index - shape_native[1], pixel_index - 1, pixel_index + 1]
        )
        pixel_neighbors_sizes[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def rectangular_central_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `pixel_neighbors` and `pixel_neighbors_sizes` arrays described in the function 
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are in the centre and thus have 4
    adjacent neighbors.

    Parameters
    ----------
    pixel_neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular 
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.
        
    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for x in range(1, shape_native[0] - 1):
        for y in range(1, shape_native[1] - 1):
            pixel_index = x * shape_native[1] + y
            pixel_neighbors[pixel_index, 0:4] = np.array(
                [
                    pixel_index - shape_native[1],
                    pixel_index - 1,
                    pixel_index + 1,
                    pixel_index + shape_native[1],
                ]
            )
            pixel_neighbors_sizes[pixel_index] = 4

    return pixel_neighbors, pixel_neighbors_sizes


@numba_util.jit()
def voronoi_neighbors_from(
    pixels: int, ridge_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the adjacent neighbors of every pixel on a Voronoi pixelization as an ndarray of shape
    [total_pixels, voronoi_pixel_with_max_neighbors], using the `ridge_points` output from the `scipy.spatial.Voronoi()`
    object.

    Entries with values of `-1` signify edge pixels which do not have neighbors. This function therefore also returns
    an ndarray with the number of neighbors of every pixel, `pixel_neighbors_sizes`, which is iterated over when using
    the `pixel_neighbors` ndarray.

    Indexing is defined in an arbritrary manner due to the irregular nature of a Voronoi pixelization.

    For example, if `pixel_neighbors[0,:] = [1, 5, 36, 2, -1, -1]`, this informs us that the first Voronoi pixel has
    4 neighbors which have indexes 1, 5, 36, 2. Correspondingly `pixel_neighbors_sizes[0] = 4`.

    Parameters
    ----------
    pixels
        The number of pixels on the Voronoi pixelization.
    ridge_points
        Contains the information on every Voronoi source pixel and its neighbors.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixel_neighbors_sizes = np.zeros(shape=(pixels))

    for ridge_index in range(ridge_points.shape[0]):
        pair0 = ridge_points[ridge_index, 0]
        pair1 = ridge_points[ridge_index, 1]
        pixel_neighbors_sizes[pair0] += 1
        pixel_neighbors_sizes[pair1] += 1

    pixel_neighbors_index = np.zeros(shape=(pixels))
    pixel_neighbors = -1 * np.ones(shape=(pixels, int(np.max(pixel_neighbors_sizes))))

    for ridge_index in range(ridge_points.shape[0]):
        pair0 = ridge_points[ridge_index, 0]
        pair1 = ridge_points[ridge_index, 1]
        pixel_neighbors[pair0, int(pixel_neighbors_index[pair0])] = pair1
        pixel_neighbors[pair1, int(pixel_neighbors_index[pair1])] = pair0
        pixel_neighbors_index[pair0] += 1
        pixel_neighbors_index[pair1] += 1

    #print('pixel_neighbors:')
    #print(pixel_neighbors)


    #print('pixel_neighbors shape:')
    #print(np.shape(pixel_neighbors))

    return pixel_neighbors, pixel_neighbors_sizes


