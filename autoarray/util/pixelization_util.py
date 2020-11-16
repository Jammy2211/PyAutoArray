import numpy as np
from autoarray import decorator_util


@decorator_util.jit()
def rectangular_neighbors_from(shape_2d: (int, int)) -> (np.ndarray, np.ndarray):
    """
    Returns the 4 adjacent neighbors of every pixel on a rectangular array or grid as an ndarray of shape
    [total_pixels, 4], using the uniformity of the rectangular grid's geometry to speed up the computation.

    Entries with values of ``-1`` signify edge pixels which do not have 4 neighbors. This function therefore also
    returns an ndarray of with the number of neighbors of every pixel, so when the neighbors are used the code knows
    how many neighbors to iterate over.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For example, on a 3x3 grid:

    - Pixel 0 is at the top-left and has two neighbors, the pixel to its right  (with index 1) and the  pixel below
    it (with index 3). Therefore, the pixel_neighbors[0,:] = [1, 3, -1, -1] and pixel_neighbors_size[0] = 2.

    - Pixel 1 is at the top-middle and has three neighbors, to its left (index 0, right (index 2) and below it
    (index 4). Therefore, pixel_neighbors[1,:] = [0, 2, 4, 1] and pixel_neighbors_size[1] = 3.

    - For pixel 4, the central pixel, pixel_neighbors[4,:] = [1, 3, 5, 7] and pixel_neighbors_size[4] = 4.

    Parameters
    ----------
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixels = shape_2d[0] * shape_2d[1]

    pixel_neighbors = -1 * np.ones(shape=(pixels, 4))
    pixel_neighbors_size = np.zeros(pixels)

    pixel_neighbors, pixel_neighbors_size = rectangular_corner_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_size=pixel_neighbors_size,
        shape_2d=shape_2d,
        pixels=pixels,
    )
    pixel_neighbors, pixel_neighbors_size = rectangular_top_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_size=pixel_neighbors_size,
        shape_2d=shape_2d,
        pixels=pixels,
    )
    pixel_neighbors, pixel_neighbors_size = rectangular_left_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_size=pixel_neighbors_size,
        shape_2d=shape_2d,
        pixels=pixels,
    )
    pixel_neighbors, pixel_neighbors_size = rectangular_right_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_size=pixel_neighbors_size,
        shape_2d=shape_2d,
        pixels=pixels,
    )
    pixel_neighbors, pixel_neighbors_size = rectangular_bottom_edge_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_size=pixel_neighbors_size,
        shape_2d=shape_2d,
        pixels=pixels,
    )
    pixel_neighbors, pixel_neighbors_size = rectangular_central_neighbors(
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_size=pixel_neighbors_size,
        shape_2d=shape_2d,
        pixels=pixels,
    )

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def rectangular_corner_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
    shape_2d: (int, int),
    pixels: int,
) -> (np.ndarray, np.ndarray):
    """
    Updates the ``pixel_neighbors`` and ``pixel_neighbors_size`` arrays described in the function
    ``rectangular_neighbors_from`` for pixels on the rectangular array or grid that are on the corners.

    Parameters
    ----------
    pixel_neighbors : np.ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : np.ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    pixels : int
        The number of pixels on the rectangular grid (e.g. shape_2d[0] * shape_2d[1]).
    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixel_neighbors[0, 0:2] = np.array([1, shape_2d[1]])
    pixel_neighbors_size[0] = 2

    pixel_neighbors[shape_2d[1] - 1, 0:2] = np.array(
        [shape_2d[1] - 2, shape_2d[1] + shape_2d[1] - 1]
    )
    pixel_neighbors_size[shape_2d[1] - 1] = 2

    pixel_neighbors[pixels - shape_2d[1], 0:2] = np.array(
        [pixels - shape_2d[1] * 2, pixels - shape_2d[1] + 1]
    )
    pixel_neighbors_size[pixels - shape_2d[1]] = 2

    pixel_neighbors[pixels - 1, 0:2] = np.array([pixels - shape_2d[1] - 1, pixels - 2])
    pixel_neighbors_size[pixels - 1] = 2

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def rectangular_top_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
    shape_2d: (int, int),
    pixels: int,
) -> (np.ndarray, np.ndarray):
    """
    Updates the ``pixel_neighbors`` and ``pixel_neighbors_size`` arrays described in the function
    ``rectangular_neighbors_from`` for pixels on the top edge of the rectangular array or grid.

    Parameters
    ----------
    pixel_neighbors : np.ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : np.ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    pixels : int
        The number of pixels on the rectangular grid (e.g. shape_2d[0] * shape_2d[1]).
    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_2d[1] - 1):
        pixel_index = pix
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [pixel_index - 1, pixel_index + 1, pixel_index + shape_2d[1]]
        )
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def rectangular_left_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
    shape_2d: (int, int),
    pixels: int,
) -> (np.ndarray, np.ndarray):
    """
    Updates the ``pixel_neighbors`` and ``pixel_neighbors_size`` arrays described in the function
    ``rectangular_neighbors_from`` for pixels on the left edge of the rectangular array or grid.

    Parameters
    ----------
    pixel_neighbors : np.ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : np.ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    pixels : int
        The number of pixels on the rectangular grid (e.g. shape_2d[0] * shape_2d[1]).
    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_2d[0] - 1):
        pixel_index = pix * shape_2d[1]
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [pixel_index - shape_2d[1], pixel_index + 1, pixel_index + shape_2d[1]]
        )
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def rectangular_right_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
    shape_2d: (int, int),
    pixels: int,
) -> (np.ndarray, np.ndarray):
    """
    Updates the ``pixel_neighbors`` and ``pixel_neighbors_size`` arrays described in the function
    ``rectangular_neighbors_from`` for pixels on the right edge of the rectangular array or grid.

    Parameters
    ----------
    pixel_neighbors : np.ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : np.ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    pixels : int
        The number of pixels on the rectangular grid (e.g. shape_2d[0] * shape_2d[1]).
    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_2d[0] - 1):
        pixel_index = pix * shape_2d[1] + shape_2d[1] - 1
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [pixel_index - shape_2d[1], pixel_index - 1, pixel_index + shape_2d[1]]
        )
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def rectangular_bottom_edge_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
    shape_2d: (int, int),
    pixels: int,
) -> (np.ndarray, np.ndarray):
    """
    Updates the ``pixel_neighbors`` and ``pixel_neighbors_size`` arrays described in the function
    ``rectangular_neighbors_from`` for pixels on the bottom edge of the rectangular array or grid.

    Parameters
    ----------
    pixel_neighbors : np.ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : np.ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    pixels : int
        The number of pixels on the rectangular grid (e.g. shape_2d[0] * shape_2d[1]).
    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for pix in range(1, shape_2d[1] - 1):
        pixel_index = pixels - pix - 1
        pixel_neighbors[pixel_index, 0:3] = np.array(
            [pixel_index - shape_2d[1], pixel_index - 1, pixel_index + 1]
        )
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def rectangular_central_neighbors(
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
    shape_2d: (int, int),
    pixels: int,
) -> (np.ndarray, np.ndarray):
    """
    Updates the ``pixel_neighbors`` and ``pixel_neighbors_size`` arrays described in the function
    ``rectangular_neighbors_from`` for pixels in the central pixels of the rectangular array or grid.

    Parameters
    ----------
    pixel_neighbors : np.ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : np.ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    pixels : int
        The number of pixels on the rectangular grid (e.g. shape_2d[0] * shape_2d[1]).
    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    for x in range(1, shape_2d[0] - 1):
        for y in range(1, shape_2d[1] - 1):
            pixel_index = x * shape_2d[1] + y
            pixel_neighbors[pixel_index, 0:4] = np.array(
                [
                    pixel_index - shape_2d[1],
                    pixel_index - 1,
                    pixel_index + 1,
                    pixel_index + shape_2d[1],
                ]
            )
            pixel_neighbors_size[pixel_index] = 4

    return pixel_neighbors, pixel_neighbors_size


@decorator_util.jit()
def voronoi_neighbors_from(
    pixels: int, ridge_points: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Returns the adjacent neighbors of every pixel on a Voronoi array or grid as an ndarray of shape
    [total_pixels, voronoi_pixel_with_max_neighbors], using the ridge_points output from the ``scipy.spatial.Voronoi()
    method.

    Entries with values of ``-1`` signify edge pixels which do not have neighbors (they are required as the ndarray size
    is set by the Voronoi pixel with the most neighbors). This function therefore also returns an ndarray of with the
    number of neighbors of every pixel, so when the neighbors are used the code knows
    how many neighbors to iterate over.

    Parameters
    ----------
    shape_2d : (int, int)
        The shape of the rectangular 2D array or grid which the pixels are defined on.
    ridge_points : np.ndarray
        Contains the information on every Voronoi source pixel and its neighbors.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixel_neighbors_size = np.zeros(shape=(pixels))

    for ridge_index in range(ridge_points.shape[0]):
        pair0 = ridge_points[ridge_index, 0]
        pair1 = ridge_points[ridge_index, 1]
        pixel_neighbors_size[pair0] += 1
        pixel_neighbors_size[pair1] += 1

    pixel_neighbors_index = np.zeros(shape=(pixels))
    pixel_neighbors = -1 * np.ones(shape=(pixels, int(np.max(pixel_neighbors_size))))

    for ridge_index in range(ridge_points.shape[0]):
        pair0 = ridge_points[ridge_index, 0]
        pair1 = ridge_points[ridge_index, 1]
        pixel_neighbors[pair0, int(pixel_neighbors_index[pair0])] = pair1
        pixel_neighbors[pair1, int(pixel_neighbors_index[pair1])] = pair0
        pixel_neighbors_index[pair0] += 1
        pixel_neighbors_index[pair1] += 1

    return pixel_neighbors, pixel_neighbors_size
