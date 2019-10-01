from autoarray import decorator_util
import numpy as np

from autoarray.util import mask_util
from autoarray.mapping_util import grid_mapping_util


@decorator_util.jit()
def centres_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin):
    """Determine the (y,x) arc-second central coordinates of an array from its shape, pixel-scales and origin.

     The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the arc-second centre is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D array.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the centre is shifted to.

    Returns
    --------
    tuple (float, float)
        The (y,x) arc-second central coordinates of the input array.

    Examples
    --------
    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=(5,5), pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    y_centre_arcsec = float(shape[0] - 1) / 2 + (origin[0] / pixel_scales[0])
    x_centre_arcsec = float(shape[1] - 1) / 2 - (origin[1] / pixel_scales[1])

    return (y_centre_arcsec, x_centre_arcsec)


@decorator_util.jit()
def grid_centre_from_grid_1d(grid_1d):
    centre_y = (np.max(grid_1d[:, 0]) + np.min(grid_1d[:, 0])) / 2.0
    centre_x = (np.max(grid_1d[:, 1]) + np.min(grid_1d[:, 1])) / 2.0
    return (centre_y, centre_x)


@decorator_util.jit()
def grid_1d_from_mask_pixel_scales_sub_size_and_origin(
    mask, pixel_scales, sub_size, origin=(0.0, 0.0)
):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    Coordinates are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked and therefore included as part of the calculated \
        sub-grid.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    --------
    ndarray
        A sub grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_1d = sub_grid_1d_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_size(mask, sub_size)

    grid_1d = np.zeros(shape=(total_sub_pixels, 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(
        shape=mask.shape, pixel_scales=pixel_scales, origin=origin
    )

    sub_index = 0

    y_sub_half = pixel_scales[0] / 2
    y_sub_step = pixel_scales[0] / (sub_size)

    x_sub_half = pixel_scales[1] / 2
    x_sub_step = pixel_scales[1] / (sub_size)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            if not mask[y, x]:

                y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
                x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

                for y1 in range(sub_size):
                    for x1 in range(sub_size):

                        grid_1d[sub_index, 0] = -(
                            y_arcsec - y_sub_half + y1 * y_sub_step + (y_sub_step / 2.0)
                        )
                        grid_1d[sub_index, 1] = (
                            x_arcsec - x_sub_half + x1 * x_sub_step + (x_sub_step / 2.0)
                        )
                        sub_index += 1

    return grid_1d


def grid_2d_from_mask_pixel_scales_sub_size_and_origin(
    mask, pixel_scales, sub_size, origin=(0.0, 0.0)
):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    Coordinates are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked and therefore included as part of the calculated \
        sub-grid.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    --------
    ndarray
        A sub grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    sub_grid_1d = sub_grid_1d_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_1d = grid_1d_from_mask_pixel_scales_sub_size_and_origin(
        mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
    )

    return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
        sub_grid_1d=grid_1d, mask=mask, sub_size=sub_size
    )


def grid_1d_from_shape_pixel_scales_sub_size_and_origin(
    shape, pixel_scales, sub_size, origin=(0.0, 0.0)
):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    Coordinates are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0]. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_pixels**2*sub_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the sub-grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    --------
    ndarray
        A sub grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    sub_grid_1d = sub_grid_1d_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    return grid_1d_from_mask_pixel_scales_sub_size_and_origin(
        mask=np.full(fill_value=False, shape=shape),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


def grid_2d_from_shape_pixel_scales_sub_size_and_origin(
    shape, pixel_scales, sub_size, origin=(0.0, 0.0)
):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    Coordinates are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0]. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_pixels**2*sub_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the sub-grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    --------
    ndarray
        A sub grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    sub_grid_1d = sub_grid_1d_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    return grid_2d_from_mask_pixel_scales_sub_size_and_origin(
        mask=np.full(fill_value=False, shape=shape),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


@decorator_util.jit()
def grid_pixels_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
    grid_arcsec_1d, shape, pixel_scales, origin=(0.0, 0.0)
):
    """ Convert a 1D grid of (y,x) arc second coordinates to a 1D grid of (y,x) pixel coordinate values. Pixel
    coordinates are returned as floats such that they include the decimal offset from each pixel's top-left corner
    relative to the input arc-second coordinate.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    the highest (most positive) y arc-second coordinate and lowest (most negative) x arc-second coordinate on the gird.

    The arc-second grid is defined by an origin and coordinates are shifted to this origin before computing their \
    1D grid pixel coordinate values.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel value coordinates.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted to.

    Returns
    --------
    ndarray
        A grid of (y,x) pixel-value coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = np.zeros((grid_arcsec_1d.shape[0], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(
        shape=shape, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_arcsec_1d.shape[0]):

        grid_pixels_1d[i, 0] = (
            (-grid_arcsec_1d[i, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5
        )
        grid_pixels_1d[i, 1] = (
            (grid_arcsec_1d[i, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5
        )

    return grid_pixels_1d


@decorator_util.jit()
def grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
    grid_arcsec_1d, shape, pixel_scales, origin=(0.0, 0.0)
):
    """ Convert a 1D grid of (y,x) arc second coordinates to a 1D grid of (y,x) pixel values. Pixel coordinates are \
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    the highest (most positive) y arc-second coordinate and lowest (most negative) x arc-second coordinate on the gird.

    The arc-second coordinate grid is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel indexes.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted

    Returns
    --------
    ndarray
        A grid of (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = np.zeros((grid_arcsec_1d.shape[0], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(
        shape=shape, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_arcsec_1d.shape[0]):

        grid_pixels_1d[i, 0] = int(
            (-grid_arcsec_1d[i, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5
        )
        grid_pixels_1d[i, 1] = int(
            (grid_arcsec_1d[i, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5
        )

    return grid_pixels_1d


@decorator_util.jit()
def grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
    grid_arcsec_1d, shape, pixel_scales, origin=(0.0, 0.0)
):
    """ Convert a 1D grid of (y,x) arc second coordinates to a 1D grid of (y,x) pixel 1D indexes. Pixel coordinates
    are returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
    downwards.

    For example:

    The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
    The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
    The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

    The arc-second coordinate grid is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to 1D pixel indexes.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted.

    Returns
    --------
    ndarray
        A grid of 1d pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_pixels_from_grid_arcsec_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
        grid_arcsec_1d=grid_arcsec_1d,
        shape=shape,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    grid_pixel_indexes_1d = np.zeros(grid_pixels_1d.shape[0])

    for i in range(grid_pixels_1d.shape[0]):

        grid_pixel_indexes_1d[i] = int(
            grid_pixels_1d[i, 0] * shape[1] + grid_pixels_1d[i, 1]
        )

    return grid_pixel_indexes_1d


@decorator_util.jit()
def grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
    grid_pixels_1d, shape, pixel_scales, origin=(0.0, 0.0)
):
    """ Convert a 1D grid of (y,x) pixel coordinates to a 1D grid of (y,x) arc second values.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    the highest (most positive) y arc-second coordinate and lowest (most negative) x arc-second coordinate on the gird.

    The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
    origin after computing their values from the 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_pixels_1d: ndarray
        The grid of (y,x) coordinates in pixel values which is converted to arc-second coordinates.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted.

    Returns
    --------
    ndarray
        A grid of 1d arc-second coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_pixels_1d = np.array([[0,0], [0,1], [1,0], [1,1])
    grid_pixels_1d = grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_arcsec_1d = np.zeros((grid_pixels_1d.shape[0], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(
        shape=shape, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_arcsec_1d.shape[0]):

        grid_arcsec_1d[i, 0] = (
            -(grid_pixels_1d[i, 0] - centres_arcsec[0] - 0.5) * pixel_scales[0]
        )
        grid_arcsec_1d[i, 1] = (
            grid_pixels_1d[i, 1] - centres_arcsec[1] - 0.5
        ) * pixel_scales[1]

    return grid_arcsec_1d


@decorator_util.jit()
def grid_pixel_centres_2d_from_grid_arcsec_2d_shape_and_pixel_scales(
    grid_arcsec_2d, shape, pixel_scales, origin=(0.0, 0.0)
):
    """ Convert a 2D grid of (y,x) arc second coordinates to a 2D grid of (y,x) pixel values. Pixel coordinates are \
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    the highest (most positive) y arc-second coordinate and lowest (most negative) x arc-second coordinate on the gird.

    The arc-second coordinate grid is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (y_dimension, x_dimension, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel indexes.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted

    Returns
    --------
    ndarray
        A grid of (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d = np.zeros((grid_arcsec_2d.shape[0], grid_arcsec_2d.shape[1], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(
        shape=shape, pixel_scales=pixel_scales, origin=origin
    )

    for y in range(grid_arcsec_2d.shape[0]):
        for x in range(grid_arcsec_2d.shape[1]):
            grid_pixels_2d[y, x, 0] = int(
                (-grid_arcsec_2d[y, x, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5
            )
            grid_pixels_2d[y, x, 1] = int(
                (grid_arcsec_2d[y, x, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5
            )

    return grid_pixels_2d


@decorator_util.jit()
def furthest_grid_1d_index_from_grid_1d_indexes_and_centre(
    grid_1d, grid_1d_indexes, coordinate
):

    distance_from_centre = 0.0

    for grid_1d_index in grid_1d_indexes:
        y = grid_1d[grid_1d_index, 0]
        x = grid_1d[grid_1d_index, 1]
        distance_from_centre_new = (x - coordinate[1]) ** 2 + (y - coordinate[0]) ** 2
        if distance_from_centre_new >= distance_from_centre:
            distance_from_centre = distance_from_centre_new
            furthest_grid_1d_index = grid_1d_index

    return furthest_grid_1d_index
