from autoarray import decorator_util
import numpy as np

from autoarray.util import array_util, mask_util


@decorator_util.jit()
def centres_from(shape_2d, pixel_scales, origin):
    """Determine the (y,x) arc-second central coordinates of an array from its shape, pixel-scales and origin.

     The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
     ----------
    shape_2d : (int, int)
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

    y_centre_arcsec = float(shape_2d[0] - 1) / 2 + (origin[0] / pixel_scales[0])
    x_centre_arcsec = float(shape_2d[1] - 1) / 2 - (origin[1] / pixel_scales[1])

    return y_centre_arcsec, x_centre_arcsec


@decorator_util.jit()
def grid_centre_from(grid_1d):
    centre_y = (np.max(grid_1d[:, 0]) + np.min(grid_1d[:, 0])) / 2.0
    centre_x = (np.max(grid_1d[:, 1]) + np.min(grid_1d[:, 1])) / 2.0
    return centre_y, centre_x


@decorator_util.jit()
def grid_1d_via_mask_from(mask, pixel_scales, sub_size, origin=(0.0, 0.0)):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    GridCoordinates are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0. \
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

    total_sub_pixels = mask_util.total_sub_pixels_from(mask, sub_size)

    grid_1d = np.zeros(shape=(total_sub_pixels, 2))

    centres_arcsec = centres_from(
        shape_2d=mask.shape, pixel_scales=pixel_scales, origin=origin
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


def grid_2d_via_mask_from(mask, pixel_scales, sub_size, origin=(0.0, 0.0)):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    GridCoordinates are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0. \
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

    grid_1d = grid_1d_via_mask_from(
        mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
    )

    return sub_grid_2d_from(sub_grid_1d=grid_1d, mask=mask, sub_size=sub_size)


def grid_1d_via_shape_2d_from(shape_2d, pixel_scales, sub_size, origin=(0.0, 0.0)):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    GridCoordinates are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0]. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_pixels**2*sub_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    shape_2d : (int, int)
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
    return grid_1d_via_mask_from(
        mask=np.full(fill_value=False, shape=shape_2d),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


def grid_2d_via_shape_2d_from(shape_2d, pixel_scales, sub_size, origin=(0.0, 0.0)):
    """ For a sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_size, sub_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this 2D mask array.

    GridCoordinates are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0]. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_pixels**2*sub_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    shape_2d : (int, int)
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
    return grid_2d_via_mask_from(
        mask=np.full(fill_value=False, shape=shape_2d),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


@decorator_util.jit()
def grid_pixels_1d_from(grid_scaled_1d, shape_2d, pixel_scales, origin=(0.0, 0.0)):
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
    grid_scaled_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel value coordinates.
    shape_2d : (int, int)
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

    grid_pixels_1d = np.zeros((grid_scaled_1d.shape[0], 2))

    centres_arcsec = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_scaled_1d.shape[0]):

        grid_pixels_1d[i, 0] = (
            (-grid_scaled_1d[i, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5
        )
        grid_pixels_1d[i, 1] = (
            (grid_scaled_1d[i, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5
        )

    return grid_pixels_1d


@decorator_util.jit()
def grid_pixel_centres_1d_from(
    grid_scaled_1d, shape_2d, pixel_scales, origin=(0.0, 0.0)
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
    grid_scaled_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel indexes.
    shape_2d : (int, int)
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

    grid_pixels_1d = np.zeros((grid_scaled_1d.shape[0], 2))

    centres_arcsec = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_scaled_1d.shape[0]):

        grid_pixels_1d[i, 0] = int(
            (-grid_scaled_1d[i, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5
        )
        grid_pixels_1d[i, 1] = int(
            (grid_scaled_1d[i, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5
        )

    return grid_pixels_1d


@decorator_util.jit()
def grid_pixel_indexes_1d_from(
    grid_scaled_1d, shape_2d, pixel_scales, origin=(0.0, 0.0)
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
    grid_scaled_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to 1D pixel indexes.
    shape_2d : (int, int)
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

    grid_pixels_1d = grid_pixel_centres_1d_from(
        grid_scaled_1d=grid_scaled_1d,
        shape_2d=shape_2d,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    grid_pixel_indexes_1d = np.zeros(grid_pixels_1d.shape[0])

    for i in range(grid_pixels_1d.shape[0]):

        grid_pixel_indexes_1d[i] = int(
            grid_pixels_1d[i, 0] * shape_2d[1] + grid_pixels_1d[i, 1]
        )

    return grid_pixel_indexes_1d


@decorator_util.jit()
def grid_scaled_1d_from(grid_pixels_1d, shape_2d, pixel_scales, origin=(0.0, 0.0)):
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
    shape_2d : (int, int)
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

    centres_arcsec = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
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
def grid_pixel_centres_2d_from(
    grid_arcsec_2d, shape_2d, pixel_scales, origin=(0.0, 0.0)
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
    shape_2d : (int, int)
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

    centres_arcsec = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
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
def furthest_grid_1d_index_from(grid_1d, grid_1d_indexes, coordinate):

    distance_from_centre = 0.0

    for grid_1d_index in grid_1d_indexes:
        y = grid_1d[grid_1d_index, 0]
        x = grid_1d[grid_1d_index, 1]
        distance_from_centre_new = (x - coordinate[1]) ** 2 + (y - coordinate[0]) ** 2
        if distance_from_centre_new >= distance_from_centre:
            distance_from_centre = distance_from_centre_new
            furthest_grid_1d_index = grid_1d_index

    return furthest_grid_1d_index


def sub_grid_1d_from(sub_grid_2d, mask, sub_size):
    """For a 2D grid and mask, map the values of all unmasked pixels to a 1D grid.

    The pixel coordinate origin is at the top left corner of the 2D grid and goes right-wards and downwards, such \
    that for an grid of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D grid will correspond to index 0 of the 1D grid.
    - pixel [0,1] of the 2D grid will correspond to index 1 of the 1D grid.
    - pixel [1,0] of the 2D grid will correspond to index 4 of the 1D grid.

    Parameters
     ----------
    mask : ndgrid
        A 2D grid of bools, where *False* values are unmasked and included in the util.
    sub_grid_2d : ndgrid
        The 2D grid of values which are mapped to a 1D grid.

    Returns
    --------
    ndgrid
        A 1D grid of values mapped from the 2D grid with dimensions (total_unmasked_pixels).

    Examples
    --------
    mask = np.grid([[True, False, True],
                     [False, False, False]
                     [True, False, True]])

    grid_2d = np.grid([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                        [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
                        [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]]])

    grid_1d = map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask=mask, grid_2d=grid_2d)
    """

    sub_grid_1d_y = array_util.sub_array_1d_from(
        sub_array_2d=sub_grid_2d[:, :, 0], mask=mask, sub_size=sub_size
    )

    sub_grid_1d_x = array_util.sub_array_1d_from(
        sub_array_2d=sub_grid_2d[:, :, 1], mask=mask, sub_size=sub_size
    )

    return np.stack((sub_grid_1d_y, sub_grid_1d_x), axis=-1)


def sub_grid_2d_from(sub_grid_1d, mask, sub_size):
    """For a 1D array that was computed by util unmasked values from a 2D array of shape (rows, columns), map its \
    values back to the original 2D array where masked values are set to zero.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels, \
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
     ----------
    sub_grid_1d : ndarray
        The 1D array of values which are mapped to a 2D array.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.
    sub_one_to_two : ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    --------
    ndarray
        A 2D array of values mapped from the 1D array with dimensions shape.

    Examples
    --------
    one_to_two = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    array_1d = np.array([[2.0, 4.0, 5.0, 6.0, 8.0])

    array_2d = map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d=array_1d, shape=(3,3),
                                                                                  one_to_two=one_to_two)
    """

    sub_grid_2d_y = array_util.sub_array_2d_from(
        sub_array_1d=sub_grid_1d[:, 0], mask=mask, sub_size=sub_size
    )

    sub_grid_2d_x = array_util.sub_array_2d_from(
        sub_array_1d=sub_grid_1d[:, 1], mask=mask, sub_size=sub_size
    )

    return np.stack((sub_grid_2d_y, sub_grid_2d_x), axis=-1)


@decorator_util.jit()
def grid_upscaled_1d_from(grid_1d, upscale_factor, pixel_scales):
    """From an input 1D grid, return an upscaled 1D grid where (y,x) coordinates are added at an upscaled resolution
    to each grid coordinate, analogous to a sub-grid.

    Parameters
    ----------
    grid_1d : ndarray
        The irregular 1D grid of (y,x) coordinates over which a square uniform grid is overlaid.
    upscale_factor : int
        The upscaled resolution at which the new grid coordinates are computed.
    pixel_scales : (float, float)
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    grid_upscaled_1d = np.zeros(shape=(grid_1d.shape[0] * upscale_factor ** 2, 2))

    upscale_index = 0

    y_upscale_half = pixel_scales[0] / 2
    y_upscale_step = pixel_scales[0] / upscale_factor

    x_upscale_half = pixel_scales[1] / 2
    x_upscale_step = pixel_scales[1] / upscale_factor

    for grid_index in range(grid_1d.shape[0]):

        y_grid = grid_1d[grid_index, 0]
        x_grid = grid_1d[grid_index, 1]

        for y in range(upscale_factor):
            for x in range(upscale_factor):

                grid_upscaled_1d[upscale_index, 0] = (
                    y_grid
                    + y_upscale_half
                    - y * y_upscale_step
                    - (y_upscale_step / 2.0)
                )
                grid_upscaled_1d[upscale_index, 1] = (
                    x_grid
                    - x_upscale_half
                    + x * x_upscale_step
                    + (x_upscale_step / 2.0)
                )

                upscale_index += 1

    return grid_upscaled_1d
