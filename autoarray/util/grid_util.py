from autoarray import decorator_util
import numpy as np

from autoarray.util import array_util, mask_util


@decorator_util.jit()
def centres_from(
    shape_2d: (int, int), pixel_scales: (float, float), origin: (float, float)
) -> (float, float):
    """
    Returns the (y,x) scaled central coordinates of an array or grid from its 2D shape, pixel-scales and origin.

    The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
    ----------
    shape_2d : (int, int)
        The (y,x) shape of the 2D array or grid the scaled centre is computed for.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D array.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the centre is shifted to.

    Returns
    -------
    tuple (float, float)
        The (y,x) scaled central coordinates of the input array.

    Examples
    --------
    centres_scaled = centres_from(shape_2d=(5,5), pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    y_centre_scaled = float(shape_2d[0] - 1) / 2 + (origin[0] / pixel_scales[0])
    x_centre_scaled = float(shape_2d[1] - 1) / 2 - (origin[1] / pixel_scales[1])

    return y_centre_scaled, x_centre_scaled


@decorator_util.jit()
def grid_centre_from(grid_1d: np.ndarray) -> (float, float):
    """
    Returns the centre of a grid from a 1D grid.

    Parameters
    ----------
    grid_1d :  np.ndarray
        The 1D grid of values which are mapped to a 2D array.

    Returns
    -------
    (float, float)
        The (y,x) central coordinates of the grid.
    """
    centre_y = (np.max(grid_1d[:, 0]) + np.min(grid_1d[:, 0])) / 2.0
    centre_x = (np.max(grid_1d[:, 1]) + np.min(grid_1d[:, 1])) / 2.0
    return centre_y, centre_x


@decorator_util.jit()
def grid_1d_via_mask_from(
    mask: np.ndarray,
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
    a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates a the centre of every sub-pixel defined by this 2D mask array.

    Grid are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
    ----------
    mask : np.ndarray
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        sub-grid.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
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

    centres_scaled = centres_from(
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

                y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
                x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

                for y1 in range(sub_size):
                    for x1 in range(sub_size):

                        grid_1d[sub_index, 0] = -(
                            y_scaled - y_sub_half + y1 * y_sub_step + (y_sub_step / 2.0)
                        )
                        grid_1d[sub_index, 1] = (
                            x_scaled - x_sub_half + x1 * x_sub_step + (x_sub_step / 2.0)
                        )
                        sub_index += 1

    return grid_1d


def grid_2d_via_mask_from(
    mask: np.ndarray,
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into a
    finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

    Grids are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
    ----------
    mask : np.ndarray
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        sub-grid.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
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


def grid_1d_via_shape_2d_from(
    shape_2d: (int, int),
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into a
    finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

    Grid are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0].
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_pixels**2*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
    ----------
    shape_2d : (int, int)
        The (y,x) shape of the 2D array the sub-grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
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


def grid_2d_via_shape_2d_from(
    shape_2d: (int, int),
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into a
    finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

    Grid are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0].
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_pixels**2*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
    ----------
    shape_2d : (int, int)
        The (y,x) shape of the 2D array the sub-grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
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


# @decorator_util.jit()
def grid_radii_scaled_1d_from(
    extent: np.ndarray,
    centre: (float, float),
    pixel_scales: (float, float),
    sub_size: int,
) -> np.ndarray:
    """
    Determine a radial grid of points from a region of coordinates defined by an extent and with a (y,x). This
    functions operates as follows:

    - Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance of
    the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and x axes).

    - Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
    pixel_scale in the x dimension is used).

    - Determine the number of pixels between the centre and the edge of the region using the longest path between the
    two chosen above.

    - Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values iterate
    from the centre in increasing steps of the pixel-scale.

    A schematric is shown below:

    -------------------
    |                 |
    |<- - -  - ->x    | x = centre
    |                 | <-> = longest radial path from centre to extent edge
    |                 |
    -------------------

    Using the centre x above, this function finds the longest radial path to the edge of the extent window.

    The returned `grid_radii` represents a radial set of points that in 1D sample the 2D grid outwards from its centre.
    This grid stores the radial coordinates as (y,x) values (where all y values are the same) as opposed to a 1D data
    structure so that it can be used in functions which require that a 2D grid structure is input.

    Parameters
    ----------
    extent : np.ndarray
        The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
    centre : (float, flloat)
        The (y,x) central coordinate which the radial grid is traced outwards from.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.

    Returns
    -------
    ndarray
        A radial set of points sampling the longest distance from the centre to the edge of the extent in along the
        positive x-axis.
    """

    distance_to_positive_x = extent[1] - centre[1]
    distance_to_positive_y = extent[3] - centre[0]
    distance_to_negative_x = centre[1] - extent[0]
    distance_to_negative_y = centre[0] - extent[2]

    scaled_distance = max(
        [
            distance_to_positive_x,
            distance_to_positive_y,
            distance_to_negative_x,
            distance_to_negative_y,
        ]
    )

    if (scaled_distance == distance_to_positive_y) or (
        scaled_distance == distance_to_negative_y
    ):
        pixel_scale = pixel_scales[0]
    else:
        pixel_scale = pixel_scales[1]

    shape_1d = sub_size * int((scaled_distance / pixel_scale)) + 1

    grid_radii_scaled = np.zeros((shape_1d, 2))

    grid_radii_scaled[:, 0] += centre[0]

    radii = centre[1]

    for i in range(shape_1d):

        grid_radii_scaled[i, 1] = radii
        radii += pixel_scale / sub_size

    return grid_radii_scaled


@decorator_util.jit()
def grid_pixels_1d_from(
    grid_scaled_1d: np.ndarray,
    shape_2d: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a 1D grid of (y,x) scaled coordinates to a 1D grid of (y,x) pixel coordinate values. Pixel
    coordinates are returned as floats such that they include the decimal offset from each pixel's top-left corner
    relative to the input scaled coordinate.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled grid is defined by an origin and coordinates are shifted to this origin before computing their
    1D grid pixel coordinate values.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_scaled_1d: np.ndarray
        The grid of (y,x) coordinates in scaled units which is converted to pixel value coordinates.
    shape_2d : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    ndarray
        A grid of (y,x) pixel-value coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_scaled_1d_to_grid_pixels_1d(grid_scaled_1d=grid_scaled_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = np.zeros((grid_scaled_1d.shape[0], 2))

    centres_scaled = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_scaled_1d.shape[0]):

        grid_pixels_1d[i, 0] = (
            (-grid_scaled_1d[i, 0] / pixel_scales[0]) + centres_scaled[0] + 0.5
        )
        grid_pixels_1d[i, 1] = (
            (grid_scaled_1d[i, 1] / pixel_scales[1]) + centres_scaled[1] + 0.5
        )

    return grid_pixels_1d


@decorator_util.jit()
def grid_pixel_centres_1d_from(
    grid_scaled_1d: np.ndarray,
    shape_2d: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a 1D grid of (y,x) scaled coordinates to a 1D grid of (y,x) pixel values. Pixel coordinates are
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_scaled_1d: np.ndarray
        The grid of (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_2d : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    ndarray
        A grid of (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_scaled_1d_to_grid_pixel_centres_1d(grid_scaled_1d=grid_scaled_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = np.zeros((grid_scaled_1d.shape[0], 2))

    centres_scaled = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_scaled_1d.shape[0]):

        grid_pixels_1d[i, 0] = int(
            (-grid_scaled_1d[i, 0] / pixel_scales[0]) + centres_scaled[0] + 0.5
        )
        grid_pixels_1d[i, 1] = int(
            (grid_scaled_1d[i, 1] / pixel_scales[1]) + centres_scaled[1] + 0.5
        )

    return grid_pixels_1d


@decorator_util.jit()
def grid_pixel_indexes_1d_from(
    grid_scaled_1d: np.ndarray,
    shape_2d: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a 1D grid of (y,x) scaled coordinates to a 1D grid of (y,x) pixel 1D indexes. Pixel coordinates
    are returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then
    downwards.

    For example:

    The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
    The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
    The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_scaled_1d: np.ndarray
        The grid of (y,x) coordinates in scaled units which is converted to 1D pixel indexes.
    shape_2d : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    ndarray
        A grid of 1d pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_pixels_from_grid_scaled_1d(grid_scaled_1d=grid_scaled_1d, shape=(2,2),
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
def grid_scaled_1d_from(
    grid_pixels_1d: np.ndarray,
    shape_2d: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a 1D grid of (y,x) pixel coordinates to a 1D grid of (y,x) scaled values.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
    origin after computing their values from the 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_pixels_1d: np.ndarray
        The grid of (y,x) coordinates in pixel values which is converted to scaled coordinates.
    shape_2d : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    ndarray
        A grid of 1d scaled coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_pixels_1d = np.array([[0,0], [0,1], [1,0], [1,1])
    grid_pixels_1d = grid_pixels_1d_to_grid_scaled_1d(grid_pixels_1d=grid_pixels_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_scaled_1d = np.zeros((grid_pixels_1d.shape[0], 2))

    centres_scaled = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
    )

    for i in range(grid_scaled_1d.shape[0]):

        grid_scaled_1d[i, 0] = (
            -(grid_pixels_1d[i, 0] - centres_scaled[0] - 0.5) * pixel_scales[0]
        )
        grid_scaled_1d[i, 1] = (
            grid_pixels_1d[i, 1] - centres_scaled[1] - 0.5
        ) * pixel_scales[1]

    return grid_scaled_1d


@decorator_util.jit()
def grid_pixel_centres_2d_from(
    grid_scaled_2d: np.ndarray,
    shape_2d: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a 2D grid of (y,x) scaled coordinates to a 2D grid of (y,x) pixel values. Pixel coordinates are
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (y_dimension, x_dimension, 2).

    Parameters
    ----------
    grid_scaled_1d: np.ndarray
        The grid of (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_2d : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    ndarray
        A grid of (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_scaled_1d_to_grid_pixel_centres_1d(grid_scaled_1d=grid_scaled_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d = np.zeros((grid_scaled_2d.shape[0], grid_scaled_2d.shape[1], 2))

    centres_scaled = centres_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
    )

    for y in range(grid_scaled_2d.shape[0]):
        for x in range(grid_scaled_2d.shape[1]):
            grid_pixels_2d[y, x, 0] = int(
                (-grid_scaled_2d[y, x, 0] / pixel_scales[0]) + centres_scaled[0] + 0.5
            )
            grid_pixels_2d[y, x, 1] = int(
                (grid_scaled_2d[y, x, 1] / pixel_scales[1]) + centres_scaled[1] + 0.5
            )

    return grid_pixels_2d


@decorator_util.jit()
def furthest_grid_1d_index_from(
    grid_1d: np.ndarray, grid_1d_indexes: np.ndarray, coordinate: (float, float)
) -> int:

    distance_from_centre = 0.0

    for grid_1d_index in grid_1d_indexes:
        y = grid_1d[grid_1d_index, 0]
        x = grid_1d[grid_1d_index, 1]
        distance_from_centre_new = (x - coordinate[1]) ** 2 + (y - coordinate[0]) ** 2
        if distance_from_centre_new >= distance_from_centre:
            distance_from_centre = distance_from_centre_new
            furthest_grid_1d_index = grid_1d_index

    return furthest_grid_1d_index


def sub_grid_1d_from(
    sub_grid_2d: np.ndarray, mask: np.ndarray, sub_size: int
) -> np.ndarray:
    """For a 2D grid and mask, map the values of all unmasked pixels to a 1D grid.

    The pixel coordinate origin is at the top left corner of the 2D grid and goes right-wards and downwards, such
    that for an grid of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D grid will correspond to index 0 of the 1D grid.
    - pixel [0,1] of the 2D grid will correspond to index 1 of the 1D grid.
    - pixel [1,0] of the 2D grid will correspond to index 4 of the 1D grid.

    Parameters
    ----------
    mask : ndgrid
        A 2D grid of bools, where `False` values are unmasked and included in the util.
    sub_grid_2d : ndgrid
        The 2D grid of values which are mapped to a 1D grid.

    Returns
    -------
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


def sub_grid_2d_from(
    sub_grid_1d: np.ndarray, mask: np.ndarray, sub_size: int
) -> np.ndarray:
    """For a 1D array that was computed by util unmasked values from a 2D array of shape (total_y_pixels, total_x_pixels), map its
    values back to the original 2D array where masked values are set to zero.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels,
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
    ----------
    sub_grid_1d : np.ndarray
        The 1D array of values which are mapped to a 2D array.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.
    sub_one_to_two : np.ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    -------
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
def grid_upscaled_1d_from(
    grid_1d: np.ndarray, upscale_factor: int, pixel_scales: (float, float)
) -> np.ndarray:
    """
    From an input 1D grid, return an upscaled 1D grid where (y,x) coordinates are added at an upscaled resolution
    to each grid coordinate, analogous to a sub-grid.

    Parameters
    ----------
    grid_1d : np.ndarray
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


def grid_of_points_within_radius(
    radius: float, centre: (float, float), grid: np.ndarray
):
    y_inside = []
    x_inside = []

    for i in range(len(grid[:, 0])):
        if (grid[i, 0] - centre[0]) ** 2 + (grid[i, 1] - centre[1]) ** 2 > radius ** 2:
            y_inside.append(grid[i, 0])
            x_inside.append(grid[i, 1])

    return np.asarray(y_inside, x_inside)
