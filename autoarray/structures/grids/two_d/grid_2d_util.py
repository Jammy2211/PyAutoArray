from autoarray import decorator_util
import numpy as np

from autoarray.mask import mask_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.arrays.two_d import array_2d_util


@decorator_util.jit()
def grid_2d_centre_from(grid_2d_slim: np.ndarray) -> (float, float):
    """
    Returns the centre of a grid from a 1D grid.

    Parameters
    ----------
    grid_2d_slim :  np.ndarray
        The 1D grid of values which are mapped to a 2D array.

    Returns
    -------
    (float, float)
        The (y,x) central coordinates of the grid.
    """
    centre_y = (np.max(grid_2d_slim[:, 0]) + np.min(grid_2d_slim[:, 0])) / 2.0
    centre_x = (np.max(grid_2d_slim[:, 1]) + np.min(grid_2d_slim[:, 1])) / 2.0
    return centre_y, centre_x


@decorator_util.jit()
def grid_2d_slim_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
    a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates a the centre of every sub-pixel defined by this 2D mask array.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked coordinates are therefore
    removed and not included in the slimmed grid.

    Grid2D are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    Parameters
    ----------
    mask_2d : np.ndarray
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
        A slimmed sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_slim = grid_2d_slim_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), sub_size=1, origin=(0.0, 0.0))
    """

    total_sub_pixels = mask_2d_util.total_sub_pixels_2d_from(mask_2d, sub_size)

    grid_slim = np.zeros(shape=(total_sub_pixels, 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, origin=origin
    )

    sub_index = 0

    y_sub_half = pixel_scales[0] / 2
    y_sub_step = pixel_scales[0] / (sub_size)

    x_sub_half = pixel_scales[1] / 2
    x_sub_step = pixel_scales[1] / (sub_size)

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):

            if not mask_2d[y, x]:

                y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
                x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

                for y1 in range(sub_size):
                    for x1 in range(sub_size):

                        grid_slim[sub_index, 0] = -(
                            y_scaled - y_sub_half + y1 * y_sub_step + (y_sub_step / 2.0)
                        )
                        grid_slim[sub_index, 1] = (
                            x_scaled - x_sub_half + x1 * x_sub_step + (x_sub_step / 2.0)
                        )
                        sub_index += 1

    return grid_slim


def grid_2d_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into a
    finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

    The sub-grid is returned in its native dimensions with shape (total_y_pixels*sub_size, total_x_pixels*sub_size).
    y coordinates are stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked pixels are
    given values (0.0, 0.0).

    Grids are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    Parameters
    ----------
    mask_2d : np.ndarray
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
        array. The sub grid array has dimensions (total_y_pixels*sub_size, total_x_pixels*sub_size).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_2d = grid_2d_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), sub_size=1, origin=(0.0, 0.0))
    """

    grid_2d_slim = grid_2d_slim_via_mask_from(
        mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
    )

    return grid_2d_native_from(
        grid_2d_slim=grid_2d_slim, mask_2d=mask_2d, sub_size=sub_size
    )


def grid_2d_slim_via_shape_native_from(
    shape_native: (int, int),
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into a
    finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

    The sub-grid is returned in its slimmed dimensions with shape (total_pixels**2*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Grid2D are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0].
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    Parameters
    ----------
    shape_native : (int, int)
        The (y,x) shape of the 2D array the sub-grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, float)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The sub grid is slimmed and has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_2d_slim = grid_2d_slim_via_shape_native_from(shape_native=(3,3), pixel_scales=(0.5, 0.5), sub_size=2, origin=(0.0, 0.0))
    """
    return grid_2d_slim_via_mask_from(
        mask_2d=np.full(fill_value=False, shape=shape_native),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


def grid_2d_via_shape_native_from(
    shape_native: (int, int),
    pixel_scales: (float, float),
    sub_size: int,
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided
    into a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes
    the (y,x) scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

    The sub-grid is returned in its native dimensions with shape (total_y_pixels*sub_size, total_x_pixels*sub_size).
    y coordinates are stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Grids are defined from the top-left corner, where the first sub-pixel corresponds to index [0,0].
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    Parameters
    ----------
    shape_native : (int, int)
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
        array. The sub grid array has dimensions (total_y_pixels*sub_size, total_x_pixels*sub_size).

    Examples
    --------
    grid_2d = grid_2d_via_shape_native_from(shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=2, origin=(0.0, 0.0))
    """
    return grid_2d_via_mask_from(
        mask_2d=np.full(fill_value=False, shape=shape_native),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


@decorator_util.jit()
def grid_scaled_2d_slim_radii_from(
    extent: np.ndarray,
    centre: (float, float),
    pixel_scales: (float, float),
    sub_size: int,
) -> np.ndarray:
    """
    Determine a radial grid of points from a region of coordinates defined by an extent and with a (y,x) centre. This
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

    shape_slim = sub_size * int((scaled_distance / pixel_scale)) + 1

    grid_scaled_2d_slim_radii = np.zeros((shape_slim, 2))

    grid_scaled_2d_slim_radii[:, 0] += centre[0]

    radii = centre[1]

    for slim_index in range(shape_slim):

        grid_scaled_2d_slim_radii[slim_index, 1] = radii
        radii += pixel_scale / sub_size

    return grid_scaled_2d_slim_radii


@decorator_util.jit()
def grid_pixels_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2d (y,x) scaled coordinates to a slimmed grid of 2d (y,x) pixel coordinate values. Pixel
    coordinates are returned as floats such that they include the decimal offset from each pixel's top-left corner
    relative to the input scaled coordinate.

    The input and output grids are both slimmed and therefore shape (total_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled grid is defined by an origin and coordinates are shifted to this origin before computing their
    1D grid pixel coordinate values.

    Parameters
    ----------
    grid_scaled_2d_slim: np.ndarray
        The slimmed grid of 2D (y,x) coordinates in scaled units which are converted to pixel value coordinates.
    shape_native : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    ndarray
        A slimmed grid of 2D (y,x) pixel-value coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d_slim = np.zeros((grid_scaled_2d_slim.shape[0], 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    for slim_index in range(grid_scaled_2d_slim.shape[0]):

        grid_pixels_2d_slim[slim_index, 0] = (
            (-grid_scaled_2d_slim[slim_index, 0] / pixel_scales[0])
            + centres_scaled[0]
            + 0.5
        )
        grid_pixels_2d_slim[slim_index, 1] = (
            (grid_scaled_2d_slim[slim_index, 1] / pixel_scales[1])
            + centres_scaled[1]
            + 0.5
        )

    return grid_pixels_2d_slim


@decorator_util.jit()
def grid_pixel_centres_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2D (y,x) scaled coordinates to a slimmed grid of 2D (y,x) pixel values. Pixel coordinates
    are returned as integers such that they map directly to the pixel they are contained within.

    The input and output grids are both slimmed and therefore shape (total_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_scaled_2d_slim: np.ndarray
        The slimmed grid of 2D (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_native : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    ndarray
        A slimmed grid of 2D (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d_slim = np.zeros((grid_scaled_2d_slim.shape[0], 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    for slim_index in range(grid_scaled_2d_slim.shape[0]):

        grid_pixels_2d_slim[slim_index, 0] = int(
            (-grid_scaled_2d_slim[slim_index, 0] / pixel_scales[0])
            + centres_scaled[0]
            + 0.5
        )
        grid_pixels_2d_slim[slim_index, 1] = int(
            (grid_scaled_2d_slim[slim_index, 1] / pixel_scales[1])
            + centres_scaled[1]
            + 0.5
        )

    return grid_pixels_2d_slim


@decorator_util.jit()
def grid_pixel_indexes_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2D (y,x) scaled coordinates to a slimmed grid of pixel indexes. Pixel coordinates are
    returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then downwards.

    The input and output grids are both slimmed and have shapes (total_pixels, 2) and (total_pixels,).

    For example:

    The pixel at the top-left, whose native index is [0,0], corresponds to slimmed pixel index 0.
    The fifth pixel on the top row, whose native index is [0,5], corresponds to slimmed pixel index 4.
    The first pixel on the second row, whose native index is [0,1], has slimmed pixel index 10 if a row has 10 pixels.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_scaled_2d_slim: np.ndarray
        The slimmed grid of 2D (y,x) coordinates in scaled units which is converted to slimmed pixel indexes.
    shape_native : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    ndarray
        A grid of slimmed pixel indexes with dimensions (total_pixels,).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixel_indexes_2d_slim = grid_pixel_indexes_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d_slim = grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d_slim,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    grid_pixel_indexes_2d_slim = np.zeros(grid_pixels_2d_slim.shape[0])

    for slim_index in range(grid_pixels_2d_slim.shape[0]):

        grid_pixel_indexes_2d_slim[slim_index] = int(
            grid_pixels_2d_slim[slim_index, 0] * shape_native[1]
            + grid_pixels_2d_slim[slim_index, 1]
        )

    return grid_pixel_indexes_2d_slim


@decorator_util.jit()
def grid_scaled_2d_slim_from(
    grid_pixels_2d_slim: np.ndarray,
    shape_native: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2D (y,x) pixel coordinates to a slimmed grid of 2D (y,x) scaled values.

    The input and output grids are both slimmed and therefore shape (total_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
    origin after computing their values from the 1D grid pixel indexes.

    Parameters
    ----------
    grid_pixels_2d_slim: np.ndarray
        The slimmed grid of (y,x) coordinates in pixel values which is converted to scaled coordinates.
    shape_native : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    ndarray
        A slimmed grid of 2d scaled coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_pixels_2d_slim = np.array([[0,0], [0,1], [1,0], [1,1])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_pixels_2d_slim=grid_pixels_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_scaled_2d_slim = np.zeros((grid_pixels_2d_slim.shape[0], 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    for slim_index in range(grid_scaled_2d_slim.shape[0]):

        grid_scaled_2d_slim[slim_index, 0] = (
            -(grid_pixels_2d_slim[slim_index, 0] - centres_scaled[0] - 0.5)
            * pixel_scales[0]
        )
        grid_scaled_2d_slim[slim_index, 1] = (
            grid_pixels_2d_slim[slim_index, 1] - centres_scaled[1] - 0.5
        ) * pixel_scales[1]

    return grid_scaled_2d_slim


@decorator_util.jit()
def grid_pixel_centres_2d_from(
    grid_scaled_2d: np.ndarray,
    shape_native: (int, int),
    pixel_scales: (float, float),
    origin: (float, float) = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a native grid of 2D (y,x) scaled coordinates to a native grid of 2D (y,x) pixel values. Pixel coordinates
    are returned as integers such that they map directly to the pixel they are contained within.

    The input and output grids are both native resolution and therefore have shape (y_pixels, x_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_scaled_2d: np.ndarray
        The native grid of 2D (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_native : (int, int)
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    ndarray
        A native grid of 2D (y,x) pixel indexes with dimensions (y_pixels, x_pixels, 2).

    Examples
    --------
    grid_scaled_2d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixel_centres_2d = grid_pixel_centres_2d_from(grid_scaled_2d=grid_scaled_2d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d = np.zeros((grid_scaled_2d.shape[0], grid_scaled_2d.shape[1], 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
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
def furthest_grid_2d_slim_index_from(
    grid_2d_slim: np.ndarray, slim_indexes: np.ndarray, coordinate: (float, float)
) -> int:

    distance_from_centre = 0.0

    for slim_index in slim_indexes:

        y = grid_2d_slim[slim_index, 0]
        x = grid_2d_slim[slim_index, 1]
        distance_from_centre_new = (x - coordinate[1]) ** 2 + (y - coordinate[0]) ** 2

        if distance_from_centre_new >= distance_from_centre:
            distance_from_centre = distance_from_centre_new
            furthest_grid_2d_slim_index = slim_index

    return furthest_grid_2d_slim_index


def grid_2d_slim_from(
    grid_2d_native: np.ndarray, mask: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a native 2D grid and mask of shape [total_y_pixels, total_x_pixels, 2], map the values of all unmasked
    pixels to a slimmed grid of shape [total_unmasked_pixels, 2].

    The pixel coordinate origin is at the top left corner of the native grid and goes right-wards and downwards, such
    that for an grid of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D grid will correspond to index 0 of the 1D grid.
    - pixel [0,1] of the 2D grid will correspond to index 1 of the 1D grid.
    - pixel [1,0] of the 2D grid will correspond to index 4 of the 1D grid.

    Parameters
    ----------
    grid_2d_native : ndarray
        The native grid of (y,x) values which are mapped to the slimmed grid.
    mask_2d : np.ndarray
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.
    sub_size : int
        The size (sub_size x sub_size) of each unmasked pixels sub-array.

    Returns
    -------
    ndarray
        A 1D grid of values mapped from the 2D grid with dimensions (total_unmasked_pixels).
    """

    grid_1d_slim_y = array_2d_util.array_2d_slim_from(
        array_2d_native=grid_2d_native[:, :, 0], mask_2d=mask, sub_size=sub_size
    )

    grid_1d_slim_x = array_2d_util.array_2d_slim_from(
        array_2d_native=grid_2d_native[:, :, 1], mask_2d=mask, sub_size=sub_size
    )

    return np.stack((grid_1d_slim_y, grid_1d_slim_x), axis=-1)


def grid_2d_native_from(
    grid_2d_slim: np.ndarray, mask_2d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a slimmed 2D grid of shape [total_unmasked_pixels, 2], that was computed by extracting the unmasked values
    from a native 2D grid of shape [total_y_pixels, total_x_pixels, 2], map the slimmed grid's coordinates back to the
    native 2D grid where masked values are set to zero.

    This uses a 1D array 'slim_to_native' where each index gives the 2D pixel indexes of the grid's native unmasked
    pixels, for example:

    - If slim_to_native[0] = [0,0], the first value of the 1D array maps to the pixels [0,0,:] of the native 2D grid.
    - If slim_to_native[1] = [0,1], the second value of the 1D array maps to the pixels [0,1,:] of the native 2D grid.
    - If slim_to_native[4] = [1,1], the fifth value of the 1D array maps to the pixels [1,1,:] of the native 2D grid.

    Parameters
    ----------
    grid_2d_slim : np.ndarray
        The (y,x) values of the slimmed 2D grid which are mapped to the native 2D grid.
    mask_2d : np.ndarray
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.
    sub_size : int
        The size (sub_size x sub_size) of each unmasked pixels sub-array.

    Returns
    -------
    ndarray
        A NumPy array of shape [total_y_pixels, total_x_pixels, 2] corresponding to the (y,x) values of the native 2D
        mapped from the slimmed grid.
    """

    grid_2d_native_y = array_2d_util.array_2d_native_from(
        array_2d_slim=grid_2d_slim[:, 0], mask_2d=mask_2d, sub_size=sub_size
    )

    grid_2d_native_x = array_2d_util.array_2d_native_from(
        array_2d_slim=grid_2d_slim[:, 1], mask_2d=mask_2d, sub_size=sub_size
    )

    return np.stack((grid_2d_native_y, grid_2d_native_x), axis=-1)


@decorator_util.jit()
def grid_2d_slim_upscaled_from(
    grid_slim: np.ndarray, upscale_factor: int, pixel_scales: (float, float)
) -> np.ndarray:
    """
    From an input slimmed 2D grid, return an upscaled slimmed 2D grid where (y,x) coordinates are added at an
    upscaled resolution to each grid coordinate, analogous to a sub-grid.

    Parameters
    ----------
    grid_slim : np.ndarray
        The slimmed grid of (y,x) coordinates over which a square uniform grid is overlaid.
    upscale_factor : int
        The upscaled resolution at which the new grid coordinates are computed.
    pixel_scales : (float, float)
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    grid_2d_slim_upscaled = np.zeros(
        shape=(grid_slim.shape[0] * upscale_factor ** 2, 2)
    )

    upscale_index = 0

    y_upscale_half = pixel_scales[0] / 2
    y_upscale_step = pixel_scales[0] / upscale_factor

    x_upscale_half = pixel_scales[1] / 2
    x_upscale_step = pixel_scales[1] / upscale_factor

    for slim_index in range(grid_slim.shape[0]):

        y_grid = grid_slim[slim_index, 0]
        x_grid = grid_slim[slim_index, 1]

        for y in range(upscale_factor):
            for x in range(upscale_factor):

                grid_2d_slim_upscaled[upscale_index, 0] = (
                    y_grid
                    + y_upscale_half
                    - y * y_upscale_step
                    - (y_upscale_step / 2.0)
                )
                grid_2d_slim_upscaled[upscale_index, 1] = (
                    x_grid
                    - x_upscale_half
                    + x * x_upscale_step
                    + (x_upscale_step / 2.0)
                )

                upscale_index += 1

    return grid_2d_slim_upscaled


def grid_2d_of_points_within_radius(
    radius: float, centre: (float, float), grid_2d: np.ndarray
):
    y_inside = []
    x_inside = []

    for i in range(len(grid_2d[:, 0])):
        if (grid_2d[i, 0] - centre[0]) ** 2 + (
            grid_2d[i, 1] - centre[1]
        ) ** 2 > radius ** 2:
            y_inside.append(grid_2d[i, 0])
            x_inside.append(grid_2d[i, 1])

    return np.asarray(y_inside, x_inside)
