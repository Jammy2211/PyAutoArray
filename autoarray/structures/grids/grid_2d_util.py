from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray import numba_util
from autoarray import type as ty


def convert_grid(grid: Union[np.ndarray, List]) -> np.ndarray:

    try:
        grid = grid.array
    except AttributeError:
        pass

    return jnp.asarray(grid)


def check_grid_slim(grid, shape_native):
    if shape_native is None:
        raise exc.GridException(
            f"""
            The input grid is not in its native shape (an ndarray / list of shape [total_y_pixels, total_x_pixels, 2])
            and the shape_native parameter has not been input the Grid2D function.

            Either change the input array to be its native shape or input its shape_native input the function.

            The shape of the input array is {grid.shape}
            """
        )

    if shape_native and len(shape_native) != 2:
        raise exc.GridException(
            """
            The input shape_native parameter is not a tuple of type (int, int).
            """
        )

def check_grid_2d(grid_2d: np.ndarray):
    if grid_2d.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid_2d.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")


def check_grid_2d_and_mask_2d(grid_2d: np.ndarray, mask_2d: Mask2D):
    if len(grid_2d.shape) == 2:
        if grid_2d.shape[0] != mask_2d.pixels_in_mask:
            raise exc.GridException(
                f"""
                The input 2D grid does not have the same number of values as pixels in
                the mask.

                The shape of the input grid_2d is {grid_2d.shape}.
                The mask shape_native is {mask_2d.shape_native}.
                The mask number of pixels is {mask_2d.pixels_in_mask}. 
                """
            )

    elif len(grid_2d.shape) == 3:
        if (grid_2d.shape[0], grid_2d.shape[1]) != mask_2d.shape_native:
            raise exc.GridException(
                f"""
                The input 2D grid is not the same dimensions as the mask
                (e.g. the mask 2D shape.)

                The shape of the input grid_2d is {grid_2d.shape}.
                The mask shape_native is {mask_2d.shape_native}.
                """
            )


def convert_grid_2d(
    grid_2d: Union[np.ndarray, List], mask_2d: Mask2D, store_native: bool = False
) -> np.ndarray:
    """
    The `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function performs the following and checks and conversions on the input:

    1: If the input is a list, convert it to an ndarray.
    2: Check that the number of coordinates in the grid is identical to that of the mask.
    3) Map the input ndarray to its `slim` representation.

    For a Grid2D, `slim` refers to a 2D NumPy array of shape [total_coordinates, 2] and `native` a 3D NumPy array of
    shape [total_y_coordinates, total_x_coordinates, 2]

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to a ndarray if it is a list.
    mask_2d
        The mask of the output Array2D.
    store_native
        If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels, 2]. This avoids
        mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
    """

    grid_2d = convert_grid(grid=grid_2d)

    check_grid_2d_and_mask_2d(grid_2d=grid_2d, mask_2d=mask_2d)

    is_native = len(grid_2d.shape) == 3

    if is_native:
        grid_2d[:, :, 0] *= np.invert(mask_2d)
        grid_2d[:, :, 1] *= np.invert(mask_2d)

    if is_native == store_native:
        return grid_2d
    elif not store_native:
        return grid_2d_slim_from(
            grid_2d_native=np.array(grid_2d),
            mask=np.array(mask_2d),
        )
    return grid_2d_native_from(
        grid_2d_slim=np.array(grid_2d),
        mask_2d=np.array(mask_2d),
    )


def convert_grid_2d_to_slim(
    grid_2d: Union[np.ndarray, List], mask_2d: Mask2D
) -> np.ndarray:
    """
    he `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function checks the dimensions of the input `grid_2d` and maps it to its `slim` representation.

    For a Grid2D, `slim` refers to a 2D NumPy array of shape [total_coordinates, 2].

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to its silm representation.
    mask_2d
        The mask of the output Array2D.
    """
    if len(grid_2d.shape) == 2:
        return grid_2d
    return grid_2d_slim_from(
        grid_2d_native=grid_2d,
        mask=mask_2d,
    )


def convert_grid_2d_to_native(
    grid_2d: Union[np.ndarray, List], mask_2d: Mask2D
) -> np.ndarray:
    """
    he `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function checks the dimensions of the input `grid_2d` and maps it to its `native` representation.

    For a Grid2D, `native` refers to a 2D NumPy array of shape [total_y_coordinates, total_x_coordinates, 2].

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to its native representation.
    mask_2d
        The mask of the output Array2D.
    """
    if len(grid_2d.shape) == 3:
        return grid_2d
    return grid_2d_native_from(
        grid_2d_slim=grid_2d,
        mask_2d=mask_2d,
    )


def grid_2d_centre_from(grid_2d_slim: np.ndarray) -> Tuple[float, float]:
    """
    Returns the centre of a grid from a 1D grid.

    Parameters
    ----------
    grid_2d_slim
        The 1D grid of values which are mapped to a 2D array.

    Returns
    -------
    (float, float)
        The (y,x) central coordinates of the grid.
    """
    centre_y = (np.max(grid_2d_slim[:, 0]) + np.min(grid_2d_slim[:, 0])) / 2.0
    centre_x = (np.max(grid_2d_slim[:, 1]) + np.min(grid_2d_slim[:, 1])) / 2.0
    return centre_y, centre_x


def grid_2d_slim_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is on a 2D mask with shape (total_y_pixels, total_x_pixels). This routine
    computes the (y,x) scaled coordinates a the centre of every pixel defined by this 2D mask array.

    The grid is returned on an array of shape (total_unmasked_pixels, 2). y coordinates are stored in the 0 index of
    the second dimension, x coordinates in the 1 index. Masked coordinates are therefore removed and not included in
    the slimmed grid.

    Grid2D are defined from the top-left corner, where the first unmasked pixel corresponds to index 0.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        grid.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A slimmed grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_unmasked_pixels, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_slim = grid_2d_slim_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = jnp.array(centres_scaled)
    pixel_scales = jnp.array(pixel_scales)
    sign = jnp.array([-1.0, 1.0])
    return (
        (jnp.stack(jnp.nonzero(~mask_2d.astype(bool))).T - centres_scaled)
        * sign
        * pixel_scales
    )


def grid_2d_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is on a 2D mask with shape (total_y_pixels, total_x_pixels). This routine computes
    the (y,x) scaled coordinates at the centre of every pixel defined by this 2D mask array.

    The grid is returned in its native dimensions with shape (total_y_pixels, total_x_pixels). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked pixels are given
    values (0.0, 0.0).

    Grids are defined from the top-left corner, where the first unmasked pixel corresponds to index 0.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        grid.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_y_pixels, total_x_pixels).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_2d = grid_2d_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_2d_slim = grid_2d_slim_via_mask_from(
        mask_2d=mask_2d, pixel_scales=pixel_scales, origin=origin
    )

    return grid_2d_native_from(
        grid_2d_slim=grid_2d_slim,
        mask_2d=mask_2d,
    )


def grid_2d_slim_via_shape_native_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is in a 2D mask with shape (total_y_pixels, total_x_pixels). This routine computes
    the (y,x) scaled coordinates at the centre of every pixel defined by this 2D mask array.

    The grid is returned in its slimmed dimensions with shape (total_pixels, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Grid2D are defined from the top-left corner, where the first pixel corresponds to index [0,0].

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the grid of coordinates is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid is slimmed and has dimensions (total_unmasked_pixels, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_2d_slim = grid_2d_slim_via_shape_native_from(shape_native=(3,3), pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    return grid_2d_slim_via_mask_from(
        mask_2d=np.full(fill_value=False, shape=shape_native),
        pixel_scales=pixel_scales,
        origin=origin,
    )


def grid_2d_via_shape_native_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is in a 2D mask with shape (total_y_pixels, total_x_pixels). This routine computes
    the (y,x) scaled coordinates at the centre of every pixel defined by this 2D mask array.

    The grid is returned in its native dimensions with shape (total_y_pixels, total_x_pixels).
    y coordinates are stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Grids are defined from the top-left corner, where the first pixel corresponds to index [0,0].

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the grid of coordinates is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_y_pixels, total_x_pixels).

    Examples
    --------
    grid_2d = grid_2d_via_shape_native_from(shape_native=(3, 3), pixel_scales=(1.0, 1.0), origin=(0.0, 0.0))
    """
    return grid_2d_via_mask_from(
        mask_2d=np.full(fill_value=False, shape=shape_native),
        pixel_scales=pixel_scales,
        origin=origin,
    )


@numba_util.jit()
def _radial_projected_shape_slim_from(
    extent: np.ndarray,
    centre: Tuple[float, float],
    pixel_scales: ty.PixelScales,
) -> int:
    """
    The function `grid_scaled_2d_slim_radial_projected_from()` determines a projected radial grid of points from a 2D
    region of coordinates defined by an extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

    To do this, the function first performs these 3 steps:

    1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance of
    the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and x axes).

    2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
    pixel_scale in the x dimension is used).

    3) Determine the number of pixels between the centre and the edge of the region using the longest path between the
    two chosen above.

    A schematic is shown below:

    -------------------
    |                 |
    |<- - -  - ->x    | x = centre
    |                 | <-> = longest radial path from centre to extent edge
    |                 |
    -------------------

    Using the centre x above, this function finds the longest radial path to the edge of the extent window.

    This function returns the integer number of pixels given by this radial grid, which is then used to create
    the radial grid.

    Parameters
    ----------
    extent
        The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
    centre : (float, flloat)
        The (y,x) central coordinate which the radial grid is traced outwards from.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.

    Returns
    -------
    int
        The 1D integer shape of a radial set of points sampling the longest distance from the centre to the edge of the
        extent in along the positive x-axis.
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

    return int((scaled_distance / pixel_scale)) + 1


@numba_util.jit()
def grid_scaled_2d_slim_radial_projected_from(
    extent: np.ndarray,
    centre: Tuple[float, float],
    pixel_scales: ty.PixelScales,
    shape_slim: Optional[int] = 0,
) -> np.ndarray:
    """
    Determine a projected radial grid of points from a 2D region of coordinates defined by an
    extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

    This functions operates as follows:

    1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance of
    the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and x axes).

    2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
    pixel_scale in the x dimension is used).

    3) Determine the number of pixels between the centre and the edge of the region using the longest path between the
    two chosen above.

    4) Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values iterate
    from the centre in increasing steps of the pixel-scale.

    5) Rotate these radial coordinates by the input `angle` clockwise.

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
    extent
        The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
    centre : (float, flloat)
        The (y,x) central coordinate which the radial grid is traced outwards from.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    shape_slim
        Manually choose the shape of the 1D projected grid that is returned. If 0, the border based on the 2D grid is
        used (due to numba None cannot be used as a default value).

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

    if shape_slim == 0:
        shape_slim = int((scaled_distance / pixel_scale)) + 1

    grid_scaled_2d_slim_radii = np.zeros((shape_slim, 2))

    grid_scaled_2d_slim_radii[:, 0] += centre[0]

    radii = centre[1]

    for slim_index in range(shape_slim):
        grid_scaled_2d_slim_radii[slim_index, 1] = radii
        radii += pixel_scale

    return grid_scaled_2d_slim_radii


@numba_util.jit()
def relocated_grid_via_jit_from(grid, border_grid):
    """
    Relocate the coordinates of a grid to its border if they are outside the border, where the border is
    defined as all pixels at the edge of the grid's mask (see *mask._border_1d_indexes*).

    This is performed as follows:

    1: Use the mean value of the grid's y and x coordinates to determine the origin of the grid.
    2: Compute the radial distance of every grid coordinate from the origin.
    3: For every coordinate, find its nearest pixel in the border.
    4: Determine if it is outside the border, by comparing its radial distance from the origin to its paired
    border pixel's radial distance.
    5: If its radial distance is larger, use the ratio of radial distances to move the coordinate to the
    border (if its inside the border, do nothing).

    The method can be used on uniform or irregular grids, however for irregular grids the border of the
    'image-plane' mask is used to define border pixels.

    Parameters
    ----------
    grid
        The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
    border_grid : Grid2D
        The grid of border (y,x) coordinates.
    """

    grid_relocated = np.zeros(grid.shape)
    grid_relocated[:, :] = grid[:, :]

    border_origin = np.zeros(2)
    border_origin[0] = np.mean(border_grid[:, 0])
    border_origin[1] = np.mean(border_grid[:, 1])
    border_grid_radii = np.sqrt(
        np.add(
            np.square(np.subtract(border_grid[:, 0], border_origin[0])),
            np.square(np.subtract(border_grid[:, 1], border_origin[1])),
        )
    )
    border_min_radii = np.min(border_grid_radii)

    grid_radii = np.sqrt(
        np.add(
            np.square(np.subtract(grid[:, 0], border_origin[0])),
            np.square(np.subtract(grid[:, 1], border_origin[1])),
        )
    )

    for pixel_index in range(grid.shape[0]):
        if grid_radii[pixel_index] > border_min_radii:
            closest_pixel_index = np.argmin(
                np.square(grid[pixel_index, 0] - border_grid[:, 0])
                + np.square(grid[pixel_index, 1] - border_grid[:, 1])
            )

            move_factor = (
                border_grid_radii[closest_pixel_index] / grid_radii[pixel_index]
            )

            if move_factor < 1.0:
                grid_relocated[pixel_index, :] = (
                    move_factor * (grid[pixel_index, :] - border_origin[:])
                    + border_origin[:]
                )

    return grid_relocated


@numba_util.jit()
def furthest_grid_2d_slim_index_from(
    grid_2d_slim: np.ndarray, slim_indexes: np.ndarray, coordinate: Tuple[float, float]
) -> int:
    distance_to_centre = 0.0

    for slim_index in slim_indexes:
        y = grid_2d_slim[slim_index, 0]
        x = grid_2d_slim[slim_index, 1]
        distance_to_centre_new = (x - coordinate[1]) ** 2 + (y - coordinate[0]) ** 2

        if distance_to_centre_new >= distance_to_centre:
            distance_to_centre = distance_to_centre_new
            furthest_grid_2d_slim_index = slim_index

    return furthest_grid_2d_slim_index


def grid_2d_slim_from(
    grid_2d_native: np.ndarray,
    mask: np.ndarray,
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
    mask_2d
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A 1D grid of values mapped from the 2D grid with dimensions (total_unmasked_pixels).
    """

    grid_1d_slim_y = array_2d_util.array_2d_slim_from(
        array_2d_native=np.array(grid_2d_native[:, :, 0]),
        mask_2d=np.array(mask),
    )

    grid_1d_slim_x = array_2d_util.array_2d_slim_from(
        array_2d_native=np.array(grid_2d_native[:, :, 1]),
        mask_2d=np.array(mask),
    )

    return np.stack((grid_1d_slim_y, grid_1d_slim_x), axis=-1)


def grid_2d_native_from(
    grid_2d_slim: np.ndarray,
    mask_2d: np.ndarray,
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
    grid_2d_slim
        The (y,x) values of the slimmed 2D grid which are mapped to the native 2D grid.
    mask_2d
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A NumPy array of shape [total_y_pixels, total_x_pixels, 2] corresponding to the (y,x) values of the native 2D
        mapped from the slimmed grid.
    """

    grid_2d_native_y = array_2d_util.array_2d_native_from(
        array_2d_slim=grid_2d_slim[:, 0],
        mask_2d=mask_2d,
    )

    grid_2d_native_x = array_2d_util.array_2d_native_from(
        array_2d_slim=grid_2d_slim[:, 1],
        mask_2d=mask_2d,
    )

    return np.stack((grid_2d_native_y, grid_2d_native_x), axis=-1)


@numba_util.jit()
def grid_2d_slim_upscaled_from(
    grid_slim: np.ndarray, upscale_factor: int, pixel_scales: ty.PixelScales
) -> np.ndarray:
    """
    From an input slimmed 2D grid, return an upscaled slimmed 2D grid where (y,x) coordinates are added at an
    upscaled resolution to each grid coordinate.

    Parameters
    ----------
    grid_slim
        The slimmed grid of (y,x) coordinates over which a square uniform grid is overlaid.
    upscale_factor
        The upscaled resolution at which the new grid coordinates are computed.
    pixel_scales
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    grid_2d_slim_upscaled = np.zeros(shape=(grid_slim.shape[0] * upscale_factor**2, 2))

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
    radius: float, centre: Tuple[float, float], grid_2d: np.ndarray
):
    y_inside = []
    x_inside = []

    for i in range(len(grid_2d[:, 0])):
        if (grid_2d[i, 0] - centre[0]) ** 2 + (
            grid_2d[i, 1] - centre[1]
        ) ** 2 > radius**2:
            y_inside.append(grid_2d[i, 0])
            x_inside.append(grid_2d[i, 1])

    return np.asarray(y_inside, x_inside)


def compute_polygon_area(points):
    x = points[:, 1]
    y = points[:, 0]

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def grid_pixels_in_mask_pixels_from(
    grid, shape_native, pixel_scales, origin
) -> np.ndarray:
    """
    Returns an array containing the number of pixels of one grid in every pixel of another masked grid.

    For example, image-mesh adaption may be performed on a 3.0" circular mask of data. The high weight pixels
    may have 3 or more mesh pixels per image pixel, whereas low weight regions may have zero pixels. The array
    returned by this function gives the integer number of pixels in each data pixel.

    Parameters
    ----------
    grid_pixel_centres
        The 2D integer index of every image pixel that each image-mesh pixel falls within.
    shape_native
        The 2D shape of the data's mask, which the number of image-mesh pixels that fall within eac pixel is counted.

    Returns
    -------
    An array containing the integer number of image-mesh pixels that fall without each of the data's mask.
    """
    grid_pixel_centres = geometry_util.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    ).astype("int")

    mesh_pixels_per_image_pixel = np.zeros(shape=shape_native)

    # Assuming grid_pixel_centres is a 2D array where each row contains (y, x) indices.
    y_indices = grid_pixel_centres[:, 0]
    x_indices = grid_pixel_centres[:, 1]

    # Use np.add.at to increment the specific indices in a safe and efficient manner
    np.add.at(mesh_pixels_per_image_pixel, (y_indices, x_indices), 1)

    return mesh_pixels_per_image_pixel