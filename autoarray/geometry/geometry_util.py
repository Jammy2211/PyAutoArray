import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union

from autoarray import type as ty


def convert_shape_native_1d(shape_native: Union[int, Tuple[int]]) -> Tuple[int]:
    """
    Converts an input `shape_native` of type int to a tuple (int,). If the input is already a (int, ) tuple it
    is unchanged

    This enables users to input `shape_native` as a single value and have the type automatically converted to type
    (int,) which is used internally for data structures.

    Parameters
    ----------
    pixel_scales
        The input pixel

    Returns
    -------
    pixel_scales
        The `shape_native` as a tuple of format (int,).
    """

    if type(shape_native) is int:
        shape_native = (shape_native,)

    return shape_native


def convert_pixel_scales_1d(pixel_scales: ty.PixelScales) -> Tuple[float]:
    """
    Converts an input pixel-scale of type float to a tuple (float,). If the input is already a (float, ) it is
    unchanged

    This enables users to input the pixel scale as a single value and have the type automatically converted to type
    (float, float) which is used for rectangular grids.

    Parameters
    ----------
    pixel_scales
        The input pixel

    Returns
    -------
    pixel_scales
        The pixel_scales as a tuple of format (float,).
    """

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales,)

    return pixel_scales


def central_pixel_coordinates_1d_from(
    shape_slim: Tuple[int],
) -> Union[Tuple[float], Tuple[float]]:
    """
    Returns the central pixel coordinates of a data structure of any dimension (e.g. in 1D a `Line`, 1d an `Array2D`,
    1d a `Frame2D`, etc.) from the shape of that data structure.

    Examples of the central pixels are as follows:

    - For a 3x3 image, the central pixel is pixel [1, 1].
    - For a 4x4 image, the central pixel is [1.5, 1.5].

    Parameters
    ----------
    shape_slim : tuple(int)
        The dimensions of the data structure, which can be in 1D, 1d or higher dimensions.

    Returns
    -------
    central_pixel_coordinates : tuple(float)
        The central pixel coordinates of the data structure.

    """

    return (float(shape_slim[0] - 1) / 2,)


def central_scaled_coordinate_1d_from(
    shape_slim: Tuple[float],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float] = (0.0, 0.0),
):
    """
    Returns the central coordinates of a 1d data structure (e.g. a`Grid1D`) in scaled units.

    This is computed by using the data structure's shape and converting it to scaled units using an input
    pixel-coordinates to scaled-coordinate conversion factor `pixel_scales`.

    The origin of the scaled grid can also be input and moved from (0.0,).

    Parameters
    ----------
    shape_slim
        The 1d shape of the data structure whose central scaled coordinates are computed.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 1d data structure.
    origin
        The (y,x) scaled units origin of the coordinate system the central scaled coordinate is computed on.

    Returns
    -------
    central_scaled_coordinates_1d
        The central coordinates of the 1d data structure in scaled units.
    """

    central_pixel_coordinates = central_pixel_coordinates_1d_from(shape_slim=shape_slim)

    x_pixel = central_pixel_coordinates[0] - (origin[0] / pixel_scales[0])

    return (x_pixel,)


def pixel_coordinates_1d_from(
    scaled_coordinates_1d: Tuple[float],
    shape_slim: Tuple[int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float] = (0.0, 0.0),
) -> Union[Tuple[float], Tuple[float]]:
    central_pixel_coordinates = central_pixel_coordinates_1d_from(shape_slim=shape_slim)

    x_pixel = int(
        (scaled_coordinates_1d[0] - origins[0]) / pixel_scales[0]
        + central_pixel_coordinates[0]
        + 0.5
    )

    return (x_pixel,)


def scaled_coordinates_1d_from(
    pixel_coordinates_1d: Tuple[float],
    shape_slim: Tuple[int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float] = (0.0, 0.0),
) -> Union[Tuple[float], Tuple[float]]:
    central_scaled_coordinates = central_scaled_coordinate_1d_from(
        shape_slim=shape_slim, pixel_scales=pixel_scales, origin=origins
    )

    x_pixel = pixel_scales[0] * (
        pixel_coordinates_1d[0] - central_scaled_coordinates[0]
    )

    return (x_pixel,)


def convert_pixel_scales_2d(pixel_scales: ty.PixelScales) -> Tuple[float, float]:
    """
    Converts an input pixel-scale of type float to a pixel-scale of tuple (float, float). If the input is already
    type (float, float) it is unchanged

    This enables users to input the pixel scale as a single value and have the type automatically converted to type
    (float, float) which is used for rectangular grids.

    Parameters
    ----------
    pixel_scales or (float, float)
        The input pixel

    Returns
    -------
    pixel_scales
        The pixel_scales of type (float, float).
    """

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales, pixel_scales)

    return pixel_scales


def central_pixel_coordinates_2d_from(
    shape_native: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Returns the central pixel coordinates of a 2D geometry (and therefore a 2D data structure like an ``Array2D``)
    from the shape of that data structure.

    Examples of the central pixels are as follows:

    - For a 3x3 image, the central pixel is pixel [1, 1].
    - For a 4x4 image, the central pixel is [1.5, 1.5].

    Parameters
    ----------
    shape_native
        The dimensions of the data structure, which can be in 1D, 2D or higher dimensions.

    Returns
    -------
    The central pixel coordinates of the data structure.
    """
    return (float(shape_native[0] - 1) / 2, float(shape_native[1] - 1) / 2)


def central_scaled_coordinate_2d_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Returns the central scaled coordinates of a 2D geometry (and therefore a 2D data structure like an ``Array2D``)
    from the shape of that data structure.

    This is computed by using the data structure's shape and converting it to scaled units using an input
    pixel-coordinates to scaled-coordinate conversion factor `pixel_scales`.

    The origin of the scaled grid can also be input and moved from (0.0, 0.0).

    Parameters
    ----------
    shape_native
        The 2D shape of the data structure whose central scaled coordinates are computed.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D data structure.
    origin
        The (y,x) scaled units origin of the coordinate system the central scaled coordinate is computed on.

    Returns
    -------
    The central coordinates of the 2D data structure in scaled units.
    """

    central_pixel_coordinates = central_pixel_coordinates_2d_from(
        shape_native=shape_native
    )

    y_pixel = central_pixel_coordinates[0] + (origin[0] / pixel_scales[0])
    x_pixel = central_pixel_coordinates[1] - (origin[1] / pixel_scales[1])

    return (y_pixel, x_pixel)


def pixel_coordinates_2d_from(
    scaled_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Convert a 2D (y,x) scaled coordinate to a 2D (y,x) pixel coordinate, which are returned as floats such that they
    include the decimal offset from each pixel's top-left corner relative to the input scaled coordinate.

    The conversion is performed according to a 2D geometry on a uniform grid, where the pixel coordinate origin is at
    the top left corner, such that the pixel [0,0] corresponds to the highest (most positive) y scaled coordinate
    and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate is defined by an origin and coordinates are shifted to this origin before computing their
    1D grid pixel coordinate values.

    Parameters
    ----------
    scaled_coordinates_2d
        The 2D (y,x) coordinates in scaled units which are converted to pixel coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    A 2D (y,x) pixel-value coordinate.

    Examples
    --------

    scaled_coordinates_2d = (1.0, 1.0)

    grid_pixels_2d_slim = pixel_coordinates_2d_from(
        scaled_coordinates_2d=scaled_coordinates_2d,
        shape=(2,2),
        pixel_scales=(0.5, 0.5),
        origin=(0.0, 0.0)
    )
    """

    central_pixel_coordinates = central_pixel_coordinates_2d_from(
        shape_native=shape_native
    )

    y_pixel = int(
        (-scaled_coordinates_2d[0] + origins[0]) / pixel_scales[0]
        + central_pixel_coordinates[0]
        + 0.5
    )
    x_pixel = int(
        (scaled_coordinates_2d[1] - origins[1]) / pixel_scales[1]
        + central_pixel_coordinates[1]
        + 0.5
    )

    return (y_pixel, x_pixel)


def scaled_coordinates_2d_from(
    pixel_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Convert a 2D (y,x) pixel coordinates to a 2D (y,x) scaled values.

    The conversion is performed according to a 2D geometry on a uniform grid, where the pixel coordinate origin is at
    the top left corner, such that the pixel [0,0] corresponds to the highest (most positive) y scaled coordinate
    and lowest (most negative) x scaled coordinate on the gird.

    The scaled coordinate is defined by an origin and coordinates are shifted to this origin before computing their
    1D grid pixel coordinate values.

    Parameters
    ----------
    scaled_coordinates_2d
        The 2D (y,x) coordinates in scaled units which are converted to pixel coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    A 2D (y,x) pixel-value coordinate.

    Examples
    --------

    scaled_coordinates_2d = (1.0, 1.0)

    grid_pixels_2d_slim = pixel_coordinates_2d_from(
        scaled_coordinates_2d=scaled_coordinates_2d,
        shape=(2,2),
        pixel_scales=(0.5, 0.5),
        origin=(0.0, 0.0)
    )
    """
    central_scaled_coordinates = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origins
    )

    y_pixel = pixel_scales[0] * -(
        pixel_coordinates_2d[0] - central_scaled_coordinates[0]
    )
    x_pixel = pixel_scales[1] * (
        pixel_coordinates_2d[1] - central_scaled_coordinates[1]
    )

    return (y_pixel, x_pixel)


def transform_grid_2d_to_reference_frame(
    grid_2d: np.ndarray, centre: Tuple[float, float], angle: float
) -> np.ndarray:
    """
    Transform a 2D grid of (y,x) coordinates to a new reference frame.

    This transformation includes:

     1) A translation to a new (y,x) centre value, by subtracting the centre from every coordinate on the grid.
     2) A rotation of the grid around this new centre, which is performed clockwise from an input angle.

    Parameters
    ----------
    grid
        The 2d grid of (y, x) coordinates which are transformed to a new reference frame.
    """
    grid_2d = jnp.asarray(grid_2d)
    centre = jnp.asarray(centre)

    # Inject a tiny dynamic dependency on `angle` to prevent heavy constant folding
    # (adds zero at runtime; negligible cost)
    dynamic_zero = jnp.zeros_like(grid_2d) * angle
    shifted = (grid_2d + dynamic_zero) - centre

    radius = jnp.sqrt(jnp.sum(shifted * shifted, axis=1))
    theta  = jnp.arctan2(shifted[:, 0], shifted[:, 1]) - jnp.deg2rad(angle)
    return jnp.vstack(
        [
            radius * jnp.sin(theta),
            radius * jnp.cos(theta),
        ]
    ).T


def transform_grid_2d_from_reference_frame(
    grid_2d: np.ndarray, centre: Tuple[float, float], angle: float
) -> np.ndarray:
    """
    Transform a 2D grid of (y,x) coordinates to a new reference frame, which is the reverse frame computed via the
    method `transform_grid_2d_to_reference_frame`.

    This transformation includes:

     1) A translation to a new (y,x) centre value, by adding the centre to every coordinate on the grid.
     2) A rotation of the grid around this new centre, which is performed counter-clockwise from an input angle.

    Parameters
    ----------
    grid
        The 2d grid of (y, x) coordinates which are transformed to a new reference frame.
    """

    cos_angle = jnp.cos(jnp.radians(angle))
    sin_angle = jnp.sin(jnp.radians(angle))

    y = jnp.add(
        jnp.add(
            jnp.multiply(grid_2d[:, 1], sin_angle),
            jnp.multiply(grid_2d[:, 0], cos_angle),
        ),
        centre[0],
    )
    x = jnp.add(
        jnp.add(
            jnp.multiply(grid_2d[:, 1], cos_angle),
            -jnp.multiply(grid_2d[:, 0], sin_angle),
        ),
        centre[1],
    )
    return jnp.vstack((y, x)).T


def grid_pixels_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
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
    grid_scaled_2d_slim
        The slimmed grid of 2D (y,x) coordinates in scaled units which are converted to pixel value coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    A slimmed grid of 2D (y,x) pixel-value coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )
    centres_scaled = centres_scaled
    pixel_scales = pixel_scales
    sign = np.array([-1, 1])
    return (sign * grid_scaled_2d_slim / pixel_scales) + centres_scaled + 0.5


def grid_pixel_centres_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
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
    grid_scaled_2d_slim
        The slimmed grid of 2D (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    A slimmed grid of 2D (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = np.array(centres_scaled)
    pixel_scales = np.array(pixel_scales)
    sign = np.array([-1.0, 1.0])
    return ((sign * grid_scaled_2d_slim / pixel_scales) + centres_scaled + 0.5).astype(
        int
    )


def grid_pixel_indexes_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
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
    grid_scaled_2d_slim
        The slimmed grid of 2D (y,x) coordinates in scaled units which is converted to slimmed pixel indexes.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
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

    return (
        (grid_pixels_2d_slim * np.array([shape_native[1], 1])).sum(axis=1).astype(int)
    )


def grid_scaled_2d_slim_from(
    grid_pixels_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
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
    grid_pixels_2d_slim
        The slimmed grid of (y,x) coordinates in pixel values which is converted to scaled coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    A slimmed grid of 2d scaled coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_pixels_2d_slim = np.array([[0,0], [0,1], [1,0], [1,1])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_pixels_2d_slim=grid_pixels_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = np.array(centres_scaled)
    pixel_scales = np.array(pixel_scales)
    sign = np.array([-1, 1])
    return (grid_pixels_2d_slim - centres_scaled - 0.5) * pixel_scales * sign


def grid_pixel_centres_2d_from(
    grid_scaled_2d: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
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
    grid_scaled_2d
        The native grid of 2D (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    A native grid of 2D (y,x) pixel indexes with dimensions (y_pixels, x_pixels, 2).

    Examples
    --------
    grid_scaled_2d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixel_centres_2d = grid_pixel_centres_2d_from(grid_scaled_2d=grid_scaled_2d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = np.array(centres_scaled)
    pixel_scales = np.array(pixel_scales)
    sign = np.array([-1.0, 1.0])
    return ((sign * grid_scaled_2d / pixel_scales) + centres_scaled + 0.5).astype(int)


def extent_symmetric_from(
    extent: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Given an input extent of the form (x_min, x_max, y_min, y_max), this function returns an extent which is
    symmetric about the origin.

    For example, if the sepration from x_min to x_max is 2.0 and the separation from y_min to y_max is 1.0, the
    returned extent expands the y-axis to also have a separation of 2.0.

    Parameters
    ----------
    extent
        The extent which is to be made symmetric about the origin.

    Returns
    -------
    The new extent which is symmetric about the origin.
    """

    y_min = extent[2]
    y_max = extent[3]
    x_min = extent[0]
    x_max = extent[1]

    y_sep = y_max - y_min
    x_sep = x_max - x_min

    if y_sep > x_sep:
        x_min -= (y_sep - x_sep) / 2
        x_max += (y_sep - x_sep) / 2
    elif x_sep > y_sep:
        y_min -= (x_sep - y_sep) / 2
        y_max += (x_sep - y_sep) / 2

    return (x_min, x_max, y_min, y_max)
