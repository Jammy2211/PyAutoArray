from typing import Tuple, Union
import numpy as np

from autoarray import decorator_util


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


def convert_pixel_scales_1d(pixel_scales: Union[float, Tuple[float]]) -> Tuple[float]:
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


@decorator_util.jit()
def central_pixel_coordinates_1d_from(
    shape_slim: Tuple[int]
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


@decorator_util.jit()
def central_scaled_coordinate_1d_from(
    shape_slim: Tuple[float],
    pixel_scales: Tuple[float],
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


@decorator_util.jit()
def pixel_coordinates_1d_from(
    scaled_coordinates_1d: Tuple[float],
    shape_slim: Tuple[int],
    pixel_scales: Tuple[float],
    origins: Tuple[float] = (0.0, 0.0),
) -> Union[Tuple[float], Tuple[float]]:

    central_pixel_coordinates = central_pixel_coordinates_1d_from(shape_slim=shape_slim)

    x_pixel = int(
        (scaled_coordinates_1d[0] - origins[0]) / pixel_scales[0]
        + central_pixel_coordinates[0]
        + 0.5
    )

    return (x_pixel,)


@decorator_util.jit()
def scaled_coordinates_1d_from(
    pixel_coordinates_1d: Tuple[float],
    shape_slim: Tuple[int],
    pixel_scales: Tuple[float],
    origins: Tuple[float] = (0.0, 0.0),
) -> Union[Tuple[float], Tuple[float]]:

    central_scaled_coordinates = central_scaled_coordinate_1d_from(
        shape_slim=shape_slim, pixel_scales=pixel_scales, origin=origins
    )

    x_pixel = pixel_scales[0] * (
        pixel_coordinates_1d[0] - central_scaled_coordinates[0]
    )

    return (x_pixel,)


def convert_pixel_scales_2d(
    pixel_scales: Union[float, Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Converts an input pixel-scale of type float to a pixel-scale of tuple (float, float). If the input is already
    type (float, float) it is unchanged

    This enables users to input the pixel scale as a single value and have the type automatically converted to type
    (float, float) which is used for rectangular grids.

    Parameters
    ----------
    pixel_scales : float or (float, float)
        The input pixel

    Returns
    -------
    pixel_scales
        The pixel_scales of type (float, float).
    """

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales, pixel_scales)

    return pixel_scales


@decorator_util.jit()
def central_pixel_coordinates_2d_from(
    shape_native: Tuple[int, int]
) -> Union[Tuple[float], Tuple[float, float]]:
    """
    Returns the central pixel coordinates of a data structure of any dimension (e.g. in 1D a `Line`, 2D an `Array2D`,
    2d a `Frame2D`, etc.) from the shape of that data structure.

    Examples of the central pixels are as follows:

    - For a 3x3 image, the central pixel is pixel [1, 1].
    - For a 4x4 image, the central pixel is [1.5, 1.5].

    Parameters
    ----------
    shape_native : tuple(int)
        The dimensions of the data structure, which can be in 1D, 2D or higher dimensions.

    Returns
    -------
    central_pixel_coordinates : tuple(float)
        The central pixel coordinates of the data structure.

    """
    return (float(shape_native[0] - 1) / 2, float(shape_native[1] - 1) / 2)


@decorator_util.jit()
def central_scaled_coordinate_2d_from(
    shape_native: Tuple[float, float],
    pixel_scales: Union[float, Tuple[float, float]],
    origin: Tuple[float, float] = (0.0, 0.0),
):
    """
    Returns the central coordinates of a 2D data structure (e.g. a `Frame2D`, `Grid2D`) in scaled units.

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
    central_scaled_coordinates_2d
        The central coordinates of the 2D data structure in scaled units.
    """

    central_pixel_coordinates = central_pixel_coordinates_2d_from(
        shape_native=shape_native
    )

    y_pixel = central_pixel_coordinates[0] + (origin[0] / pixel_scales[0])
    x_pixel = central_pixel_coordinates[1] - (origin[1] / pixel_scales[1])

    return (y_pixel, x_pixel)


@decorator_util.jit()
def pixel_coordinates_2d_from(
    scaled_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: Union[float, Tuple[float, float]],
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Union[Tuple[float], Tuple[float, float]]:

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


@decorator_util.jit()
def scaled_coordinates_2d_from(
    pixel_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: Union[float, Tuple[float, float]],
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Union[Tuple[float], Tuple[float, float]]:

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
    grid : ndarray
        The 2d grid of (y, x) coordinates which are transformed to a new reference frame.
    """
    shifted_grid_2d = np.subtract(grid_2d, centre)
    radius = np.sqrt(np.sum(shifted_grid_2d ** 2.0, 1))
    theta_coordinate_to_profile = np.arctan2(
        shifted_grid_2d[:, 0], shifted_grid_2d[:, 1]
    ) - np.radians(angle)
    return np.vstack(
        radius
        * (np.sin(theta_coordinate_to_profile), np.cos(theta_coordinate_to_profile))
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
    grid : ndarray
        The 2d grid of (y, x) coordinates which are transformed to a new reference frame.
    """

    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))

    y = np.add(
        np.add(
            np.multiply(grid_2d[:, 1], sin_angle), np.multiply(grid_2d[:, 0], cos_angle)
        ),
        centre[0],
    )
    x = np.add(
        np.add(
            np.multiply(grid_2d[:, 1], cos_angle),
            -np.multiply(grid_2d[:, 0], sin_angle),
        ),
        centre[1],
    )
    return np.vstack((y, x)).T
