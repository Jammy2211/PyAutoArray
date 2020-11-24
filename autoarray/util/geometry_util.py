import typing


def convert_pixel_scales_2d(
    pixel_scales: typing.Union[float, typing.Tuple[float, float]]
) -> typing.Tuple[float, float]:
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


def central_pixel_coordinates_from(shape: typing.Tuple[float]) -> typing.Tuple[float]:
    """
    Returns the central pixel coordinates of a data structure of any dimension (e.g. in 1D a `Line`, 2D an `Array`,
    2d a `Frame`, etc.) from the shape of that data structure.

    Examples of the central pixels are as follows:

    - For a 3x3 image, the central pixel is pixel [1, 1].
    - For a 4x4 image, the central pixel is [1.5, 1.5].

    Parameters
    ----------
    shape : tuple(int)
        The dimensions of the data structure, which can be in 1D, 2D or higher dimensions.

    Returns
    -------
    central_pixel_coordinates : tuple(float)
        The central pixel coordinates of the data structure.

    """
    return tuple([float(dim - 1) / 2 for dim in shape])


def central_scaled_coordinate_2d_from(shape_2d, pixel_scales, origin=(0.0, 0.0)):
    """
    Returns the central coordinates of a 2D data structure (e.g. a `Frame`, `Grid`) in scaled units.

    This is computed by using the data structure's shape and converting it to scaled units using an input
    pixel-coordinates to scaled-coordinate conversion factor `pixel_scales`.

    The origin of the scaled grid can also be input and moved from (0.0, 0.0).

    Parameters
    ----------
    shape_2d : (int, int)
        The 2D shape of the data structure whose central scaled coordinates are computed.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D data structure.
    origin : (float, float)
        The (y,x) scaled units origin of the coordinate system the central scaled coordinate is computed on.

    Returns
    -------
    central_scaled_coordinates_2d : (float, float)
        The central coordinates of the 2D data structure in scaled units.
    """

    pixel_scales = convert_pixel_scales_2d(pixel_scales=pixel_scales)

    central_pixel_coordinates = central_pixel_coordinates_from(shape=shape_2d)

    y_pixel = central_pixel_coordinates[0] + (origin[0] / pixel_scales[0])
    x_pixel = central_pixel_coordinates[1] - (origin[1] / pixel_scales[1])

    return (y_pixel, x_pixel)


def pixel_coordinates_2d_from(
    scaled_coordinates_2d, shape_2d, pixel_scales, origins=(0.0, 0.0)
):

    pixel_scales = convert_pixel_scales_2d(pixel_scales=pixel_scales)

    central_pixel_coordinates = central_pixel_coordinates_from(shape=shape_2d)

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
    pixel_coordinates_2d, shape_2d, pixel_scales, origins=(0.0, 0.0)
):

    pixel_scales = convert_pixel_scales_2d(pixel_scales=pixel_scales)

    central_scaled_coordinates = central_scaled_coordinate_2d_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origins
    )

    y_pixel = pixel_scales[0] * -(
        pixel_coordinates_2d[0] - central_scaled_coordinates[0]
    )
    x_pixel = pixel_scales[1] * (
        pixel_coordinates_2d[1] - central_scaled_coordinates[1]
    )

    return (y_pixel, x_pixel)
