def central_pixel_coordinates_from(shape):
    return tuple([float(dim - 1) / 2 for dim in shape])


def central_scaled_coordinates_from(shape, pixel_scales, origins=None):

    if type(pixel_scales) is float:
        pixel_scales = tuple([pixel_scales for i in range(len(shape))])

    if origins is None:
        origins = tuple([0.0 for i in range(len(shape))])

    central_pixel_coordinates = central_pixel_coordinates_from(shape=shape)

    y_pixel = central_pixel_coordinates[0] + (origins[0] / pixel_scales[0])
    x_pixel = central_pixel_coordinates[1] - (origins[1] / pixel_scales[1])

    return (y_pixel, x_pixel)


def pixel_coordinates_from_scaled_coordinates(
    scaled_coordinates, shape, pixel_scales, origins=None
):

    if type(pixel_scales) is float:
        pixel_scales = tuple([pixel_scales for i in range(len(shape))])

    central_pixel_coordinates = central_pixel_coordinates_from(shape=shape)

    y_pixel = int(
        (-scaled_coordinates[0] + origins[0]) / pixel_scales[0]
        + central_pixel_coordinates[0]
        + 0.5
    )
    x_pixel = int(
        (scaled_coordinates[1] - origins[1]) / pixel_scales[1]
        + central_pixel_coordinates[1]
        + 0.5
    )

    return (y_pixel, x_pixel)


def scaled_coordinates_from_pixel_coordinates(
    pixel_coordinates, shape, pixel_scales, origins=None
):

    if type(pixel_scales) is float:
        pixel_scales = tuple([pixel_scales for i in range(len(shape))])

    central_scaled_coordinates = central_scaled_coordinates_from(
        shape=shape, pixel_scales=pixel_scales, origins=origins
    )

    y_pixel = pixel_scales[0] * -(pixel_coordinates[0] - central_scaled_coordinates[0])
    x_pixel = pixel_scales[1] * (pixel_coordinates[1] - central_scaled_coordinates[1])

    return (y_pixel, x_pixel)
