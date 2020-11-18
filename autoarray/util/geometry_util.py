def convert_pixel_scales_2d(pixel_scales):

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales, pixel_scales)

    return pixel_scales


def central_pixel_coordinates_from(shape):
    return tuple([float(dim - 1) / 2 for dim in shape])


def central_scaled_coordinate_2d_from(shape_2d, pixel_scales, origin=(0.0, 0.0)):

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
