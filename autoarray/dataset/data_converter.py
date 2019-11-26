from autoarray.structures import arrays

import numpy as np

# TODO : The way these work dpeending on if the input array (e.g. the weight map) are an auto array or not is a bit
# TODO : clunky. For example, we can currently overwrite pixel_scales. We need to decade how we approach this.


def array_using_correct_dimensions(
    array, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)
):

    if isinstance(array, arrays.AbstractArray):
        return array

    if len(array.shape) == 1:
        return arrays.Array.manual_1d(
            array=array,
            shape_2d=array.shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )
    elif len(array.shape) == 2:
        return arrays.Array.manual_2d(
            array=array, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )


def noise_map_from_weight_map(
    weight_map, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)
):
    """Setup the noise-map from a weight map, which is a form of noise-map that comes via HST image-reduction and \
    the software package MultiDrizzle.

    The variance in each pixel is computed as:

    Variance = 1.0 / sqrt(weight_map).

    The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
    the analysis.

    Parameters
    -----------
    pixel_scales : float
        The size of each pixel in arc seconds.
    weight_map : ndarray
        The weight-value of each pixel which is converted to a variance.
    """
    np.seterr(divide="ignore")
    noise_map = 1.0 / np.sqrt(weight_map)
    noise_map[noise_map > 1.0e8] = 1.0e8
    return array_using_correct_dimensions(
        array=noise_map, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
    )


def noise_map_from_inverse_noise_map(
    inverse_noise_map, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)
):
    """Setup the noise-map from an root-mean square standard deviation map, which is a form of noise-map that \
    comes via HST image-reduction and the software package MultiDrizzle.

    The variance in each pixel is computed as:

    Variance = 1.0 / inverse_std_map.

    The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
    the analysis.

    Parameters
    -----------
    pixel_scales : float
        The size of each pixel in arc seconds.
    inverse_noise_map : ndarray
        The inverse noise_map value of each pixel which is converted to a variance.
    """
    return array_using_correct_dimensions(
        array=1.0 / inverse_noise_map,
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


def noise_map_from_image_and_background_noise_map(
    image,
    background_noise_map,
    exposure_time_map,
    gain=None,
    convert_from_electrons=False,
    convert_from_adus=False,
    pixel_scales=None,
    sub_size=1,
    origin=(0.0, 0.0),
):

    if not convert_from_electrons and not convert_from_adus:
        noise_map = (
            np.sqrt(
                np.abs(
                    ((background_noise_map) * exposure_time_map) ** 2.0
                    + (image) * exposure_time_map
                )
            )
            / exposure_time_map
        )

    elif convert_from_electrons:
        noise_map = np.sqrt(np.abs(background_noise_map ** 2.0 + image))
    elif convert_from_adus:
        noise_map = (
            np.sqrt(np.abs((gain * background_noise_map) ** 2.0 + gain * image)) / gain
        )

    return array_using_correct_dimensions(
        array=noise_map, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
    )


def poisson_noise_map_from_image_and_exposure_time_map(
    image,
    exposure_time_map,
    gain=None,
    convert_from_electrons=False,
    convert_from_adus=False,
    pixel_scales=None,
    sub_size=1,
    origin=(0.0, 0.0),
):
    if not convert_from_electrons and not convert_from_adus:
        noise_map = np.sqrt(np.abs(image) * exposure_time_map) / exposure_time_map

    elif convert_from_electrons:
        noise_map = np.sqrt(np.abs(image))
    elif convert_from_adus:
        noise_map = np.sqrt(gain * np.abs(image)) / gain

    return array_using_correct_dimensions(
        array=noise_map, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
    )
