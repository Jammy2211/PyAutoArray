import numpy as np

from autoarray import exc
from autoarray.dataset import data_converter
from autoarray.structures import arrays


def load_noise_map(
    noise_map_path,
    noise_map_hdu,
    pixel_scales,
    image=None,
    background_noise_map=None,
    exposure_time_map=None,
    convert_noise_map_from_weight_map=False,
    convert_noise_map_from_inverse_noise_map=False,
    noise_map_from_image_and_background_noise_map=False,
    convert_from_electrons=False,
    gain=None,
    convert_from_adus=False,
):
    """Factory for loading the noise-map from a .fits file.

    This factory also includes a number of routines for converting the noise-map from from other unit_label (e.g. \
    a weight map) or computing the noise-map from other unblurred_image_1d (e.g. the imaging image and background noise-map).

    Parameters
    ----------
    noise_map_path : str
        The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
    noise_map_hdu : int
        The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    image : ndarray
        The image-image, which the noise-map can be calculated using.
    background_noise_map : ndarray
        The background noise-map, which the noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the noise-map can be calculated using.
    convert_noise_map_from_weight_map : bool
        If True, the noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_noise_map_from_inverse_noise_map : bool
        If True, the noise-map loaded from the .fits file is converted from an inverse noise-map to a noise-map (see \
        *NoiseMap.from_inverse_noise_map).
    noise_map_from_image_and_background_noise_map : bool
        If True, the noise-map is computed from the observed image and background noise-map \
        (see NoiseMap.from_image_and_background_noise_map).
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in units of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in units of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """
    noise_map_options = sum(
        [
            convert_noise_map_from_weight_map,
            convert_noise_map_from_inverse_noise_map,
            noise_map_from_image_and_background_noise_map,
        ]
    )

    if noise_map_options > 1:
        raise exc.DataException(
            "You have specified more than one method to load the noise_map map, e.g.:"
            "convert_noise_map_from_weight_map | "
            "convert_noise_map_from_inverse_noise_map |"
            "noise_map_from_image_and_background_noise_map"
        )

    if noise_map_options == 0 and noise_map_path is not None:
        return arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
    elif convert_noise_map_from_weight_map and noise_map_path is not None:
        weight_map = arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
        return data_converter.noise_map_from_weight_map(weight_map=weight_map)
    elif convert_noise_map_from_inverse_noise_map and noise_map_path is not None:
        inverse_noise_map = arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
        return data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )
    elif noise_map_from_image_and_background_noise_map:

        if background_noise_map is None:
            raise exc.DataException(
                "Cannot compute the noise-map from the image and background noise_map map if a "
                "background noise_map map is not supplied."
            )

        if (
            not (convert_from_electrons or convert_from_adus)
            and exposure_time_map is None
        ):
            raise exc.DataException(
                "Cannot compute the noise-map from the image and background noise_map map if an "
                "exposure-time (or exposure time map) is not supplied to convert to adus"
            )

        if convert_from_adus and gain is None:
            raise exc.DataException(
                "Cannot compute the noise-map from the image and background noise_map map if a"
                "gain is not supplied to convert from adus"
            )

        return data_converter.noise_map_from_image_and_background_noise_map(
            image=image,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            convert_from_electrons=convert_from_electrons,
            gain=gain,
            convert_from_adus=convert_from_adus,
        )
    else:
        raise exc.DataException(
            "A noise_map map was not loaded, specify a noise_map_path or option to compute a noise_map map."
        )


def load_background_noise_map(
    background_noise_map_path,
    background_noise_map_hdu,
    pixel_scales,
    convert_background_noise_map_from_weight_map=False,
    convert_background_noise_map_from_inverse_noise_map=False,
):
    """Factory for loading the background noise-map from a .fits file.

    This factory also includes a number of routines for converting the background noise-map from from other unit_label (e.g. \
    a weight map).

    Parameters
    ----------
    background_noise_map_path : str
        The path to the background_noise_map .fits file containing the background noise-map \
        (e.g. '/path/to/background_noise_map.fits')
    background_noise_map_hdu : int
        The hdu the background_noise_map is contained in the .fits file specified by *background_noise_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    """
    background_noise_map_options = sum(
        [
            convert_background_noise_map_from_weight_map,
            convert_background_noise_map_from_inverse_noise_map,
        ]
    )

    if background_noise_map_options == 0 and background_noise_map_path is not None:
        return arrays.Array.from_fits(
            file_path=background_noise_map_path,
            hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
    elif (
        convert_background_noise_map_from_weight_map
        and background_noise_map_path is not None
    ):
        weight_map = arrays.Array.from_fits(
            file_path=background_noise_map_path,
            hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_weight_map(weight_map=weight_map)
    elif (
        convert_background_noise_map_from_inverse_noise_map
        and background_noise_map_path is not None
    ):
        inverse_noise_map = arrays.Array.from_fits(
            file_path=background_noise_map_path,
            hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )
    else:
        return None


def load_poisson_noise_map(
    poisson_noise_map_path,
    poisson_noise_map_hdu,
    pixel_scales,
    convert_poisson_noise_map_from_weight_map=False,
    convert_poisson_noise_map_from_inverse_noise_map=False,
    poisson_noise_map_from_image=False,
    image=None,
    exposure_time_map=None,
    convert_from_electrons=False,
    gain=None,
    convert_from_adus=False,
):
    """Factory for loading the Poisson noise-map from a .fits file.

    This factory also includes a number of routines for converting the Poisson noise-map from from other unit_label (e.g. \
    a weight map) or computing the Poisson noise_map from other unblurred_image_1d (e.g. the imaging image).

    Parameters
    ----------
    poisson_noise_map_path : str
        The path to the poisson_noise_map .fits file containing the Poisson noise-map \
         (e.g. '/path/to/poisson_noise_map.fits')
    poisson_noise_map_hdu : int
        The hdu the poisson_noise_map is contained in the .fits file specified by *poisson_noise_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    convert_poisson_noise_map_from_weight_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_poisson_noise_map_from_inverse_noise_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    poisson_noise_map_from_image : bool
        If True, the Poisson noise-map is estimated using the image.
    image : ndarray
        The image, which the Poisson noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the Poisson noise-map can be calculated using.
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in units of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in units of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """
    poisson_noise_map_options = sum(
        [
            convert_poisson_noise_map_from_weight_map,
            convert_poisson_noise_map_from_inverse_noise_map,
            poisson_noise_map_from_image,
        ]
    )

    if poisson_noise_map_options == 0 and poisson_noise_map_path is not None:
        return arrays.Array.from_fits(
            file_path=poisson_noise_map_path,
            hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
    elif poisson_noise_map_from_image:

        if (
            not (convert_from_electrons or convert_from_adus)
            and exposure_time_map is None
        ):
            raise exc.DataException(
                "Cannot compute the Poisson noise-map from the image if an "
                "exposure-time (or exposure time map) is not supplied to convert to adus"
            )

        if convert_from_adus and gain is None:
            raise exc.DataException(
                "Cannot compute the Poisson noise-map from the image if a"
                "gain is not supplied to convert from adus"
            )

        return data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=image,
            exposure_time_map=exposure_time_map,
            convert_from_electrons=convert_from_electrons,
            gain=gain,
            convert_from_adus=convert_from_adus,
        )

    elif (
        convert_poisson_noise_map_from_weight_map and poisson_noise_map_path is not None
    ):
        weight_map = arrays.Array.from_fits(
            file_path=poisson_noise_map_path,
            hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_weight_map(weight_map=weight_map)
    elif (
        convert_poisson_noise_map_from_inverse_noise_map
        and poisson_noise_map_path is not None
    ):
        inverse_noise_map = arrays.Array.from_fits(
            file_path=poisson_noise_map_path,
            hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )
    else:
        return None


def load_background_sky_map(
    background_sky_map_path, background_sky_map_hdu, pixel_scales
):
    """Factory for loading the background sky from a .fits file.

    Parameters
    ----------
    background_sky_map_path : str
        The path to the background_sky_map .fits file containing the background sky map \
        (e.g. '/path/to/background_sky_map.fits').
    background_sky_map_hdu : int
        The hdu the background_sky_map is contained in the .fits file specified by *background_sky_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    """
    if background_sky_map_path is not None:
        return arrays.Array.from_fits(
            file_path=background_sky_map_path,
            hdu=background_sky_map_hdu,
            pixel_scales=pixel_scales,
        )
    else:
        return None
