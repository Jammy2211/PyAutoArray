import logging
import numpy as np

from autoarray.dataset.imaging.imaging import Imaging
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.arrays.kernel_2d import Kernel2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.dataset import preprocess

logger = logging.getLogger(__name__)


class SimulatorImaging:
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: Kernel2D = None,
        normalize_psf: bool = True,
        read_noise: float = None,
        add_poisson_noise: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """
        A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        psf : Kernel2D
            An arrays describing the PSF kernel of the image.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        normalize_psf
            If `True`, the PSF kernel is normalized so all values sum to 1.0.
        read_noise
            The level of read-noise added to the simulated imaging by drawing from a Gaussian distribution with
            sigma equal to the value `read_noise`.
        add_poisson_noise
            Whether Poisson noise corresponding to photon count statistics on the imaging observation is added.
        noise_if_add_noise_false
            If noise is not added to the simulated dataset a `noise_map` must still be returned. This value gives
            the value of noise assigned to every pixel in the noise-map.
        noise_seed
            The random seed used to add random noise, where -1 corresponds to a random seed every run.
        """

        if psf is not None:
            if psf is not None and normalize_psf:
                psf = psf.normalized
            self.psf = psf
        else:
            self.psf = Kernel2D.no_blur(pixel_scales=1.0)

        self.exposure_time = exposure_time
        self.background_sky_level = background_sky_level

        self.read_noise = read_noise
        self.add_poisson_noise = add_poisson_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    def via_image_from(self, image: Array2D):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image
            The image before simulating which has noise added, PSF convolution, etc performed to it.
        """

        exposure_time_map = Array2D.full(
            fill_value=self.exposure_time,
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
        )

        background_sky_map = Array2D.full(
            fill_value=self.background_sky_level,
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
        )

        image = self.psf.convolved_array_from(array=image)

        image = image + background_sky_map

        if self.add_poisson_noise is True:
            image = preprocess.data_eps_with_poisson_noise_added(
                data_eps=image,
                exposure_time_map=exposure_time_map,
                seed=self.noise_seed,
            )

            noise_map = preprocess.noise_map_via_data_eps_and_exposure_time_map_from(
                data_eps=image, exposure_time_map=exposure_time_map
            )

        else:
            noise_map = Array2D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=image.shape_native,
                pixel_scales=image.pixel_scales,
            )

        if np.isnan(noise_map).any():
            raise exc.DatasetException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        image = image - background_sky_map

        mask = Mask2D.unmasked(
            shape_native=image.shape_native, pixel_scales=image.pixel_scales
        )

        image = Array2D.manual_mask(array=image, mask=mask)

        return Imaging(image=image, psf=self.psf, noise_map=noise_map)
