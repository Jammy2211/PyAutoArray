import logging
import numpy as np

from autoarray.dataset.imaging.dataset import Imaging
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
        add_poisson_noise: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """
        Simulations observations of imaging data, including simulation of the image, noise-map, PSF, etc. as
        an `Imaging` object.

        The simulation of an `Imaging` dataset uses the following steps:

        1) Receive as input the raw image which is simulated via the steps below.
        2) Convolve the image with the Point Spread Function of the simulated dataset.
        3) Use input values of the background sky level in every pixel of the image to add the background sky to
           the PSF convolved image.
        4) Add Poisson noise to the image, which represents noise due to whether photons hits the CCD and are converted
           to photo-electrons which are succcessfully detected by the CCD and converted to counts.
        5) Subtract the background sky from the image, so that the returned simulated dataset is background sky
           subtracted.

        The inputs of the `SimulatorImaging` object can toggle these steps on and off, for example if `psf=None` the
        PSF convolution step is omitted.

        Parameters
        ----------
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        psf
            An array describing the PSF kernel of the image.
        normalize_psf
            If `True`, the PSF kernel is normalized so all values sum to 1.0.
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

        self.add_poisson_noise = add_poisson_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    def via_image_from(self, image: Array2D) -> Imaging:
        """
        Simulate an `Imaging` dataset from an input image.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        image
            The 2D image from which the Imaging dataset is simulated.
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

        mask = Mask2D.all_false(
            shape_native=image.shape_native, pixel_scales=image.pixel_scales
        )

        image = Array2D(values=image, mask=mask)

        return Imaging(data=image, psf=self.psf, noise_map=noise_map)
