import numpy as np

from autoarray.dataset.interferometer.interferometer import Interferometer
from autoarray.operators.transformer import TransformerDFT
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray import exc
from autoarray.dataset import preprocess


class AbstractSimulatorInterferometer:
    def __init__(
        self,
        uv_wavelengths,
        exposure_time: float,
        transformer_class=TransformerDFT,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        real_space_shape_native
            The shape of the observation. Note that we do not simulator a full Imaging array (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        real_space_pixel_scales
            The size of each pixel in scaled units.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time_map
            The exposure time of an observation using this data_type.
        """

        self.uv_wavelengths = uv_wavelengths
        self.exposure_time = exposure_time
        self.transformer_class = transformer_class
        self.noise_sigma = noise_sigma
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    def via_image_from(self, image):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        real_space_image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and UVPlane read-out).
        real_space_pixel_scales
            The scale of each pixel in scaled units
        exposure_time_map
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated image is blurred with.
        add_poisson_noise: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        transformer = self.transformer_class(
            uv_wavelengths=self.uv_wavelengths, real_space_mask=image.mask
        )

        visibilities = transformer.visibilities_from(image=image)

        if self.noise_sigma is not None:
            visibilities = preprocess.data_with_complex_gaussian_noise_added(
                data=visibilities, sigma=self.noise_sigma, seed=self.noise_seed
            )
            noise_map = VisibilitiesNoiseMap.full(
                fill_value=self.noise_sigma, shape_slim=(visibilities.shape[0],)
            )
        else:
            noise_map = VisibilitiesNoiseMap.full(
                fill_value=self.noise_if_add_noise_false,
                shape_slim=(visibilities.shape[0],),
            )

        if np.isnan(noise_map).any():
            raise exc.DatasetException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        return Interferometer(
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=transformer.uv_wavelengths,
            real_space_mask=image.mask,
        )


class SimulatorInterferometer(AbstractSimulatorInterferometer):

    pass
