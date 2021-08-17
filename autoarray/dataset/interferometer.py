import logging
import numpy as np
import copy
from typing import List, Optional, Tuple, Type, Union

from autoarray.dataset.abstract_dataset import AbstractSettingsDataset
from autoarray.dataset.abstract_dataset import AbstractDataset
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray import exc
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.dataset import preprocess


logger = logging.getLogger(__name__)


class SettingsInterferometer(AbstractSettingsDataset):
    def __init__(
        self,
        grid_class=Grid2D,
        grid_inversion_class=Grid2D,
        sub_size=1,
        sub_size_inversion=1,
        fractional_accuracy: float = 0.9999,
        sub_steps: List[int] = None,
        pixel_scales_interp: Optional[Union[float, Tuple[float, float]]] = None,
        signal_to_noise_limit: Optional[float] = None,
        transformer_class=TransformerNUFFT,
    ):
        """
          The lens dataset is the collection of data_type (image, noise-map), a mask, grid, convolver \
          and other utilities that are used for modeling and fitting an image of a strong lens.

          Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
          for lens calculations.

          Parameters
          ----------
        grid_class : ag.Grid2D
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D`,
            `Grid2DIterate` and `Grid2DInterpolate` (see the `Grid2D` documentation for a description of these options).
        grid_inversion_class : ag.Grid2D
            The type of grid used to create the grid that maps the `Inversion` source pixels to the data's image-pixels.
            The options are `Grid2D`, `Grid2DIterate` and `Grid2DInterpolate` (see the `Grid2D` documentation for a
            description of these options).
        sub_size
            If the grid and / or grid_inversion use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy : float
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the fractional accuracy it
            uses when evaluating functions.
        sub_steps : [int]
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        pixel_scales_interp : float or (float, float)
            If the grid and / or grid_inversion use a `Grid2DInterpolate`, this sets the resolution of the interpolation
            grid.
        signal_to_noise_limit : float
            If input, the dataset's noise-map is rescaled such that no pixel has a signal-to-noise above the
            signa to noise limit.
          """

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            sub_size_inversion=sub_size_inversion,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
            signal_to_noise_limit=signal_to_noise_limit,
        )

        self.transformer_class = transformer_class


class Interferometer(AbstractDataset):
    def __init__(
        self,
        visibilities,
        noise_map,
        uv_wavelengths,
        real_space_mask,
        settings=SettingsInterferometer(),
        name=None,
    ):

        self.real_space_mask = real_space_mask

        super().__init__(
            data=visibilities, noise_map=noise_map, name=name, settings=settings
        )

        self.uv_wavelengths = uv_wavelengths

        self.transformer = self.settings.transformer_class(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

    @classmethod
    def from_fits(
        cls,
        visibilities_path,
        noise_map_path,
        uv_wavelengths_path,
        real_space_mask,
        visibilities_hdu=0,
        noise_map_hdu=0,
        uv_wavelengths_hdu=0,
        settings=SettingsInterferometer(),
    ):
        """Factory for loading the interferometer data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the interferometer-data_type.

        This factory also includes a number of routines for converting the interferometer-data_type from unit_label not supported by PyAutoLens \
        (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        """

        visibilities = Visibilities.from_fits(
            file_path=visibilities_path, hdu=visibilities_hdu
        )

        noise_map = VisibilitiesNoiseMap.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )

        uv_wavelengths = array_2d_util.numpy_array_2d_from_fits(
            file_path=uv_wavelengths_path, hdu=uv_wavelengths_hdu
        )

        return Interferometer(
            real_space_mask=real_space_mask,
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            settings=settings,
        )

    def apply_settings(self, settings):

        return Interferometer(
            visibilities=self.visibilities,
            noise_map=self.noise_map,
            uv_wavelengths=self.uv_wavelengths,
            real_space_mask=self.real_space_mask,
            settings=settings,
            name=self.name,
        )

    @property
    def mask(self):
        return self.real_space_mask

    @property
    def visibilities(self):
        return self.data

    @property
    def amplitudes(self):
        return self.visibilities.amplitudes

    @property
    def phases(self):
        return self.visibilities.phases

    @property
    def uv_distances(self):
        return np.sqrt(
            np.square(self.uv_wavelengths[:, 0]) + np.square(self.uv_wavelengths[:, 1])
        )

    @property
    def dirty_image(self):
        return self.transformer.image_from_visibilities(visibilities=self.visibilities)

    @property
    def dirty_noise_map(self):
        return self.transformer.image_from_visibilities(visibilities=self.noise_map)

    @property
    def dirty_signal_to_noise_map(self):
        return self.transformer.image_from_visibilities(
            visibilities=self.signal_to_noise_map
        )

    @property
    def dirty_inverse_noise_map(self):
        return self.transformer.image_from_visibilities(
            visibilities=self.inverse_noise_map
        )

    def modified_visibilities_from_visibilities(self, visibilities):

        interferometer = copy.deepcopy(self)
        interferometer.data = Visibilities(visibilities=visibilities)
        return interferometer

    @property
    def signal_to_noise_map(self):

        signal_to_noise_map_real = np.divide(
            np.real(self.data), np.real(self.noise_map)
        )
        signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
        signal_to_noise_map_imag = np.divide(
            np.imag(self.data), np.imag(self.noise_map)
        )
        signal_to_noise_map_imag[signal_to_noise_map_imag < 0] = 0.0

        return signal_to_noise_map_real + 1j * signal_to_noise_map_imag

    def signal_to_noise_limited_from(self, signal_to_noise_limit, mask=None):

        interferometer = copy.deepcopy(self)

        noise_map_limit_real = np.where(
            np.real(self.signal_to_noise_map) > signal_to_noise_limit,
            np.real(self.visibilities) / signal_to_noise_limit,
            np.real(self.noise_map),
        )

        noise_map_limit_imag = np.where(
            np.imag(self.signal_to_noise_map) > signal_to_noise_limit,
            np.imag(self.visibilities) / signal_to_noise_limit,
            np.imag(self.noise_map),
        )

        interferometer.noise_map = VisibilitiesNoiseMap(
            visibilities=noise_map_limit_real + 1j * noise_map_limit_imag
        )

        return interferometer

    def output_to_fits(
        self,
        visibilities_path=None,
        noise_map_path=None,
        uv_wavelengths_path=None,
        overwrite=False,
    ):

        if visibilities_path is not None:
            self.visibilities.output_to_fits(
                file_path=visibilities_path, overwrite=overwrite
            )

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)

        if self.uv_wavelengths is not None and uv_wavelengths_path is not None:
            array_2d_util.numpy_array_2d_to_fits(
                array_2d=self.uv_wavelengths,
                file_path=uv_wavelengths_path,
                overwrite=overwrite,
            )


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
        real_space_pixel_scales : float
            The size of each pixel in scaled units.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        """

        self.uv_wavelengths = uv_wavelengths
        self.exposure_time = exposure_time
        self.transformer_class = transformer_class
        self.noise_sigma = noise_sigma
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    def from_image(self, image, name=None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        real_space_image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and UVPlane read-out).
        real_space_pixel_scales: float
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

        visibilities = transformer.visibilities_from_image(image=image)

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
            name=name,
        )


class SimulatorInterferometer(AbstractSimulatorInterferometer):

    pass
