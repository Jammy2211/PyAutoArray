import logging
import numpy as np

from autoconf import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.interferometer.settings import SettingsInterferometer
from autoarray.dataset.interferometer.w_tilde import WTildeInterferometer
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.structures.arrays import array_2d_util

logger = logging.getLogger(__name__)


class Interferometer(AbstractDataset):
    def __init__(
        self,
        visibilities: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        uv_wavelengths: np.ndarray,
        real_space_mask,
        settings: SettingsInterferometer = SettingsInterferometer(),
    ):
        """
        A class containing an interferometer dataset, including the visibilities data, noise-map and the
        uv-plane baseline wavelengths.

        Parameters
        ----------
        visibilities
            The array of the visibilities data, containing by real and complex values.
        noise_map
            An array describing the RMS standard deviation error in each visibility.
        uv_wavelengths
            The uv-plane baseline wavelengths.
        real_space_mask
            A 2D mask in real-space (e.g. not Fourier space like the visibilities) which defines in real space
            how calculations are performed.
        settings
            Controls settings of how the dataset is set up (e.g. the types of grids used to perform calculations).
        """
        self.real_space_mask = real_space_mask

        super().__init__(data=visibilities, noise_map=noise_map, settings=settings)

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
        settings: SettingsInterferometer = SettingsInterferometer(),
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

        uv_wavelengths = array_2d_util.numpy_array_2d_via_fits_from(
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
        )

    @cached_property
    def w_tilde(self):
        """
        The w_tilde formalism of the linear algebra equations precomputes the Fourier Transform of all the visibilities
        given the `uv_wavelengths` (see `inversion.inversion_util`).

        The `WTilde` object stores these precomputed values in the interferometer dataset ensuring they are only
        computed once per analysis.

        This uses lazy allocation such that the calculation is only performed when the wtilde matrices are used,
        ensuring efficient set up of the `Interferometer` class.

        Returns
        -------
        WTildeInterferometer
            Precomputed values used for the w tilde formalism of linear algebra calculations.
        """

        from autoarray.inversion.inversion import inversion_util_secret

        logger.info("INTERFEROMETER - Computing W-Tilde... May take a moment.")

        curvature_preload = inversion_util_secret.w_tilde_curvature_preload_interferometer_from(
            noise_map_real=self.noise_map.real,
            uv_wavelengths=self.uv_wavelengths,
            shape_masked_pixels_2d=self.transformer.grid.mask.shape_native_masked_pixels,
            grid_radians_2d=self.transformer.grid.mask.unmasked_grid_sub_1.in_radians.native,
        )

        w_matrix = inversion_util_secret.w_tilde_via_preload_from(
            w_tilde_preload=curvature_preload,
            native_index_for_slim_index=self.real_space_mask.native_index_for_slim_index,
        )

        dirty_image = self.transformer.image_from(
            visibilities=self.visibilities.real * self.noise_map.real**-2.0
            + 1j * self.visibilities.imag * self.noise_map.imag**-2.0,
            use_adjoint_scaling=True,
        )

        return WTildeInterferometer(
            w_matrix=w_matrix,
            curvature_preload=curvature_preload,
            dirty_image=dirty_image,
            real_space_mask=self.real_space_mask,
            noise_map_value=self.noise_map[0],
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
        return self.transformer.image_from(visibilities=self.visibilities)

    @property
    def dirty_noise_map(self):
        return self.transformer.image_from(visibilities=self.noise_map)

    @property
    def dirty_signal_to_noise_map(self):
        return self.transformer.image_from(visibilities=self.signal_to_noise_map)

    @property
    def dirty_inverse_noise_map(self):
        return self.transformer.image_from(visibilities=self.inverse_noise_map)

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

    @property
    def blurring_grid(self):
        return None

    @property
    def convolver(self):
        return None

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
