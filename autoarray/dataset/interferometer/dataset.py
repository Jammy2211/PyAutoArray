import logging
import numpy as np
from typing import Optional

from autoconf import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.interferometer.w_tilde import WTildeInterferometer
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.grids.over_sample.abstract import AbstractOverSample
from autoarray.structures.grids.over_sample.uniform import OverSampleUniform
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.structures.arrays import array_2d_util

logger = logging.getLogger(__name__)


class Interferometer(AbstractDataset):
    def __init__(
        self,
        data: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        uv_wavelengths: np.ndarray,
        real_space_mask,
        transformer_class=TransformerNUFFT,
        over_sample: Optional[AbstractOverSample] = OverSampleUniform(sub_size=1),
        over_sample_pixelization: Optional[AbstractOverSample] = OverSampleUniform(sub_size=4),
    ):
        """
        An interferometer dataset, containing the visibilities data, noise-map, real-space msk, Fourier transformer and
        associated quantities for calculations like the grid.

        This object is the input to the `FitInterferometer` object, which fits the dataset with model visibilities
        and quantifies the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the interferometer data are available and used for the following tasks:

        - `data`: The visibilities data, which shows the signal that is analysed and fitted with model visibilities.

        - `noise_map`: The RMS standard deviation error in every visibility, which is used to compute the chi-squared
        value and likelihood of a fit.

        - `uv_wavelengths`: The baselines of the interferometer which are used to Fourier transform a real space
        image to the uv-plane.

        `real_space_mask`: Defines in real space where the signal is present. This mask is used to transform images to
        Fourier space via the Fourier transform. The grids contained in the settings are aligned with this mask.

        Datasets also contains following properties:

        - `grid`: A grids of (y,x) coordinates which align with the image pixels, whereby each coordinate corresponds to
        the centre of an image pixel. This may be used in fits to calculate the model image of the imaging data.

        - `grid_pixelization`: A grid of (y,x) coordinates which align with the pixels of a pixelization. This grid
        is specifically used for pixelizations computed via the `invserion` module, which often use different
        oversampling and sub-size values to the grid above.

        The `over_sample` and `over_sample_pixelization` define how over sampling is performed for these grids.

        This is used in the project PyAutoGalaxy to load imaging data of a galaxy and fit it with galaxy light profiles.
        It is used in PyAutoLens to load imaging data of a strong lens and fit it with a lens model.

        Parameters
        ----------
        data
            The array of the visibilities data containing the signal that is fitted.
        noise_map
            An array describing the RMS standard deviation error in each visibility used for computing quantities like the
            chi-squared in a fit.
        uv_wavelengths
            The baselines of the interferometer which are used to Fourier transform a real space
            image to the uv-plane.
        real_space_mask
            Defines in real space where the signal is present. This mask is used to transform images to
            Fourier space via the Fourier transform. The grids contained in the settings are aligned with this mask.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        over_sample
            How over sampling is performed for the grid which performs calculations not associated with a pixelization.
            In PyAutoGalaxy and PyAutoLens this is light profile calculations.
        over_sample_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        transformer_class
            The class of the Fourier Transform which maps images from real space to Fourier space visibilities and
            the uv-plane.
        """
        self.real_space_mask = real_space_mask

        super().__init__(
            data=data,
            noise_map=noise_map,
            over_sample=over_sample,
            over_sample_pixelization=over_sample_pixelization,
        )

        self.uv_wavelengths = uv_wavelengths

        self.transformer = transformer_class(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

    @classmethod
    def from_fits(
        cls,
        data_path,
        noise_map_path,
        uv_wavelengths_path,
        real_space_mask,
        visibilities_hdu=0,
        noise_map_hdu=0,
        uv_wavelengths_hdu=0,
        transformer_class=TransformerNUFFT,
        over_sample: Optional[AbstractOverSample] = None,
        over_sample_pixelization: Optional[AbstractOverSample] = None,
    ):
        """
        Factory for loading the interferometer data_type from .fits files, as well as computing properties like the
        noise-map, exposure-time map, etc. from the interferometer-data_type.

        This factory also includes a number of routines for converting the interferometer-data_type from unit_label
        not supported by PyAutoLens (e.g. adus, electrons) to electrons per second.
        """

        visibilities = Visibilities.from_fits(file_path=data_path, hdu=visibilities_hdu)

        noise_map = VisibilitiesNoiseMap.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )

        uv_wavelengths = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=uv_wavelengths_path, hdu=uv_wavelengths_hdu
        )

        return Interferometer(
            real_space_mask=real_space_mask,
            data=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            transformer_class=transformer_class,
            over_sample=over_sample,
            over_sample_pixelization=over_sample_pixelization,
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

        logger.info("INTERFEROMETER - Computing W-Tilde... May take a moment.")

        from autoarray.inversion.inversion import inversion_util_secret

        curvature_preload = (
            inversion_util_secret.w_tilde_curvature_preload_interferometer_from(
                noise_map_real=np.array(self.noise_map.real),
                uv_wavelengths=np.array(self.uv_wavelengths),
                shape_masked_pixels_2d=np.array(
                    self.transformer.grid.mask.shape_native_masked_pixels
                ),
                grid_radians_2d=np.array(
                    self.transformer.grid.mask.derive_grid.all_false.in_radians.native
                ),
            )
        )

        w_matrix = inversion_util_secret.w_tilde_via_preload_from(
            w_tilde_preload=curvature_preload,
            native_index_for_slim_index=self.real_space_mask.derive_indexes.native_for_slim,
        )

        dirty_image = self.transformer.image_from(
            visibilities=self.data.real * self.noise_map.real**-2.0
            + 1j * self.data.imag * self.noise_map.imag**-2.0,
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
        return self.data.amplitudes

    @property
    def phases(self):
        return self.data.phases

    @property
    def uv_distances(self):
        return np.sqrt(
            np.square(self.uv_wavelengths[:, 0]) + np.square(self.uv_wavelengths[:, 1])
        )

    @property
    def dirty_image(self):
        return self.transformer.image_from(visibilities=self.data)

    @property
    def dirty_noise_map(self):
        return self.transformer.image_from(visibilities=self.noise_map)

    @property
    def dirty_signal_to_noise_map(self):
        return self.transformer.image_from(visibilities=self.signal_to_noise_map)

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

        return self.data.with_new_array(
            signal_to_noise_map_real + 1j * signal_to_noise_map_imag
        )

    @property
    def blurring_grid(self):
        return None

    @property
    def convolver(self):
        return None

    def output_to_fits(
        self,
        data_path=None,
        noise_map_path=None,
        uv_wavelengths_path=None,
        overwrite=False,
    ):
        if data_path is not None:
            self.data.output_to_fits(file_path=data_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)

        if self.uv_wavelengths is not None and uv_wavelengths_path is not None:
            array_2d_util.numpy_array_2d_to_fits(
                array_2d=self.uv_wavelengths,
                file_path=uv_wavelengths_path,
                overwrite=overwrite,
            )
