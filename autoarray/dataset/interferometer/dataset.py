import logging
import numpy as np
from typing import Optional

from autoconf import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.interferometer.w_tilde import WTildeInterferometer
from autoarray.dataset.grids import GridsDataset
from autoarray.dataset.over_sampling import OverSamplingDataset
from autoarray.operators.transformer import TransformerNUFFT
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
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
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

        The `over_sampling` and `over_sampling_pixelization` define how over sampling is performed for these grids.

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
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grid_pixelization).
        transformer_class
            The class of the Fourier Transform which maps images from real space to Fourier space visibilities and
            the uv-plane.
        """
        self.real_space_mask = real_space_mask

        super().__init__(
            data=data,
            noise_map=noise_map,
            over_sampling=over_sampling,
        )

        self.uv_wavelengths = uv_wavelengths

        self.transformer = transformer_class(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

    @cached_property
    def grids(self):
        return GridsDataset(mask=self.real_space_mask, over_sampling=self.over_sampling)

    def apply_over_sampling(
        self,
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
    ) -> "Interferometer":
        """
        Apply new over sampling objects to the grid and grid pixelization of the dataset.

        This method is used to change the over sampling of the grid and grid pixelization, for example when the
        user wishes to perform over sampling with a higher sub grid size or with an iterative over sampling strategy.

        The `grid` and grid_pixelization` are cached properties which after use are stored in memory for efficiency.
        This function resets the cached properties so that the new over sampling is used in the grid and grid
        pixelization.

        The `default_galaxy_mode` parameter is used to set up default over sampling for galaxy light profiles in
        the project PyAutoGalaxy. This sets up the over sampling such that there is high over sampling in the centre
        of the mask, where the galaxy is located, and lower over sampling in the outer regions of the mask. It
        does this based on the pixel scale, which gives a good estimate of how large the central region
        requiring over sampling is.

        Parameters
        ----------
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grid_pixelization).
        """

        uniform = over_sampling.uniform or self.over_sampling.uniform
        non_uniform = over_sampling.non_uniform or self.over_sampling.non_uniform
        pixelization = over_sampling.pixelization or self.over_sampling.pixelization

        over_sampling = OverSamplingDataset(
            uniform=uniform,
            non_uniform=non_uniform,
            pixelization=pixelization,
        )

        return Interferometer(
            data=self.data,
            noise_map=self.noise_map,
            uv_wavelengths=self.uv_wavelengths,
            real_space_mask=self.real_space_mask,
            transformer_class=self.transformer.__class__,
            over_sampling=over_sampling,
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
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
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
            over_sampling=over_sampling,
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

    @property
    def convolver(self):
        return None
