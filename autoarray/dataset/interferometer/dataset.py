import logging
import numpy as np
from pathlib import Path

from autoconf import cached_property
from autoconf.fitsable import ndarray_via_fits_from, output_to_fits

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.interferometer.w_tilde import WTildeInterferometer
from autoarray.dataset.grids import GridsDataset
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.inversion.interferometer import inversion_interferometer_util

logger = logging.getLogger(__name__)


class Interferometer(AbstractDataset):
    def __init__(
        self,
        data: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        uv_wavelengths: np.ndarray,
        real_space_mask: Mask2D,
        transformer_class=TransformerNUFFT,
        dft_preload_transform: bool = True,
        preprocessing_directory=None,
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

        The dataset also has a number of (y,x) grids of coordinates associated with it, which map to the centres
        of its image pixels. They are used for performing calculations which map directly to the data and have
        over sampling calculations built in which approximate the 2D line integral of these calculations within a
        pixel. This is explained in more detail in the `GridsDataset` class.

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
        transformer_class
            The class of the Fourier Transform which maps images from real space to Fourier space visibilities and
            the uv-plane.
        dft_preload_transform
            If True, precomputes and stores the cosine and sine terms for the Fourier transform.
            This accelerates repeated transforms but consumes additional memory (~1GB+ for large datasets).
        """
        self.real_space_mask = real_space_mask

        super().__init__(
            data=data,
            noise_map=noise_map,
            over_sample_size_lp=1,
            over_sample_size_pixelization=1,
        )

        self.uv_wavelengths = uv_wavelengths

        self.transformer = transformer_class(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=dft_preload_transform,
        )

        self.preprocessing_directory = (
            Path(preprocessing_directory)
            if preprocessing_directory is not None
            else None
        )

        self.grids = GridsDataset(
            mask=self.real_space_mask,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
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
        dft_preload_transform: bool = True,
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

        uv_wavelengths = ndarray_via_fits_from(
            file_path=uv_wavelengths_path, hdu=uv_wavelengths_hdu
        )

        return Interferometer(
            real_space_mask=real_space_mask,
            data=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            transformer_class=transformer_class,
            dft_preload_transform=dft_preload_transform,
        )

    def w_tilde_preprocessing(self):

        from astropy.io import fits

        if self.preprocessing_directory.is_dir():
            filename = "{}/curvature_preload.fits".format(self.preprocessing_directory)

            if not self.preprocessing_directory.isfile(filename):
                print("The file {} does not exist".format(filename))
                logger.info("INTERFEROMETER - Computing W-Tilde... May take a moment.")

                curvature_preload = inversion_interferometer_util.w_tilde_curvature_preload_interferometer_from(
                    noise_map_real=self.noise_map.real,
                    uv_wavelengths=self.uv_wavelengths,
                    shape_masked_pixels_2d=self.transformer.grid.mask.shape_native_masked_pixels,
                    grid_radians_2d=self.transformer.grid.mask.unmasked_grid_sub_1.in_radians.native,
                )

                fits.writeto(filename, data=curvature_preload)

    @property
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

        try:
            import numba
        except ModuleNotFoundError:
            raise exc.InversionException(
                "Inversion w-tilde functionality (pixelized reconstructions) is "
                "disabled if numba is not installed.\n\n"
                "This is because the run-times without numba are too slow.\n\n"
                "Please install numba, which is described at the following web page:\n\n"
                "https://pyautolens.readthedocs.io/en/latest/installation/overview.html"
            )

        curvature_preload = (
            inversion_interferometer_util.w_tilde_curvature_preload_interferometer_from(
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

        w_matrix = inversion_interferometer_util.w_tilde_via_preload_from(
            w_tilde_preload=curvature_preload,
            native_index_for_slim_index=np.array(
                self.real_space_mask.derive_indexes.native_for_slim
            ).astype("int"),
        )

        dirty_image = self.transformer.image_from(
            visibilities=self.data.real * self.noise_map.real**-2.0
            + 1j * self.data.imag * self.noise_map.imag**-2.0,
            use_adjoint_scaling=True,
        )

        return WTildeInterferometer(
            w_matrix=w_matrix,
            curvature_preload=curvature_preload,
            dirty_image=np.array(dirty_image.array),
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
            np.real(self.data.array), np.real(self.noise_map.array)
        )
        signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
        signal_to_noise_map_imag = np.divide(
            np.imag(self.data.array), np.imag(self.noise_map.array)
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
            output_to_fits(
                values=self.uv_wavelengths,
                file_path=uv_wavelengths_path,
                overwrite=overwrite,
            )

    @property
    def psf(self):
        return None
