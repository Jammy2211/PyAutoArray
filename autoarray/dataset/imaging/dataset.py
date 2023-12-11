import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

from autoconf import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.imaging.settings import SettingsImaging
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.arrays.kernel_2d import Kernel2D
from autoarray.mask.mask_2d import Mask2D
from autoarray import type as ty

from autoarray import exc
from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)


class Imaging(AbstractDataset):
    def __init__(
        self,
        data: Array2D,
        noise_map: Optional[Array2D] = None,
        psf: Optional[Kernel2D] = None,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        settings: SettingsImaging = SettingsImaging(),
        pad_for_convolver: bool = False,
        check_noise_map: bool = True,
    ):
        """
        A class containing an imaging dataset, including the image data, noise-map and a point spread function (PSF).

        Parameters
        ----------
        data
            The array of the image data, for example in units of electrons per second.
        noise_map
            An array describing the RMS standard deviation error in each pixel, for example in units of electrons per
            second.
        psf
            An array describing the Point Spread Function kernel of the image which accounts for diffraction due to the
            telescope optics via 2D convolution.
        settings
            Controls settings of how the dataset is set up (e.g. the types of grids used to perform calculations).
        check_noise_map
            If True, the noise-map is checked to ensure all values are above zero.
        """

        self.unmasked = None

        self.pad_for_convolver = pad_for_convolver

        if pad_for_convolver and psf is not None:
            try:
                data.mask.derive_mask.blurring_from(
                    kernel_shape_native=psf.shape_native
                )
            except exc.MaskException:
                data = data.padded_before_convolution_from(
                    kernel_shape=psf.shape_native, mask_pad_value=1
                )
                if noise_map is not None:
                    noise_map = noise_map.padded_before_convolution_from(
                        kernel_shape=psf.shape_native, mask_pad_value=1
                    )
                logger.info(
                    f"The image and noise map of the `Imaging` objected have been padded to the dimensions"
                    f"{data.shape}. This is because the blurring region around the mask (which defines where"
                    f"PSF flux may be convolved into the masked region) extended beyond the edge of the image."
                    f""
                    f"This can be prevented by using a smaller mask, smaller PSF kernel size or manually padding"
                    f"the image and noise-map yourself."
                )

        super().__init__(
            data=data,
            noise_map=noise_map,
            noise_covariance_matrix=noise_covariance_matrix,
            settings=settings,
        )

        if self.noise_map.native is not None and check_noise_map:
            if ((self.noise_map.native <= 0.0) * np.invert(self.noise_map.mask)).any():
                zero_entries = np.argwhere(self.noise_map.native <= 0.0)

                raise exc.DatasetException(
                    f"""
                    A value in the noise-map of the dataset is {np.min(self.noise_map)}. 

                    This is less than or equal to zero, and therefore an ill-defined value which must be corrected.
                    
                    The 2D indexes of the arrays in the native noise map array are {zero_entries}.
                    """
                )

        if psf is not None and settings.use_normalized_psf:
            psf = Kernel2D.no_mask(
                values=psf.native, pixel_scales=psf.pixel_scales, normalize=True
            )

        self.psf = psf

    @cached_property
    def blurring_grid(self) -> Grid2D:
        """
        Returns a blurring-grid from a mask and the 2D shape of the PSF kernel.

        A blurring grid consists of all pixels that are masked (and therefore have their values set to (0.0, 0.0)),
        but are close enough to the unmasked pixels that their values will be convolved into the unmasked those pixels.
        This when computing images from light profile objects.

        This uses lazy allocation such that the calculation is only performed when the blurring grid is used, ensuring
        efficient set up of the `Imaging` class.

        Returns
        -------
        The blurring grid given the mask of the imaging data.
        """

        return self.grid.blurring_grid_via_kernel_shape_from(
            kernel_shape_native=self.psf.shape_native
        )

    @cached_property
    def convolver(self):
        """
        Returns a `Convolver` from a mask and 2D PSF kernel.

        The `Convolver` stores in memory the array indexing between the mask and PSF, enabling efficient 2D PSF
        convolution of images and matrices used for linear algebra calculations (see `operators.convolver`).

        This uses lazy allocation such that the calculation is only performed when the convolver is used, ensuring
        efficient set up of the `Imaging` class.

        Returns
        -------
        Convolver
            The convolver given the masked imaging data's mask and PSF.
        """

        return Convolver(mask=self.mask, kernel=self.psf)

    @cached_property
    def w_tilde(self):
        """
        The w_tilde formalism of the linear algebra equations precomputes the convolution of every pair of masked
        noise-map values given the PSF (see `inversion.inversion_util`).

        The `WTilde` object stores these precomputed values in the imaging dataset ensuring they are only computed once
        per analysis.

        This uses lazy allocation such that the calculation is only performed when the wtilde matrices are used,
        ensuring efficient set up of the `Imaging` class.

        Returns
        -------
        WTildeImaging
            Precomputed values used for the w tilde formalism of linear algebra calculations.
        """

        logger.info("IMAGING - Computing W-Tilde... May take a moment.")

        (
            curvature_preload,
            indexes,
            lengths,
        ) = inversion_imaging_util.w_tilde_curvature_preload_imaging_from(
            noise_map_native=self.noise_map.native,
            kernel_native=self.psf.native,
            native_index_for_slim_index=self.mask.derive_indexes.native_for_slim,
        )

        return WTildeImaging(
            curvature_preload=curvature_preload,
            indexes=indexes.astype("int"),
            lengths=lengths.astype("int"),
            noise_map_value=self.noise_map[0],
        )

    @classmethod
    def from_fits(
        cls,
        pixel_scales: ty.PixelScales,
        data_path: Union[Path, str],
        noise_map_path: Union[Path, str],
        data_hdu: int = 0,
        noise_map_hdu: int = 0,
        psf_path: Optional[Union[Path, str]] = None,
        psf_hdu: int = 0,
        noise_covariance_matrix: Optional[np.ndarray] = None,
    ) -> "Imaging":
        """
        Load an imaging dataset from multiple .fits file.

        For each attribute of the imaging data (e.g. `data`, `noise_map`, `pre_cti_data`) the path to
        the .fits and the `hdu` containing the data can be specified.

        The `noise_map` assumes the noise value in each `data` value are independent, where these values are the
        the RMS standard deviation error in each pixel.

        A `noise_covariance_matrix` can be input instead, which represents the covariance between noise values in
        the data and can be used to fit the data accounting for correlations (the `noise_map` is the diagonal values
        of this matrix).

        If the dataset has a mask associated with it (e.g. in a `mask.fits` file) the file must be loaded separately
        via the `Mask2D` object and applied to the imaging after loading via fits using the `from_fits` method.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        data_path
            The path to the data .fits file containing the image data (e.g. '/path/to/image.fits').
        data_hdu
            The hdu the image data is contained in the .fits file specified by `data_path`.
        psf_path
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits').
        psf_hdu
            The hdu the psf is contained in the .fits file specified by `psf_path`.
        noise_map_path
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits').
        noise_map_hdu
            The hdu the noise map is contained in the .fits file specified by `noise_map_path`.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value.
        """

        data = Array2D.from_fits(
            file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
        )

        noise_map = Array2D.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )

        if psf_path is not None:
            psf = Kernel2D.from_fits(
                file_path=psf_path,
                hdu=psf_hdu,
                pixel_scales=pixel_scales,
                normalize=False,
            )

        else:
            psf = None

        return Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            noise_covariance_matrix=noise_covariance_matrix,
        )

    def apply_mask(self, mask: Mask2D) -> "Imaging":
        """
        Apply a mask to the imaging dataset, whereby the mask is applied to the image data, noise-map and other
        quantities one-by-one.

        The original unmasked imaging data is stored as the `self.unmasked` attribute. This is used to ensure that if
        the `apply_mask` function is called multiple times, every mask is always applied to the original unmasked
        imaging dataset.

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image.
        """
        if self.data.mask.is_all_false:
            unmasked_dataset = self
        else:
            unmasked_dataset = self.unmasked

        data = Array2D(values=unmasked_dataset.data.native, mask=mask.derive_mask.sub_1)

        noise_map = Array2D(
            values=unmasked_dataset.noise_map.native, mask=mask.derive_mask.sub_1
        )

        if unmasked_dataset.noise_covariance_matrix is not None:
            noise_covariance_matrix = unmasked_dataset.noise_covariance_matrix

            noise_covariance_matrix = np.delete(
                noise_covariance_matrix, mask.derive_indexes.masked_slim, 0
            )
            noise_covariance_matrix = np.delete(
                noise_covariance_matrix, mask.derive_indexes.masked_slim, 1
            )

        else:
            noise_covariance_matrix = None

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            noise_covariance_matrix=noise_covariance_matrix,
            settings=self.settings,
            pad_for_convolver=True,
        )

        dataset.unmasked = unmasked_dataset

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return dataset

    def apply_settings(self, settings: SettingsImaging) -> "Imaging":
        """
        Returns a new instance of the imaging with the input `SettingsImaging` applied to them.

        This can be used to update settings like the types of grids associated with the dataset that are used
        to perform calculations or putting a limit of the dataset's signal-to-noise.

        Parameters
        ----------
        settings
            The settings for the imaging data that control things like the grids used for calculations.
        """
        return Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            settings=settings,
            pad_for_convolver=self.pad_for_convolver,
        )

    def output_to_fits(
        self,
        data_path: Union[Path, str],
        psf_path: Optional[Union[Path, str]] = None,
        noise_map_path: Optional[Union[Path, str]] = None,
        overwrite: bool = False,
    ):
        """
        Output an imaging dataset to multiple .fits file.

        For each attribute of the imaging data (e.g. `data`, `noise_map`) the path to
        the .fits can be specified, with `hdu=0` assumed automatically.

        If the `data` has been masked, the masked data is output to .fits files. A mask can be separately output to
        a file `mask.fits` via the `Mask` objects `output_to_fits` method.

        Parameters
        ----------
        data_path
            The path to the data .fits file where the image data is output (e.g. '/path/to/data.fits').
        psf_path
            The path to the psf .fits file where the psf is output (e.g. '/path/to/psf.fits').
        noise_map_path
            The path to the noise_map .fits where the noise_map is output (e.g. '/path/to/noise_map.fits').
        overwrite
            If `True`, the .fits files are overwritten if they already exist, if `False` they are not and an
            exception is raised.
        """
        self.data.output_to_fits(file_path=data_path, overwrite=overwrite)

        if self.psf is not None and psf_path is not None:
            self.psf.output_to_fits(file_path=psf_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
