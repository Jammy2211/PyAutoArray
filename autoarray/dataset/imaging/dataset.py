import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

from autoconf import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.grids import GridsDataset
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.arrays.kernel_2d import Kernel2D
from autoarray.mask.mask_2d import Mask2D
from autoarray import type as ty

from autoarray import exc
from autoarray.operators.over_sampling import over_sample_util
from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)


class Imaging(AbstractDataset):
    def __init__(
        self,
        data: Array2D,
        noise_map: Optional[Array2D] = None,
        psf: Optional[Kernel2D] = None,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        over_sample_size_lp: Union[int, Array2D] = 4,
        over_sample_size_pixelization: Union[int, Array2D] = 4,
        pad_for_psf: bool = False,
        use_normalized_psf: Optional[bool] = True,
        check_noise_map: bool = True,
    ):
        """
        An imaging dataset, containing the image data, noise-map, PSF and associated quantities
        for calculations like the grid.

        This object is the input to the `FitImaging` object, which fits the dataset with a model image and quantifies
        the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the imaging data are available and used for the following tasks:

        - `data`: The image data, which shows the signal that is analysed and fitted with a model image.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        - `psf`: The Point Spread Function of the data, used to perform 2D convolution on images to produce a model
        image which is compared to the data.

        The dataset also has a number of (y,x) grids of coordinates associated with it, which map to the centres
        of its image pixels. They are used for performing calculations which map directly to the data and have
        over sampling calculations built in which approximate the 2D line integral of these calculations within a
        pixel. This is explained in more detail in the `GridsDataset` class.

        Parameters
        ----------
        data
            The array of the image data containing the signal that is fitted (in PyAutoGalaxy and PyAutoLens the
            recommended units are electrons per second).
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit (in PyAutoGalaxy and PyAutoLens the recommended units are electrons per second).
        psf
            The Point Spread Function kernel of the image which accounts for diffraction due to the telescope optics
            via 2D convolution.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        pad_for_psf
            The PSF convolution may extend beyond the edges of the image mask, which can lead to edge effects in the
            convolved image. If `True`, the image and noise-map are padded to ensure the PSF convolution does not
            extend beyond the edge of the image.
        use_normalized_psf
            If `True`, the PSF kernel values are rescaled such that they sum to 1.0. This can be important for ensuring
            the PSF kernel does not change the overall normalization of the image when it is convolved with it.
        check_noise_map
            If True, the noise-map is checked to ensure all values are above zero.
        """

        self.unmasked = None

        self.pad_for_psf = pad_for_psf

        if pad_for_psf and psf is not None:
            try:
                data.mask.derive_mask.blurring_from(
                    kernel_shape_native=psf.shape_native
                )
            except exc.MaskException:
                over_sample_size_lp = (
                    over_sample_util.over_sample_size_convert_to_array_2d_from(
                        over_sample_size=over_sample_size_lp, mask=data.mask
                    )
                )
                over_sample_size_lp = (
                    over_sample_size_lp.padded_before_convolution_from(
                        kernel_shape=psf.shape_native, mask_pad_value=1
                    )
                )

                over_sample_size_pixelization = (
                    over_sample_util.over_sample_size_convert_to_array_2d_from(
                        over_sample_size=over_sample_size_pixelization, mask=data.mask
                    )
                )
                over_sample_size_pixelization = (
                    over_sample_size_pixelization.padded_before_convolution_from(
                        kernel_shape=psf.shape_native, mask_pad_value=1
                    )
                )

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
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

        self.use_normalized_psf = use_normalized_psf

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

        if psf is not None and use_normalized_psf:
            psf = Kernel2D.no_mask(
                values=psf.native._array, pixel_scales=psf.pixel_scales, normalize=True
            )

        self.psf = psf

    @cached_property
    def grids(self):
        return GridsDataset(
            mask=self.data.mask,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            psf=self.psf,
        )

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
            noise_map_native=np.array(self.noise_map.native),
            kernel_native=np.array(self.psf.native),
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
        check_noise_map: bool = True,
        over_sample_size_lp: Union[int, Array2D] = 4,
        over_sample_size_pixelization: Union[int, Array2D] = 4,
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
        check_noise_map
            If True, the noise-map is checked to ensure all values are above zero.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
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
            check_noise_map=check_noise_map,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
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

        data = Array2D(values=unmasked_dataset.data.native, mask=mask)

        noise_map = Array2D(values=unmasked_dataset.noise_map.native, mask=mask)

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

        over_sample_size_lp = Array2D(values=self.over_sample_size_lp.native, mask=mask)
        over_sample_size_pixelization = Array2D(
            values=self.over_sample_size_pixelization.native, mask=mask
        )

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            noise_covariance_matrix=noise_covariance_matrix,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
            pad_for_psf=True,
        )

        dataset.unmasked = unmasked_dataset

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return dataset

    def apply_noise_scaling(
        self,
        mask: Mask2D,
        noise_value: float = 1e8,
        signal_to_noise_value: Optional[float] = None,
        should_zero_data: bool = True,
    ) -> "Imaging":
        """
        Apply a mask to the imaging dataset using noise scaling, whereby the maskmay zero the data and increase
        noise-map values to change how they enter the likelihood calculation.

        Given this data region is masked, it is likely thr data itself should not be included and therefore
        the masked data values are set to zero. This can be disabled by setting `should_zero_data=False`.

        Two forms of scaling are supported depending on whether the `signal_to_noise_value` is input:

        - `noise_value`: The noise-map values in the masked region are set to this value, typically a very large value,
        such that they are never included in the likelihood calculation.

        - `signal_to_noise_value`: The noise-map values in the masked region are set to values such that they give
        this signal-to-noise ratio. This overwrites the `noise_value` parameter.

        For certain modeling tasks, the mask defines regions of the data that are used to calculate the likelihood.
        For example, all data points in a mask may be used to create a pixel-grid, which is used in the likelihood.
        When data points are moved via `apply_mask`, they would be omitted from this grid entirely, which would
        lead to an incorrect likelihood calculation. Noise scaling retains these data points in the likelihood
        calculation, but ensures they do not contribute to the fit.

        This function can only be applied before actual masking.

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image and noise-map, to scale the noise-map values to large values.
        noise_value
            The value that the noise-map values are set to in the masked region where noise scaling is applied.
        signal_to_noise_value
            The noise-map values are instead set to values such that they give this signal-to-noise_maps ratio.
            This overwrites the noise_value parameter.
        should_zero_data
            If True, the data values in the masked region are set to zero.
        """

        if signal_to_noise_value is None:
            noise_map = self.noise_map.native
            noise_map[mask == False] = noise_value
        else:
            noise_map = np.where(
                mask == False,
                np.median(self.data.native[mask.derive_mask.edge == False])
                / signal_to_noise_value,
                self.noise_map.native,
            )

        if should_zero_data:
            data = np.where(np.invert(mask), 0.0, self.data.native)
        else:
            data = self.data.native

        data_unmasked = Array2D.no_mask(
            values=data,
            shape_native=self.data.shape_native,
            pixel_scales=self.data.pixel_scales,
        )

        noise_map_unmasked = Array2D.no_mask(
            values=noise_map,
            shape_native=self.noise_map.shape_native,
            pixel_scales=self.noise_map.pixel_scales,
        )

        data = Array2D(values=data, mask=self.data.mask)

        noise_map = Array2D(values=noise_map, mask=self.data.mask)

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            pad_for_psf=False,
            check_noise_map=False,
        )

        if self.unmasked is not None:
            dataset.unmasked = self.unmasked
            dataset.unmasked.data = data_unmasked
            dataset.unmasked.noise_map = noise_map_unmasked

        logger.info(
            f"IMAGING - Data noise scaling applied, a total of {mask.pixels_in_mask} pixels were scaled to large noise values."
        )

        return dataset

    def apply_over_sampling(
        self,
        over_sample_size_lp: Union[int, Array2D] = None,
        over_sample_size_pixelization: Union[int, Array2D] = None,
    ) -> "AbstractDataset":
        """
        Apply new over sampling objects to the grid and grid pixelization of the dataset.

        This method is used to change the over sampling of the grid and grid pixelization, for example when the
        user wishes to perform over sampling with a higher sub grid size or with an iterative over sampling strategy.

        The `grid` and grids.pixelization` are cached properties which after use are stored in memory for efficiency.
        This function resets the cached properties so that the new over sampling is used in the grid and grid
        pixelization.

        Parameters
        ----------
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        """

        return Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            over_sample_size_lp=over_sample_size_lp or self.over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization
            or self.over_sample_size_pixelization,
            pad_for_psf=False,
            check_noise_map=False,
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
