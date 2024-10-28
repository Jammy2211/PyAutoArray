import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

from autoconf import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.grids import GridsDataset
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.dataset.over_sampling import OverSamplingDataset
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
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
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
        pad_for_convolver: bool = False,
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
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grids.pixelization).
        pad_for_convolver
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
            over_sampling=over_sampling,
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
                values=psf.native, pixel_scales=psf.pixel_scales, normalize=True
            )

        self.psf = psf

    @cached_property
    def grids(self):
        return GridsDataset(
            mask=self.data.mask, over_sampling=self.over_sampling, psf=self.psf
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
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
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
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grids.pixelization).
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
            over_sampling=over_sampling,
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

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            noise_covariance_matrix=noise_covariance_matrix,
            over_sampling=self.over_sampling,
            pad_for_convolver=True,
        )

        dataset.unmasked = unmasked_dataset

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return dataset

    def apply_noise_scaling(self, mask: Mask2D, noise_value: float = 1e8) -> "Imaging":
        """
        Apply a mask to the imaging dataset using noise scaling, whereby the mask increases noise-map values to be
        extremely large such that they are never included in the likelihood calculation, but it does
        not remove the image data values, which are set to zero.

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
        """
        data = np.where(np.invert(mask), 0.0, self.data.native)
        data = Array2D.no_mask(
            values=data,
            shape_native=self.data.shape_native,
            pixel_scales=self.data.pixel_scales,
        )

        noise_map = self.noise_map.native
        noise_map[mask == False] = noise_value
        noise_map = Array2D.no_mask(
            values=noise_map,
            shape_native=self.data.shape_native,
            pixel_scales=self.data.pixel_scales,
        )

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sampling=self.over_sampling,
            pad_for_convolver=False,
        )

        logger.info(
            f"IMAGING - Data noise scaling applied, a total of {mask.pixels_in_mask} pixels were scaled to large noise values."
        )

        return dataset

    def apply_over_sampling(
        self,
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
    ) -> "AbstractDataset":
        """
        Apply new over sampling objects to the grid and grid pixelization of the dataset.

        This method is used to change the over sampling of the grid and grid pixelization, for example when the
        user wishes to perform over sampling with a higher sub grid size or with an iterative over sampling strategy.

        The `grid` and grids.pixelization` are cached properties which after use are stored in memory for efficiency.
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
            This class controls over sampling for all the different grids (e.g. `grid`, `grids.pixelization).
        """

        uniform = over_sampling.uniform or self.over_sampling.uniform
        non_uniform = over_sampling.non_uniform or self.over_sampling.non_uniform
        pixelization = over_sampling.pixelization or self.over_sampling.pixelization

        over_sampling = OverSamplingDataset(
            uniform=uniform,
            non_uniform=non_uniform,
            pixelization=pixelization,
        )

        return Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            over_sampling=over_sampling,
            pad_for_convolver=False,
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
