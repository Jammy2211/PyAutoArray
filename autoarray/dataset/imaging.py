import copy
import logging
import numpy as np
from typing import List, Optional

from autoconf import cached_property

from autoarray.dataset.abstract_dataset import AbstractWTilde
from autoarray.dataset.abstract_dataset import AbstractSettingsDataset
from autoarray.dataset.abstract_dataset import AbstractDataset
from autoarray.structures.two_d.array_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.structures.two_d.grids.grid_2d import Grid2D
from autoarray.structures.kernel_2d import Kernel2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.inversion.linear_eqn import leq_util
from autoarray.dataset import preprocess

logger = logging.getLogger(__name__)


class WTildeImaging(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        indexes: np.ndim,
        lengths: np.ndarray,
        noise_map_value: float,
    ):
        """
        Packages together all derived data quantities necessary to fit `Imaging` data using an ` Inversion` via the
        w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing the
        blurring operations performed using the imaging's PSF.

        Parameters
        ----------
        curvature_preload
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        indexes
            The image-pixel indexes of the curvature preload matrix, which are used to compute the curvature matrix
            efficiently when performing an inversion.
        lengths
            The lengths of how many indexes each curvature preload contains, again used to compute the curvature
            matrix efficienctly.
        noise_map_value
            The first value of the noise-map used to construct the curvature preload, which is used as a sanity
            check when performing the inversion to ensure the preload corresponds to the data being fitted.
        """
        super().__init__(
            curvature_preload=curvature_preload, noise_map_value=noise_map_value
        )

        self.indexes = indexes
        self.lengths = lengths


class SettingsImaging(AbstractSettingsDataset):
    def __init__(
        self,
        grid_class=Grid2D,
        grid_inversion_class=Grid2D,
        sub_size: int = 1,
        sub_size_inversion=4,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: List[int] = None,
        signal_to_noise_limit: Optional[float] = None,
        signal_to_noise_limit_radii: Optional[float] = None,
        use_normalized_psf: Optional[bool] = True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each
        for lens calculations.

        Parameters
        ----------
        grid_class : ag.Grid2D
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D`,
            and `Grid2DIterate` (see the `Grid2D` documentation for a description of these options).
        grid_inversion_class : ag.Grid2D
            The type of grid used to create the grid that maps the `LEq` source pixels to the data's image-pixels.
            The options are `Grid2D` and `Grid2DIterate`.
            (see the `Grid2D` documentation for a description of these options).
        sub_size
            If the grid and / or grid_inversion use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the fractional accuracy it
            uses when evaluating functions, where the fraction accuracy is the ratio of the values computed using
            two grids at a higher and lower sub-grid size.
        relative_accuracy
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the relative accuracy it
            uses when evaluating functions, where the relative accuracy is the absolute difference of the values
            computed using two grids at a higher and lower sub-grid size.
        sub_steps : [int]
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        signal_to_noise_limit
            If input, the dataset's noise-map is rescaled such that no pixel has a signal-to-noise above the
            signa to noise limit.
        psf_shape_2d
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        """

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            sub_size_inversion=sub_size_inversion,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            signal_to_noise_limit_radii=signal_to_noise_limit_radii,
        )

        self.use_normalized_psf = use_normalized_psf


class Imaging(AbstractDataset):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        psf: Kernel2D = None,
        settings=SettingsImaging(),
        name: str = None,
        pad_for_convolver=False,
    ):
        """
        A class containing the data, noise-map and point spread function of a 2D imaging dataset.

        Parameters
        ----------
        image
            The array of the image data, in units of electrons per second.
        noise_map : Array2D
            An array describing the RMS standard deviation error in each pixel in units of electrons per second.
        psf
            An array describing the Point Spread Function kernel of the image.
        settings
            Controls settings of how the dataset is set up (e.g. the types of grids used to perform calculations).
        """

        self.unmasked = None

        self.pad_for_convolver = pad_for_convolver

        if pad_for_convolver and psf is not None:

            try:
                image.mask.blurring_mask_from(kernel_shape_native=psf.shape_native)
            except exc.MaskException:
                image = image.padded_before_convolution_from(
                    kernel_shape=psf.shape_native, mask_pad_value=1
                )
                noise_map = noise_map.padded_before_convolution_from(
                    kernel_shape=psf.shape_native, mask_pad_value=1
                )
                logger.info(
                    f"The image and noise map of the `Imaging` objected have been padded to the dimensions"
                    f"{image.shape}. This is because the blurring region around the mask (which defines where"
                    f"PSF flux may be convolved into the masked region) extended beyond the edge of the image."
                    f""
                    f"This can be prevented by using a smaller mask, smaller PSF kernel size or manually padding"
                    f"the image and noise-map yourself."
                )

        super().__init__(data=image, noise_map=noise_map, settings=settings, name=name)

        self.psf_unormalized = psf

        if psf is not None:

            self.psf_normalized = Kernel2D.manual_native(
                array=psf.native, pixel_scales=psf.pixel_scales, normalize=True
            )

    @property
    def psf(self):
        if self.settings.use_normalized_psf:
            return self.psf_normalized
        return self.psf_unormalized

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
        np.ndarray
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
        noise-map values given the PSF (see `inversion.linear_eqn_util`).

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

        curvature_preload, indexes, lengths = leq_util.w_tilde_curvature_preload_imaging_from(
            noise_map_native=self.noise_map.native,
            kernel_native=self.psf.native,
            native_index_for_slim_index=self.mask.native_index_for_slim_index,
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
        image_path,
        pixel_scales,
        noise_map_path,
        image_hdu=0,
        noise_map_hdu=0,
        psf_path=None,
        psf_hdu=0,
        name=None,
    ):
        """
        Factory for loading the imaging data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the imaging-data.

        This factory also includes a number of routines for converting the imaging-data from unit_label not
        supported by PyAutoLens (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        noise_map_non_constant
        name
        image_path : str
            The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
        pixel_scales
            The size of each pixel in scaled units.
        image_hdu
            The hdu the image is contained in the .fits file specified by *image_path*.
        psf_path : str
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits')
        psf_hdu
            The hdu the psf is contained in the .fits file specified by *psf_path*.
        noise_map_path : str
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
        noise_map_hdu
            The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
        """

        image = Array2D.from_fits(
            file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
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

        return Imaging(image=image, noise_map=noise_map, psf=psf, name=name)

    def apply_mask(self, mask: Mask2D) -> "Imaging":
        """
        Apply a mask to the imaging dataset, whereby the mask is applied to the image data and noise-map one-by-one.

        The original unmasked imaging data is stored as the `self.unmasked` attribute. This is used to ensure that if
        the `apply_mask` function is called multiple times, every mask is always applied to the original unmasked
        imaging dataset.

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image.
        """
        if self.image.mask.is_all_false:
            unmasked_imaging = self
        else:
            unmasked_imaging = self.unmasked

        image = Array2D.manual_mask(
            array=unmasked_imaging.image.native, mask=mask.mask_sub_1
        )

        noise_map = Array2D.manual_mask(
            array=unmasked_imaging.noise_map.native, mask=mask.mask_sub_1
        )

        imaging = Imaging(
            image=image,
            noise_map=noise_map,
            psf=self.psf_unormalized,
            settings=self.settings,
            name=self.name,
            pad_for_convolver=True,
        )

        imaging.unmasked = unmasked_imaging

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return imaging

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
            image=self.image,
            noise_map=self.noise_map,
            psf=self.psf_unormalized,
            settings=settings,
            name=self.name,
            pad_for_convolver=self.pad_for_convolver,
        )

    @property
    def shape_native(self):
        return self.data.shape_native

    @property
    def image(self):
        return self.data

    @property
    def pixel_scales(self):
        return self.data.pixel_scales

    def signal_to_noise_limited_from(self, signal_to_noise_limit, mask=None):

        imaging = copy.deepcopy(self)

        if mask is None:
            mask = Mask2D.unmasked(
                shape_native=self.shape_native, pixel_scales=self.pixel_scales
            )

        noise_map_limit = np.where(
            (self.signal_to_noise_map.native > signal_to_noise_limit) & (mask == False),
            np.abs(self.image.native) / signal_to_noise_limit,
            self.noise_map.native,
        )

        imaging.noise_map = Array2D.manual_mask(
            array=noise_map_limit, mask=self.image.mask
        )

        return imaging

    def output_to_fits(
        self, image_path, psf_path=None, noise_map_path=None, overwrite=False
    ):
        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)

        if self.psf is not None and psf_path is not None:
            self.psf.output_to_fits(file_path=psf_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)


class AbstractSimulatorImaging:
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: Kernel2D = None,
        normalize_psf: bool = True,
        read_noise: float = None,
        add_poisson_noise: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """
        A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        psf : Kernel2D
            An arrays describing the PSF kernel of the image.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        normalize_psf : bool
            If `True`, the PSF kernel is normalized so all values sum to 1.0.
        read_noise
            The level of read-noise added to the simulated imaging by drawing from a Gaussian distribution with
            sigma equal to the value `read_noise`.
        add_poisson_noise : bool
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

        self.read_noise = read_noise
        self.add_poisson_noise = add_poisson_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed


class SimulatorImaging(AbstractSimulatorImaging):
    def via_image_from(self, image: Array2D, name: str = None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image : Array2D
            The image before simulating which has noise added, PSF convolution, etc performed to it.
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

        mask = Mask2D.unmasked(
            shape_native=image.shape_native, pixel_scales=image.pixel_scales
        )

        image = Array2D.manual_mask(array=image, mask=mask)

        return Imaging(image=image, psf=self.psf, noise_map=noise_map, name=name)
