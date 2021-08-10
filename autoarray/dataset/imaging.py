import logging

import numpy as np
import copy

from autoarray import exc
from autoarray.dataset import abstract_dataset, preprocess
from autoarray.inversion.inversion import inversion_util
from autoarray.mask import mask_2d as msk
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures import kernel_2d
from autoarray.operators import convolver

logger = logging.getLogger(__name__)


class WTilde:
    def __init__(self, curvature_preload, indexes, lengths):

        self.curvature_preload = curvature_preload
        self.indexes = indexes
        self.lengths = lengths


class SettingsImaging(abstract_dataset.AbstractSettingsDataset):
    def __init__(
        self,
        grid_class=grid_2d.Grid2D,
        grid_inversion_class=grid_2d.Grid2D,
        sub_size=1,
        sub_size_inversion=4,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
        signal_to_noise_limit_radii=None,
        use_normalized_psf=True,
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
            `Grid2DIterate` and `Grid2DInterpolate` (see the `Grid2D` documentation for a description of these options).
        grid_inversion_class : ag.Grid2D
            The type of grid used to create the grid that maps the `Inversion` source pixels to the data's image-pixels.
            The options are `Grid2D`, `Grid2DIterate` and `Grid2DInterpolate` 
            (see the `Grid2D` documentation for a description of these options).
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
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
            signal_to_noise_limit=signal_to_noise_limit,
            signal_to_noise_limit_radii=signal_to_noise_limit_radii,
        )

        self.use_normalized_psf = use_normalized_psf


class Imaging(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        image: array_2d.Array2D,
        noise_map: array_2d.Array2D,
        psf: kernel_2d.Kernel2D = None,
        settings=SettingsImaging(),
        name: str = None,
        pad_for_convolver=False,
    ):
        """
        A class containing the data, noise-map and point spread function of a 2D imaging dataset.

        Parameters
        ----------
        image : aa.Array2D
            The array of the image data, in units of electrons per second.
        noise_map : Array2D
            An array describing the RMS standard deviation error in each pixel in units of electrons per second.
        psf : aa.Array2D
            An array describing the Point Spread Function kernel of the image.
        mask: msk.Mask2D
            The 2D mask that is applied to the image.
        """

        self.unmasked = None

        self.pad_for_convolver = pad_for_convolver

        if pad_for_convolver and psf is not None:

            try:
                image.mask.blurring_mask_from_kernel_shape(
                    kernel_shape_native=psf.shape_native
                )
            except exc.MaskException:
                image = image.padded_before_convolution_from(
                    kernel_shape=psf.shape_native, mask_pad_value=1
                )
                noise_map = noise_map.padded_before_convolution_from(
                    kernel_shape=psf.shape_native, mask_pad_value=1
                )
                print(
                    f"The image and noise map of the `Imaging` objected had been padded to the dimensions"
                    f"{image.shape}. This is because the blurring region around its mask, which defines where"
                    f"PSF flux may be convolved into the masked region, extended beyond the edge of the image."
                    f""
                    f"This can be prevented by using a smaller mask, smaller PSF kernel size or manually padding"
                    f"the image and noise-map yourself."
                )

        super().__init__(data=image, noise_map=noise_map, settings=settings, name=name)

        self.psf_unormalized = psf

        if psf is not None:

            self.psf_normalized = kernel_2d.Kernel2D.manual_native(
                array=psf.native, pixel_scales=psf.pixel_scales, normalize=True
            )

        self._convolver = None
        self._blurring_grid = None
        self._w_tilde = None

    def __array_finalize__(self, obj):
        if isinstance(obj, Imaging):
            try:
                for key, value in obj.__dict__.items():
                    setattr(self, key, value)
            except AttributeError:
                logger.debug(
                    "Original object in Imaging.__array_finalize__ missing one or more attributes"
                )

    @property
    def psf(self):
        if self.settings.use_normalized_psf:
            return self.psf_normalized
        return self.psf_unormalized

    @property
    def blurring_grid(self):
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

        if self._blurring_grid is None:

            self._blurring_grid = self.grid.blurring_grid_from_kernel_shape(
                kernel_shape_native=self.psf.shape_native
            )

        return self._blurring_grid

    @property
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
        if self._convolver is None:

            self._convolver = convolver.Convolver(mask=self.mask, kernel=self.psf)

        return self._convolver

    @property
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
        WTilde
            Precomputed values used for the w tilde formalism of linear algebra calculations.
        """

        if self._w_tilde is None:

            preload, indexes, lengths = inversion_util.w_tilde_curvature_preload_imaging_from(
                noise_map_native=self.noise_map.native,
                kernel_native=self.psf.native,
                native_index_for_slim_index=self.mask._native_index_for_slim_index,
            )

            self._w_tilde = WTilde(
                curvature_preload=preload,
                indexes=indexes.astype("int"),
                lengths=lengths.astype("int"),
            )

        return self._w_tilde

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
        pixel_scales : float
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

        image = array_2d.Array2D.from_fits(
            file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
        )

        noise_map = array_2d.Array2D.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )

        if psf_path is not None:

            psf = kernel_2d.Kernel2D.from_fits(
                file_path=psf_path,
                hdu=psf_hdu,
                pixel_scales=pixel_scales,
                normalize=False,
            )

        else:

            psf = None

        return Imaging(image=image, noise_map=noise_map, psf=psf, name=name)

    def apply_mask(self, mask):

        if self.image.mask.is_all_false:
            unmasked_imaging = self
        else:
            unmasked_imaging = self.unmasked

        image = array_2d.Array2D.manual_mask(
            array=unmasked_imaging.image.native, mask=mask.mask_sub_1
        )

        noise_map = array_2d.Array2D.manual_mask(
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

        return imaging

    def apply_settings(self, settings):

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
            mask = msk.Mask2D.unmasked(
                shape_native=self.shape_native, pixel_scales=self.pixel_scales
            )

        noise_map_limit = np.where(
            (self.signal_to_noise_map.native > signal_to_noise_limit) & (mask == False),
            np.abs(self.image.native) / signal_to_noise_limit,
            self.noise_map.native,
        )

        imaging.noise_map = array_2d.Array2D.manual_mask(
            array=noise_map_limit, mask=self.image.mask
        )

        return imaging

    def modify_image_and_noise_map(self, image, noise_map):

        imaging = copy.deepcopy(self)

        imaging.data = image
        imaging.noise_map = noise_map

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
        psf: kernel_2d.Kernel2D = None,
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
        exposure_time : float
            The exposure time of the simulated imaging.
        background_sky_level : float
            The level of the background sky of the simulated imaging.
        normalize_psf : bool
            If `True`, the PSF kernel is normalized so all values sum to 1.0.
        read_noise : float
            The level of read-noise added to the simulated imaging by drawing from a Gaussian distribution with
            sigma equal to the value `read_noise`.
        add_poisson_noise : bool
            Whether Poisson noise corresponding to photon count statistics on the imaging observation is added.
        noise_if_add_noise_false : float
            If noise is not added to the simulated dataset a `noise_map` must still be returned. This value gives
            the value of noise assigned to every pixel in the noise-map.
        noise_seed
            The random seed used to add random noise, where -1 corresponds to a random seed every run.
        """

        if psf is not None and normalize_psf:
            psf = psf.normalized

        self.psf = psf

        self.exposure_time = exposure_time
        self.background_sky_level = background_sky_level

        self.read_noise = read_noise
        self.add_poisson_noise = add_poisson_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed


class SimulatorImaging(AbstractSimulatorImaging):
    def from_image(self, image: array_2d.Array2D, name: str = None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image : array_2d.Array2D
            The image before simulating which has noise added, PSF convolution, etc performed to it.
        """

        exposure_time_map = array_2d.Array2D.full(
            fill_value=self.exposure_time,
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
        )

        background_sky_map = array_2d.Array2D.full(
            fill_value=self.background_sky_level,
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
        )

        if self.psf is not None:
            psf = self.psf
        else:
            psf = kernel_2d.Kernel2D.no_blur(pixel_scales=image.pixel_scales)

        image = psf.convolved_array_from_array(array=image)

        image = image + background_sky_map

        if self.add_poisson_noise is True:
            image = preprocess.data_eps_with_poisson_noise_added(
                data_eps=image,
                exposure_time_map=exposure_time_map,
                seed=self.noise_seed,
            )

            noise_map = preprocess.noise_map_from_data_eps_and_exposure_time_map(
                data_eps=image, exposure_time_map=exposure_time_map
            )

        else:
            noise_map = array_2d.Array2D.full(
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

        mask = msk.Mask2D.unmasked(
            shape_native=image.shape_native, pixel_scales=image.pixel_scales
        )

        image = array_2d.Array2D.manual_mask(array=image, mask=mask)

        return Imaging(image=image, psf=self.psf, noise_map=noise_map, name=name)
