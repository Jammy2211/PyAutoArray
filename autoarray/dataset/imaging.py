import logging

import numpy as np
import copy

from autoconf import conf
from autoarray import exc
from autoarray.dataset import abstract_dataset, preprocess
from autoarray.mask import mask_2d as msk
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.structures import kernel_2d
from autoarray.operators import convolver

logger = logging.getLogger(__name__)


class AbstractImaging(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        image: array_2d.Array2D,
        noise_map: array_2d.Array2D,
        psf: kernel_2d.Kernel2D = None,
        positions: grid_2d_irregular.Grid2DIrregular = None,
        name: str = None,
    ):
        """A class containing the data, noise-map and point spread function of a 2D imaging dataset.

        Parameters
        ----------
        image : aa.Array2D
            The array of the image data, in units of electrons per second.
        noise_map : aa.Array2D
            An array describing the RMS standard deviation error in each pixel in units of electrons per second.
        psf : aa.Array2D
            An array describing the Point Spread Function kernel of the image.
        """

        super().__init__(
            data=image, noise_map=noise_map, positions=positions, name=name
        )

        self.psf = psf

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
            array=noise_map_limit,
            mask=self.image.mask,
            store_slim=self.noise_map.store_slim,
        )

        return imaging


class AbstractSettingsMaskedImaging(abstract_dataset.AbstractSettingsMaskedDataset):
    def __init__(
        self,
        grid_class=grid_2d.Grid2D,
        grid_inversion_class=grid_2d.Grid2D,
        sub_size=2,
        sub_size_inversion=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
        signal_to_noise_limit_radii=None,
        psf_shape_2d=None,
        renormalize_psf=True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
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
        sub_size : int
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
        psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
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

        self.psf_shape_2d = psf_shape_2d
        self.renormalize_psf = renormalize_psf

    @property
    def tag_no_inversion(self):
        return (
            f"{conf.instance['notation']['settings_tags']['imaging']['imaging']}"
            f"[{self.grid_tag_no_inversion}"
            f"{self.signal_to_noise_limit_tag}"
            f"{self.psf_shape_tag}]"
        )

    @property
    def tag_with_inversion(self):
        return (
            f"{conf.instance['notation']['settings_tags']['imaging']['imaging']}"
            f"[{self.grid_tag_with_inversion}"
            f"{self.signal_to_noise_limit_tag}"
            f"{self.psf_shape_tag}]"
        )

    def psf_reshaped_and_renormalized_from_psf(self, psf):

        if psf is not None:

            if self.psf_shape_2d is None:
                psf_shape_2d = psf.shape_native
            else:
                psf_shape_2d = self.psf_shape_2d

            return kernel_2d.Kernel2D.manual_native(
                array=psf.resized_from(new_shape=psf_shape_2d).native,
                pixel_scales=psf.pixel_scales,
                renormalize=self.renormalize_psf,
            )

    @property
    def psf_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder as follows:

        image_psf_shape = 1 -> settings
        image_psf_shape = 2 -> settings_image_psf_shape_2
        image_psf_shape = 2 -> settings_image_psf_shape_2
        """
        if self.psf_shape_2d is None:
            return ""
        y = str(self.psf_shape_2d[0])
        x = str(self.psf_shape_2d[1])
        return (
            "__"
            + conf.instance["notation"]["settings_tags"]["imaging"]["psf_shape"]
            + "_"
            + y
            + "x"
            + x
        )


class AbstractMaskedImaging(abstract_dataset.AbstractMaskedDataset):
    def __init__(self, imaging, mask, settings=AbstractSettingsMaskedImaging()):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask2D
            The 2D mask that is applied to the image.
        psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        pixel_scales_interp : float
            If `True`, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        super().__init__(dataset=imaging, mask=mask, settings=settings)

        self.image = array_2d.Array2D.manual_mask(
            array=self.dataset.image.native,
            mask=mask.mask_sub_1,
            store_slim=imaging.image.store_slim,
        )

        self.noise_map = array_2d.Array2D.manual_mask(
            array=self.dataset.noise_map.native,
            mask=mask.mask_sub_1,
            store_slim=imaging.noise_map.store_slim,
        )

        psf = copy.deepcopy(imaging.psf)

        self.psf = settings.psf_reshaped_and_renormalized_from_psf(psf=psf)

        if self.psf is not None:

            self.convolver = convolver.Convolver(mask=mask, kernel=self.psf)
            self.blurring_grid = self.grid.blurring_grid_from_kernel_shape(
                kernel_shape_native=self.psf.shape_native
            )

    @property
    def imaging(self):
        return self.dataset

    @property
    def data(self):
        return self.image

    @property
    def signal_to_noise_map(self):
        return self.data / self.noise_map

    def modify_image_and_noise_map(self, image, noise_map):

        masked_imaging = copy.deepcopy(self)

        masked_imaging.image = image
        masked_imaging.noise_map = noise_map

        return masked_imaging


class SettingsMaskedImaging(AbstractSettingsMaskedImaging):

    pass


class Imaging(AbstractImaging):
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
        positions_path=None,
        name=None,
    ):
        """Factory for loading the imaging data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the imaging-data.

        This factory also includes a number of routines for converting the imaging-data from unit_label not supported by PyAutoLens \
        (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        renormalize_psf
        noise_map_non_constant
        name
        image_path : str
            The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
        pixel_scales : float
            The size of each pixel in scaled units.
        image_hdu : int
            The hdu the image is contained in the .fits file specified by *image_path*.
        psf_path : str
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits')
        psf_hdu : int
            The hdu the psf is contained in the .fits file specified by *psf_path*.
        noise_map_path : str
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
        noise_map_hdu : int
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
                renormalize=True,
            )

        else:

            psf = None

        if positions_path is not None:

            if ".dat" in positions_path:

                positions = grid_2d_irregular.Grid2DIrregular.from_dat_file(
                    file_path=positions_path
                )
                positions_path = positions_path.replace(".dat", ".json")
                positions.output_to_json(file_path=positions_path, overwrite=True)

            positions = grid_2d_irregular.Grid2DIrregular.from_json(
                file_path=positions_path
            )

        else:

            positions = None

        return Imaging(
            image=image, noise_map=noise_map, psf=psf, positions=positions, name=name
        )

    def output_to_fits(
        self, image_path, psf_path=None, noise_map_path=None, overwrite=False
    ):
        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)

        if self.psf is not None and psf_path is not None:
            self.psf.output_to_fits(file_path=psf_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)


class MaskedImaging(AbstractMaskedImaging):

    pass


class AbstractSimulatorImaging:
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: kernel_2d.Kernel2D = None,
        renormalize_psf: bool = True,
        read_noise: float = None,
        add_poisson_noise: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        psf : Kernel2D
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of the simulated imaging.
        background_sky_level : float
            The level of the background sky of the simulated imaging.
        renormalize_psf : bool
            If `True`, the PSF kernel is renormalized so all values sum to 1.0.
        read_noise : float
            The level of read-noise added to the simulated imaging by drawing from a Gaussian distribution with
            sigma equal to the value `read_noise`.
        add_poisson_noise : bool
            Whether Poisson noise corresponding to photon count statistics on the imaging observation is added.
        noise_if_add_noise_false : float
            If noise is not added to the simulated dataset a `noise_map` must still be returned. This value gives
            the value of noise assigned to every pixel in the noise-map.
        noise_seed : int
            The random seed used to add random noise, where -1 corresponds to a random seed every run.
        """

        if psf is not None and renormalize_psf:
            psf = psf.renormalized

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
