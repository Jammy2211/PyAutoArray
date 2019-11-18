import os

from autoarray.util import array_util
from autoarray.structures import grids, kernel
from autoarray.dataset import imaging, interferometer
from autoarray.operators import transformer


class ImagingSimulator(object):
    def __init__(
        self,
        shape_2d,
        pixel_scales,
        sub_size,
        psf,
        exposure_time,
        background_level,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        origin=(0.0, 0.0),
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of an observation using this data_type.
        background_level : float
            The level of the background sky of an observationg using this data_type.
        """

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        self.shape_2d = shape_2d
        self.pixel_scales = pixel_scales
        self.sub_size = sub_size
        self.origin = origin
        self.psf = psf
        self.exposure_time = exposure_time
        self.background_level = background_level
        self.add_noise = add_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    @classmethod
    def lsst(
        cls,
        shape=(101, 101),
        pixel_scales=0.2,
        sub_size=8,
        psf_shape_2d=(31, 31),
        psf_sigma=0.5,
        exposure_time=100.0,
        background_level=1.0,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """Default settings for an observation with the Large Synotpic Survey Telescope.

        This can be customized by over-riding the default input values."""
        psf = kernel.Kernel.from_gaussian(
            shape_2d=psf_shape_2d, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return cls(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            psf=psf,
            exposure_time=exposure_time,
            background_level=background_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def euclid(
        cls,
        shape=(151, 151),
        pixel_scales=0.1,
        sub_size=8,
        psf_shape_2d=(31, 31),
        psf_sigma=0.1,
        exposure_time=565.0,
        background_level=1.0,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """Default settings for an observation with the Euclid space satellite.

        This can be customized by over-riding the default input values."""
        psf = kernel.Kernel.from_gaussian(
            shape_2d=psf_shape_2d, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return cls(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            psf=psf,
            exposure_time=exposure_time,
            background_level=background_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def hst(
        cls,
        shape=(251, 251),
        pixel_scales=0.05,
        sub_size=8,
        psf_shape_2d=(31, 31),
        psf_sigma=0.05,
        exposure_time=2000.0,
        background_level=1.0,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """Default settings for an observation with the Hubble Space Telescope.

        This can be customized by over-riding the default input values."""
        psf = kernel.Kernel.from_gaussian(
            shape_2d=psf_shape_2d, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return cls(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            psf=psf,
            exposure_time=exposure_time,
            background_level=background_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def hst_up_sampled(
        cls,
        shape=(401, 401),
        pixel_scales=0.03,
        sub_size=8,
        psf_shape_2d=(31, 31),
        psf_sigma=0.05,
        exposure_time=2000.0,
        background_level=1.0,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """Default settings for an observation with the Hubble Space Telescope which has been upscaled to a higher \
        pixel-scale to better sample the PSF.

        This can be customized by over-riding the default input values."""
        psf = kernel.Kernel.from_gaussian(
            shape_2d=psf_shape_2d, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return cls(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            psf=psf,
            exposure_time=exposure_time,
            background_level=background_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def keck_adaptive_optics(
        cls,
        shape=(751, 751),
        pixel_scales=0.01,
        sub_size=8,
        psf_shape_2d=(31, 31),
        psf_sigma=0.025,
        exposure_time=1000.0,
        background_level=1.0,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """Default settings for an observation using Keck Adaptive Optics imaging.

        This can be customized by over-riding the default input values."""
        psf = kernel.Kernel.from_gaussian(
            shape_2d=psf_shape_2d, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return cls(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            psf=psf,
            exposure_time=exposure_time,
            background_level=background_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @property
    def grid(self):
        return grids.Grid.uniform(
            shape_2d=self.shape_2d,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def from_image(self, image, name=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """
        return imaging.SimulatedImaging.simulate(
            image=image,
            exposure_time=self.exposure_time,
            psf=self.psf,
            background_level=self.background_level,
            add_noise=self.add_noise,
            noise_if_add_noise_false=self.noise_if_add_noise_false,
            noise_seed=self.noise_seed,
            name=name,
        )


class InterferometerSimulator(object):
    def __init__(
        self,
        real_space_shape_2d,
        real_space_pixel_scales,
        uv_wavelengths,
        sub_size,
        exposure_time,
        background_level,
        primary_beam=None,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        origin=(0.0, 0.0),
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        real_space_shape_2d : (int, int)
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        real_space_pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of an observation using this data_type.
        background_level : float
            The level of the background sky of an observationg using this data_type.
        """

        if type(real_space_pixel_scales) is float:
            real_space_pixel_scales = (real_space_pixel_scales, real_space_pixel_scales)

        self.real_space_shape_2d = real_space_shape_2d
        self.real_space_pixel_scales = real_space_pixel_scales
        self.uv_wavelengths = uv_wavelengths
        self.sub_size = sub_size
        self.origin = origin
        self.transformer = transformer.Transformer(
            uv_wavelengths=self.uv_wavelengths,
            grid_radians=self.grid.in_1d_binned.in_radians,
        )
        self.exposure_time = exposure_time
        self.background_level = background_level
        self.primary_beam = primary_beam
        self.noise_sigma = noise_sigma
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    @property
    def grid(self):
        return grids.Grid.uniform(
            shape_2d=self.real_space_shape_2d,
            pixel_scales=self.real_space_pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    @classmethod
    def sma(
        cls,
        real_space_shape_2d=(151, 151),
        real_space_pixel_scales=(0.05, 0.05),
        sub_size=8,
        primary_beam_shape_2d=None,
        primary_beam_sigma=None,
        exposure_time=100.0,
        background_level=1.0,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """Default settings for an observation with the Large Synotpic Survey Telescope.

        This can be customized by over-riding the default input values."""

        uv_wavelengths_path = "{}/dataset/sma_uv_wavelengths.fits".format(
            os.path.dirname(os.path.realpath(__file__))
        )

        uv_wavelengths = array_util.numpy_array_1d_from_fits(
            file_path=uv_wavelengths_path, hdu=0
        )

        if primary_beam_shape_2d is not None and primary_beam_sigma is not None:
            primary_beam = kernel.Kernel.from_gaussian(
                shape_2d=primary_beam_shape_2d,
                sigma=primary_beam_sigma,
                pixel_scales=real_space_pixel_scales,
            )
        else:
            primary_beam = None

        return cls(
            real_space_shape_2d=real_space_shape_2d,
            real_space_pixel_scales=real_space_pixel_scales,
            uv_wavelengths=uv_wavelengths,
            sub_size=sub_size,
            primary_beam=primary_beam,
            exposure_time=exposure_time,
            background_level=background_level,
            noise_sigma=noise_sigma,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_real_space_image(self, real_space_image):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        real_space_image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """
        return interferometer.SimulatedInterferometer.simulate(
            real_space_image=real_space_image,
            real_space_pixel_scales=self.real_space_pixel_scales,
            exposure_time=self.exposure_time,
            transformer=self.transformer,
            primary_beam=self.primary_beam,
            background_level=self.background_level,
            noise_sigma=self.noise_sigma,
            noise_if_add_noise_false=self.noise_if_add_noise_false,
            noise_seed=self.noise_seed,
        )
