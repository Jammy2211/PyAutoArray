import autoarray as aa

from autoarray.data import imaging

class ImagingSimulator(object):

    def __init__(
        self, shape_2d, pixel_scales, psf, exposure_time, background_sky_level, add_noise=True, noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulate a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of an observation using this data_type.
        background_sky_level : float
            The level of the background sky of an observationg using this data_type.
        """

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        self.shape = shape_2d
        self.pixel_scales = pixel_scales
        self.psf = psf
        self.exposure_time = exposure_time
        self.background_sky_level = background_sky_level
        self.add_noise = add_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    @classmethod
    def lsst(
        cls,
        shape=(101, 101),
        pixel_scales=0.2,
        psf_shape=(31, 31),
        psf_sigma=0.5,
        exposure_time=100.0,
        background_sky_level=1.0,
        add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Large Synotpic Survey Telescope.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def euclid(
        cls,
        shape=(151, 151),
        pixel_scales=0.1,
        psf_shape=(31, 31),
        psf_sigma=0.1,
        exposure_time=565.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Euclid space satellite.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def hst(
        cls,
        shape=(251, 251),
        pixel_scales=0.05,
        psf_shape=(31, 31),
        psf_sigma=0.05,
        exposure_time=2000.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Hubble Space Telescope.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def hst_up_sampled(
        cls,
        shape=(401, 401),
        pixel_scales=0.03,
        psf_shape=(31, 31),
        psf_sigma=0.05,
        exposure_time=2000.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Hubble Space Telescope which has been upscaled to a higher \
        pixel-scale to better sample the PSF.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed
        )

    @classmethod
    def keck_adaptive_optics(
        cls,
        shape=(751, 751),
        pixel_scales=0.01,
        psf_shape=(31, 31),
        psf_sigma=0.025,
        exposure_time=1000.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation using Keck Adaptive Optics imaging.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_image(
        self,
        image,
        name=None,
    ):
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
            background_sky_level=self.background_sky_level,
            add_noise=self.add_noise,
            noise_if_add_noise_false=self.noise_if_add_noise_false,
            noise_seed=self.noise_seed,
            name=name,
        )
