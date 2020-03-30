import logging
import numpy as np
import copy

import autoarray as aa

from autoarray import exc
from autoarray.dataset import abstract_dataset, preprocess
from autoarray.structures import arrays, visibilities as vis, kernel
from autoarray.operators import transformer


logger = logging.getLogger(__name__)


class Interferometer(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        visibilities,
        noise_map,
        uv_wavelengths,
        primary_beam=None,
        name=None,
        metadata=None,
    ):

        super().__init__(
            data=visibilities, noise_map=noise_map, name=name, metadata=metadata
        )

        self.uv_wavelengths = uv_wavelengths
        self.primary_beam = primary_beam

    @classmethod
    def manual(cls, visibilities, noise_map, uv_wavelengths, primary_beam=None):
        return Interferometer(
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            primary_beam=primary_beam,
        )

    @classmethod
    def from_fits(
        cls,
        visibilities_path,
        noise_map_path,
        uv_wavelengths_path,
        visibilities_hdu=0,
        noise_map_hdu=0,
        uv_wavelengths_hdu=0,
        primary_beam_path=None,
        primary_beam_hdu=0,
    ):
        """Factory for loading the interferometer data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the interferometer-data_type.

        This factory also includes a number of routines for converting the interferometer-data_type from unit_label not supported by PyAutoLens \
        (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        lens_name
        image_path : str
            The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
        image_hdu : int
            The hdu the image is contained in the .fits file specified by *image_path*.
        image_hdu : int
            The hdu the image is contained in the .fits file that *image_path* points too.
        resized_interferometer_shape : (int, int) | None
            If input, the interferometer structures that are image sized, e.g. the image, noise-maps) are resized to these dimensions.
        resized_interferometer_origin_pixels : (int, int) | None
            If the interferometer structures are resized, this defines a new origin (in pixels) around which recentering occurs.
        resized_interferometer_origin_arcsec : (float, float) | None
            If the interferometer structures are resized, this defines a new origin (in arc-seconds) around which recentering occurs.
        primary_beam_path : str
            The path to the primary_beam .fits file containing the primary_beam (e.g. '/path/to/primary_beam.fits')
        primary_beam_hdu : int
            The hdu the primary_beam is contained in the .fits file specified by *primary_beam_path*.
        resized_primary_beam_shape_2d : (int, int) | None
            If input, the primary_beam is resized to these dimensions.
        renormalize_psf : bool
            If True, the PrimaryBeam is renoralized such that all elements sum to 1.0.
        noise_map_path : str
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
        noise_map_hdu : int
            The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
        noise_map_from_image_and_background_noise_map : bool
            If True, the noise-map is computed from the observed image and background noise-map \
            (see NoiseMap.from_image_and_background_noise_map).
        convert_noise_map_from_weight_map : bool
            If True, the noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
            *NoiseMap.from_weight_map).
        convert_noise_map_from_inverse_noise_map : bool
            If True, the noise-map loaded from the .fits file is converted from an inverse noise-map to a noise-map (see \
            *NoiseMap.from_inverse_noise_map).
        background_noise_map_path : str
            The path to the background_noise_map .fits file containing the background noise-map \
            (e.g. '/path/to/background_noise_map.fits')
        background_noise_map_hdu : int
            The hdu the background_noise_map is contained in the .fits file specified by *background_noise_map_path*.
        convert_background_noise_map_from_weight_map : bool
            If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
            *NoiseMap.from_weight_map).
        convert_background_noise_map_from_inverse_noise_map : bool
            If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
            noise-map (see *NoiseMap.from_inverse_noise_map).
        poisson_noise_map_path : str
            The path to the poisson_noise_map .fits file containing the Poisson noise-map \
             (e.g. '/path/to/poisson_noise_map.fits')
        poisson_noise_map_hdu : int
            The hdu the poisson_noise_map is contained in the .fits file specified by *poisson_noise_map_path*.
        poisson_noise_map_from_image : bool
            If True, the Poisson noise-map is estimated using the image.
        convert_poisson_noise_map_from_weight_map : bool
            If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
            *NoiseMap.from_weight_map).
        convert_poisson_noise_map_from_inverse_noise_map : bool
            If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
            noise-map (see *NoiseMap.from_inverse_noise_map).
        exposure_time_map_path : str
            The path to the exposure_time_map .fits file containing the exposure time map \
            (e.g. '/path/to/exposure_time_map.fits')
        exposure_time_map_hdu : int
            The hdu the exposure_time_map is contained in the .fits file specified by *exposure_time_map_path*.
        exposure_time_map_from_single_value : float
            The exposure time of the interferometer imaging, which is used to compute the exposure-time map as a single value \
            (see *ExposureTimeMap.from_single_value*).
        exposure_time_map_from_inverse_noise_map : bool
            If True, the exposure-time map is computed from the background noise-map \
            (see *ExposureTimeMap.from_background_noise_map*)
        background_sky_map_path : str
            The path to the background_sky_map .fits file containing the background sky map \
            (e.g. '/path/to/background_sky_map.fits').
        background_sky_map_hdu : int
            The hdu the background_sky_map is contained in the .fits file specified by *background_sky_map_path*.
        convert_from_electrons : bool
            If True, the input unblurred_image_1d are in units of electrons and all converted to electrons / second using the exposure \
            time map.
        gain : float
            The image gain, used for convert from ADUs.
        convert_from_adus : bool
            If True, the input unblurred_image_1d are in units of adus and all converted to electrons / second using the exposure \
            time map and gain.
        """

        visibilities = aa.Visibilities.from_fits(
            file_path=visibilities_path, hdu=visibilities_hdu
        )

        noise_map = aa.Visibilities.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )

        uv_wavelengths = aa.util.array.numpy_array_2d_from_fits(
            file_path=uv_wavelengths_path, hdu=uv_wavelengths_hdu
        )

        if primary_beam_path is not None:
            primary_beam = aa.Kernel.from_fits(
                file_path=primary_beam_path, hdu=primary_beam_hdu, renormalize=True
            )
        else:
            primary_beam = None

        return Interferometer(
            visibilities=visibilities,
            primary_beam=primary_beam,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
        )

    @property
    def visibilities(self):
        return self.data

    @property
    def amplitudes(self):
        return self.visibilities.amplitudes

    @property
    def phases(self):
        return self.visibilities.phases

    @property
    def uv_distances(self):
        return np.sqrt(
            np.square(self.uv_wavelengths[:, 0]) + np.square(self.uv_wavelengths[:, 1])
        )

    def modified_visibilities_from_visibilities(self, visibilities):

        interferometer = copy.deepcopy(self)
        interferometer.data = visibilities
        return interferometer

    def resized_primary_beam_from_new_shape_2d(self, new_shape_2d):

        interferometer = copy.deepcopy(self)
        interferometer.primary_beam = self.primary_beam.resized_from_new_shape(
            new_shape=new_shape_2d
        )
        return interferometer

    def output_to_fits(
        self,
        visibilities_path=None,
        noise_map_path=None,
        primary_beam_path=None,
        uv_wavelengths_path=None,
        overwrite=False,
    ):

        if primary_beam_path is not None:
            self.primary_beam.output_to_fits(
                file_path=primary_beam_path, overwrite=overwrite
            )

        if visibilities_path is not None:
            self.visibilities.output_to_fits(
                file_path=visibilities_path, overwrite=overwrite
            )

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)

        if self.uv_wavelengths is not None and uv_wavelengths_path is not None:
            aa.util.array.numpy_array_2d_to_fits(
                array_2d=self.uv_wavelengths,
                file_path=uv_wavelengths_path,
                overwrite=overwrite,
            )


class MaskedInterferometer(abstract_dataset.AbstractMaskedDataset):
    def __init__(
        self,
        interferometer,
        visibilities_mask,
        real_space_mask,
        transformer_class=transformer.TransformerNUFFT,
        primary_beam_shape_2d=None,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
        renormalize_primary_beam=True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, primary_beam), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, primary_beam, etc.)
        real_space_mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        primary_beam_shape_2d : (int, int)
            The shape of the primary_beam used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the primary_beam relative to the input image primary_beam, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        self.interferometer = interferometer

        super(MaskedInterferometer, self).__init__(
            mask=real_space_mask,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
        )

        if self.interferometer.primary_beam is None:
            self.primary_beam_shape_2d = None
        elif (
            primary_beam_shape_2d is None
            and self.interferometer.primary_beam is not None
        ):
            self.primary_beam_shape_2d = self.interferometer.primary_beam.shape_2d
        else:
            self.primary_beam_shape_2d = primary_beam_shape_2d

        if self.primary_beam_shape_2d is not None:
            self.primary_beam = kernel.Kernel.manual_2d(
                array=interferometer.primary_beam.resized_from_new_shape(
                    new_shape=self.primary_beam_shape_2d
                ).in_2d,
                renormalize=renormalize_primary_beam,
            )

        self.transformer = transformer_class(
            uv_wavelengths=interferometer.uv_wavelengths,
            grid=self.grid.in_1d_binned.in_radians,
        )

        self.visibilities = interferometer.visibilities
        self.noise_map = interferometer.noise_map
        self.visibilities_mask = visibilities_mask

    @property
    def data(self):
        return self.visibilities

    @property
    def uv_distances(self):
        return self.interferometer.uv_distances

    @property
    def real_space_mask(self):
        return self.mask

    def signal_to_noise_map(self):
        return self.visibilities / self.noise_map


class SimulatorInterferometer:
    def __init__(
        self,
        uv_wavelengths,
        exposure_time_map,
        background_sky_map=None,
        transformer_class=transformer.TransformerDFT,
        primary_beam=None,
        renormalize_primary_beam=True,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
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
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        background_sky_map : float
            The level of the background sky of an observationg using this data_type.
        """

        self.uv_wavelengths = uv_wavelengths
        self.exposure_time_map = exposure_time_map
        self.background_sky_map = background_sky_map
        self.transformer_class = transformer_class

        if primary_beam is not None and renormalize_primary_beam:
            primary_beam = primary_beam.renormalized

        self.primary_beam = primary_beam
        self.noise_sigma = noise_sigma
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    def from_image(self, image, name=None, metadata=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        real_space_image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and UVPlane read-out).
        real_space_pixel_scales: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        transformer = self.transformer_class(
            uv_wavelengths=self.uv_wavelengths,
            grid=image.mask.geometry.unmasked_grid.in_1d_binned.in_radians,
        )

        if self.background_sky_map is not None:
            background_sky_map = self.background_sky_map
        else:
            background_sky_map = arrays.Array.zeros(
                shape_2d=image.shape_2d, pixel_scales=image.pixel_scales
            )

        image = image + background_sky_map

        visibilities = transformer.visibilities_from_image(image=image)

        if self.noise_sigma is not None:
            visibilities = preprocess.data_with_gaussian_noise_added(
                data=visibilities, sigma=self.noise_sigma, seed=self.noise_seed
            )
            noise_map = vis.Visibilities.full(
                fill_value=self.noise_sigma, shape_1d=(visibilities.shape[0],)
            )
        else:
            noise_map = vis.Visibilities.full(
                fill_value=self.noise_if_add_noise_false,
                shape_1d=(visibilities.shape[0],),
            )

        if np.isnan(noise_map).any():
            raise exc.DataException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        return Interferometer(
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=transformer.uv_wavelengths,
            primary_beam=self.primary_beam,
            name=name,
            metadata=metadata,
        )
