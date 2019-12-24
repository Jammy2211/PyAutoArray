import logging

import numpy as np
from scipy.stats import norm

from autoarray import exc
from autoarray.dataset import abstract_dataset, data_converter
from autoarray.mask import mask as msk
from autoarray.masked import masked_structures
from autoarray.structures import kernel, arrays

logger = logging.getLogger(__name__)


class AbstractImagingDataSet(abstract_dataset.AbstractDataset):
    @property
    def image(self):
        return self.data

    @property
    def pixel_scales(self):
        return self.data.pixel_scales

    @property
    def pixel_scale(self):
        return self.data.pixel_scale

    @property
    def shape(self):
        return self.image.shape_2d

    @property
    def shape_2d(self):
        return self.image.shape_2d

    def binned_from_bin_up_factor(self, bin_up_factor):

        image = self.image.binned_from_bin_up_factor(
            bin_up_factor=bin_up_factor, method="mean"
        )
        psf = self.psf.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=1.0 / bin_up_factor, renormalize=False
        )
        noise_map = (
            self.noise_map.binned_from_bin_up_factor(
                bin_up_factor=bin_up_factor, method="quadrature"
            )
            if self.noise_map is not None
            else None
        )

        background_noise_map = (
            self.background_noise_map.binned_from_bin_up_factor(
                bin_up_factor=bin_up_factor, method="quadrature"
            )
            if self.background_noise_map is not None
            else None
        )

        poisson_noise_map = (
            self.poisson_noise_map.binned_from_bin_up_factor(
                bin_up_factor=bin_up_factor, method="quadrature"
            )
            if self.poisson_noise_map is not None
            else None
        )

        exposure_time_map = (
            self.exposure_time_map.binned_from_bin_up_factor(
                bin_up_factor=bin_up_factor, method="sum"
            )
            if self.exposure_time_map is not None
            else None
        )

        background_sky_map = (
            self.background_sky_map.binned_from_bin_up_factor(
                bin_up_factor=bin_up_factor, method="mean"
            )
            if self.background_sky_map is not None
            else None
        )

        return Imaging(
            image=image,
            psf=psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            name=self.name,
        )

    def resized_from_new_shape(self, new_shape):

        image = self.image.resized_from_new_shape(new_shape=new_shape)

        noise_map = (
            self.noise_map.resized_from_new_shape(new_shape=new_shape)
            if self.noise_map is not None
            else None
        )

        background_noise_map = (
            self.background_noise_map.resized_from_new_shape(new_shape=new_shape)
            if self.background_noise_map is not None
            else None
        )

        poisson_noise_map = (
            self.poisson_noise_map.resized_from_new_shape(new_shape=new_shape)
            if self.poisson_noise_map is not None
            else None
        )

        exposure_time_map = (
            self.exposure_time_map.resized_from_new_shape(new_shape=new_shape)
            if self.exposure_time_map is not None
            else None
        )

        background_sky_map = (
            self.background_sky_map.resized_from_new_shape(new_shape=new_shape)
            if self.background_sky_map is not None
            else None
        )

        return Imaging(
            image=image,
            psf=self.psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            name=self.name,
        )

    def resized_psf_from_new_shape(self, new_shape):
        psf = self.psf.resized_from_new_shape(new_shape=new_shape)
        return Imaging(
            image=self.image,
            psf=psf,
            noise_map=self.noise_map,
            background_noise_map=self.background_noise_map,
            poisson_noise_map=self.poisson_noise_map,
            exposure_time_map=self.exposure_time_map,
            background_sky_map=self.background_sky_map,
            name=self.name,
        )

    def modified_image_from_image(self, image):

        return Imaging(
            image=image,
            psf=self.psf,
            noise_map=self.noise_map,
            background_noise_map=self.background_noise_map,
            poisson_noise_map=self.poisson_noise_map,
            exposure_time_map=self.exposure_time_map,
            background_sky_map=self.background_sky_map,
            name=self.name,
        )

    def add_poisson_noise_to_data(self, seed=-1):

        image_with_sky = self.image + self.background_sky_map

        image_with_sky_and_noise = image_with_sky + generate_poisson_noise(
            image=image_with_sky, exposure_time_map=self.exposure_time_map, seed=seed
        )

        image_with_noise = image_with_sky_and_noise - self.background_sky_map

        return Imaging(
            image=image_with_noise,
            psf=self.psf,
            noise_map=self.noise_map,
            background_noise_map=self.background_noise_map,
            poisson_noise_map=self.poisson_noise_map,
            name=self.name,
        )

    def data_in_electrons(self):

        image = self.array_from_counts_to_electrons_per_second(array=self.image)
        noise_map = self.array_from_counts_to_electrons_per_second(array=self.noise_map)
        background_noise_map = self.array_from_counts_to_electrons_per_second(
            array=self.background_noise_map
        )
        poisson_noise_map = self.array_from_counts_to_electrons_per_second(
            array=self.poisson_noise_map
        )
        background_sky_map = self.array_from_counts_to_electrons_per_second(
            array=self.background_sky_map
        )

        return Imaging(
            image=image,
            psf=self.psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=self.exposure_time_map,
            background_sky_map=background_sky_map,
            name=self.name,
        )

    def data_in_adus_from_gain(self, gain):

        image = self.array_from_adus_to_electrons_per_second(
            array=self.image, gain=gain
        )
        noise_map = self.array_from_adus_to_electrons_per_second(
            array=self.noise_map, gain=gain
        )
        background_noise_map = self.array_from_adus_to_electrons_per_second(
            array=self.background_noise_map, gain=gain
        )
        poisson_noise_map = self.array_from_adus_to_electrons_per_second(
            array=self.poisson_noise_map, gain=gain
        )
        background_sky_map = self.array_from_adus_to_electrons_per_second(
            array=self.background_sky_map, gain=gain
        )

        return Imaging(
            image=image,
            psf=self.psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=self.exposure_time_map,
            background_sky_map=background_sky_map,
            name=self.name,
        )

    def signal_to_noise_limited_from_signal_to_noise_limit(self, signal_to_noise_limit):

        noise_map_limit = np.where(
            self.signal_to_noise_map > signal_to_noise_limit,
            np.abs(self.image) / signal_to_noise_limit,
            self.noise_map,
        )

        noise_map_limit = masked_structures.MaskedArray.manual_1d(
            array=noise_map_limit, mask=self.image.mask
        )

        return Imaging(
            image=self.image,
            psf=self.psf,
            noise_map=noise_map_limit,
            background_noise_map=self.background_noise_map,
            poisson_noise_map=self.poisson_noise_map,
            exposure_time_map=self.exposure_time_map,
            background_sky_map=self.background_sky_map,
            name=self.name,
        )

    @property
    def background_noise_map_counts(self):
        """ The background noise_maps mappers in unit_label of counts."""
        return self.array_from_electrons_per_second_to_counts(self.background_noise_map)

    @property
    def estimated_noise_map_counts(self):
        """ The estimated noise_maps mappers of the image (using its background noise_maps mappers and image values
        in counts) in counts.
        """
        return np.sqrt(
            (np.abs(self.image_counts) + np.square(self.background_noise_map_counts))
        )

    @property
    def estimated_noise_map(self):
        """ The estimated noise_maps mappers of the image (using its background noise_maps mappers and image values
        in counts) in electrons per second.
        """
        return self.array_from_counts_to_electrons_per_second(
            self.estimated_noise_map_counts
        )

    def background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning data_to_image located at the edge(s) of an image
        into a histogram and fitting a Gaussian profiles to this histogram. The standard deviation (sigma) of this
        Gaussian gives a signal_to_noise_ratio estimate.

        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the background signal_to_noise_ratio.

        """

        edges = []

        for edge_no in range(no_edges):
            top_edge = self.image.in_2d[
                edge_no, edge_no : self.image.shape_2d[1] - edge_no
            ]
            bottom_edge = self.image.in_2d[
                self.image.shape_2d[0] - 1 - edge_no,
                edge_no : self.image.shape_2d[1] - edge_no,
            ]
            left_edge = self.image.in_2d[
                edge_no + 1 : self.image.shape_2d[0] - 1 - edge_no, edge_no
            ]
            right_edge = self.image.in_2d[
                edge_no + 1 : self.image.shape_2d[0] - 1 - edge_no,
                self.image.shape_2d[1] - 1 - edge_no,
            ]

            edges = np.concatenate(
                (edges, top_edge, bottom_edge, right_edge, left_edge)
            )

        return norm.fit(edges)[1]

    def output_to_fits(
        self,
        image_path,
        psf_path=None,
        noise_map_path=None,
        background_noise_map_path=None,
        poisson_noise_map_path=None,
        exposure_time_map_path=None,
        background_sky_map_path=None,
        overwrite=False,
    ):
        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)

        if self.psf is not None and psf_path is not None:
            self.psf.output_to_fits(file_path=psf_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)

        if (
            self.background_noise_map is not None
            and background_noise_map_path is not None
        ):
            self.background_noise_map.output_to_fits(
                file_path=background_noise_map_path, overwrite=overwrite
            )

        if self.poisson_noise_map is not None and poisson_noise_map_path is not None:
            self.poisson_noise_map.output_to_fits(
                file_path=poisson_noise_map_path, overwrite=overwrite
            )

        if self.exposure_time_map is not None and exposure_time_map_path is not None:
            self.exposure_time_map.output_to_fits(
                file_path=exposure_time_map_path, overwrite=overwrite
            )

        if self.background_sky_map is not None and background_sky_map_path is not None:
            self.background_sky_map.output_to_fits(
                file_path=background_sky_map_path, overwrite=overwrite
            )


class Imaging(AbstractImagingDataSet):
    def __init__(
        self,
        image,
        noise_map,
        psf=None,
        background_noise_map=None,
        poisson_noise_map=None,
        exposure_time_map=None,
        background_sky_map=None,
        name=None,
        **kwargs
    ):
        """A collection of 2D imaging dataset(an image, noise-map, psf, etc.)

        Parameters
        ----------
        image : aa.Array
            The array of the image data_type, in unit_label of electrons per second.
        psf : PSF
            An array describing the PSF kernel of the image.
        noise_map : NoiseMap | float | ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in unit_label of electrons per
            second.
        background_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the background sky noise_map,
            preferably in unit_label of electrons per second.
        poisson_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the Poisson counts of the source,
            preferably in unit_label of electrons per second.
        exposure_time_map : aa.Array
            An array describing the effective exposure time in each imaging pixel.
        background_sky_map : aa.Scaled
            An array describing the background sky.
        """

        super(Imaging, self).__init__(
            data=image, noise_map=noise_map, exposure_time_map=exposure_time_map
        )

        self.psf = psf
        self.__name = name
        self.background_noise_map = background_noise_map
        self.poisson_noise_map = poisson_noise_map
        self.background_sky_map = background_sky_map

    @property
    def name(self) -> str:
        return self.__name

    @classmethod
    def manual(
        cls,
        image,
        noise_map,
        psf=None,
        background_noise_map=None,
        poisson_noise_map=None,
        exposure_time_map=None,
        background_sky_map=None,
    ):
        return Imaging(
            image=image,
            psf=psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
        )

    @classmethod
    def from_fits(
        cls,
        image_path,
        pixel_scales=None,
        image_hdu=0,
        resized_imaging_shape=None,
        noise_map_path=None,
        noise_map_hdu=0,
        noise_map_from_image_and_background_noise_map=False,
        noise_map_non_constant=False,
        psf_path=None,
        psf_hdu=0,
        resized_psf_shape=None,
        renormalize_psf=True,
        convert_noise_map_from_weight_map=False,
        convert_noise_map_from_inverse_noise_map=False,
        background_noise_map_path=None,
        background_noise_map_hdu=0,
        convert_background_noise_map_from_weight_map=False,
        convert_background_noise_map_from_inverse_noise_map=False,
        poisson_noise_map_path=None,
        poisson_noise_map_hdu=0,
        poisson_noise_map_from_image=False,
        convert_poisson_noise_map_from_weight_map=False,
        convert_poisson_noise_map_from_inverse_noise_map=False,
        exposure_time_map_path=None,
        exposure_time_map_hdu=0,
        exposure_time_map_from_single_value=None,
        exposure_time_map_from_inverse_noise_map=False,
        background_sky_map_path=None,
        background_sky_map_hdu=0,
        convert_from_electrons=False,
        gain=None,
        convert_from_adus=False,
        lens_name=None,
    ):
        """Factory for loading the imaging data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the imaging-data.

        This factory also includes a number of routines for converting the imaging-data from unit_label not supported by PyAutoLens \
        (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        lens_name
        image_path : str
            The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
        pixel_scales : float
            The size of each pixel in arc seconds.
        image_hdu : int
            The hdu the image is contained in the .fits file specified by *image_path*.
        image_hdu : int
            The hdu the image is contained in the .fits file that *image_path* points too.
        resized_imaging_shape : (int, int) | None
            If input, the imaging structures that are image sized, e.g. the image, noise-maps) are resized to these dimensions.
        resized_imaging_origin_pixels : (int, int) | None
            If the imaging structures are resized, this defines a new origin (in pixels) around which recentering occurs.
        resized_imaging_origin_arcsec : (float, float) | None
            If the imaging structures are resized, this defines a new origin (in arc-seconds) around which recentering occurs.
        psf_path : str
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits')
        psf_hdu : int
            The hdu the psf is contained in the .fits file specified by *psf_path*.
        resized_psf_shape : (int, int) | None
            If input, the psf is resized to these dimensions.
        renormalize_psf : bool
            If True, the PSF is renoralized such that all elements sum to 1.0.
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
            The exposure time of the imaging, which is used to compute the exposure-time map as a single value \
            (see *ExposureTimeMap.from_single_value*).
        exposure_time_map_from_inverse_noise_map : bool
            If True, the exposure-time map is computed from the background noise_map map \
            (see *ExposureTimeMap.from_background_noise_map*)
        background_sky_map_path : str
            The path to the background_sky_map .fits file containing the background sky map \
            (e.g. '/path/to/background_sky_map.fits').
        background_sky_map_hdu : int
            The hdu the background_sky_map is contained in the .fits file specified by *background_sky_map_path*.
        convert_from_electrons : bool
            If True, the input unblurred_image_1d are in unit_label of electrons and all converted to electrons / second using the exposure \
            time map.
        gain : float
            The image gain, used for convert from ADUs.
        convert_from_adus : bool
            If True, the input unblurred_image_1d are in unit_label of adus and all converted to electrons / second using the exposure \
            time map and gain.
        """

        image = abstract_dataset.load_image(
            image_path=image_path, image_hdu=image_hdu, pixel_scales=pixel_scales
        )

        background_noise_map = load_background_noise_map(
            background_noise_map_path=background_noise_map_path,
            background_noise_map_hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
            convert_background_noise_map_from_weight_map=convert_background_noise_map_from_weight_map,
            convert_background_noise_map_from_inverse_noise_map=convert_background_noise_map_from_inverse_noise_map,
        )

        if background_noise_map is not None:
            inverse_noise_map = 1.0 / background_noise_map
        else:
            inverse_noise_map = None

        exposure_time_map = abstract_dataset.load_exposure_time_map(
            exposure_time_map_path=exposure_time_map_path,
            exposure_time_map_hdu=exposure_time_map_hdu,
            pixel_scales=pixel_scales,
            shape=image.mask.shape,
            exposure_time=exposure_time_map_from_single_value,
            exposure_time_map_from_inverse_noise_map=exposure_time_map_from_inverse_noise_map,
            inverse_noise_map=inverse_noise_map,
        )

        poisson_noise_map = load_poisson_noise_map(
            poisson_noise_map_path=poisson_noise_map_path,
            poisson_noise_map_hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
            convert_poisson_noise_map_from_weight_map=convert_poisson_noise_map_from_weight_map,
            convert_poisson_noise_map_from_inverse_noise_map=convert_poisson_noise_map_from_inverse_noise_map,
            image=image,
            exposure_time_map=exposure_time_map,
            poisson_noise_map_from_image=poisson_noise_map_from_image,
            convert_from_electrons=convert_from_electrons,
            gain=gain,
            convert_from_adus=convert_from_adus,
        )

        noise_map = load_noise_map(
            noise_map_path=noise_map_path,
            noise_map_hdu=noise_map_hdu,
            pixel_scales=pixel_scales,
            image=image,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            convert_noise_map_from_weight_map=convert_noise_map_from_weight_map,
            convert_noise_map_from_inverse_noise_map=convert_noise_map_from_inverse_noise_map,
            noise_map_from_image_and_background_noise_map=noise_map_from_image_and_background_noise_map,
            convert_from_electrons=convert_from_electrons,
            gain=gain,
            convert_from_adus=convert_from_adus,
        )

        if noise_map_non_constant:
            if np.allclose(noise_map, noise_map[0] * np.ones(shape=noise_map.shape)):
                noise_map = noise_map + (
                    0.001 * noise_map[0] * np.random.uniform(size=noise_map.shape_1d)
                )

        if psf_path is not None:

            psf = kernel.Kernel.from_fits(
                file_path=psf_path,
                hdu=psf_hdu,
                pixel_scales=pixel_scales,
                renormalize=renormalize_psf,
            )

        else:

            psf = None

        background_sky_map = load_background_sky_map(
            background_sky_map_path=background_sky_map_path,
            background_sky_map_hdu=background_sky_map_hdu,
            pixel_scales=pixel_scales,
        )

        imaging = Imaging(
            image=image,
            pixel_scales=pixel_scales,
            psf=psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            gain=gain,
            name=lens_name,
        )

        if resized_imaging_shape is not None:
            imaging = imaging.resized_from_new_shape(new_shape=resized_imaging_shape)

        if resized_psf_shape is not None and imaging.psf is not None:
            imaging = imaging.resized_psf_from_new_shape(new_shape=resized_psf_shape)

        if convert_from_electrons:
            imaging = imaging.data_in_electrons()
        elif convert_from_adus:
            imaging = imaging.data_in_adus_from_gain(gain=gain)

        return imaging

    @classmethod
    def simulate(
        cls,
        image,
        exposure_time,
        psf=None,
        exposure_time_map=None,
        background_level=0.0,
        background_sky_map=None,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
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

        if psf is None:
            psf = kernel.Kernel.no_blur(pixel_scales=image.pixel_scales)
            image_needs_trimming = False
        else:
            image_needs_trimming = True

        if exposure_time_map is None:
            exposure_time_map = arrays.Array.full(
                fill_value=exposure_time, shape_2d=image.shape_2d
            )

        if background_sky_map is None:
            background_sky_map = arrays.Array.full(
                fill_value=background_level, shape_2d=image.shape_2d
            )

        image += background_sky_map

        image = psf.convolved_array_from_array(array=image)

        if image_needs_trimming:
            image = image.trimmed_from_kernel_shape(kernel_shape_2d=psf.shape_2d)
            exposure_time_map = exposure_time_map.trimmed_from_kernel_shape(
                kernel_shape_2d=psf.shape_2d
            )
            background_sky_map = background_sky_map.trimmed_from_kernel_shape(
                kernel_shape_2d=psf.shape_2d
            )

        if add_noise is True:
            noise_realization = generate_poisson_noise(
                image, exposure_time_map, noise_seed
            )
            image += noise_realization
            image_counts = np.multiply(image, exposure_time_map)
            noise_map = np.divide(np.sqrt(image_counts), exposure_time_map)
            noise_map = masked_structures.MaskedArray.manual_1d(
                array=noise_map, mask=noise_map.mask
            )
        else:
            noise_map = arrays.Array.full(
                fill_value=noise_if_add_noise_false,
                shape_2d=image.shape_2d,
                pixel_scales=image.pixel_scales,
            )
            noise_realization = None

        if np.isnan(noise_map).any():
            raise exc.DataException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        image -= background_sky_map

        # ESTIMATE THE BACKGROUND NOISE MAP FROM THE BACKGROUND SKY MAP

        background_noise_map_counts = np.sqrt(
            np.multiply(background_sky_map, exposure_time_map)
        )
        background_noise_map = np.divide(background_noise_map_counts, exposure_time_map)

        # ESTIMATE THE POISSON NOISE MAP FROM THE IMAGE

        image_counts = np.multiply(image, exposure_time_map)
        poisson_noise_map = np.divide(np.sqrt(np.abs(image_counts)), exposure_time_map)

        mask = msk.Mask.unmasked(
            shape_2d=image.shape_2d, pixel_scales=image.pixel_scales
        )

        image = masked_structures.MaskedArray.manual_1d(array=image, mask=mask)
        background_noise_map = masked_structures.MaskedArray.manual_1d(
            array=background_noise_map, mask=mask
        )
        poisson_noise_map = masked_structures.MaskedArray.manual_1d(
            array=poisson_noise_map, mask=mask
        )

        return SimulatedImaging(
            image=image,
            psf=psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            noise_realization=noise_realization,
            name=name,
        )

    def __array_finalize__(self, obj):
        if isinstance(obj, Imaging):
            try:
                self.data = obj.data
                self.psf = obj.psf
                self.noise_map = obj.noise_map
                self.background_noise_map = obj.background_noise_map
                self.poisson_noise_map = obj.poisson_noise_map
                self.exposure_time_map = obj.exposure_time_map
                self.background_sky_map = obj.background_sky_map
                self.origin = obj.origin
            except AttributeError:
                logger.debug(
                    "Original object in Imaging.__array_finalize__ missing one or more attributes"
                )


class SimulatedImaging(Imaging):
    def __init__(
        self,
        image,
        psf,
        noise_map=None,
        background_noise_map=None,
        poisson_noise_map=None,
        exposure_time_map=None,
        background_sky_map=None,
        noise_realization=None,
        name=None,
        **kwargs
    ):

        super(SimulatedImaging, self).__init__(
            image=image,
            psf=psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            name=name,
            kwargs=kwargs,
        )

        self.noise_realization = noise_realization

    def __array_finalize__(self, obj):
        if isinstance(obj, SimulatedImaging):
            try:
                self.data = obj.data
                self.psf = obj.psf
                self.noise_map = obj.noise_map
                self.background_noise_map = obj.background_noise_map
                self.poisson_noise_map = obj.poisson_noise_map
                self.exposure_time_map = obj.exposure_time_map
                self.background_sky_map = obj.background_sky_map
                self.background_noise_realization = obj.background_noise_realization
                self.poisson_noise_realization = obj.poisson_noise_realization
                self.origin = obj.origin
            except AttributeError:
                logger.debug(
                    "Original object in Imaging.__array_finalize__ missing one or more attributes"
                )


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is \
    positive, that seed is used for all runs, thereby giving reproducible results.

    Parameters
    ----------
    seed : int
        The seed of the random number generator.
    """
    if seed == -1:
        seed = np.random.randint(
            0, int(1e9)
        )  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


def generate_poisson_noise(image, exposure_time_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_maps-mappers from an image.

    Values are computed from a Poisson distribution using the image's input values in unit_label of counts.

    Parameters
    ----------
    image : ndarray
        The 2D image, whose values in counts are used to draw Poisson noise_maps values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed : int
        The seed of the random number generator, used for the random noise_maps maps.

    Returns
    -------
    poisson_noise_map: ndarray
        An array describing simulated poisson noise_maps
    """
    setup_random_seed(seed)
    image_counts = np.multiply(image, exposure_time_map)
    return image - np.divide(
        np.random.poisson(image_counts, image.shape), exposure_time_map
    )


def load_noise_map(
    noise_map_path,
    noise_map_hdu,
    pixel_scales,
    image=None,
    background_noise_map=None,
    exposure_time_map=None,
    convert_noise_map_from_weight_map=False,
    convert_noise_map_from_inverse_noise_map=False,
    noise_map_from_image_and_background_noise_map=False,
    convert_from_electrons=False,
    gain=None,
    convert_from_adus=False,
):
    """Factory for loading the noise-map from a .fits file.

    This factory also includes a number of routines for converting the noise-map from from other unit_label (e.g. \
    a weight map) or computing the noise-map from other unblurred_image_1d (e.g. the imaging image and background noise-map).

    Parameters
    ----------
    noise_map_path : str
        The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
    noise_map_hdu : int
        The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    image : ndarray
        The image-image, which the noise-map can be calculated using.
    background_noise_map : ndarray
        The background noise-map, which the noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the noise-map can be calculated using.
    convert_noise_map_from_weight_map : bool
        If True, the noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_noise_map_from_inverse_noise_map : bool
        If True, the noise-map loaded from the .fits file is converted from an inverse noise-map to a noise-map (see \
        *NoiseMap.from_inverse_noise_map).
    background_noise_map_path : str
        The path and filename of the .fits image containing the background noise-map.
    background_noise_map_hdu : int
        The hdu the background noise-map is contained in the .fits file that *background_noise_map_path* points too.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    noise_map_from_image_and_background_noise_map : bool
        If True, the noise-map is computed from the observed image and background noise-map \
        (see NoiseMap.from_image_and_background_noise_map).
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in unit_label of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in unit_label of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """
    noise_map_options = sum(
        [
            convert_noise_map_from_weight_map,
            convert_noise_map_from_inverse_noise_map,
            noise_map_from_image_and_background_noise_map,
        ]
    )

    if noise_map_options > 1:
        raise exc.DataException(
            "You have specified more than one method to load the noise_map map, e.g.:"
            "convert_noise_map_from_weight_map | "
            "convert_noise_map_from_inverse_noise_map |"
            "noise_map_from_image_and_background_noise_map"
        )

    if noise_map_options == 0 and noise_map_path is not None:
        return arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
    elif convert_noise_map_from_weight_map and noise_map_path is not None:
        weight_map = arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
        return data_converter.noise_map_from_weight_map(weight_map=weight_map)
    elif convert_noise_map_from_inverse_noise_map and noise_map_path is not None:
        inverse_noise_map = arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
        return data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )
    elif noise_map_from_image_and_background_noise_map:

        if background_noise_map is None:
            raise exc.DataException(
                "Cannot compute the noise-map from the image and background noise_map map if a "
                "background noise_map map is not supplied."
            )

        if (
            not (convert_from_electrons or convert_from_adus)
            and exposure_time_map is None
        ):
            raise exc.DataException(
                "Cannot compute the noise-map from the image and background noise_map map if an "
                "exposure-time (or exposure time map) is not supplied to convert to adus"
            )

        if convert_from_adus and gain is None:
            raise exc.DataException(
                "Cannot compute the noise-map from the image and background noise_map map if a"
                "gain is not supplied to convert from adus"
            )

        return data_converter.noise_map_from_image_and_background_noise_map(
            image=image,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            convert_from_electrons=convert_from_electrons,
            gain=gain,
            convert_from_adus=convert_from_adus,
        )
    else:
        raise exc.DataException(
            "A noise_map map was not loaded, specify a noise_map_path or option to compute a noise_map map."
        )


def load_background_noise_map(
    background_noise_map_path,
    background_noise_map_hdu,
    pixel_scales,
    convert_background_noise_map_from_weight_map=False,
    convert_background_noise_map_from_inverse_noise_map=False,
):
    """Factory for loading the background noise-map from a .fits file.

    This factory also includes a number of routines for converting the background noise-map from from other unit_label (e.g. \
    a weight map).

    Parameters
    ----------
    background_noise_map_path : str
        The path to the background_noise_map .fits file containing the background noise-map \
        (e.g. '/path/to/background_noise_map.fits')
    background_noise_map_hdu : int
        The hdu the background_noise_map is contained in the .fits file specified by *background_noise_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    """
    background_noise_map_options = sum(
        [
            convert_background_noise_map_from_weight_map,
            convert_background_noise_map_from_inverse_noise_map,
        ]
    )

    if background_noise_map_options == 0 and background_noise_map_path is not None:
        return arrays.Array.from_fits(
            file_path=background_noise_map_path,
            hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
    elif (
        convert_background_noise_map_from_weight_map
        and background_noise_map_path is not None
    ):
        weight_map = arrays.Array.from_fits(
            file_path=background_noise_map_path,
            hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_weight_map(weight_map=weight_map)
    elif (
        convert_background_noise_map_from_inverse_noise_map
        and background_noise_map_path is not None
    ):
        inverse_noise_map = arrays.Array.from_fits(
            file_path=background_noise_map_path,
            hdu=background_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )
    else:
        return None


def load_poisson_noise_map(
    poisson_noise_map_path,
    poisson_noise_map_hdu,
    pixel_scales,
    convert_poisson_noise_map_from_weight_map=False,
    convert_poisson_noise_map_from_inverse_noise_map=False,
    poisson_noise_map_from_image=False,
    image=None,
    exposure_time_map=None,
    convert_from_electrons=False,
    gain=None,
    convert_from_adus=False,
):
    """Factory for loading the Poisson noise-map from a .fits file.

    This factory also includes a number of routines for converting the Poisson noise-map from from other unit_label (e.g. \
    a weight map) or computing the Poisson noise_map from other unblurred_image_1d (e.g. the imaging image).

    Parameters
    ----------
    poisson_noise_map_path : str
        The path to the poisson_noise_map .fits file containing the Poisson noise-map \
         (e.g. '/path/to/poisson_noise_map.fits')
    poisson_noise_map_hdu : int
        The hdu the poisson_noise_map is contained in the .fits file specified by *poisson_noise_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    convert_poisson_noise_map_from_weight_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_poisson_noise_map_from_inverse_noise_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    poisson_noise_map_from_image : bool
        If True, the Poisson noise-map is estimated using the image.
    image : ndarray
        The image, which the Poisson noise-map can be calculated using.
    background_noise_map : ndarray
        The background noise-map, which the Poisson noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the Poisson noise-map can be calculated using.
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in unit_label of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in unit_label of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """
    poisson_noise_map_options = sum(
        [
            convert_poisson_noise_map_from_weight_map,
            convert_poisson_noise_map_from_inverse_noise_map,
            poisson_noise_map_from_image,
        ]
    )

    if poisson_noise_map_options == 0 and poisson_noise_map_path is not None:
        return arrays.Array.from_fits(
            file_path=poisson_noise_map_path,
            hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
    elif poisson_noise_map_from_image:

        if (
            not (convert_from_electrons or convert_from_adus)
            and exposure_time_map is None
        ):
            raise exc.DataException(
                "Cannot compute the Poisson noise-map from the image if an "
                "exposure-time (or exposure time map) is not supplied to convert to adus"
            )

        if convert_from_adus and gain is None:
            raise exc.DataException(
                "Cannot compute the Poisson noise-map from the image if a"
                "gain is not supplied to convert from adus"
            )

        return data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=image,
            exposure_time_map=exposure_time_map,
            convert_from_electrons=convert_from_electrons,
            gain=gain,
            convert_from_adus=convert_from_adus,
        )

    elif (
        convert_poisson_noise_map_from_weight_map and poisson_noise_map_path is not None
    ):
        weight_map = arrays.Array.from_fits(
            file_path=poisson_noise_map_path,
            hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_weight_map(weight_map=weight_map)
    elif (
        convert_poisson_noise_map_from_inverse_noise_map
        and poisson_noise_map_path is not None
    ):
        inverse_noise_map = arrays.Array.from_fits(
            file_path=poisson_noise_map_path,
            hdu=poisson_noise_map_hdu,
            pixel_scales=pixel_scales,
        )
        return data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )
    else:
        return None


def load_background_sky_map(
    background_sky_map_path, background_sky_map_hdu, pixel_scales
):
    """Factory for loading the background sky from a .fits file.

    Parameters
    ----------
    background_sky_map_path : str
        The path to the background_sky_map .fits file containing the background sky map \
        (e.g. '/path/to/background_sky_map.fits').
    background_sky_map_hdu : int
        The hdu the background_sky_map is contained in the .fits file specified by *background_sky_map_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    """
    if background_sky_map_path is not None:
        return arrays.Array.from_fits(
            file_path=background_sky_map_path,
            hdu=background_sky_map_hdu,
            pixel_scales=pixel_scales,
        )
    else:
        return None
