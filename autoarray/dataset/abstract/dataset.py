import copy
import logging
import numpy as np
import warnings
from typing import Optional, Union

from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray import exc
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.over_sampling.abstract import AbstractOverSampling
from autoarray.operators.over_sampling.uniform import OverSamplingUniform
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoconf import cached_property


logger = logging.getLogger(__name__)


class AbstractDataset:
    def __init__(
        self,
        data: Structure,
        noise_map: Structure,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        over_sampling: Optional[AbstractOverSampling] = None,
        over_sampling_non_uniform: Optional[AbstractOverSampling] = None,
        over_sampling_pixelization: Optional[AbstractOverSampling] = None,
    ):
        """
        An abstract dataset, containing the image data, noise-map, PSF and associated quantities for calculations
        like the grid.

        This object is extended with specific dataset types, such as an `Imaging` or `Interferometer` dataset,
        and can be used for fitting data with model data and quantifying the goodness-of-fit.

        The following quantities are abstract and used by any dataset type:

        - `data`: The image data, which shows the signal that is analysed and fitted with a model image.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        Datasets also contains following properties:

        - `grid`: A grids of (y,x) coordinates which align with the image pixels, whereby each coordinate corresponds to
        the centre of an image pixel. This may be used in fits to calculate the model image of the imaging data.

        - `grid_pixelization`: A grid of (y,x) coordinates which align with the pixels of a pixelization. This grid
        is specifically used for pixelizations computed via the `inversion` module, which often use different
        oversampling and sub-size values to the grid above.

        The `over_sampling` and `over_sampling_pixelization` define how over sampling is performed for these grids.

        This is used in the project PyAutoGalaxy to load imaging data of a galaxy and fit it with galaxy light profiles.
        It is used in PyAutoLens to load imaging data of a strong lens and fit it with a lens model.

        Parameters
        ----------
        data
            The array of the image data containing the signal that is fitted (in PyAutoGalaxy and PyAutoLens the
            recommended units are electrons per second).
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit (in PyAutoGalaxy and PyAutoLens the recommended units are electrons per second).
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        over_sampling
            The over sampling scheme, which divides the grid into a sub grid of smaller pixels when computing values
            (e.g. images) from the grid so as to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sampling_non_uniform
            The over sampling scheme when the grid input into a function is not a uniform grid. This is used
            by **PyAutoLens** when the grid has been deflected and ray-traced and therefore some of the default
            over sampling schemes are not appropriate.
        over_sampling_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        """

        self.data = data
        self.over_sampling = over_sampling
        self.over_sampling_non_uniform = over_sampling_non_uniform
        self.over_sampling_pixelization = over_sampling_pixelization

        self.noise_covariance_matrix = noise_covariance_matrix

        if noise_map is None:
            try:
                noise_map = Array2D.no_mask(
                    values=np.diag(noise_covariance_matrix),
                    shape_native=data.shape_native,
                    pixel_scales=data.shape_native,
                )

                logger.info(
                    """
                    No noise map was input into the Imaging class, but a `noise_covariance_matrix` was.
    
                    Using the diagonal of the `noise_covariance_matrix` to create the `noise_map`. 
    
                    This `noise-map` is used only for visualization where it is not appropriate to plot covariance.
                    """
                )

            except ValueError as e:
                raise exc.DatasetException(
                    """
                    No noise map or noise_covariance_matrix was passed to the Imaging object.
                    """
                ) from e

        self.noise_map = noise_map

    @cached_property
    def grid(self) -> Union[Grid1D, Grid2D]:
        """
        Returns the grid of (y,x) Cartesian coordinates of every pixel in the masked data structure.

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data structure.
        """

        return Grid2D.from_mask(
            mask=self.mask,
            over_sampling=self.over_sampling,
            over_sampling_non_uniform=self.over_sampling_non_uniform
            or OverSamplingUniform(sub_size=1),
        )

    @cached_property
    def grid_pixelization(self) -> Grid2D:
        """
        Returns the grid of (y,x) Cartesian coordinates of every pixel in the masked data structure which is used
        specifically for pixelization reconstructions (e.g. an `inversion`).

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        A pixelization often uses a different grid of coordinates compared to the main `grid` of the data structure.
        A common example is that a pixelization may use a higher `sub_size` than the main grid, in order to better
        prevent aliasing effects.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data structure, used for pixelization / inversion calculations.
        """

        over_sampling = self.over_sampling_pixelization

        if over_sampling is None:
            over_sampling = OverSamplingUniform(sub_size=4)

        return Grid2D.from_mask(
            mask=self.mask,
            over_sampling=over_sampling,
        )

    @cached_property
    def over_sampler_non_uniform(self):
        return self.grid.over_sampling_non_uniform.over_sampler_from(mask=self.mask)

    @cached_property
    def over_sampler_pixelization(self):
        return self.grid_pixelization.over_sampling.over_sampler_from(mask=self.mask)

    @cached_property
    def border_relocator(self) -> BorderRelocator:
        return BorderRelocator(
            mask=self.mask, sub_size=self.grid_pixelization.over_sampling.sub_size
        )

    @property
    def shape_native(self):
        return self.mask.shape_native

    @property
    def shape_slim(self):
        return self.data.shape_slim

    @property
    def pixel_scales(self):
        return self.mask.pixel_scales

    @property
    def mask(self) -> Union[Mask1D, Mask2D]:
        return self.data.mask

    @property
    def signal_to_noise_map(self) -> Structure:
        """
        The estimated signal-to-noise_maps mappers of the image.

        Warnings arise when masked native noise-maps are used, whose masked entries are given values of 0.0. We
        use the warnings module to suppress these RunTimeWarnings.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = self.data / self.noise_map
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self) -> float:
        """
        The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers.
        """
        return np.max(self.signal_to_noise_map)

    @cached_property
    def noise_covariance_matrix_inv(self) -> np.ndarray:
        """
        Returns the inverse of the noise covariance matrix, which is used when computing a chi-squared which accounts
        for covariance via a fit.
        """
        return np.linalg.inv(self.noise_covariance_matrix)

    def trimmed_after_convolution_from(self, kernel_shape) -> "AbstractDataset":
        dataset = copy.copy(self)

        dataset.data = dataset.data.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )
        dataset.noise_map = dataset.noise_map.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )

        return dataset

    def apply_over_sampling(
        self,
        over_sampling: Optional[AbstractOverSampling] = None,
        over_sampling_non_uniform: Optional[AbstractOverSampling] = None,
        over_sampling_pixelization: Optional[AbstractOverSampling] = None,
    ) -> "AbstractDataset":
        """
        Apply new over sampling objects to the grid and grid pixelization of the dataset.

        This method is used to change the over sampling of the grid and grid pixelization, for example when the
        user wishes to perform over sampling with a higher sub grid size or with an iterative over sampling strategy.

        The `grid` and grid_pixelization` are cached properties which after use are stored in memory for efficiency.
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
            The new over sampling object to apply to the grid.
        over_sampling_non_uniform
            The over sampling scheme when the grid input into a function is not a uniform grid. This is used
            by **PyAutoLens** when the grid has been deflected and ray-traced and therefore some of the default
            over sampling schemes are not appropriate.
        over_sampling_pixelization
            The new over sampling object to apply to the grid pixelization.
        """

        if over_sampling is not None:
            self.over_sampling = over_sampling
            try:
                del self.__dict__["grid"]
            except KeyError:
                pass

        if over_sampling_non_uniform is not None:
            self.over_sampling_non_uniform = over_sampling_non_uniform
            try:
                del self.__dict__["grid"]
            except KeyError:
                pass
            try:
                del self.__dict__["over_sampler_non_uniform"]
            except KeyError:
                pass

        if over_sampling_pixelization is not None:
            self.over_sampling_pixelization = over_sampling_pixelization
            try:
                del self.__dict__["grid_pixelization"]
            except KeyError:
                pass
            try:
                del self.__dict__["over_sampler_pixelization"]
            except KeyError:
                pass
            try:
                del self.__dict__["border_relocator"]
            except KeyError:
                pass

        return self
