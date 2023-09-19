import copy
import logging
import numpy as np
from typing import Optional, Union
import warnings

from autoconf import cached_property
from autoconf import conf

from autoarray.dataset.abstract.settings import AbstractSettingsDataset
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc

logger = logging.getLogger(__name__)


class AbstractDataset:
    def __init__(
        self,
        data: Structure,
        noise_map: Structure,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        settings: AbstractSettingsDataset = AbstractSettingsDataset(),
    ):
        """
        A collection of abstract data structures for different types of data (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        dataucture
            The array of the image data, in units of electrons per second.
        noise_mapucture
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """

        self.data = data
        self.settings = settings

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

        mask_grid = self.mask.mask_new_sub_size_from(
            mask=self.mask, sub_size=self.settings.sub_size
        )
        return self.settings.grid_from(mask=mask_grid)

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
        mask_inversion = self.mask.mask_new_sub_size_from(
            mask=self.mask, sub_size=self.settings.sub_size_pixelization
        )

        return self.settings.grid_pixelization_from(mask=mask_inversion)

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

        Warnings airse when masked native noise-maps are used, whose masked entries are given values of 0.0. We
        uses the warnings module to surpress these RunTimeWarnings.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = np.divide(self.data, self.noise_map)
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
