import copy
import numpy as np
from typing import List, Optional, Type, Union
import warnings

from autoconf import cached_property
from autoconf import conf

from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.iterate_2d import Grid2DIterate
from autoarray.structures.vectors.uniform import VectorYX2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc


class AbstractWTilde:
    def __init__(self, curvature_preload, noise_map_value):
        """
        Packages together all derived data quantities necessary to fit `data (e.g. `Imaging`, Interferometer`) using
        an ` Inversion` via the w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing
        operations like blurring or a Fourier transform.

        Parameters
        ----------
        curvature_preload
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        noise_map_value
            The first value of the noise-map used to construct the curvature preload, which is used as a sanity
            check when performing the inversion to ensure the preload corresponds to the data being fitted.
        """
        self.curvature_preload = curvature_preload
        self.noise_map_value = noise_map_value

    def check_noise_map(self, noise_map):

        if noise_map[0] != self.noise_map_value:
            raise exc.InversionException(
                "The preloaded values of WTildeImaging are not consistent with the noise-map passed to them, thus "
                "they cannot be used for the inversion."
                ""
                f"The value of the noise map is {noise_map[0]} whereas in WTildeImaging it is {self.noise_map_value}"
                ""
                "Update WTildeImaging or do not use the w_tilde formalism to perform the Inversion."
            )


def grid_via_grid_class_from(
    mask: Union[Mask1D, Mask2D],
    grid_class: Union[Type[Grid1D], Type[Grid2D]],
    fractional_accuracy: float,
    relative_accuracy: Optional[float],
    sub_steps: List[int],
) -> Optional[Union[Grid1D, Grid2D, Grid2DIterate]]:

    if mask.pixel_scales is None:
        return None

    if grid_class is None:
        if isinstance(mask, Mask1D):
            grid_class = Grid1D
        elif isinstance(mask, Mask2D):
            grid_class = Grid2D

    if grid_class is Grid1D:

        return Grid1D.from_mask(mask=mask)

    if grid_class is Grid2D:

        return Grid2D.from_mask(mask=mask)

    elif grid_class is Grid2DIterate:

        return Grid2DIterate.from_mask(
            mask=mask,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
        )


class AbstractSettingsDataset:
    def __init__(
        self,
        grid_class: Optional[Union[Type[Grid1D], Type[Grid2D]]] = None,
        grid_pixelization_class: Optional[Union[Type[Grid1D], Type[Grid2D]]] = None,
        sub_size: int = 2,
        sub_size_pixelization: int = 2,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: Optional[List[int]] = None,
    ):
        """
        A dataset is a collection of data structures (e.g. the data, noise-map, PSF), a mask, grid, convolver
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        grid_class : ag.Grid2D
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D` and
            `Grid2DIterate` (see the `Grid2D` documentation for a description of these options).
        grid_pixelization_class : ag.Grid2D
            The type of grid used to create the grid that maps the `Inversion` source pixels to the data's image-pixels.
            The options are `Grid2D` and `Grid2DIterate` (see the `Grid2D` documentation for a
            description of these options).
        sub_size
            If the grid and / or grid_pixelization use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the fractional accuracy it
            uses when evaluating functions, where the fraction accuracy is the ratio of the values computed using
            two grids at a higher and lower sub-grid size.
        relative_accuracy
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the relative accuracy it
            uses when evaluating functions, where the relative accuracy is the absolute difference of the values
            computed using two grids at a higher and lower sub-grid size.
        sub_steps : [int]
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        """

        self.grid_class = grid_class
        self.grid_pixelization_class = grid_pixelization_class
        self.sub_size = sub_size
        self.sub_size_pixelization = sub_size_pixelization
        self.fractional_accuracy = fractional_accuracy
        self.relative_accuracy = relative_accuracy

        if sub_steps is None:
            sub_steps = [2, 4, 8, 16]

        self.sub_steps = sub_steps

    def grid_from(self, mask) -> Union[Grid1D, Grid2D]:

        return grid_via_grid_class_from(
            mask=mask,
            grid_class=self.grid_class,
            fractional_accuracy=self.fractional_accuracy,
            relative_accuracy=self.relative_accuracy,
            sub_steps=self.sub_steps,
        )

    def grid_pixelization_from(self, mask) -> Union[Grid1D, Grid2D]:

        return grid_via_grid_class_from(
            mask=mask,
            grid_class=self.grid_pixelization_class,
            fractional_accuracy=self.fractional_accuracy,
            relative_accuracy=self.relative_accuracy,
            sub_steps=self.sub_steps,
        )


class AbstractDataset:
    def __init__(
        self,
        data: Union[Array1D, Array2D, VectorYX2D, Visibilities],
        noise_map: Union[Array1D, Array2D, VectorYX2D, VisibilitiesNoiseMap],
        noise_covariance_matrix: Optional[np.ndarray] = None,
        settings=AbstractSettingsDataset(),
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
        self.noise_map = noise_map
        self.settings = settings

        mask = self.mask

        self.noise_covariance_matrix = noise_covariance_matrix

        if noise_map is None and noise_covariance_matrix is not None:

            logger.info(
                """
                No noise map was input into the Imaging class, but a `noise_covariance_matrix` was.

                Using the diagonal of the `noise_covariance_matrix` to create the `noise_map`. 

                This `noise-map` is used only for visualization where it is not appropriate to plot covariance.
                """
            )

            noise_map = Array2D.manual_slim(
                array=np.diag(noise_covariance_matrix),
                shape_native=image.shape_native,
                pixel_scales=image.shape_native,
            )

        elif noise_map is None and noise_covariance_matrix is None:

            raise exc.DatasetException(
                """
                No noise map or noise_covariance_matrix was passed to the Imaging object.
                """
            )

        self.noise_map = noise_map

        if conf.instance["general"]["structures"]["use_dataset_grids"]:

            mask_grid = mask.mask_new_sub_size_from(
                mask=mask, sub_size=settings.sub_size
            )
            self.grid = settings.grid_from(mask=mask_grid)

            mask_inversion = mask.mask_new_sub_size_from(
                mask=mask, sub_size=settings.sub_size_pixelization
            )

            self.grid_pixelization = settings.grid_pixelization_from(
                mask=mask_inversion
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
    def inverse_noise_map(self) -> Union[Array1D, Array2D]:
        return 1.0 / self.noise_map

    @property
    def signal_to_noise_map(self) -> Union[Array1D, Array2D]:
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

    @property
    def absolute_signal_to_noise_map(self) -> Union[Array1D, Array2D]:
        """
        The estimated absolute_signal-to-noise_maps mappers of the image.
        """
        return np.divide(np.abs(self.data), self.noise_map)

    @property
    def absolute_signal_to_noise_max(self) -> float:
        """
        The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers.
        """
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self) -> Union[Array1D, Array2D]:
        """
        The potential chi-squared-map of the imaging data_type. This represents how much each pixel can contribute to
        the chi-squared-map, assuming the model fails to fit it at all (e.g. model value = 0.0).
        """
        return np.square(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_max(self) -> float:
        """
        The maximum value of the potential chi-squared-map.
        """
        return np.max(self.potential_chi_squared_map)

    @cached_property
    def noise_covariance_matrix_inv(self) -> Grid2D:
        """
        Returns the inverse of the noise covariance matrix, which is used when computing a chi-squared which accounts
        for covariance via a fit.

        Returns
        -------
        The inverse of the noise covariance matrix.
        """
        return np.linalg.inv(self.noise_covariance_matrix)

    def trimmed_after_convolution_from(self, kernel_shape) -> "AbstractDataset":

        imaging = copy.copy(self)

        imaging.data = imaging.data.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )
        imaging.noise_map = imaging.noise_map.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )

        return imaging
