import pickle

import numpy as np
import copy

from autoconf import conf
from autoarray.structures import arrays, grids
from autoarray.mask import mask as msk


def grid_from_mask_and_grid_class(
    mask, grid_class, fractional_accuracy, sub_steps, pixel_scales_interp
):
    if mask.pixel_scales is None:
        return None

    if grid_class is grids.Grid:

        return grids.Grid.from_mask(mask=mask)

    elif grid_class is grids.GridIterate:

        return grids.GridIterate.from_mask(
            mask=mask, fractional_accuracy=fractional_accuracy, sub_steps=sub_steps
        )

    elif grid_class is grids.GridInterpolate:

        return grids.GridInterpolate.from_mask(
            mask=mask, pixel_scales_interp=pixel_scales_interp
        )


class AbstractDataset:
    def __init__(self, data, noise_map, positions=None, name=None):
        """A collection of abstract 2D for different data_type classes (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        data : arrays.Array
            The array of the image data, in units of electrons per second.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the image.
        noise_map : ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        self.data = data
        self.noise_map = noise_map
        self.positions = positions
        self._name = name if name is not None else "dataset"

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def load(cls, filename) -> "AbstractDataset":
        """
        Load the dataset at the specified filename

        Parameters
        ----------
        filename
            The filename containing the dataset

        Returns
        -------
        The dataset
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def mapping(self):
        return self.data.mask.mapping

    @property
    def geometry(self):
        return self.data.mask.geometry

    @property
    def inverse_noise_map(self):
        return 1.0 / self.noise_map

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)

    @property
    def absolute_signal_to_noise_map(self):
        """The estimated absolute_signal-to-noise_maps mappers of the image."""
        return arrays.Array(
            array=np.divide(np.abs(self.data), self.noise_map), mask=self.data.mask
        )

    @property
    def absolute_signal_to_noise_max(self):
        """The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self):
        """The potential chi-squared-map of the imaging data_type. This represents how much each pixel can contribute to \
        the chi-squared-map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return arrays.Array(
            array=np.square(self.absolute_signal_to_noise_map), mask=self.data.mask
        )

    @property
    def potential_chi_squared_max(self):
        """The maximum value of the potential chi-squared-map"""
        return np.max(self.potential_chi_squared_map)

    def modify_noise_map(self, noise_map):

        imaging = copy.deepcopy(self)

        imaging.noise_map = noise_map

        return imaging


class AbstractSettingsMaskedDataset:
    def __init__(
        self,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        grid_class : ag.Grid
            The type of grid used to create the image from the *Galaxy* and *Plane*. The options are *Grid*,
            *GridIterate* and *GridInterpolate* (see the *Grids* documentation for a description of these options).
        grid_inversion_class : ag.Grid
            The type of grid used to create the grid that maps the *Inversion* source pixels to the data's image-pixels.
            The options are *Grid*, *GridIterate* and *GridInterpolate* (see the *Grids* documentation for a
            description of these options).
        sub_size : int
            If the grid and / or grid_inversion use a *Grid*, this sets the sub-size used by the *Grid*.
        fractional_accuracy : float
            If the grid and / or grid_inversion use a *GridIterate*, this sets the fractional accuracy it
            uses when evaluating functions.
        sub_steps : [int]
            If the grid and / or grid_inversion use a *GridIterate*, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        pixel_scales_interp : float or (float, float)
            If the grid and / or grid_inversion use a *GridInterpolate*, this sets the resolution of the interpolation
            grid.
        signal_to_noise_limit : float
            If input, the dataset's noise-map is rescaled such that no pixel has a signal-to-noise above the
            signa to noise limit.
        """

        self.grid_class = grid_class
        self.grid_inversion_class = grid_inversion_class
        self.sub_size = sub_size
        self.fractional_accuracy = fractional_accuracy

        if sub_steps is None:
            sub_steps = [2, 4, 8, 16]

        self.sub_steps = sub_steps
        self.pixel_scales_interp = pixel_scales_interp
        self.signal_to_noise_limit = signal_to_noise_limit

    def grid_from_mask(self, mask):
        return grid_from_mask_and_grid_class(
            mask=mask,
            grid_class=self.grid_class,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            pixel_scales_interp=self.pixel_scales_interp,
        )

    def grid_inversion_from_mask(self, mask):
        return grid_from_mask_and_grid_class(
            mask=mask,
            grid_class=self.grid_inversion_class,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            pixel_scales_interp=self.pixel_scales_interp,
        )

    @property
    def tag_no_inversion(self):
        return self.grid_tag_no_inversion + self.signal_to_noise_limit_tag

    @property
    def tag_with_inversion(self):
        return +self.grid_tag_with_inversion + self.signal_to_noise_limit_tag

    @property
    def grid_tag_no_inversion(self):
        """Generate a tag describing the the grid and grid_inversions used by the phase.

        This assumes both grids were used in the analysis.
        """

        return (
            "__"
            + conf.instance.tag.get("dataset", "grid")
            + "_"
            + self.grid_sub_size_tag
            + self.grid_fractional_accuracy_tag
            + self.grid_pixel_scales_interp_tag
        )

    @property
    def grid_tag_with_inversion(self):
        """Generate a tag describing the the grid and grid_inversions used by the phase.

        This assumes both grids were used in the analysis.
        """
        return (
            "__"
            + conf.instance.tag.get("dataset", "grid")
            + "_"
            + self.grid_sub_size_tag
            + self.grid_fractional_accuracy_tag
            + self.grid_pixel_scales_interp_tag
            + "_"
            + conf.instance.tag.get("dataset", "grid_inversion")
            + "_"
            + self.grid_inversion_sub_size_tag
            + self.grid_inversion_fractional_accuracy_tag
            + self.grid_inversion_pixel_scales_interp_tag
        )

    @property
    def grid_sub_size_tag(self):
        """Generate a sub-size tag, to customize phase names based on the sub-grid size used, of the Grid class.

        This changes the phase settings folder as follows:

        sub_size = None -> settings
        sub_size = 1 -> settings_sub_size_2
        sub_size = 4 -> settings_sub_size_4
        """
        if not self.grid_class is grids.Grid:
            return ""
        return conf.instance.tag.get("dataset", "sub_size") + "_" + str(self.sub_size)

    @property
    def grid_fractional_accuracy_tag(self):
        """Generate a fractional accuracy tag, to customize phase names based on the fractional accuracy of the
        GridIterate class.

        This changes the phase settings folder as follows:

        fraction_accuracy = 0.5 -> settings__facc_0.5
        fractional_accuracy = 0.999999 = 4 -> settings__facc_0.999999
        """
        if not self.grid_class is grids.GridIterate:
            return ""
        return (
            conf.instance.tag.get("dataset", "fractional_accuracy")
            + "_"
            + str(self.fractional_accuracy)
        )

    @property
    def grid_pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the
        GridInterpolate.

        This changes the phase settings folder as follows:

        pixel_scales_interp = None -> settings
        pixel_scales_interp = 0.1 -> settings___grid_interp_0.1
        """
        if not self.grid_class is grids.GridInterpolate:
            return ""
        if self.pixel_scales_interp is None:
            return ""
        return conf.instance.tag.get(
            "dataset", "pixel_scales_interp"
        ) + "_{0:.3f}".format(self.pixel_scales_interp)

    @property
    def grid_inversion_sub_size_tag(self):
        """Generate a sub-size tag, to customize phase names based on the sub-grid size used, of the Grid class.

        This changes the phase settings folder as follows:

        sub_size = None -> settings
        sub_size = 1 -> settings__grid_sub_size_2
        sub_size = 4 -> settings__grid_inv_sub_size_4
        """
        if not self.grid_inversion_class is grids.Grid:
            return ""
        return conf.instance.tag.get("dataset", "sub_size") + "_" + str(self.sub_size)

    @property
    def grid_inversion_fractional_accuracy_tag(self):
        """Generate a fractional accuracy tag, to customize phase names based on the fractional accuracy of the
        GridIterate class.

        This changes the phase settings folder as follows:

        fraction_accuracy = 0.5 -> settings__facc_0.5
        fractional_accuracy = 0.999999 = 4 -> settings__facc_0.999999
        """
        if not self.grid_inversion_class is grids.GridIterate:
            return ""
        return (
            conf.instance.tag.get("dataset", "fractional_accuracy")
            + "_"
            + str(self.fractional_accuracy)
        )

    @property
    def grid_inversion_pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the
        GridInterpolate.

        This changes the phase settings folder as follows:

        pixel_scales_interp = None -> settings
        pixel_scales_interp = 0.1 -> settings___grid_interp_0.1
        """
        if not self.grid_inversion_class is grids.GridInterpolate:
            return ""
        if self.pixel_scales_interp is None:
            return ""
        return conf.instance.tag.get(
            "dataset", "pixel_scales_interp"
        ) + "_{0:.3f}".format(self.pixel_scales_interp)

    @property
    def signal_to_noise_limit_tag(self):
        """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
        the dataset being fitted.

        This changes the phase settings folder as follows:

        signal_to_noise_limit = None -> settings
        signal_to_noise_limit = 2 -> settings_snr_2
        signal_to_noise_limit = 10 -> settings_snr_10
        """
        if self.signal_to_noise_limit is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("dataset", "signal_to_noise_limit")
            + "_"
            + str(self.signal_to_noise_limit)
        )


class AbstractMaskedDataset:
    def __init__(self, dataset, mask, settings=AbstractSettingsMaskedDataset()):

        if mask.sub_size != settings.sub_size:
            mask = msk.Mask.manual(
                mask=mask,
                pixel_scales=mask.pixel_scales,
                sub_size=settings.sub_size,
                origin=mask.origin,
            )

        if settings.signal_to_noise_limit is not None:

            dataset = dataset.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=settings.signal_to_noise_limit
            )

        self.dataset = dataset
        self.mask = mask
        self.settings = settings

        self.grid = settings.grid_from_mask(mask=mask)

        self.grid_inversion = settings.grid_inversion_from_mask(mask=mask)

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def positions(self):
        return self.dataset.positions

    def modify_noise_map(self, noise_map):

        masked_imaging = copy.deepcopy(self)

        masked_imaging.noise_map = noise_map

        return masked_imaging
