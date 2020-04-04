import pickle

import numpy as np

from autoarray.structures import arrays, grids


class AbstractDataset:
    def __init__(self, data, noise_map, name=None, metadata=None):
        """A collection of abstract 2D for different data_type classes (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        data : arrays.Array
            The array of the image data, in units of electrons per second.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the image.
        noise_map : NoiseMap | float | ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        self.data = data
        self.noise_map = noise_map
        self._name = name
        self.metadata = dict() if metadata is None else metadata

    @property
    def name(self) -> str:
        return self._name

    def save(self, directory: str):
        """
        Save this instance as a pickle with the dataset name in the given directory.

        Parameters
        ----------
        directory
            The directory to save into
        """
        with open(f"{directory}/{self.name}.pickle", "wb") as f:
            pickle.dump(self, f)

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
        return self.mapping.array_stored_1d_from_array_1d(
            array_1d=np.divide(np.abs(self.data), self.noise_map)
        )

    @property
    def absolute_signal_to_noise_max(self):
        """The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self):
        """The potential chi-squared map of the imaging data_type. This represents how much each pixel can contribute to \
        the chi-squared map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return self.mapping.array_stored_1d_from_array_1d(
            array_1d=np.square(self.absolute_signal_to_noise_map)
        )

    @property
    def potential_chi_squared_max(self):
        """The maximum value of the potential chi-squared map"""
        return np.max(self.potential_chi_squared_map)


class AbstractMaskedDataset:
    def __init__(
        self,
        mask,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):

        self.mask = mask

        ### GRIDS ###

        if mask.pixel_scales is not None:

            self.grid = grids.MaskedGrid.from_mask(mask=mask)

            if pixel_scale_interpolation_grid is not None:

                self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

                self.grid = self.grid.new_grid_with_interpolator(
                    pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
                )

        else:

            self.grid = None

        self.inversion_pixel_limit = inversion_pixel_limit
        self.inversion_uses_border = inversion_uses_border
