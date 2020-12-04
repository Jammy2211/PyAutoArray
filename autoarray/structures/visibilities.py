import logging

import numpy as np
import pylops

from autoarray.structures import grids
from autoarray.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractVisibilities(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(cls, visibilities, *args, **kwargs):
        """
        A collection of (real, imag) visibilities which are used to represent the data in an `Interferometer` dataset.

        The (real, imag) visibilities are stored as a 1D complex NumPy array of shape [total_visibilities]. These can
        be mapped to a 2D real NumPy array of shape [total_visibilities, 2] and a `GridIrregular` data structure
        which is used for plotting the visibilities in 2D in the complex plane.

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The vectors input to this function can have any of the following forms (they will be converted to the 1D
        complex NumPy array structure and can be converted back using the object's properties):

        [1.0+1.0j, 2.0+2.0j]
        [[1.0, 1.0], [2.0, 2.0]]

        Parameters
        ----------
        visibilities : np.ndarray
            The (real, imag) visibilities values.
        """

        if type(visibilities) is list:
            visibilities = np.asarray(visibilities)

        if "float" in str(visibilities.dtype):
            if visibilities.shape[1] == 2:
                visibilities = (
                    np.apply_along_axis(lambda args: [complex(*args)], 1, visibilities)
                    .astype("complex128")
                    .ravel()
                )

        return visibilities.view(cls)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractVisibilities, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(AbstractVisibilities, self).__setstate__(state[0:-1])

    @property
    def in_1d(self):
        return self

    @property
    def in_array(self):
        """
        Returns the 1D complex NumPy array of values with shape [total_visibilities] as a NumPy float array of
        shape [total_visibilities, 2].
        """
        return np.stack((np.real(self), np.imag(self)), axis=-1)

    @property
    def in_grid(self):
        """Returns the 1D complex NumPy array of values as an irregular grid."""
        return grids.GridIrregular(grid=self.in_array)

    @property
    def shape_1d(self):
        return self.shape[0]

    @property
    @array_util.Memoizer()
    def amplitudes(self):
        return np.sqrt(np.square(self.real) + np.square(self.imag))

    @property
    @array_util.Memoizer()
    def phases(self):
        return np.arctan2(self.imag, self.real)

    def output_to_fits(self, file_path, overwrite=False):
        """
        Output the visibilities to a .fits file.

        The complex values are converted to a 2D NumPy float array of shape [total_visiblities, 2] before being
        written to `.fits` format via the `in_array` property.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        overwrite : bool
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_util.numpy_array_2d_to_fits(
            array_2d=self.in_array, file_path=file_path, overwrite=overwrite
        )

    @property
    def scaled_maxima(self):
        """The maximum values of the visibilities if they are treated as a 2D grid in the complex plane."""
        return (np.max(self.real), np.max(self.imag))

    @property
    def scaled_minima(self):
        """The minimum values of the visibilities if they are treated as a 2D grid in the complex plane."""
        return (np.min(self.real), np.min(self.imag))

    @property
    def extent(self):
        """The extent of the visibilities if they are treated as a 2D grid in the complex plane."""
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


class Visibilities(AbstractVisibilities):
    @classmethod
    def manual_1d(cls, visibilities):
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) by inputting (real, imag) values as a 1D complex
        NumPy array or 2D NumPy float array or list, for example:

        visibilities=np.array([1.0+1.0j, 2.0+2.0j, 3.0+3.0j, 4.0+4.0j])
        visibilities=np.array([[1.0+1.0], [2.0+2.0], [3.0+3.0], [4.0+4.0]])
        visibilities=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        Parameters
        ----------
        visibilities : np.ndarray
            The (real, imag) visibilities values.
        """
        return cls(visibilities=visibilities)

    @classmethod
    def full(cls, fill_value, shape_1d):
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) where all (real, imag) values are filled with an
        input fill value, analogous to the method numpy ndarray.full.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_2d.

        Parameters
        ----------
        fill_value : float
            The value all real and imaginary array elements are filled with.
        shape_1d : int
            The 1D shape of output visibilities.
        """
        return cls.manual_1d(
            visibilities=np.full(
                fill_value=fill_value + fill_value * 1j, shape=(shape_1d[0],)
            )
        )

    @classmethod
    def ones(cls, shape_1d):
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) where all (real, imag) values are filled with ones,
        analogous to the method np.ones().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_2d.

        Parameters
        ----------
        shape_1d : int
            The 1D shape of output visibilities.
        """
        return cls.full(fill_value=1.0, shape_1d=shape_1d)

    @classmethod
    def zeros(cls, shape_1d):
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) where all (real, imag) values are filled with zeros,
        analogous to the method np.zeros().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_2d.

        Parameters
        ----------
        shape_1d : int
            The 1D shape of output visibilities.
        """
        return cls.full(fill_value=0.0, shape_1d=shape_1d)

    @classmethod
    def from_fits(cls, file_path, hdu):
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) by loading the(real, imag) values from a .fits file.

        The `.fits` file stores these values as a real set of values of shape [total_visibilities, 2] which are
        converted to a 1d complex NumPy array.

        Parameters
        ----------
        file_path : str
            The path the file is loaded from, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        hdu : int
            The Header-Data Unit of the .fits file the visibilitiy data is loaded from.
        """
        visibilities_1d = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu
        )
        return cls.manual_1d(visibilities=visibilities_1d)


class VisibilitiesNoiseMap(Visibilities):

    # noinspection PyUnusedLocal
    def __new__(cls, visibilities, *args, **kwargs):
        """
        A collection of (real, imag) visibilities noise-map values which are used to represent the noise-map in
        an `Interferometer` dataset.

        This data structure behaves the same as the `Visibilities` structure (see `AbstractVisibilities.__new__`). The
        only difference is that it includes a `WeightOperator` used by `Inversion`'s which use `LinearOperators` and
        the library `PyLops` to fit `Interferometer` data.

        Parameters
        ----------
        visibilities : np.ndarray
            The (real, imag) visibilities values.
        """

        if type(visibilities) is list:
            visibilities = np.asarray(visibilities)

        if "float" in str(visibilities.dtype):
            if visibilities.shape[1] == 2:
                visibilities = (
                    np.apply_along_axis(lambda args: [complex(*args)], 1, visibilities)
                    .astype("complex128")
                    .ravel()
                )

        obj = super(VisibilitiesNoiseMap, cls).__new__(
            cls=cls, visibilities=visibilities
        )

        weights = np.reciprocal(np.real(visibilities)) + 1j * np.reciprocal(
            np.imag(visibilities)
        )

        obj.Wop = pylops.Diagonal(np.real(weights.ravel()), dtype="complex128")

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "Wop"):
            self.Wop = obj.Wop
