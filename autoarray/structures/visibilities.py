import logging
import numpy as np
from typing import List, Tuple, Union

from autoconf import cached_property

from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.structures.arrays import array_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractVisibilities(Structure):

    # noinspection PyUnusedLocal
    def __new__(cls, visibilities: Union[np.ndarray, List[complex]], *args, **kwargs):
        """
        A collection of (real, imag) visibilities which are used to represent the data in an `Interferometer` dataset.

        The (real, imag) visibilities are stored as a 1D complex NumPy array of shape [total_visibilities]. These can
        be mapped to a 2D real NumPy array of shape [total_visibilities, 2] and a `Grid2DIrregular` data structure
        which is used for plotting the visibilities in 2D in the complex plane.

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The vectors input to this function can have any of the following forms (they will be converted to the 1D
        complex NumPy array structure and can be converted back using the object's properties):

        [1.0+1.0j, 2.0+2.0j]
        [[1.0, 1.0], [2.0, 2.0]]

        Parameters
        ----------
        visibilities
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

        obj = visibilities.view(cls)

        obj.ordered_1d = np.concatenate(
            (np.real(visibilities), np.imag(visibilities)), axis=0
        )

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "ordered_1d"):
            self.ordered_1d = obj.ordered_1d

    @property
    def slim(self) -> "AbstractVisibilities":
        return self

    @property
    def in_array(self) -> np.ndarray:
        """
        Returns the 1D complex NumPy array of values with shape [total_visibilities] as a NumPy float array of
        shape [total_visibilities, 2].
        """
        return np.stack((np.real(self), np.imag(self)), axis=-1)

    @property
    def in_grid(self) -> Grid2DIrregular:
        """
        Returns the 1D complex NumPy array of values as an irregular grid.
        """
        return Grid2DIrregular(grid=self.in_array)

    @property
    def shape_slim(self) -> int:
        return self.shape[0]

    @cached_property
    def amplitudes(self) -> np.ndarray:
        return np.sqrt(np.square(self.real) + np.square(self.imag))

    @cached_property
    def phases(self) -> np.ndarray:
        return np.arctan2(self.imag, self.real)

    def output_to_fits(self, file_path: str, overwrite: bool = False):
        """
        Output the visibilities to a .fits file.

        The complex values are converted to a 2D NumPy float array of shape [total_visiblities, 2] before being
        written to `.fits` format via the `in_array` property.

        Parameters
        ----------
        file_path
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        overwrite
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.in_array, file_path=file_path, overwrite=overwrite
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        """
        The maximum values of the visibilities if they are treated as a 2D grid in the complex plane.
        """
        return np.max(self.real), np.max(self.imag)

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        """
        The minimum values of the visibilities if they are treated as a 2D grid in the complex plane.
        """
        return np.min(self.real), np.min(self.imag)

    @property
    def extent(self) -> np.ndarray:
        """
        The extent of the visibilities if they are treated as a 2D grid in the complex plane.
        """
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


class Visibilities(AbstractVisibilities):
    @classmethod
    def manual_slim(
        cls, visibilities: Union[np.ndarray, List[complex]]
    ) -> "Visibilities":
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) by inputting (real, imag) values as a 1D complex
        NumPy array or 2D NumPy float array or list, for example:

        visibilities=np.array([1.0+1.0j, 2.0+2.0j, 3.0+3.0j, 4.0+4.0j])
        visibilities=np.array([[1.0+1.0], [2.0+2.0], [3.0+3.0], [4.0+4.0]])
        visibilities=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        Parameters
        ----------
        visibilities
            The (real, imag) visibilities values.
        """
        return cls(visibilities)

    @classmethod
    def full(cls, fill_value: float, shape_slim: Tuple[int]) -> "Visibilities":
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) where all (real, imag) values are filled with an
        input fill value, analogous to the method numpy ndarray.full.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all real and imaginary array elements are filled with.
        shape_slim
            The 1D shape of output visibilities.
        """
        return cls.manual_slim(
            visibilities=np.full(
                fill_value=fill_value + fill_value * 1j, shape=(shape_slim[0],)
            )
        )

    @classmethod
    def ones(cls, shape_slim: Tuple[int]) -> "Visibilities":
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) where all (real, imag) values are filled with ones,
        analogous to the method np.ones().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        shape_slim
            The 1D shape of output visibilities.
        """
        return cls.full(fill_value=1.0, shape_slim=shape_slim)

    @classmethod
    def zeros(cls, shape_slim: Tuple[int]) -> "Visibilities":
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) where all (real, imag) values are filled with zeros,
        analogous to the method np.zeros().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        shape_slim
            The 1D shape of output visibilities.
        """
        return cls.full(fill_value=0.0, shape_slim=shape_slim)

    @classmethod
    def from_fits(cls, file_path: str, hdu: int) -> "Visibilities":
        """
        Create `Visibilities` (see `AbstractVisibilities.__new__`) by loading the(real, imag) values from a .fits file.

        The `.fits` file stores these values as a real set of values of shape [total_visibilities, 2] which are
        converted to a 1d complex NumPy array.

        Parameters
        ----------
        file_path : str
            The path the file is loaded from, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        hdu
            The Header-Data Unit of the .fits file the visibilitiy data is loaded from.
        """
        visibilities_1d = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=hdu
        )
        return cls.manual_slim(visibilities=visibilities_1d)


class VisibilitiesNoiseMap(Visibilities):

    # noinspection PyUnusedLocal
    def __new__(cls, visibilities: Union[np.ndarray, List[complex]], *args, **kwargs):
        """
        A collection of (real, imag) visibilities noise-map values which are used to represent the noise-map in
        an `Interferometer` dataset.

        This data structure behaves the same as the `Visibilities` structure (see `AbstractVisibilities.__new__`). The
        only difference is that it includes a `WeightOperator` used by `LEq`'s which use `LinearOperators` and
        the library `PyLops` to fit `Interferometer` data.

        Parameters
        ----------
        visibilities
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

        obj.ordered_1d = np.concatenate(
            (np.real(visibilities), np.imag(visibilities)), axis=0
        )

        weight_list = 1.0 / obj.in_array ** 2.0

        obj.weight_list_ordered_1d = np.concatenate(
            (weight_list[:, 0], weight_list[:, 1]), axis=0
        )

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "ordered_1d"):
            self.ordered_1d = obj.ordered_1d

        if hasattr(obj, "weight_list_ordered_1d"):
            self.weight_list_ordered_1d = obj.weight_list_ordered_1d
