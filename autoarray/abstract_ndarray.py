from __future__ import annotations

from copy import copy

from abc import ABC
from abc import abstractmethod
import numpy as np

from autoarray.numpy_wrapper import numpy as npw, register_pytree_node, Array

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.structures.abstract_structure import Structure

from autoarray.structures.arrays import array_2d_util
from autoconf import conf


def to_new_array(func):
    def wrapper(self, *args, **kwargs):
        return self.with_new_array(func(self, *args, **kwargs))

    return wrapper


def unwrap_array(func):
    def wrapper(self, other):
        try:
            return func(self, other.array)
        except AttributeError:
            return func(self, other)

    return wrapper


class AbstractNDArray(ABC):
    def __init__(self, array):
        while isinstance(array, AbstractNDArray):
            array = array.array
        self._array = array
        try:
            register_pytree_node(
                type(self),
                self.instance_flatten,
                self.instance_unflatten,
            )
        except ValueError:
            pass

    __no_flatten__ = ()

    @classmethod
    def instance_flatten(cls, instance):
        keys, values = zip(
            *sorted(
                {
                    key: value
                    for key, value in instance.__dict__.items()
                    if key not in cls.__no_flatten__
                }.items()
            )
        )
        return values, keys

    @staticmethod
    def flip_hdu_for_ds9(values):
        if conf.instance["general"]["fits"]["flip_for_ds9"]:
            return np.flipud(values)
        return values

    @classmethod
    def instance_unflatten(cls, aux_data, children):
        instance = cls.__new__(cls)
        for key, value in zip(aux_data, children[1:]):
            setattr(instance, key, value)
        return instance

    def with_new_array(self, array):
        new_array = self.copy()
        new_array._array = array
        return new_array

    def copy(self):
        new = copy(self)
        new._array = self._array.copy()
        return new

    def __iter__(self):
        return iter(self._array)

    @to_new_array
    def sqrt(self):
        return np.sqrt(self._array)

    @property
    def array(self):
        return self._array

    @unwrap_array
    def __lt__(self, other):
        return self._array < other

    @unwrap_array
    def __le__(self, other):
        return self._array <= other

    @unwrap_array
    def __gt__(self, other):
        return self._array > other

    @unwrap_array
    def __ge__(self, other):
        return self._array >= other

    @unwrap_array
    def __eq__(self, other):
        return self._array == other

    @to_new_array
    @unwrap_array
    def __pow__(self, other):
        return self._array**other

    @to_new_array
    @unwrap_array
    def __add__(self, other):
        return self._array + other

    @to_new_array
    @unwrap_array
    def __radd__(self, other):
        return other + self._array

    @to_new_array
    @unwrap_array
    def __sub__(self, other):
        return self._array - other

    @to_new_array
    @unwrap_array
    def __rsub__(self, other):
        return other - self._array

    @unwrap_array
    def __ne__(self, other):
        return self._array != other

    @to_new_array
    @unwrap_array
    def __mul__(self, other):
        return self._array * other

    @to_new_array
    @unwrap_array
    def __rmul__(self, other):
        return other * self._array

    @to_new_array
    def __neg__(self):
        return -self._array

    def __invert__(self):
        return ~self._array

    def __divmod__(self, other):
        return divmod(self._array, other)

    def __rdivmod__(self, other):
        return divmod(other, self._array)

    @to_new_array
    @unwrap_array
    def __truediv__(self, other):
        return self._array / other

    @to_new_array
    @unwrap_array
    def __rtruediv__(self, other):
        return other / self._array

    @to_new_array
    def __abs__(self):
        return abs(self._array)

    def sum(self, *args, **kwargs):
        return self._array.sum(*args, **kwargs)

    def __float__(self):
        return float(self._array)

    @property
    @abstractmethod
    def native(self) -> Structure:
        """
        Returns the data structure in its `native` format which contains all unmaksed values to the native dimensions.
        """

    def output_to_fits(self, file_path: str, overwrite: bool = False):
        """
        Output the grid to a .fits file.

        Parameters
        ----------
        file_path
            The path the file is output to, including the filename and the .fits extension, e.g. '/path/to/filename.fits'
        overwrite
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.native.array, file_path=file_path, overwrite=overwrite
        )

    @property
    def shape(self):
        try:
            return self._array.shape
        except AttributeError:
            return ()

    @property
    def size(self):
        return self._array.size

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self):
        return self._array.ndim

    def max(self, *args, **kwargs):
        return self._array.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self._array.min(*args, **kwargs)

    @to_new_array
    def reshape(self, *args, **kwargs):
        return self._array.reshape(*args, **kwargs)

    def __getattr__(self, item):
        if item != "__setstate__":
            try:
                return getattr(self._array, item)
            except AttributeError:
                pass
        raise AttributeError(
            f"{self.__class__.__name__} does not have attribute {item}"
        )

    def __getitem__(self, item):
        result = self._array[item]
        if isinstance(item, slice):
            result = self.with_new_array(result)
        if isinstance(result, np.ndarray):
            result = self.with_new_array(result)
        return result

    def __setitem__(self, key, value):
        if isinstance(key, (np.ndarray, AbstractNDArray, Array)):
            self._array = npw.where(key, value, self._array)
        else:
            self._array[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.shape}"

    def __array__(self, dtype=None):
        if dtype:
            return self._array.astype(dtype)
        return self._array

    def __len__(self):
        return len(self._array)

    @to_new_array
    def astype(self, dtype):
        return self._array.astype(dtype)

    @property
    @to_new_array
    def real(self):
        return self._array.real

    @property
    @to_new_array
    def imag(self):
        return self._array.imag

    def all(self):
        return self._array.all()
