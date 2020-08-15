import logging

import numpy as np
import pylops

from autoarray.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractVisibilities(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(cls, visibilities_1d, *args, **kwargs):
        """ A hyper array with square-pixels.

        Parameters
        ----------
        array_1d: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the hyper array's coordinate system.
        """
        obj = visibilities_1d.view(cls)
        obj._as_complex = None
        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "_as_complex"):
            self._as_complex = obj._as_complex

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

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        super_result = super(AbstractVisibilities, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)

    @property
    def in_1d(self):
        return self

    @property
    def shape_1d(self):
        return self.shape[0]

    @property
    def in_1d_flipped(self):
        return Visibilities.manual_1d(visibilities=np.fliplr(self))

    @property
    def real(self):
        return self[:, 0]

    @property
    def imag(self):
        return self[:, 1]

    @property
    @array_util.Memoizer()
    def amplitudes(self):
        return np.sqrt(np.square(self.real) + np.square(self.imag))

    @property
    @array_util.Memoizer()
    def phases(self):
        return np.arctan2(self.imag, self.real)

    @property
    def as_complex(self):
        if self._as_complex is None:
            self._as_complex = np.apply_along_axis(
                lambda args: [complex(*args)], 1, self
            )
        return self._as_complex

    def output_to_fits(self, file_path, overwrite=False):
        array_util.numpy_array_2d_to_fits(
            array_2d=self, file_path=file_path, overwrite=overwrite
        )

    @property
    def scaled_maxima(self):
        return (np.max(self.real), np.max(self.imag))

    @property
    def scaled_minima(self):
        return (np.min(self.real), np.min(self.imag))

    @property
    def extent(self):
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

        if type(visibilities) is list:
            visibilities = np.asarray(visibilities)

        return cls(visibilities_1d=visibilities)

    @classmethod
    def full(cls, fill_value, shape_1d):
        return cls.manual_1d(
            visibilities=np.full(fill_value=fill_value, shape=(shape_1d[0], 2))
        )

    @classmethod
    def ones(cls, shape_1d):
        return cls.full(fill_value=1.0, shape_1d=shape_1d)

    @classmethod
    def zeros(cls, shape_1d):
        return cls.full(fill_value=0.0, shape_1d=shape_1d)

    @classmethod
    def from_fits(cls, file_path, hdu):
        visibilities_1d = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu
        )
        return cls.manual_1d(visibilities=visibilities_1d)


class VisibilitiesNoiseMap(Visibilities):

    # noinspection PyUnusedLocal
    def __new__(cls, visibilities_1d, *args, **kwargs):
        """ A hyper array with square-pixels.

        Parameters
        ----------
        array_1d: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the hyper array's coordinate system.
        """
        obj = super(VisibilitiesNoiseMap, cls).__new__(
            cls=cls, visibilities_1d=visibilities_1d
        )
        obj.Wop = pylops.Diagonal(1.0 / obj.as_complex.ravel())
        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "_as_complex"):
            self._as_complex = obj._as_complex

        if hasattr(obj, "Wop"):
            self.Wop = obj.Wop

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
