import logging

import numpy as np

from autoarray import exc
from autoarray.mask import geometry, regions

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractMask(np.ndarray):

    pixel_scales = None

    # noinspection PyUnusedLocal
    def __new__(
        cls,
        mask: np.ndarray,
        origin: tuple,
        pixel_scales: tuple,
        sub_size: int = 1,
        *args,
        **kwargs
    ):
        """An abstract class for a mask that represents data structure that can be in 1D, 2D or other shapes.

         When applied to data it extracts or masks the unmasked image pixels corresponding to mask entries that are
        `False` or 0).

        The mask also defines the geometry of the data structure it is paired with, for example how its pixels convert
        to physical units via the ``pixel_scales`` and ``origin`` parameters and a sub-grid which is used for
        perform calculations via super-sampling.

         Parameters
         ----------
         mask : np.ndarray
             The ``ndarray`` containing the ``bool``'s representing the ``mask``, where `False` signifies an entry is
             unmasked and used in calculations.
         pixel_scales : (float, float) or float
             The scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``, it is
             converted to a ``(float, float)`` structure.
         origin : (float, float)
             The origin of the ``mask``'s coordinate system in scaled units.
        """

        # noinspection PyArgumentList
        mask = mask.astype("bool")
        obj = mask.view(cls)
        obj.sub_size = sub_size
        obj.pixel_scales = pixel_scales
        obj.origin = origin
        return obj

    def __array_finalize__(self, obj):

        if isinstance(obj, AbstractMask):
            self.sub_size = obj.sub_size
            self.pixel_scales = obj.pixel_scales
            self.origin = obj.origin
        else:
            self.sub_size = 1
            self.pixel_scales = None

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
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
        super().__setstate__(state[0:-1])

    @property
    def pixel_scale(self):
        """
        For a mask with dimensions two or above check that are pixel scales are the same, and if so return this
        single value as a ``float``.
        """

        for pixel_scale in self.pixel_scales:
            if pixel_scale != self.pixel_scales[0]:
                raise exc.MaskException(
                    "Cannot return a pixel_scale for a grid where each dimension has a "
                    "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]"
                )

        return self.pixel_scales[0]

    @property
    def geometry(self):
        """The Geometry class contains methods describing the Mask2D's geometry, for example the Grid of unmasked
        pixels.

        See the Geometry class for a full description."""
        return geometry.Geometry(mask=self)

    @property
    def regions(self):
        """The Region class contains methods describing regions on the Mask2D, for example the pixel indexes of Mask2D
        pixels on its edge.

        See the Region class for a full description."""
        return regions.Regions(mask=self)

    @property
    def sub_length(self) -> int:
        """
        The total number of sub-pixels in a give pixel,

        For example, a sub-size of 3x3 means every pixel has 9 sub-pixels.
        """
        return int(self.sub_size ** 2.0)

    @property
    def sub_fraction(self) -> float:
        """
        The fraction of the area of a pixel every sub-pixel contains.

        For example, a sub-size of 3x3 mean every pixel contains 1/9 the area.
        """
        return 1.0 / self.sub_length

    def output_to_fits(self, file_path: str, overwrite: bool = False):
        raise NotImplementedError()

    @property
    def pixels_in_mask(self) -> int:
        """
        The total number of unmasked pixels (values are `False`) in the mask.
        """
        return int(np.size(self) - np.sum(self))

    @property
    def is_all_true(self) -> bool:
        """
        Returns `True` if all pixels in a mask are `True`, else returns `False`.
        """
        return self.pixels_in_mask == 0

    @property
    def is_all_false(self) -> bool:
        """
        Returns `False` if all pixels in a mask are `False`, else returns `True`.
        """
        return self.pixels_in_mask == np.size(self)

    @property
    def sub_pixels_in_mask(self) -> int:
        """
        The total number of unmasked sub-pixels (values are `False`) in the mask.
        """
        return self.sub_size ** 2 * self.pixels_in_mask

    @property
    def shape_1d(self) -> int:
        """
        The 1D shape of the mask, which is equivalent to the total number of unmasked pixels in the mask.
        """
        return self.pixels_in_mask

    @property
    def sub_shape_1d(self) -> int:
        """
        The 1D shape of the masks's sub-grid, which is equivalent to the total number of unmasked pixels in the mask.
        """
        return int(self.pixels_in_mask * self.sub_size ** 2.0)

    @property
    def mask_sub_1(self) -> "AbstractMask":
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of ``sub_size`` `.
        """
        return self.__class__(
            mask=self, sub_size=1, pixel_scales=self.pixel_scales, origin=self.origin
        )

    def mask_new_sub_size_from_mask(self, mask, sub_size=1) -> "AbstractMask":
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of an input``sub_size`` `.
        """
        return self.__class__(
            mask=mask,
            sub_size=sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def binned_pixel_scales_from_bin_up_factor(self, bin_up_factor: int) -> tuple:
        """
        If the mask is binned up to a coarser resolution, computes the ``pixel_scales`` of the binned up mask

        Parameters
        ----------
        bin_up_factor : int
            The factor by which the mask is binned up to a coarser resolution.
        """
        if self.pixel_scales is not None:
            return tuple(
                pixel_scale * bin_up_factor for pixel_scale in self.pixel_scales
            )
