from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoconf import cached_property

from autoarray.mask.abstract_mask import Mask

from autoarray import exc
from autoarray import type as ty
from autoarray.geometry.geometry_2d import Geometry2D
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask2D(Mask):

    # noinspection PyUnusedLocal
    def __new__(
        cls,
        mask: np.ndarray,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        *args,
        **kwargs,
    ):
        """
        A 2D mask, representing a uniform rectangular grid of neighboring rectangular pixels.

        When applied to 2D data it extracts or masks the unmasked image pixels corresponding to mask entries that are
        `False` or 0).

        The mask defines the geometry of the 2D uniform grid of pixels for the 2D data structure it is paired with,
        for example the grid's `pixel scales` (y,x) `origin`. The 2D uniform grid may also be sub-gridded,
        whereby every pixel is sub-divided into a uniform grid of sub-pixels which are all used to perform
        calculations more accurate.

        The mask includes tols to map the 2D data structure between 2D representations (that include all  data-points
        irrespective of if they are masked or not) and 1D data structures (that only contain the unmasked data).

        Parameters
        ----------
        mask: np.ndarray
            The `ndarray` of shape [total_y_pixels, total_x_pixels] containing the `bool`'s representing the
            `mask`, where `False` signifies an entry is unmasked and used in calculations.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        obj = Mask.__new__(
            cls=cls,
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )
        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj=obj)

        if isinstance(obj, Mask2D):
            pass
        else:
            self.origin = (0.0, 0.0)

    @property
    def geometry(self):
        return Geometry2D(shape_native=self.shape_native, pixel_scales=self.pixel_scales)

    @classmethod
    def manual(
        cls,
        mask: Union[np.ndarray, list],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see `Mask2D.__new__`) by inputting the array values in 2D, for example:

        mask=np.array([[False, False],
                       [True, False]])

        mask=[[False, False],
               [True, False]]

        Parameters
        ----------
        mask or list
            The `bool` values of the mask input as an `np.ndarray` of shape [total_y_pixels, total_x_pixels] or a
            list of lists.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return Mask2D(
            mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def unmasked(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Create a mask where all pixels are `False` and therefore unmasked.

        Parameters
        ----------
        shape_native
            The 2D shape of the mask that is created.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        return cls.manual(
            mask=np.full(shape=shape_native, fill_value=False),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular(
        cls,
        shape_native: Tuple[int, int],
        radius: float,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within a circle of input radius.

        The `radius` and `centre` are both input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        radius
            The radius in scaled units of the circle within which pixels are `False` and unmasked.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the circle used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_circular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            radius=radius,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular_annular(
        cls,
        shape_native: Tuple[int, int],
        inner_radius: float,
        outer_radius: float,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an annulus of input
        inner radius and outer radius.

        The `inner_radius`, `outer_radius` and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        inner_radius
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius
            The outer radius in scaled units of the annulus within which pixels are `False` and unmasked.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the annulus used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_circular_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular_anti_annular(
        cls,
        shape_native: Tuple[int, int],
        inner_radius: float,
        outer_radius: float,
        outer_radius_2: float,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an inner circle and second
        outer circle, forming an inverse annulus.

        The `inner_radius`, `outer_radius`, `outer_radius_2` and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        inner_radius
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius
            The first outer radius in scaled units of the annulus within which pixels are `True` and masked.
        outer_radius_2
            The second outer radius in scaled units of the annulus within which pixels are `False` and unmasked and
            outside of which all entries are `True` and masked.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the anti-annulus used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_circular_anti_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            outer_radius_2_scaled=outer_radius_2,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def elliptical(
        cls,
        shape_native: Tuple[int, int],
        major_axis_radius: float,
        axis_ratio: float,
        angle: float,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an ellipse.

        The `major_axis_radius`, and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        major_axis_radius
            The major-axis in scaled units of the ellipse within which pixels are unmasked.
        axis_ratio
            The axis-ratio of the ellipse within which pixels are unmasked.
        angle
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive
             x-axis).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centred of the ellipse used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_elliptical_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            major_axis_radius=major_axis_radius,
            axis_ratio=axis_ratio,
            angle=angle,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def elliptical_annular(
        cls,
        shape_native: Tuple[int, int],
        inner_major_axis_radius: float,
        inner_axis_ratio: float,
        inner_phi: float,
        outer_major_axis_radius: float,
        outer_axis_ratio: float,
        outer_phi: float,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an elliptical annulus of input
        inner and outer scaled major-axis and centre.

        The `outer_major_axis_radius`, `inner_major_axis_radius` and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        inner_major_axis_radius
            The major-axis in scaled units of the inner ellipse within which pixels are masked.
        inner_axis_ratio
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the
            positive x-axis).
        outer_major_axis_radius
            The major-axis in scaled units of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the
            positive x-axis).
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the elliptical annuli used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_elliptical_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_major_axis_radius=inner_major_axis_radius,
            inner_axis_ratio=inner_axis_ratio,
            inner_phi=inner_phi,
            outer_major_axis_radius=outer_major_axis_radius,
            outer_axis_ratio=outer_axis_ratio,
            outer_phi=outer_phi,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_pixel_coordinates(
        cls,
        shape_native: Tuple[int, int],
        pixel_coordinates: [[int, int]],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        buffer: int = 0,
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are defined from an input list of list of
        pixel coordinates.

        These may be buffed via an input `buffer`, whereby all entries in all 8 neighboring directions by this
        amount.

        Parameters
        ----------
        shape_native (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_coordinates : [[int, int]]
            The input lists of 2D pixel coordinates where `False` entries are created.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        buffer
            All input `pixel_coordinates` are buffed with `False` entries in all 8 neighboring directions by this
            amount.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        mask = mask_2d_util.mask_2d_via_pixel_coordinates_from(
            shape_native=shape_native,
            pixel_coordinates=pixel_coordinates,
            buffer=buffer,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: str,
        pixel_scales: ty.PixelScales,
        hdu: int = 0,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        resized_mask_shape: Tuple[int, int] = None,
    ) -> "Mask2D":
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path
            The full path of the fits file.
        hdu
            The HDU number in the fits file containing the image image.
        pixel_scales or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = Mask2D(
            mask=array_2d_util.numpy_array_2d_via_fits_from(
                file_path=file_path, hdu=hdu
            ),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.resized_mask_from(new_shape=resized_mask_shape)

        return mask

    @property
    def shape_native(self) -> Tuple[int, ...]:
        return self.shape

    @property
    def sub_shape_native(self) -> Tuple[int, int]:
        try:
            return (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)
        except AttributeError:
            print("bleh")

    @property
    def sub_mask(self) -> np.ndarray:

        sub_shape = (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

        return mask_2d_util.mask_2d_via_shape_native_and_native_for_slim(
            shape_native=sub_shape,
            native_for_slim=self.sub_mask_index_for_sub_mask_1d_index,
        ).astype("bool")

    def rescaled_mask_from(self, rescale_factor) -> "Mask2D":

        rescaled_mask = mask_2d_util.rescaled_mask_2d_from(
            mask_2d=self, rescale_factor=rescale_factor
        )

        return Mask2D(
            mask=rescaled_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    @property
    def mask_sub_1(self) -> "Mask2D":
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of `sub_size`.
        """
        return Mask2D(
            mask=self, sub_size=1, pixel_scales=self.pixel_scales, origin=self.origin
        )

    def resized_mask_from(self, new_shape, pad_value: int = 0.0) -> "Mask2D":
        """
        Resized the array to a new shape and at a new origin.

        Parameters
        ----------
        new_shape
            The new two-dimensional shape of the array.
        """

        mask = copy.deepcopy(self)

        resized_mask = array_2d_util.resized_array_2d_from(
            array_2d=mask, resized_shape=new_shape, pad_value=pad_value
        ).astype("bool")

        return Mask2D(
            mask=resized_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def trimmed_array_from(self, padded_array, image_shape) -> Array2D:
        """
        Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        ----------
        padded_array
            A 1D array of values which were computed using a padded grid
        """

        from autoarray.structures.arrays.uniform_2d import Array2D

        pad_size_0 = self.shape[0] - image_shape[0]
        pad_size_1 = self.shape[1] - image_shape[1]
        trimmed_array = padded_array.binned.native[
            pad_size_0 // 2 : self.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.shape[1] - pad_size_1 // 2,
        ]
        return Array2D.manual(
            array=trimmed_array,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

    def unmasked_blurred_array_from(self, padded_array, psf, image_shape) -> Array2D:
        """
        For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens dataset's padded-grid, which is a grid of (y,x) coordinates which extends over
        the entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : aa.Kernel2D
            The PSF of the image used for convolution.
        unmasked_image_1d
            The 1D unmasked image which is blurred.
        """

        blurred_image = psf.convolved_array_from(array=padded_array)

        return self.trimmed_array_from(
            padded_array=blurred_image, image_shape=image_shape
        )

    def output_to_fits(self, file_path, overwrite=False):
        """
        Write the 2D Mask to a .fits file.

        Before outputting a NumPy array, the array may be flipped upside-down using np.flipud depending on the project
        config files. This is for Astronomy projects so that structures appear the same orientation as `.fits` files
        loaded in DS9.

        Parameters
        ----------
        file_path
            The full path of the file that is output, including the file name and `.fits` extension.
        overwrite
            If `True` and a file already exists with the input file_path the .fits file is overwritten. If `False`, an
            error is raised.

        Returns
        -------
        None

        Examples
        --------
        mask = Mask2D(mask=np.full(shape=(5,5), fill_value=False))
        mask.output_to_fits(file_path='/path/to/file/filename.fits', overwrite=True)
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D shape of the mask in scaled units, computed from the 2D `shape` (units pixels) and
        the `pixel_scales` (units scaled/pixels) conversion factor.
        """
        return (
            float(self.pixel_scales[0] * self.shape[0]),
            float(self.pixel_scales[1] * self.shape[1]),
        )

    def pixel_coordinates_2d_from(
        self, scaled_coordinates_2d
    ) -> Union[Tuple[float], Tuple[float, float]]:

        return geometry_util.pixel_coordinates_2d_from(
            scaled_coordinates_2d=scaled_coordinates_2d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def scaled_coordinates_2d_from(self, pixel_coordinates_2d):

        return geometry_util.scaled_coordinates_2d_from(
            pixel_coordinates_2d=pixel_coordinates_2d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    @property
    def mask_centre(self) -> Tuple[float, float]:
        return grid_2d_util.grid_2d_centre_from(grid_2d_slim=self.masked_grid_sub_1)

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        return (
            (self.shape_native_scaled[0] / 2.0) + self.origin[0],
            (self.shape_native_scaled[1] / 2.0) + self.origin[1],
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        return (
            (-(self.shape_native_scaled[0] / 2.0)) + self.origin[0],
            (-(self.shape_native_scaled[1] / 2.0)) + self.origin[1],
        )

    @property
    def extent(self) -> np.ndarray:
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

    @property
    def edge_buffed_mask(self) -> "Mask2D":
        edge_buffed_mask = mask_2d_util.buffed_mask_2d_from(mask_2d=self).astype("bool")
        return Mask2D(
            mask=edge_buffed_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    @property
    def unmasked_grid_sub_1(self) -> Grid2D:
        """
        The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x
        value y value in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

        return Grid2D(grid=grid_slim, mask=self.unmasked_mask.mask_sub_1)

    @property
    def masked_grid(self) -> Grid2D:

        from autoarray.structures.grids.uniform_2d import Grid2D

        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )
        return Grid2D(grid=sub_grid_1d, mask=self.edge_mask.mask_sub_1)

    @property
    def masked_grid_sub_1(self) -> Grid2D:

        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_slim = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin
        )
        return Grid2D(grid=grid_slim, mask=self.mask_sub_1)

    @property
    def edge_grid_sub_1(self) -> Grid2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """

        from autoarray.structures.grids.uniform_2d import Grid2D

        edge_grid_1d = self.masked_grid_sub_1[self.edge_1d_indexes]
        return Grid2D(grid=edge_grid_1d, mask=self.edge_mask.mask_sub_1)

    @property
    def border_grid_1d(self) -> Grid2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return self.masked_grid[self.sub_border_flat_indexes]

    @property
    def border_grid_sub_1(self) -> Grid2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        border_grid_1d = self.masked_grid_sub_1[self.border_1d_indexes]
        return Grid2D(grid=border_grid_1d, mask=self.border_mask.mask_sub_1)

    def grid_pixels_from(self, grid_scaled_1d) -> Grid2D:
        """
        Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel values. Pixel coordinates are
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            A grid of (y,x) coordinates in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_pixels_1d = grid_2d_util.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return Grid2D(grid=grid_pixels_1d, mask=self.mask_sub_1)

    def grid_pixel_centres_from(self, grid_scaled_1d) -> Grid2D:
        """
        Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel values. Pixel coordinates are
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        higher y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            The grid of (y,x) coordinates in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_pixel_centres_1d = grid_2d_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

        return Grid2D(grid=grid_pixel_centres_1d, mask=self.edge_mask.mask_sub_1)

    def grid_pixel_indexes_from(self, grid_scaled_1d):
        """
        Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are
        returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then
        downwards.

        For example:

        - The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
        - The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
        - The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            The grid of (y,x) coordinates in scaled units.
        """

        from autoarray.structures.arrays.uniform_2d import Array2D

        grid_pixel_indexes_1d = grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

        return Array2D(array=grid_pixel_indexes_1d, mask=self.edge_mask.mask_sub_1)

    def grid_scaled_from(self, grid_pixels_1d) -> Grid2D:
        """
        Convert a grid of (y,x) pixel coordinates to a grid of (y,x) scaled values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        higher y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels_1d
            The grid of (y,x) coordinates in pixels.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_scaled_1d = grid_2d_util.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return Grid2D(grid=grid_scaled_1d, mask=self.edge_mask.mask_sub_1)

    def grid_scaled_for_marching_squares_from(
        self, grid_pixels_1d, shape_native
    ) -> Grid2D:

        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_scaled_1d = grid_2d_util.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels_1d,
            shape_native=shape_native,
            pixel_scales=(
                self.pixel_scales[0] / self.sub_size,
                self.pixel_scales[1] / self.sub_size,
            ),
            origin=self.origin,
        )

        grid_scaled_1d[:, 0] -= self.pixel_scales[0] / (2.0 * self.sub_size)
        grid_scaled_1d[:, 1] += self.pixel_scales[1] / (2.0 * self.sub_size)

        return Grid2D(grid=grid_scaled_1d, mask=self.edge_mask.mask_sub_1)

    @property
    def native_index_for_slim_index(self) -> np.ndarray:
        """
        A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates.
        """
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self, sub_size=1
        ).astype("int")

    @property
    def masked_1d_indexes(self) -> np.ndarray:
        """
        The 1D indexes of the mask's unmasked pixels (e.g. `value=False`).
        """
        return mask_2d_util.mask_1d_indexes_from(
            mask_2d=self, return_masked_indexes=True
        ).astype("int")

    @property
    def unmasked_1d_indexes(self) -> np.ndarray:
        """
        The 1D indexes of the mask's unmasked pixels (e.g. `value=False`).
        """
        return mask_2d_util.mask_1d_indexes_from(
            mask_2d=self, return_masked_indexes=False
        ).astype("int")

    @property
    def edge_1d_indexes(self) -> np.ndarray:
        """
        The indexes of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge
        (next to at least one pixel with a `True` value).
        """
        return mask_2d_util.edge_1d_indexes_from(mask_2d=self).astype("int")

    @property
    def edge_2d_indexes(self) -> np.ndarray:
        """
        The indexes of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge
        (next to at least one pixel with a `True` value).
        """
        return self.native_index_for_slim_index[self.edge_1d_indexes].astype("int")

    @property
    def border_1d_indexes(self) -> np.ndarray:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return mask_2d_util.border_slim_indexes_from(mask_2d=self).astype("int")

    @property
    def border_2d_indexes(self) -> np.ndarray:
        """The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return self.native_index_for_slim_index[self.border_1d_indexes].astype("int")

    @cached_property
    def sub_border_flat_indexes(self) -> np.ndarray:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return mask_2d_util.sub_border_pixel_slim_indexes_from(
            mask_2d=self, sub_size=self.sub_size
        ).astype("int")

    def blurring_mask_from(self, kernel_shape_native) -> "Mask2D":
        """
        Returns a blurring mask, which represents all masked pixels whose light will be blurred into unmasked
        pixels via PSF convolution (see grid.Grid2D.blurring_grid_from).

        Parameters
        ----------
        kernel_shape_native
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if kernel_shape_native[0] % 2 == 0 or kernel_shape_native[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_2d_util.blurring_mask_2d_from(
            mask_2d=self, kernel_shape_native=kernel_shape_native
        )

        return Mask2D(
            mask=blurring_mask,
            sub_size=1,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def unmasked_mask(self) -> "Mask2D":
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return Mask2D.unmasked(
            shape_native=self.shape_native,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def edge_mask(self) -> "Mask2D":
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        mask = np.full(fill_value=True, shape=self.shape)
        mask[self.edge_2d_indexes[:, 0], self.edge_2d_indexes[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def border_mask(self) -> "Mask2D":
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        mask = np.full(fill_value=True, shape=self.shape)
        mask[self.border_2d_indexes[:, 0], self.border_2d_indexes[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @cached_property
    def sub_mask_index_for_sub_mask_1d_index(self) -> np.ndarray:
        """
        A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates.
        """
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self, sub_size=self.sub_size
        ).astype("int")

    @cached_property
    def slim_index_for_sub_slim_index(self) -> np.ndarray:
        """
        The util between every sub-pixel and its host pixel.

        For example:

        sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd pixel.
        sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th pixel.
        """
        return mask_2d_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=self, sub_size=self.sub_size
        ).astype("int")

    @property
    def shape_native_masked_pixels(self) -> Tuple[int, int]:
        """
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.

        For example, if a mask is primarily surrounded by True entries, and there are 15 False entries going vertically
        and 12 False entries going horizontally in the central regions of the mask, then shape_masked_pixels=(15,12).
        """

        where = np.array(np.where(np.invert(self.astype("bool"))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        return (y1 - y0 + 1, x1 - x0 + 1)

    @property
    def zoom_centre(self) -> Tuple[float, float]:

        extraction_grid_1d = self.grid_pixels_from(
            grid_scaled_1d=self.masked_grid_sub_1.slim
        )
        y_pixels_max = np.max(extraction_grid_1d[:, 0])
        y_pixels_min = np.min(extraction_grid_1d[:, 0])
        x_pixels_max = np.max(extraction_grid_1d[:, 1])
        x_pixels_min = np.min(extraction_grid_1d[:, 1])

        return (
            ((y_pixels_max + y_pixels_min - 1.0) / 2.0),
            ((x_pixels_max + x_pixels_min - 1.0) / 2.0),
        )

    @property
    def zoom_offset_pixels(self) -> Tuple[float, float]:

        if self.pixel_scales is None:
            return self.central_pixel_coordinates

        return (
            self.zoom_centre[0] - self.central_pixel_coordinates[0],
            self.zoom_centre[1] - self.central_pixel_coordinates[1],
        )

    @property
    def zoom_offset_scaled(self) -> Tuple[float, float]:

        return (
            -self.pixel_scales[0] * self.zoom_offset_pixels[0],
            self.pixel_scales[1] * self.zoom_offset_pixels[1],
        )

    @property
    def zoom_region(self) -> List[int]:
        """
        The zoomed rectangular region corresponding to the square encompassing all unmasked values. This zoomed
        extraction region is a squuare, even if the mask is rectangular.

        This is used to zoom in on the region of an image that is used in an analysis for visualization.
        """

        where = np.array(np.where(np.invert(self.astype("bool"))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        # Have to convert mask to bool for invert function to work.

        ylength = y1 - y0
        xlength = x1 - x0

        if ylength > xlength:
            length_difference = ylength - xlength
            x1 += int(length_difference / 2.0)
            x0 -= int(length_difference / 2.0)
        elif xlength > ylength:
            length_difference = xlength - ylength
            y1 += int(length_difference / 2.0)
            y0 -= int(length_difference / 2.0)

        return [y0, y1 + 1, x0, x1 + 1]

    @property
    def zoom_shape_native(self) -> Tuple[int, int]:
        region = self.zoom_region
        return (region[1] - region[0], region[3] - region[2])

    @property
    def zoom_mask_unmasked(self) -> "Mask2D":
        """
        The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x
        value y value in scaled units.
        """

        return Mask2D.unmasked(
            shape_native=self.zoom_shape_native,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.zoom_offset_scaled,
        )
