from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.mask.abstract_mask import Mask

from autoarray import exc
from autoarray import type as ty
from autoarray.geometry.geometry_2d import Geometry2D
from autoarray.mask.derive.mask_2d import DeriveMask2D
from autoarray.mask.derive.grid_2d import DeriveGrid2D
from autoarray.mask.derive.indexes_2d import DeriveIndexes2D

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
        mask: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
        *args,
        **kwargs,
    ):
        """
        A 2D mask, used for masking values which are associated with a a uniform rectangular grid of pixels.

        When applied to 2D data with the same shape, values in the mask corresponding to ``False`` entries are
        unmasked and therefore used in subsequent calculations. .

        The ``Mask2D`, has in-built functionality which:

        - Maps data structures between two data representations: `slim`` (all unmasked ``False`` values in
          a 1D ``ndarray``) and ``native`` (all unmasked values in a 2D or 3D ``ndarray``).

        - Has a ``Geometry2D`` object (defined by its (y,x) ``pixel scales``, (y,x) ``origin`` and ``sub_size``)
          which defines how coordinates are converted from pixel units to scaled units.

        - Associates Cartesian ``Grid2D`` objects of (y,x) coordinates with the data structure (e.g.
          a (y,x) grid of all unmasked pixels) via the ``DeriveGrid2D`` object.

        - This includes sub-grids, which perform calculations higher resolutions which are then binned up.

        A detailed description of the 2D mask API is provided below.

        **SLIM DATA REPRESENTATION (sub-size=1)**

        Below is a visual illustration of a ``Mask2D``, where a total of 10 pixels are unmasked (values are ``False``):

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the array)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the array)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        The mask pixel index's are as follows (the positive / negative direction of the ``Grid2D`` objects associated
        with the mask are also shown on the y and x axes).

        ::

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x 0 1 x x x x +ve
             x x x 2 3 4 5 x x x  y
             x x x 6 7 8 9 x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x \/
             x x x x x x x x x x

        The ``Mask2D``'s ``slim`` data representation is an ``ndarray`` of shape [total_unmasked_pixels].

        For the ``Mask2D`` above the ``slim`` representation therefore contains 10 entries and two examples of these
        entries are:

        ::

            mask[3] = the 4th unmasked pixel's value.
            mask[6] = the 7th unmasked pixel's value.

        A Cartesian grid of (y,x) coordinates, corresponding to all ``slim`` values (e.g. unmasked pixels) is given
        by ``mask.derive_grid.masked.slim``.


        **NATIVE DATA REPRESENTATION (sub_size=1)**

        Masked data represented as an an ``ndarray`` of shape [total_y_values, total_x_values], where all masked
        entries have values of 0.0.

        For the following mask:

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the array)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the array)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        The mask has the following indexes:

        ::

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x 0 1 x x x x +ve
             x x x 2 3 4 5 x x x  y
             x x x 6 7 8 9 x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x  \/
             x x x x x x x x x x

        In the above array:

        ::

            - mask[0,0] = True (it is masked)
            - mask[0,0] = True (it is masked)
            - mask[3,3] = True (it is masked)
            - mask[3,3] = True (it is masked)
            - mask[3,4] = False (not masked)
            - mask[3,5] = False (not masked)
            - mask[4,5] = False (not masked)

        **SLIM TO NATIVE MAPPING**

        The ``Mask2D`` has functionality which maps data between the ``slim`` and ``native`` data representations.

        For the example mask above, the 1D ``ndarray`` given by ``mask.derive_indexes.slim_to_native`` is:

        ::

            slim_to_native[0] = [3,4]
            slim_to_native[1] = [3,5]
            slim_to_native[2] = [4,3]
            slim_to_native[3] = [4,4]
            slim_to_native[4] = [4,5]
            slim_to_native[5] = [4,6]
            slim_to_native[6] = [5,3]
            slim_to_native[7] = [5,4]
            slim_to_native[8] = [5,5]
            slim_to_native[9] = [5,6]

        **SUB GRIDDING**

        If the ``Mask2D`` ``sub_size`` is > 1, its ``slim`` and ``native`` data representations have entries
        corresponding to the values at the centre of every sub-pixel of each unmasked pixel.

        The sub-array indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.

        Therefore, the shapes of the sub-array are as follows:

        - ``slim`` representation: an ``ndarray`` of shape [total_unmasked_pixels*sub_size**2].
        - ``native`` representation: an ``ndarray`` of shape [total_y_values*sub_size, total_x_values*sub_size].

        Below is a visual illustration of a sub array. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the array above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x x x x x x x     x = `True` (Pixel is masked and excluded from lens)
             x 0 0 x x x x x x x     O = `False` (Pixel is not masked and included in lens)
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        If ``sub_size=2``, each unmasked pixel has 4 (2x2) sub-pixel values. For the example above, pixels 0 and 1
        each have 4 values which map to ``slim`` representation as follows:

        ::

            Pixel 0 - (2x2):

                   slim[0] = value of first sub-pixel in pixel 0.
             0 1   slim[1] = value of first sub-pixel in pixel 1.
             2 3   slim[2] = value of first sub-pixel in pixel 2.
                   slim[3] = value of first sub-pixel in pixel 3.

            Pixel 1 - (2x2):

                   slim[4] = value of first sub-pixel in pixel 0.
             4 5   slim[5] = value of first sub-pixel in pixel 1.
             6 7   slim[6] = value of first sub-pixel in pixel 2.
                   slim[7] = value of first sub-pixel in pixel 3.

        For the ``native`` data representation we get the following mappings:

        ::

            Pixel 0 - (2x2):

                   native[8, 2] = value of first sub-pixel in pixel 0.
             0 1   native[8, 3] = value of first sub-pixel in pixel 1.
             2 3   native[9, 2] = value of first sub-pixel in pixel 2.
                   native[9, 3] = value of first sub-pixel in pixel 3.

            Pixel 1 - (2x2):

                   native[10, 4] = value of first sub-pixel in pixel 0.
             4 5   native[10, 5] = value of first sub-pixel in pixel 1.
             6 7   native[11, 4] = value of first sub-pixel in pixel 2.
                   native[11, 5] = value of first sub-pixel in pixel 3.

            Other entries (all masked sub-pixels are zero):

                   native[0, 0] = 0.0 (it is masked, thus zero)
                   native[15, 12] = 0.0 (it is masked, thus zero)

        If we used a sub_size of 3, for pixel 0 we we would create a 3x3 sub-array:

        ::

                     slim[0] = value of first sub-pixel in pixel 0.
                     slim[1] = value of first sub-pixel in pixel 1.
                     slim[2] = value of first sub-pixel in pixel 2.
             0 1 2   slim[3] = value of first sub-pixel in pixel 3.
             3 4 5   slim[4] = value of first sub-pixel in pixel 4.
             6 7 8   slim[5] = value of first sub-pixel in pixel 5.
                     slim[6] = value of first sub-pixel in pixel 6.
                     slim[7] = value of first sub-pixel in pixel 7.
                     slim[8] = value of first sub-pixel in pixel 8.

        Parameters
        ----------
        mask
            The `ndarray` of shape [total_y_pixels, total_x_pixels] containing the `bool`'s representing the
            `mask`, where `False` signifies an entry is unmasked and used in calculations.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        obj = Mask.__new__(
            cls=cls,
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )
        obj.derive_indexes = DeriveIndexes2D(mask=obj)
        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj=obj)

        if isinstance(obj, Mask2D):
            self.derive_indexes = obj.derive_indexes
        else:
            self.origin = (0.0, 0.0)

    @property
    def geometry(self) -> Geometry2D:
        """
        Return the 2D geometry of the mask, representing its uniform rectangular grid of (y,x) coordinates defined by
        its ``shape_native``.
        """
        return Geometry2D(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def derive_mask(self) -> DeriveMask2D:
        return DeriveMask2D(mask=self)

    @property
    def derive_grid(self) -> DeriveGrid2D:
        return DeriveGrid2D(mask=self)

    @classmethod
    def all_false(
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
        return cls(
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

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_circular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            radius=radius,
            centre=centre,
        )

        return cls(
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

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_circular_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            centre=centre,
        )

        return cls(
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

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_circular_anti_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            outer_radius_2_scaled=outer_radius_2,
            centre=centre,
        )

        return cls(
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
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_elliptical_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            major_axis_radius=major_axis_radius,
            axis_ratio=axis_ratio,
            angle=angle,
            centre=centre,
        )

        return cls(
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
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

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

        return cls(
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

        return cls(
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
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = Mask2D(
            mask=array_2d_util.numpy_array_2d_via_fits_from(
                file_path=file_path, hdu=hdu
            ),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.derive_mask.resized_from(new_shape=resized_mask_shape)

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
        return Array2D.without_mask(
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
    def mask_centre(self) -> Tuple[float, float]:
        return grid_2d_util.grid_2d_centre_from(
            grid_2d_slim=self.derive_grid.unmasked_sub_1
        )

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

        extraction_grid_1d = self.geometry.grid_pixels_2d_from(
            grid_scaled_2d=self.derive_grid.unmasked_sub_1.slim
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
            return self.geometry.central_pixel_coordinates

        return (
            self.zoom_centre[0] - self.geometry.central_pixel_coordinates[0],
            self.zoom_centre[1] - self.geometry.central_pixel_coordinates[1],
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

        return Mask2D.all_false(
            shape_native=self.zoom_shape_native,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.zoom_offset_scaled,
        )
