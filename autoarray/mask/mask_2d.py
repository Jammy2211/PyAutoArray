import logging

import copy
import numpy as np

from autoarray import exc
from autoarray.mask import abstract_mask
from autoarray.util import array_util, binning_util, geometry_util, mask_util
from autoarray.structures import arrays

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractMask2D(abstract_mask.AbstractMask):

    # noinspection PyUnusedLocal
    def __new__(
        cls,
        mask: np.ndarray,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        *args,
        **kwargs
    ):
        """
        A 2D mask, representing a uniform rectangular grid of neighboring rectangular pixels.

        When applied to 2D data it extracts or masks the unmasked image pixels corresponding to mask entries that are
        `False` or 0).

        The mask defines the geometry of the 2D uniform grid of pixels for the 2D data structure it is paired with,
        for example the grid's ``pixel scales`` (y,x) ``origin``. The 2D uniform grid may also be sub-gridded,
        whereby every pixel is sub-divided into a uniform grid of sub-pixels which are all used to perform
        calculations more accurate.

        The mask includes tols to map the 2D data structure between 2D representations (that include all  data-points
        irrespective of if they are masked or not) and 1D data structures (that only contain the unmasked data).

        Parameters
        ----------
        mask: np.ndarray
            The ``ndarray`` of shape [total_y_pixels, total_x_pixels] containing the ``bool``'s representing the
            ``mask``, where `False` signifies an entry is unmasked and used in calculations.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        obj = abstract_mask.AbstractMask.__new__(
            cls=cls,
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )
        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj=obj)

        if isinstance(obj, AbstractMask2D):
            pass
        else:
            self.origin = (0.0, 0.0)

    @property
    def shape_2d(self):
        return self.shape

    @property
    def sub_shape_2d(self):
        try:
            return (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)
        except AttributeError:
            print("bleh")

    @property
    def sub_mask(self):

        sub_shape = (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

        return mask_util.mask_via_shape_2d_and_mask_index_for_mask_1d_index_from(
            shape_2d=sub_shape,
            mask_index_for_mask_1d_index=self.regions._sub_mask_index_for_sub_mask_1d_index,
        ).astype("bool")

    @property
    def edge_buffed_mask(self):
        edge_buffed_mask = mask_util.buffed_mask_from(mask=self).astype("bool")
        return self.__class__(
            mask=edge_buffed_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def rescaled_mask_from_rescale_factor(self, rescale_factor):
        rescaled_mask = mask_util.rescaled_mask_from(
            mask=self, rescale_factor=rescale_factor
        )
        return self.__class__(
            mask=rescaled_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def binned_mask_from_bin_up_factor(self, bin_up_factor):

        binned_up_mask = binning_util.bin_mask(mask=self, bin_up_factor=bin_up_factor)

        return self.__class__(
            mask=binned_up_mask,
            pixel_scales=self.binned_pixel_scales_from_bin_up_factor(
                bin_up_factor=bin_up_factor
            ),
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def resized_mask_from_new_shape(self, new_shape):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """

        mask = copy.deepcopy(self)

        resized_mask = array_util.resized_array_2d_from_array_2d(
            array_2d=mask, resized_shape=new_shape
        ).astype("bool")

        return self.__class__(
            mask=resized_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def trimmed_array_from_padded_array_and_image_shape(
        self, padded_array, image_shape
    ):
        """Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array : np.ndarray
            A 1D array of values which were computed using a padded grid
        """

        pad_size_0 = self.shape[0] - image_shape[0]
        pad_size_1 = self.shape[1] - image_shape[1]
        trimmed_array = padded_array.in_2d_binned[
            pad_size_0 // 2 : self.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.shape[1] - pad_size_1 // 2,
        ]
        return arrays.Array.manual(
            array=trimmed_array,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

    def unmasked_blurred_array_from_padded_array_psf_and_image_shape(
        self, padded_array, psf, image_shape
    ):
        """For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens dataset's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : aa.Kernel
            The PSF of the image used for convolution.
        unmasked_image_1d : np.ndarray
            The 1D unmasked image which is blurred.
        """

        blurred_image = psf.convolved_array_from_array(array=padded_array)

        return self.trimmed_array_from_padded_array_and_image_shape(
            padded_array=blurred_image, image_shape=image_shape
        )

    def output_to_fits(self, file_path, overwrite=False):
        """
        Write the 2D Mask to a .fits file.

        Before outputting a NumPy array, the array may be flipped upside-down using np.flipud depending on the project
        config files. This is for Astronomy projects so that structures appear the same orientation as ``.fits`` files
        loaded in DS9.

        Parameters
        ----------
        file_path : str
            The full path of the file that is output, including the file name and ``.fits`` extension.
        overwrite : bool
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
        array_util.numpy_array_2d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )


class Mask2D(AbstractMask2D):
    @classmethod
    def manual(
        cls,
        mask: np.ndarray or list,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see ``AbstractMask2D.__new__``) by inputting the array values in 2D, for example:

        mask=np.array([[False, False],
                       [True, False]])

        mask=[[False, False],
               [True, False]]

        Parameters
        ----------
        mask : np.ndarray or list
            The ``bool`` values of the mask input as an ``np.ndarray`` of shape [total_y_pixels, total_x_pixels] or a
            list of lists.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return cls(
            mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def unmasked(
        cls,
        shape_2d: (int, int),
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """Create a mask where all pixels are `False` and therefore unmasked.

        Parameters
        ----------
        shape_2d : (int, int)
            The 2D shape of the mask that is created.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        return cls.manual(
            mask=np.full(shape=shape_2d, fill_value=False),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular(
        cls,
        shape_2d: (int, int),
        radius: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within a circle of input radius.

        The ``radius`` and ``centre`` are both input in scaled units.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        radius : float
            The radius in scaled units of the circle within which pixels are `False` and unmasked.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the circle used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_util.mask_circular_from(
            shape_2d=shape_2d, pixel_scales=pixel_scales, radius=radius, centre=centre
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
        shape_2d: (int, int),
        inner_radius: float,
        outer_radius: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an annulus of input
        inner radius and outer radius.

        The ``inner_radius``, ``outer_radius`` and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        inner_radius : float
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius : float
            The outer radius in scaled units of the annulus within which pixels are `False` and unmasked.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the annulus used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_util.mask_circular_annular_from(
            shape_2d=shape_2d,
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
        shape_2d: (int, int),
        inner_radius: float,
        outer_radius: float,
        outer_radius_2: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an inner circle and second
        outer circle, forming an inverse annulus.

        The ``inner_radius``, ``outer_radius``, ``outer_radius_2`` and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        inner_radius : float
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius : float
            The first outer radius in scaled units of the annulus within which pixels are `True` and masked.
        outer_radius_2 : float
            The second outer radius in scaled units of the annulus within which pixels are `False` and unmasked and
            outside of which all entries are `True` and masked.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the anti-annulus used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_util.mask_circular_anti_annular_from(
            shape_2d=shape_2d,
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
        shape_2d: (int, int),
        major_axis_radius: float,
        axis_ratio: float,
        phi: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an ellipse.
        
        The ``major_axis_radius``, and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        major_axis_radius : float
            The major-axis in scaled units of the ellipse within which pixels are unmasked.
        axis_ratio : float
            The axis-ratio of the ellipse within which pixels are unmasked.
        phi : float
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive \
             x-axis).
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centred of the ellipse used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_util.mask_elliptical_from(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            major_axis_radius=major_axis_radius,
            axis_ratio=axis_ratio,
            phi=phi,
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
        shape_2d: (int, int),
        inner_major_axis_radius: float,
        inner_axis_ratio: float,
        inner_phi: float,
        outer_major_axis_radius: float,
        outer_axis_ratio: float,
        outer_phi: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an elliptical annulus of input
       inner and outer scaled major-axis and centre.

        The ``outer_major_axis_radius``, ``inner_major_axis_radius`` and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_2d (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        inner_major_axis_radius : float
            The major-axis in scaled units of the inner ellipse within which pixels are masked.
        inner_axis_ratio : float
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi : float
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the \
            positive x-axis).
        outer_major_axis_radius : float
            The major-axis in scaled units of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio : float
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi : float
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the \
            positive x-axis).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the elliptical annuli used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_util.mask_elliptical_annular_from(
            shape_2d=shape_2d,
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
        shape_2d: (int, int),
        pixel_coordinates: [[int, int]],
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        buffer: int = 0,
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are defined from an input list of list of
        pixel coordinates.

        These may be buffed via an input ``buffer``, whereby all entries in all 8 neighboring directions by this
        amount.

        Parameters
        ----------
        shape_2d (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_coordinates : [[int, int]]
            The input lists of 2D pixel coordinates where `False` entries are created.
        pixel_scales : (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        buffer : int
            All input ``pixel_coordinates`` are buffed with `False` entries in all 8 neighboring directions by this
            amount.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        mask = mask_util.mask_via_pixel_coordinates_from(
            shape_2d=shape_2d, pixel_coordinates=pixel_coordinates, buffer=buffer
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
        pixel_scales: (float, float),
        hdu: int = 0,
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        resized_mask_shape: (int, int) = None,
    ) -> "Mask2D":
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = cls(
            array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.resized_mask_from_new_shape(new_shape=resized_mask_shape)

        return mask
