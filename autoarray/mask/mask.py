import logging

import numpy as np

from autoarray import exc
from autoarray.structures import arrays
from autoarray.mask import geometry, regions
from autoarray.util import array_util, binning_util, mask_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(
        cls, mask, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), *args, **kwargs
    ):
        """ A 2D mask, representing a uniform rectangular grid of neighboring rectangular pixels.

        A mask s applied to an Array or Grid structure to signify which entries are used in calculations, where a
        *False* entry signifies that the mask entry is unmasked and therefore is used in calculations.

        The mask defines the geometry of the 2D uniform grid of pixels, for example their pixel scale and coordinate
        origin. The 2D uniform grid may also be sub-gridded, whereby every pixel is sub-divided into a uniform gridd
        of sub-pixels which are all used to perform calculations more accurate. See *Grid* for a detailed description
        of sub-gridding.

        Parameters
        ----------
        mask: ndarray
            The array of shape [total_y_pixels, total_x_pixels] containing the bools representing the mask, where
            *False* signifies an entry is unmasked and used in calculations.
        pixel_scales: (float, float) or float
            The (y,x) arc-second to pixel conversion factors of every pixel. If this is input as a float, it is
            converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        """
        # noinspection PyArgumentList

        mask = mask.astype("bool")
        obj = mask.view(cls)
        obj.sub_size = sub_size
        obj.pixel_scales = pixel_scales
        obj.origin = origin
        return obj

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Mask, self).__reduce__()
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
        super(Mask, self).__setstate__(state[0:-1])

    @property
    def geometry(self):
        """The Geometry class contains methods describing the Mask's geometry, for example the Grid of unmasked
        pixels.

        See the Geometry class for a full description."""
        return geometry.Geometry(mask=self)

    @property
    def regions(self):
        """The Region class contains methods describing regions on the Mask, for example the pixel indexes of Mask
        pixels on its edge.

        See the Region class for a full description."""
        return regions.Regions(mask=self)

    def __array_finalize__(self, obj):

        if isinstance(obj, Mask):
            self.sub_size = obj.sub_size
            self.pixel_scales = obj.pixel_scales
            self.origin = obj.origin
        else:
            self.sub_size = 1
            self.origin = (0.0, 0.0)
            self.pixel_scales = None

    @classmethod
    def manual(
        cls, mask, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False
    ):
        """Create a Mask (see *Mask.__new__*) by inputting the array values in 2D, for example:

        mask=np.array([[False, False],
                       [True, False]])

        mask=[[False, False],
               [True, False]]

        Parameters
        ----------
        mask : np.ndarray or list
            The bool values of the mask input as an ndarray of shape [total_y_pixels, total_x_pixels ]or a list of
            lists.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
        """
        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return Mask(
            mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def unmasked(
        cls, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False
    ):
        """Create a mask where all pixels are *False* and therefore unmasked.

        Parameters
        ----------
        mask : np.ndarray or list
            The bool values of the mask input as an ndarray of shape [total_y_pixels, total_x_pixels ]or a list of
            lists.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
        """
        return cls.manual(
            mask=np.full(shape=shape_2d, fill_value=False),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @property
    def sub_length(self):
        """The total number of sub-pixels in a give pixel.

        For example, a sub-size of 3x3 means every pixel has 9 sub-pixels."""
        return int(self.sub_size ** 2.0)

    @property
    def sub_fraction(self):
        """The fraction of the area of a pixel every sub-pixel contains.

        For example, a sub-size of 3x3 mean every pixel contains 1/9 the area."""
        return 1.0 / self.sub_length

    @classmethod
    def circular(
        cls,
        shape_2d,
        radius,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Create a Mask (see *Mask.__new__*) where all *False* entries are within a circle of input radius and
        centre.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        radius : float
            The radius (in scaled units) of the circle within which pixels are False and unmasked.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
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
        shape_2d,
        inner_radius,
        outer_radius,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Create a Mask (see *Mask.__new__*) where all *False* entries are within an annulus of input inner radius,
         outer radius and centre.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        inner_radius : float
            The inner radius (in scaled units) of the annulus within which pixels are False and unmasked.
        outer_radius : float
            The outer radius (in scaled units) of the annulus within which pixels are False and unmasked.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
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
        shape_2d,
        inner_radius,
        outer_radius,
        outer_radius_2,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Create a Mask (see *Mask.__new__*) where all *False* entries are within an inner circle and second outer
         circle, forming an inverse annulus.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        inner_radius : float
            The inner radius (in scaled units) of the annulus within which pixels are False and unmasked.
        outer_radius : float
            The first outer radius (in scaled units) of the annulus within which pixels are True and masked.
        outer_radius_2 : float
            The second outer radius (in scaled units) of the annulus within which pixels are False and unmasked and
            outside of which all entries are True and masked..
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
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
        shape_2d,
        major_axis_radius,
        axis_ratio,
        phi,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Create a Mask (see *Mask.__new__*) where all *False* entries are within a circle of input radius and
        centre.

        Parameters
        ----------
        shape_2d : (int, int)
            The (y,x) shape of the mask in units of pixels.
        major_axis_radius : float
            The major-axis (in scaled units) of the ellipse within which pixels are unmasked.
        axis_ratio : float
            The axis-ratio of the ellipse within which pixels are unmasked.
        phi : float
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive \
             x-axis).
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
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
        shape_2d,
        inner_major_axis_radius,
        inner_axis_ratio,
        inner_phi,
        outer_major_axis_radius,
        outer_axis_ratio,
        outer_phi,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within an elliptical annulus of input inner and outer arc second \
        major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_major_axis_radius : float
            The major-axis (in arc seconds) of the inner ellipse within which pixels are masked.
        inner_axis_ratio : float
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi : float
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the \
            positive x-axis).
        outer_major_axis_radius : float
            The major-axis (in arc seconds) of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio : float
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi : float
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the \
            positive x-axis).
        centre: (float, float)
            The centre of the elliptical annuli used to mask pixels.
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
        shape_2d,
        pixel_coordinates,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        buffer=0,
        invert=False,
    ):

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
        file_path,
        pixel_scales,
        hdu=0,
        sub_size=1,
        origin=(0.0, 0.0),
        resized_mask_shape=None,
    ):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
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

    def output_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_2d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def pixel_scale(self):
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.MaskException(
                "Cannot return a pixel_scale for a a grid where each dimension has a "
                "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]"
            )

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def is_all_true(self):
        return self.pixels_in_mask == 0

    @property
    def is_all_false(self):
        return self.pixels_in_mask == self.shape_2d[0] * self.shape_2d[1]

    @property
    def sub_pixels_in_mask(self):
        return self.sub_size ** 2 * self.pixels_in_mask

    @property
    def shape_1d(self):
        return self.pixels_in_mask

    @property
    def shape_2d(self):
        return self.shape

    @property
    def sub_shape_1d(self):
        return int(self.pixels_in_mask * self.sub_size ** 2.0)

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
    def mask_sub_1(self):
        return Mask(
            mask=self, sub_size=1, pixel_scales=self.pixel_scales, origin=self.origin
        )

    @property
    def edge_buffed_mask(self):
        edge_buffed_mask = mask_util.buffed_mask_from(mask=self).astype("bool")
        return Mask(
            mask=edge_buffed_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def rescaled_mask_from_rescale_factor(self, rescale_factor):
        rescaled_mask = mask_util.rescaled_mask_from(
            mask=self, rescale_factor=rescale_factor
        )
        return Mask(
            mask=rescaled_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def mask_new_sub_size_from_mask(self, mask, sub_size=1):
        return Mask(
            mask=mask,
            sub_size=sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def binned_pixel_scales_from_bin_up_factor(self, bin_up_factor):
        if self.pixel_scales is not None:
            return (
                self.pixel_scales[0] * bin_up_factor,
                self.pixel_scales[1] * bin_up_factor,
            )
        else:
            return None

    def binned_mask_from_bin_up_factor(self, bin_up_factor):

        binned_up_mask = binning_util.bin_mask(mask=self, bin_up_factor=bin_up_factor)

        return Mask(
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

        resized_mask = array_util.resized_array_2d_from_array_2d(
            array_2d=self, resized_shape=new_shape
        ).astype("bool")

        return Mask(
            mask=resized_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def trimmed_array_from_padded_array_and_image_shape(
        self, padded_array, image_shape
    ):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array : ndarray
            A 1D array of values which were computed using a padded grid
        """

        pad_size_0 = self.shape[0] - image_shape[0]
        pad_size_1 = self.shape[1] - image_shape[1]
        trimmed_array = padded_array.in_2d_binned[
            pad_size_0 // 2 : self.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.shape[1] - pad_size_1 // 2,
        ]
        return arrays.Array.manual_2d(
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
        unmasked_image_1d : ndarray
            The 1D unmasked image which is blurred.
        """

        blurred_image = psf.convolved_array_from_array(array=padded_array)

        return self.trimmed_array_from_padded_array_and_image_shape(
            padded_array=blurred_image, image_shape=image_shape
        )
