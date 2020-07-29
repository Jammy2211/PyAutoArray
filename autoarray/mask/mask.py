import logging

import numpy as np

from autoarray import exc
from autoarray.structures import abstract_structure
from autoarray.mask import abstract_mask
from autoarray.util import array_util, mask_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(abstract_mask.AbstractMask):
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

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return cls(
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


class Mask1D(np.ndarray):
    def __new__(cls, mask, pixel_scale=None, origin=0.0, *args, **kwargs):
        """ A mask, which is applied to data to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        mask: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        mask = mask.astype("bool")
        obj = mask.view(cls)
        obj.pixel_scales = pixel_scale
        obj.origin = origin
        return obj

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Mask1D, self).__reduce__()
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
        super(Mask1D, self).__setstate__(state[0:-1])

    def __array_finalize__(self, obj):

        if isinstance(obj, Mask):
            self.pixel_scale = obj.pixel_scale
            self.origin = obj.origin
        else:
            self.origin = 0.0
            self.pixel_scale = None

    @classmethod
    def manual(cls, mask, pixel_scale=None, origin=0.0, invert=False):

        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        if len(mask.shape) != 1:
            raise exc.MaskException("The input mask is not a one dimensional array")

        return Mask1D(mask=mask, pixel_scale=pixel_scale, origin=origin)

    @classmethod
    def unmasked(cls, shape_1d, pixel_scale=None, origin=0.0, invert=False):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls.manual(
            mask=np.full(shape=shape_1d, fill_value=False),
            pixel_scale=pixel_scale,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_masked_regions(cls, shape_1d, masked_regions):

        mask = cls.unmasked(shape_1d=shape_1d)
        masked_regions = list(
            map(lambda region: reg.Region(region=region), masked_regions)
        )
        for region in masked_regions:
            mask[region.x0 : region.x1] = True

        return mask

    @classmethod
    def from_cosmic_ray_map(cls, cosmic_ray_map, cosmic_ray_buffer=0):
        """
        Create the mask used for CTI Calibration, which is all False unless specific regions are input for masking.

        Parameters
        ----------
        shape_2d : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        cosmic_ray_map : Line
            2D arrays flagging where cosmic rays on the image.
        cosmic_ray_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the parallel \
            direction.
        """
        mask = cls.unmasked(shape_1d=cosmic_ray_map.shape_1d)

        cosmic_ray_mask = (cosmic_ray_map > 0.0).astype("bool")

        # TODO : refactor after unit test.

        for x in range(mask.shape[0]):
            if cosmic_ray_mask[x]:
                mask[x : x + cosmic_ray_buffer] = True

        return mask

    @classmethod
    def from_fits(cls, file_path, pixel_scale, hdu=0, origin=0.0):
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

        mask = cls(
            array_util.numpy_array_1d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scale=pixel_scale,
            origin=origin,
        )

        return mask

    def output_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_1d_to_fits(
            array_1d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def is_all_false(self):
        return self.pixels_in_mask == self.shape_1d

    @property
    def shape_1d(self):
        return self.shape[0]

    @property
    def shape_1d_scaled(self):
        return float(self.pixel_scale * self.shape_1d)

    @property
    def scaled_maxima(self):
        return (self.shape_1d_scaled / 2.0) + self.origin

    @property
    def scaled_minima(self):
        return -(self.shape_1d_scaled / 2.0) + self.origin

    @property
    def extent(self):
        return np.asarray([self.scaled_minima, self.scaled_maxima])
