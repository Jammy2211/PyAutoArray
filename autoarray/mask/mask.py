import logging

import numpy as np

from autoarray import exc
from autoarray.mask import geometry, mapping, regions
from autoarray.util import array_util, mask_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(
        cls, mask_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), *args, **kwargs
    ):
        """ A mask, which is applied to a 2D array of hyper_galaxies to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        mask_2d: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        obj = mask_2d.view(cls)
        obj.sub_size = sub_size
        obj.sub_length = int(sub_size ** 2.0)
        obj.sub_fraction = 1.0 / obj.sub_length
        obj.pixel_scales = pixel_scales
        obj.origin = origin
        obj.mapping = mapping.Mapping(mask=obj)
        obj.geometry = geometry.Geometry(mask=obj)
        obj.regions = regions.Regions(mask=obj)
        return obj

    @classmethod
    def manual(cls, mask_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False):

        mask_2d = np.asarray(mask_2d).astype('bool')

        if invert:
            mask_2d = np.invert(mask_2d)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if len(mask_2d.shape) != 2:
            raise exc.MaskException('The input mask_2d is not a two dimensional array')

        return Mask(mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)

    @classmethod
    def unmasked(cls, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls.manual(
            mask_2d=np.full(shape=shape_2d, fill_value=False), pixel_scales=pixel_scales, sub_size=sub_size, origin=origin, invert=invert
        )

    @classmethod
    def circular(cls, shape_2d, radius_arcsec, pixel_scales, sub_size=1, origin=(0.0, 0.0), centre=(0.0, 0.0), invert=False):
        """Setup a mask where unmasked pixels are within a circle of an input arc second radius and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        radius_arcsec : float
            The radius (in arc seconds) of the circle within which pixels unmasked.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        """
        mask_2d = mask_util.mask_2d_circular_from_shape_2d_pixel_scales_and_radius(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            radius_arcsec=radius_arcsec,
            centre=centre,
        )

        return cls.manual(
            mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin, invert=invert
        )

    @classmethod
    def circular_annular(cls, shape_2d, inner_radius_arcsec, outer_radius_arcsec, pixel_scales, sub_size=1, origin=(0.0, 0.0), centre=(0.0, 0.0),
                  invert=False):
        """Setup a mask where unmasked pixels are within an annulus of input inner and outer arc second radii and \
         centre.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The radius (in arc seconds) of the inner circle outside of which pixels are unmasked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are unmasked.
        centre: (float, float)
            The centre of the annulus used to mask pixels.
        """
        mask_2d = mask_util.mask_2d_circular_annular_from_shape_2d_pixel_scales_and_radii(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            centre=centre,
        )

        return cls.manual(
            mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin, invert=invert
        )

    @classmethod
    def circular_anti_annular(cls, shape_2d, inner_radius_arcsec, outer_radius_arcsec, outer_radius_2_arcsec, pixel_scales, sub_size=1,
                              origin=(0.0, 0.0), centre=(0.0, 0.0),
                              invert=False):
        """Setup a mask where unmasked pixels are outside an annulus of input inner and outer arc second radii, but \
        within a second outer radius, and at a given centre.

        This mask there has two distinct unmasked regions (an inner circle and outer annulus), with an inner annulus \
        of masked pixels.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The radius (in arc seconds) of the inner circle inside of which pixels are unmasked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are masked and outside of which they \
            are unmasked.
        outer_radius_2_arcsec : float
            The radius (in arc seconds) of the second outer circle within which pixels are unmasked and outside of \
            which they masked.
        centre: (float, float)
            The centre of the anti-annulus used to mask pixels.
        """
        mask_2d = mask_util.mask_2d_circular_anti_annular_from_shape_2d_pixel_scales_and_radii(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            outer_radius_2_arcsec=outer_radius_2_arcsec,
            centre=centre,
        )

        return cls.manual(
            mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin, invert=invert
        )

    @classmethod
    def elliptical(cls, shape_2d,
                   major_axis_radius_arcsec,
                   axis_ratio,
                   phi, pixel_scales, sub_size=1, origin=(0.0, 0.0), centre=(0.0, 0.0), invert=False):
        """ Setup a mask where unmasked pixels are within an ellipse of an input arc second major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the ellipse within which pixels are unmasked.
        axis_ratio : float
            The axis-ratio of the ellipse within which pixels are unmasked.
        phi : float
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive \
             x-axis).
        centre: (float, float)
            The centre of the ellipse used to mask pixels.
        """
        mask_2d = mask_util.mask_2d_elliptical_from_shape_2d_pixel_scales_and_radius(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            major_axis_radius_arcsec=major_axis_radius_arcsec,
            axis_ratio=axis_ratio,
            phi=phi,
            centre=centre,
        )

        return cls.manual(
            mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin, invert=invert
        )

    @classmethod
    def elliptical_annular(cls, shape_2d,
                           inner_major_axis_radius_arcsec,
                           inner_axis_ratio,
                           inner_phi,
                           outer_major_axis_radius_arcsec,
                           outer_axis_ratio,
                           outer_phi, pixel_scales, sub_size=1, origin=(0.0, 0.0), centre=(0.0, 0.0), invert=False):
        """Setup a mask where unmasked pixels are within an elliptical annulus of input inner and outer arc second \
        major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the inner ellipse within which pixels are masked.
        inner_axis_ratio : float
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi : float
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the \
            positive x-axis).
        outer_major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio : float
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi : float
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the \
            positive x-axis).
        centre: (float, float)
            The centre of the elliptical annuli used to mask pixels.
        """
        mask_2d = mask_util.mask_2d_elliptical_annular_from_shape_2d_pixel_scales_and_radius(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            inner_major_axis_radius_arcsec=inner_major_axis_radius_arcsec,
            inner_axis_ratio=inner_axis_ratio,
            inner_phi=inner_phi,
            outer_major_axis_radius_arcsec=outer_major_axis_radius_arcsec,
            outer_axis_ratio=outer_axis_ratio,
            outer_phi=outer_phi,
            centre=centre,
        )

        return cls.manual(
            mask_2d=mask_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin, invert=invert
        )

    @classmethod
    def from_fits(cls, file_path, hdu, pixel_scales, sub_size, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(
            array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    def output_fits(self, file_path, overwrite=False):

        array_util.numpy_array_2d_to_fits(
            array_2d=self, file_path=file_path, overwrite=overwrite
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
    def sub_pixels_in_mask(self):
        return self.sub_size ** 2 * self.pixels_in_mask

    @property
    def shape_2d(self):
        return self.shape

    @property
    def sub_shape_1d(self):
        return self.pixels_in_mask * self.sub_size ** 2.0

    @property
    def sub_shape_2d(self):
        return (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

    @property
    def sub_mask_2d(self):

        sub_shape = (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

        return mask_util.mask_2d_from_shape_2d_and_mask_2d_index_for_mask_1d_index(
            shape_2d=sub_shape,
            mask_2d_index_for_mask_1d_index=self.regions._sub_mask_2d_index_for_sub_mask_1d_index,
        ).astype("bool")

    @property
    def mask_sub_1(self):
        return self.mapping.mask_sub_1