import logging

import numpy as np

from autoarray import exc
from autoarray.mask import geometry, mapping, regions
from autoarray.util import array_util, binning_util, mask_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(
        cls, array_2d, *args, **kwargs
    ):
        """ A mask, which is applied to a 2D array of hyper_galaxies to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        array_2d: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        obj = array_2d.view(cls)
        return obj

    @property
    def mapping(self):
        return mapping.Mapping(mask_2d=self)

    @property
    def geometry(self):
        return self.mapping.geometry

    @property
    def regions(self):
        return self.mapping.regions

    @classmethod
    def unmasked_from_shape(
        cls, shape, invert=False
    ):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        mask = np.full(tuple(map(lambda d: int(d), shape)), False)
        if invert:
            mask = np.invert(mask)
        return cls(
            array_2d=mask,
        )

    @classmethod
    def from_fits(cls, file_path, hdu):
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
        return Mask(
            array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
        )

    def output_mask_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_2d_to_fits(
            array_2d=self, file_path=file_path, overwrite=overwrite
        )

    @property
    def is_sub(self):
        return False

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))


class ScaledMask(Mask):

    # noinspection PyUnusedLocal
    def __new__(
        cls, array_2d, pixel_scales, origin=(0.0, 0.0), *args, **kwargs
    ):
        """ A mask, which is applied to a 2D array of hyper_galaxies to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        array_2d: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        obj = super(ScaledMask, cls).__new__(cls, array_2d=array_2d)
        obj.pixel_scales = pixel_scales
        obj.origin = origin
        return obj

    @property
    def mapping(self):
        return mapping.ScaledMapping(mask_2d=self, pixel_scales=self.pixel_scales, origin=self.origin)

    @classmethod
    def unmasked_from_shape(
        cls, shape, pixel_scales, origin=(0.0, 0.0), invert=False
    ):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        mask = np.full(tuple(map(lambda d: int(d), shape)), False)
        if invert:
            mask = np.invert(mask)
        return cls(
            array_2d=mask, pixel_scales=pixel_scales, origin=origin,
        )

    @classmethod
    def from_fits(cls, file_path, hdu, pixel_scales, origin=(0.0, 0.0)):
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
        return ScaledMask(
            array_2d=array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu), pixel_scales=pixel_scales, origin=origin
        )


class ScaledSubMask(ScaledMask):

    # noinspection PyUnusedLocal
    def __new__(
        cls, array_2d, pixel_scales, sub_size, origin=(0.0, 0.0), *args, **kwargs
    ):
        """ A mask, which is applied to a 2D array of hyper_galaxies to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        array_2d: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        obj = super(ScaledSubMask, cls).__new__(cls, array_2d=array_2d, sub_size=sub_size, pixel_scales=pixel_scales, origin=origin)
        obj.sub_size = sub_size
        return obj

    @property
    def mapping(self):
        return mapping.ScaledSubMapping(mask_2d=self, sub_size=self.sub_size, pixel_scales=self.pixel_scales, origin=self.origin)

    @classmethod
    def unmasked_from_shape(
        cls, shape, pixel_scales, sub_size, origin=(0.0, 0.0), invert=False
    ):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        mask = np.full(tuple(map(lambda d: int(d), shape)), False)
        if invert:
            mask = np.invert(mask)
        return ScaledSubMask(
            array_2d=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def circular(
        cls,
        shape,
        pixel_scales,
        radius_arcsec,
        sub_size,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
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
        mask = mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=shape,
            pixel_scales=pixel_scales,
            radius_arcsec=radius_arcsec,
            centre=centre,
        )
        if invert:
            mask = np.invert(mask)
        return ScaledSubMask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def circular_annular(
        cls,
        shape,
        pixel_scales,
        sub_size,
        inner_radius_arcsec,
        outer_radius_arcsec,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
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

        mask = mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=shape,
            pixel_scales=pixel_scales,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return ScaledSubMask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def circular_anti_annular(
        cls,
        shape,
        pixel_scales,
        sub_size,
        inner_radius_arcsec,
        outer_radius_arcsec,
        outer_radius_2_arcsec,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
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

        mask = mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=shape,
            pixel_scales=pixel_scales,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            outer_radius_2_arcsec=outer_radius_2_arcsec,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return ScaledSubMask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def elliptical(
        cls,
        shape,
        pixel_scales,
        major_axis_radius_arcsec,
        axis_ratio,
        phi,
        sub_size,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
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

        mask = mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=shape,
            pixel_scales=pixel_scales,
            major_axis_radius_arcsec=major_axis_radius_arcsec,
            axis_ratio=axis_ratio,
            phi=phi,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return ScaledSubMask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def elliptical_annular(
        cls,
        shape,
        pixel_scales,
        sub_size,
        inner_major_axis_radius_arcsec,
        inner_axis_ratio,
        inner_phi,
        outer_major_axis_radius_arcsec,
        outer_axis_ratio,
        outer_phi,
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

        mask = mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=shape,
            pixel_scales=pixel_scales,
            inner_major_axis_radius_arcsec=inner_major_axis_radius_arcsec,
            inner_axis_ratio=inner_axis_ratio,
            inner_phi=inner_phi,
            outer_major_axis_radius_arcsec=outer_major_axis_radius_arcsec,
            outer_axis_ratio=outer_axis_ratio,
            outer_phi=outer_phi,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return ScaledSubMask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
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

    @property
    def is_sub(self):
        return True