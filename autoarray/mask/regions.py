import numpy as np

from autoarray import exc
from autoarray.mask import mask as msk
from autoarray.util import array_util, mask_util


class Regions:
    def __init__(self, mask):

        self.mask = mask

    @property
    def _mask_index_for_mask_1d_index(self):
        """A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates."""
        return mask_util.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=self.mask, sub_size=1
        ).astype("int")

    @property
    def _edge_1d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a *True* value).
        """
        return mask_util.edge_1d_indexes_from(mask=self.mask).astype("int")

    @property
    def _edge_2d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a *True* value).
        """
        return self._mask_index_for_mask_1d_index[self._edge_1d_indexes].astype("int")

    @property
    def _border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.border_1d_indexes_from(mask=self.mask).astype("int")

    @property
    def _border_2d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return self._mask_index_for_mask_1d_index[self._border_1d_indexes].astype("int")

    @property
    def _sub_border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.sub_border_pixel_1d_indexes_from(
            mask=self.mask, sub_size=self.mask.sub_size
        ).astype("int")

    @array_util.Memoizer()
    def blurring_mask_from_kernel_shape(self, kernel_shape_2d):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        kernel_shape_2d : (int, int)
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if kernel_shape_2d[0] % 2 == 0 or kernel_shape_2d[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_util.blurring_mask_from(
            mask=self.mask, kernel_shape_2d=kernel_shape_2d
        )

        return msk.Mask(
            mask=blurring_mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def unmasked_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return msk.Mask.unmasked(
            shape_2d=self.mask.shape_2d,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def edge_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[self._edge_2d_indexes[:, 0], self._edge_2d_indexes[:, 1]] = False
        return msk.Mask(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def border_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[self._border_2d_indexes[:, 0], self._border_2d_indexes[:, 1]] = False
        return msk.Mask(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def _sub_mask_index_for_sub_mask_1d_index(self):
        """A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates."""
        return mask_util.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=self.mask, sub_size=self.mask.sub_size
        ).astype("int")

    @property
    @array_util.Memoizer()
    def _mask_1d_index_for_sub_mask_1d_index(self):
        """The util between every sub-pixel and its host pixel.

        For example:

        - sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd pixel.
        - sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th pixel.
        """
        return mask_util.mask_1d_index_for_sub_mask_1d_index_via_mask_from(
            mask=self.mask, sub_size=self.mask.sub_size
        ).astype("int")
