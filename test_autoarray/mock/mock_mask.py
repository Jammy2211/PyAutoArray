import numpy as np

from autoarray.mask import mask as msk
from autoarray.util import mask_util
from autoarray.mask import geometry, mapping, regions
from autoarray import exc


class MockMask(msk.Mask):
    def __new__(
        cls,
        mask_2d,
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        origin=(0.0, 0.0),
        *args,
        **kwargs
    ):

        obj = mask_2d.view(cls)
        obj.pixel_scales = pixel_scales
        obj.sub_size = sub_size
        obj.origin = origin
        return obj

    def __init__(
        self,
        mask_2d,
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        origin=(0.0, 0.0),
        *args,
        **kwargs
    ):
        pass

    def blurring_mask_from_kernel_shape(self, kernel_shape_2d):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        kernel_shape_2d : (int, int)
           The shape of the psf which defines the blurring region (e.aa. the shape of the PSF)
        """

        if kernel_shape_2d[0] % 2 == 0 or kernel_shape_2d[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_util.blurring_mask_2d_from_mask_2d_and_kernel_shape_2d(
            self, kernel_shape_2d
        )

        return MockMask(mask_2d=blurring_mask, pixel_scales=self.pixel_scales)


class MockMask1D(np.ndarray):
    def __new__(cls, shape, pixel_scales=1.0, *args, **kwargs):

        array = np.full(fill_value=False, shape=shape)

        obj = np.array(array, dtype="bool").view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = (0.0, 0.0)

        return obj
