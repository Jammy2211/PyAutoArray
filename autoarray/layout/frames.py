import numpy as np

from autoarray.structures.arrays import abstract_array
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.layout import layout_util
from autoarray.mask import mask_2d as msk
from autoarray.geometry import geometry_util


class Frame2D:
    @classmethod
    def manual(
        cls, array, pixel_scales, roe_corner=(1, 0), scans=None, exposure_info=None
    ):
        """Abstract class for the geometry of a CTI Image.

        A FrameArray is stored as a 2D ndarrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the ndarrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions
        defined in this class (and its children). These routines define how an image is rotated before parallel
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : frame.Region2D
            The parallel overscan region of the frame_ci.
        serial_prescan : frame.Region2D
            The serial prescan region of the frame_ci.
        serial_overscan : frame.Region2D
            The serial overscan region of the frame_ci.
        """

        array = abstract_array.convert_array(array=array)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = msk.Mask2D.unmasked(shape_native=array.shape, pixel_scales=pixel_scales)

        scans = abstract_frame.Scans.rotated_from_roe_corner(
            roe_corner=roe_corner, shape_native=array.shape, scans=scans
        )

        return cls(
            array=frame_util.rotate_array_from_roe_corner(
                array=array, roe_corner=roe_corner
            ),
            mask=mask,
            original_roe_corner=roe_corner,
            scans=scans,
            exposure_info=exposure_info,
        )

    @classmethod
    def manual_mask(
        cls, array, mask, scans=None, roe_corner=(1, 0), exposure_info=None
    ):

        """Abstract class for the geometry of a CTI Image.

        A FrameArray is stored as a 2D ndarrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the ndarrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions
        defined in this class (and its children). These routines define how an image is rotated before parallel
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : Region2D
            The parallel overscan region of the frame_ci.
        serial_prescan : Region2D
            The serial prescan region of the frame_ci.
        serial_overscan : Region2D
            The serial overscan region of the frame_ci.
        """

        array = abstract_array.convert_array(array=array)

        array = frame_util.rotate_array_from_roe_corner(
            array=array, roe_corner=roe_corner
        )
        mask = frame_util.rotate_array_from_roe_corner(
            array=mask, roe_corner=roe_corner
        )

        array[mask == True] = 0.0

        scans = abstract_frame.Scans.rotated_from_roe_corner(
            roe_corner=roe_corner, shape_native=array.shape, scans=scans
        )

        return cls(
            array=array,
            mask=mask,
            original_roe_corner=roe_corner,
            scans=scans,
            exposure_info=exposure_info,
        )
