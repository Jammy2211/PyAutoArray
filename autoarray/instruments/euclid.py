from typing import Dict, Tuple, Optional

from autoarray.structures.header import Header
from autoarray.structures.two_d.array_2d import Array2D
from autoarray.layout.layout import Layout2D
from autoarray.layout.region import Region2D

from autoarray.layout import layout_util


def roe_corner_from(ccd_id, quadrant_id):

    row_index = ccd_id[-1]

    if (row_index in "123") and (quadrant_id == "E"):
        return (1, 0)
    elif (row_index in "123") and (quadrant_id == "F"):
        return (1, 1)
    elif (row_index in "123") and (quadrant_id == "G"):
        return (0, 1)
    elif (row_index in "123") and (quadrant_id == "H"):
        return (0, 0)
    elif (row_index in "456") and (quadrant_id == "E"):
        return (0, 1)
    elif (row_index in "456") and (quadrant_id == "F"):
        return (0, 0)
    elif (row_index in "456") and (quadrant_id == "G"):
        return (1, 0)
    elif (row_index in "456") and (quadrant_id == "H"):
        return (1, 1)


class Array2DEuclid(Array2D):
    """
    In the Euclid FPA, the quadrant id ('E', 'F', 'G', 'H') depends on whether the CCD is located
    on the left side (rows 1-3) or right side (rows 4-6) of the FPA:

    LEFT SIDE ROWS 1-2-3
    --------------------

     <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          |
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx H xxxxxxxxx] [xxxxxxxxx G xxxxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                   | of the ndarrays)
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    P   [xxxxxxxxx E xxxxxxxxx] [xxxxxxxxx F xxxxxxxxx] P          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

    [] [========= 0 =========] [========= 1 =========] []
        <---------S----------   ----------S----------->


    RIGHT SIDE ROWS 4-5-6
    ---------------------

     <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          |
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx F xxxxxxxxx] [xxxxxxxxx E xxxxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                   | of the ndarrays)
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    P   [xxxxxxxxx G xxxxxxxxx] [xxxxxxxxx H xxxxxxxxx] P          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

    [] [========= 0 =========] [========= 1 =========] []
        <---------S----------   ----------S----------->

    Therefore, to setup a quadrant image with the correct frame_geometry using its CCD id (from which
    we can extract its row number) and quadrant id, we need to first determine if the CCD is on the left / right
    side and then use its quadrant id ('E', 'F', 'G' or 'H') to pick the correct quadrant.
    """

    @classmethod
    def from_fits_header(cls, array, ext_header):
        """
        Use an input array of a Euclid quadrant and its corresponding .fits file header to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        ccd_id = ext_header["CCDID"]
        quadrant_id = ext_header["QUADID"]

        return cls.from_ccd_and_quadrant_id(
            array=array, ccd_id=ccd_id, quadrant_id=quadrant_id
        )

    @classmethod
    def from_ccd_and_quadrant_id(cls, array, ccd_id, quadrant_id):
        """
        Use an input array of a Euclid quadrant, its ccd_id and quadrant_id  to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        row_index = ccd_id[-1]

        if (row_index in "123") and (quadrant_id == "E"):
            return Array2DEuclid.bottom_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return Array2DEuclid.bottom_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return Array2DEuclid.top_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return Array2DEuclid.top_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return Array2DEuclid.top_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return Array2DEuclid.top_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return Array2DEuclid.bottom_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return Array2DEuclid.bottom_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )

    @classmethod
    def top_left(
        cls,
        array_electrons,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ):
        """
        Use an input array of a Euclid quadrant corresponding to the top-left of a Euclid CCD and rotate the quadrant
        to the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(0, 0)
        )

        header = HeaderEuclid(
            original_roe_corner=(0, 0), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.manual(array=array_electrons, pixel_scales=0.1, header=header)

    @classmethod
    def top_right(
        cls,
        array_electrons,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ):
        """
        Use an input array of a Euclid quadrant corresponding the top-left of a Euclid CCD and rotate the  quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(0, 1)
        )

        header = HeaderEuclid(
            original_roe_corner=(0, 1), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.manual(array=array_electrons, pixel_scales=0.1, header=header)

    @classmethod
    def bottom_left(
        cls,
        array_electrons,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ):
        """
        Use an input array of a Euclid quadrant corresponding to the bottom-left of a Euclid CCD and rotate the
        quadrant to the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 0)
        )

        header = HeaderEuclid(
            original_roe_corner=(1, 0), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.manual(array=array_electrons, pixel_scales=0.1, header=header)

    @classmethod
    def bottom_right(
        cls,
        array_electrons,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ):
        """
        Use an input array of a Euclid quadrant corresponding to the bottom-right of a Euclid CCD and rotate the
        quadrant to the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        header = HeaderEuclid(
            original_roe_corner=(1, 1), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.manual(array=array_electrons, pixel_scales=0.1, header=header)


class Layout2DEuclid(Layout2D):
    @classmethod
    def from_fits_header(cls, ext_header):
        """
        Use an input array of a Euclid quadrant and its corresponding .fits file header to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        ccd_id = ext_header["CCDID"]
        quadrant_id = ext_header["QUADID"]

        parallel_overscan_size = ext_header.get("PAROVRX", default=None)
        if parallel_overscan_size is None:
            parallel_overscan_size = 0
        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        return cls.from_ccd_and_quadrant_id(
            ccd_id=ccd_id,
            quadrant_id=quadrant_id,
            parallel_size=parallel_size,
            serial_size=serial_size,
            parallel_overscan_size=parallel_overscan_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
        )

    @classmethod
    def from_ccd_and_quadrant_id(
        cls,
        ccd_id,
        quadrant_id,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):
        """
        Use an input array of a Euclid quadrant, its ccd_id and quadrant_id  to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        row_index = ccd_id[-1]

        if (row_index in "123") and (quadrant_id == "E"):
            return Layout2DEuclid.bottom_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return Layout2DEuclid.bottom_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return Layout2DEuclid.top_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return Layout2DEuclid.top_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return Layout2DEuclid.top_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return Layout2DEuclid.top_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return Layout2DEuclid.bottom_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return Layout2DEuclid.bottom_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )

    @classmethod
    def top_left(
        cls,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    0,
                    parallel_overscan_size,
                    serial_prescan_size,
                    serial_size - serial_overscan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D((0, parallel_size, 0, serial_prescan_size))
        serial_overscan = Region2D(
            (
                0,
                parallel_size - parallel_overscan_size,
                serial_size - serial_overscan_size,
                serial_size,
            )
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(0, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(0, 0))

    @classmethod
    def top_right(
        cls,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    0,
                    parallel_overscan_size,
                    serial_overscan_size,
                    serial_size - serial_prescan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )
        serial_overscan = Region2D(
            (0, parallel_size - parallel_overscan_size, 0, serial_overscan_size)
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(0, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(0, 1))

    @classmethod
    def bottom_left(
        cls,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    parallel_size - parallel_overscan_size,
                    parallel_size,
                    serial_prescan_size,
                    serial_size - serial_overscan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D((0, parallel_size, 0, serial_prescan_size))
        serial_overscan = Region2D(
            (
                0,
                parallel_size - parallel_overscan_size,
                serial_size - serial_overscan_size,
                serial_size,
            )
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(1, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(1, 0))

    @classmethod
    def bottom_right(
        cls,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    parallel_size - parallel_overscan_size,
                    parallel_size,
                    serial_overscan_size,
                    serial_size - serial_prescan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )
        serial_overscan = Region2D(
            (0, parallel_size - parallel_overscan_size, 0, serial_overscan_size)
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(1, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(1, 1))


class HeaderEuclid(Header):
    def __init__(
        self,
        header_sci_obj: Dict = None,
        header_hdu_obj: Dict = None,
        original_roe_corner: Tuple[int, int] = None,
        readout_offsets: Optional[Tuple] = None,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ):

        super().__init__(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            original_roe_corner=original_roe_corner,
            readout_offsets=readout_offsets,
        )

        self.ccd_id = ccd_id
        self.quadrant_id = quadrant_id

    @property
    def row_index(self) -> str:
        if self.ccd_id is not None:
            return self.ccd_id[-1]
