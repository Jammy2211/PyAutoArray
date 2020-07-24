import numpy as np

from autoarray.structures import arrays
from autoarray.structures import frame as f
from autoarray.structures import region as reg
from autoarray.util import frame_util

from autoarray import exc


class AbstractFrame(arrays.Array):
    def __new__(
        cls, array, mask, original_roe_corner=(1, 0), scans=None, exposure_info=None
    ):
        """Abstract class for the geometry of a CTI Image.

        A f.FrameArray is stored as a 2D NumPy arrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the NumPy arrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions
        defined in this class (and its children). These routines define how an image is rotated before parallel
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : frame.Region
            The parallel overscan region of the ci_frame.
        serial_prescan : frame.Region
            The serial prescan region of the ci_frame.
        serial_overscan : frame.Region
            The serial overscan region of the ci_frame.
        """

        if type(array) is list:
            array = np.asarray(array)

        array[mask == True] = 0.0

        obj = super(AbstractFrame, cls).__new__(
            cls=cls, array=array, mask=mask, store_in_1d=False
        )

        obj.original_roe_corner = original_roe_corner
        obj.scans = scans or Scans()
        obj.exposure_info = exposure_info

        return obj

    def __array_finalize__(self, obj):

        super(AbstractFrame, self).__array_finalize__(obj)

        if isinstance(obj, AbstractFrame):
            if hasattr(obj, "roe_corner"):
                self.roe_corner = obj.roe_corner

            if hasattr(obj, "original_roe_corner"):
                self.original_roe_corner = obj.original_roe_corner

            if hasattr(obj, "scans"):
                self.scans = obj.scans

            if hasattr(obj, "exposure_info"):
                self.exposure_info = obj.exposure_info

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractFrame, self).__reduce__()
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
        super(AbstractFrame, self).__setstate__(state[0:-1])

    @property
    def original_orientation(self):
        return frame_util.rotate_array_from_roe_corner(
            array=self, roe_corner=self.original_roe_corner
        )

    @property
    def binned_across_parallel(self):
        return np.mean(np.ma.masked_array(self, self.mask), axis=0)

    @property
    def binned_across_serial(self):
        return np.mean(np.ma.masked_array(self, self.mask), axis=1)

    @property
    def parallel_overscan_frame(self):
        """Extract an arrays of all of the parallel trails in the parallel overscan region, that are to the side of a
        charge-injection scans from a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = parallel prescan       [ssssssssss] = parallel overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / parallel charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][000000000000000000000][tst]
        | [000][000000000000000000000][sts]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][000000000000000000000][tst]    | clocking
          [000][000000000000000000000][sts]    |

        []     [=====================]
               <---------S----------
        """
        return f.Frame.extracted_frame_from_frame_and_extraction_region(
            frame=self, extraction_region=self.scans.parallel_overscan
        )

    @property
    def parallel_overscan_binned_line(self):
        return self.parallel_overscan_frame.binned_across_serial

    def parallel_trail_from_y(self, y, dy):
        """GridCoordinates of a parallel trail of size dy from coordinate y"""
        return (int(y - dy), int(y + 1))

    def serial_trail_from_x(self, x, dx):
        """GridCoordinates of a serial trail of size dx from coordinate x"""
        return (int(x), int(x + 1 + dx))

    def parallel_front_edge_of_region(self, region, rows):

        reg.check_parallel_front_edge_size(region=region, rows=rows)

        y_coord = region.y0
        y_min = y_coord + rows[0]
        y_max = y_coord + rows[1]

        return reg.Region((y_min, y_max, region.x0, region.x1))

    def parallel_trails_of_region(self, region, rows=(0, 1)):
        y_coord = region.y1
        y_min = y_coord + rows[0]
        y_max = y_coord + rows[1]
        return reg.Region((y_min, y_max, region.x0, region.x1))

    def parallel_side_nearest_read_out_region(self, region, columns=(0, 1)):
        x_min, x_max = self.x_limits(region, columns)
        return reg.Region(region=(0, self.shape_2d[0], x_min, x_max))

    @property
    def serial_overscan_frame(self):
        """Extract an arrays of all of the serial trails in the serial overscan region, that are to the side of a
        charge-injection scans from a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][000000000000000000000][tst]
        | [000][000000000000000000000][sts]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][000000000000000000000][tst]    | clocking
          [000][000000000000000000000][sts]    |

        []     [=====================]
               <---------S----------
        """
        return f.Frame.extracted_frame_from_frame_and_extraction_region(
            frame=self, extraction_region=self.scans.serial_overscan
        )

    @property
    def serial_overscan_binned_line(self):
        return self.serial_overscan_frame.binned_across_parallel

    def serial_front_edge_of_region(self, region, columns=(0, 1)):
        reg.check_serial_front_edge_size(region, columns)
        x_min, x_max = self.x_limits(region, columns)
        return reg.Region(region=(region.y0, region.y1, x_min, x_max))

    def serial_trails_of_region(self, region, columns=(0, 1)):
        x_coord = region.x1
        x_min = x_coord + columns[0]
        x_max = x_coord + columns[1]
        return reg.Region(region=(region.y0, region.y1, x_min, x_max))

    def serial_entire_rows_of_region(self, region):
        return reg.Region(region=(region.y0, region.y1, 0, self.shape_2d[1]))

    def x_limits(self, region, columns):
        x_coord = region.x0
        x_min = x_coord + columns[0]
        x_max = x_coord + columns[1]
        return x_min, x_max


class Scans:
    def __init__(
        self, parallel_overscan=None, serial_prescan=None, serial_overscan=None
    ):

        if isinstance(parallel_overscan, tuple):
            parallel_overscan = reg.Region(region=parallel_overscan)

        if isinstance(serial_prescan, tuple):
            serial_prescan = reg.Region(region=serial_prescan)

        if isinstance(serial_overscan, tuple):
            serial_overscan = reg.Region(region=serial_overscan)

        self.parallel_overscan = parallel_overscan
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan

    @classmethod
    def rotated_from_roe_corner(cls, roe_corner, shape_2d, scans):

        if scans is None:
            return Scans()

        parallel_overscan = frame_util.rotate_region_from_roe_corner(
            region=scans.parallel_overscan, shape_2d=shape_2d, roe_corner=roe_corner
        )
        serial_prescan = frame_util.rotate_region_from_roe_corner(
            region=scans.serial_prescan, shape_2d=shape_2d, roe_corner=roe_corner
        )
        serial_overscan = frame_util.rotate_region_from_roe_corner(
            region=scans.serial_overscan, shape_2d=shape_2d, roe_corner=roe_corner
        )

        return Scans(
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def after_extraction(cls, frame, extraction_region):

        parallel_overscan = frame_util.region_after_extraction(
            original_region=frame.scans.parallel_overscan,
            extraction_region=extraction_region,
        )
        serial_prescan = frame_util.region_after_extraction(
            original_region=frame.scans.serial_prescan,
            extraction_region=extraction_region,
        )
        serial_overscan = frame_util.region_after_extraction(
            original_region=frame.scans.serial_overscan,
            extraction_region=extraction_region,
        )

        return Scans(
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def from_frame(cls, frame):

        return Scans(
            parallel_overscan=frame.scans.parallel_overscan,
            serial_prescan=frame.scans.serial_prescan,
            serial_overscan=frame.scans.serial_overscan,
        )

    @property
    def serial_trails_columns(self):
        return self.serial_overscan[3] - self.serial_overscan[2]
