import numpy as np

from autoarray import exc

from typing import Tuple


class Region1D(object):
    def __init__(self, region: Tuple[int, int]):
        """
        Setup a region of an `Structure1D` object (e.g. `Array1D`, `Grid1D`, etc.), which could be where the parallel
        overscan, serial overscan, etc. are.

        This is defined as a tuple of pixel indexes (x0, x1).

        For example, if an `Array1D` has `shape_native` = (10,), a region (2, 4) would be defined over the region
        `array[2,4]`.

        Parameters
        -----------
        region
            The (x0, x1) pixel indexes on the image defining the region.
        """

        if region[0] < 0 or region[1] < 0:
            raise exc.RegionException(
                "A coordinate of the Region1D was specified as negative."
            )

        if region[0] >= region[1]:
            raise exc.RegionException(
                "The first pixel index in the Region1D was equal to or greater than the second pixel index."
            )

        self.region = region

    @property
    def total_pixels(self):
        return self.x1 - self.x0

    @property
    def x0(self):
        return self[0]

    @property
    def x1(self):
        return self[1]

    def __getitem__(self, item):
        return self.region[item]

    def __eq__(self, other):
        if self.region == other:
            return True
        return super().__eq__(other)

    def __repr__(self):
        return "<Region1D {} {}>".format(*self)

    @property
    def slice(self):
        return np.s_[self.x0 : self.x1]

    @property
    def x_slice(self):
        return np.s_[self.x0 : self.x1]

    @property
    def shape(self):
        return self.x1 - self.x0


class Region2D(object):
    def __init__(self, region):
        """Setup a region of an image, which could be where the parallel overscan, serial overscan, etc. are.

        This is defined as a tuple (y0, y1, x0, x1).

        Parameters
        -----------
        region : (int,)
            The coordinates on the image of the region (y0, y1, x0, y1).
        """

        if region[0] < 0 or region[1] < 0 or region[2] < 0 or region[3] < 0:
            raise exc.RegionException(
                "A coordinate of the Region2D was specified as negative."
            )

        if region[0] >= region[1]:
            raise exc.RegionException(
                "The first row in the Region2D was equal to or greater than the second row."
            )

        if region[2] >= region[3]:
            raise exc.RegionException(
                "The first column in the Region2D was equal to greater than the second column."
            )
        self.region = region

    @property
    def total_rows(self):
        return self.y1 - self.y0

    @property
    def total_columns(self):
        return self.x1 - self.x0

    @property
    def y0(self):
        return self[0]

    @property
    def y1(self):
        return self[1]

    @property
    def x0(self):
        return self[2]

    @property
    def x1(self):
        return self[3]

    def __getitem__(self, item):
        return self.region[item]

    def __eq__(self, other):
        if self.region == other:
            return True
        return super().__eq__(other)

    def __repr__(self):
        return "<Region2D {} {} {} {}>".format(*self)

    @property
    def slice(self):
        return np.s_[self.y0 : self.y1, self.x0 : self.x1]

    @property
    def y_slice(self):
        return np.s_[self.y0 : self.y1]

    @property
    def x_slice(self):
        return np.s_[self.x0 : self.x1]

    @property
    def shape(self):
        return self.y1 - self.y0, self.x1 - self.x0

    def x_limits_from(self, columns):

        x_coord = self.x0
        x_min = x_coord + columns[0]
        x_max = x_coord + columns[1]

        return x_min, x_max

    def parallel_front_edge_region_from(self, rows):

        y_coord = self.y0
        y_min = y_coord + rows[0]
        y_max = y_coord + rows[1]

        return Region2D((y_min, y_max, self.x0, self.x1))

    def parallel_trails_of_region_from(self, rows=(0, 1)):

        y_coord = self.y1
        y_min = y_coord + rows[0]
        y_max = y_coord + rows[1]

        return Region2D((y_min, y_max, self.x0, self.x1))

    def parallel_side_nearest_read_out_region_from(self, shape_2d, columns=(0, 1)):

        x_min, x_max = self.x_limits_from(columns)

        return Region2D(region=(0, shape_2d[0], x_min, x_max))

    def serial_front_edge_of_region_from(self, columns=(0, 1)):
        x_min, x_max = self.x_limits_from(columns)
        return Region2D(region=(self.y0, self.y1, x_min, x_max))

    def serial_trails_of_region_from(self, columns=(0, 1)):

        x_coord = self.x1
        x_min = x_coord + columns[0]
        x_max = x_coord + columns[1]

        return Region2D(region=(self.y0, self.y1, x_min, x_max))

    def serial_entire_rows_of_region_from(self, shape_2d):
        return Region2D(region=(self.y0, self.y1, 0, shape_2d[1]))
