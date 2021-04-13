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
