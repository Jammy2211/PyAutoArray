import numpy as np

from autoarray import exc


class Region(object):
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
                "A coordinate of the Region was specified as negative."
            )

        if region[0] >= region[1]:
            raise exc.RegionException(
                "The first row in the Region was equal to or greater than the second row."
            )

        if region[2] >= region[3]:
            raise exc.RegionException(
                "The first column in the Region was equal to greater than the second column."
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
        return "<Region {} {} {} {}>".format(*self)

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


def check_parallel_front_edge_size(region, rows):
    # TODO: are these checks important?
    if (
        rows[0] < 0
        or rows[1] < 1
        or rows[1] > region.y1 - region.y0
        or rows[0] >= rows[1]
    ):
        raise exc.CIPatternException(
            "The number of rows to extract from the leading edge is bigger than the entire"
            "ci ci_region"
        )


def check_serial_front_edge_size(region, columns):

    if (
        columns[0] < 0
        or columns[1] < 1
        or columns[1] > region.x1 - region.x0
        or columns[0] >= columns[1]
    ):
        raise exc.CIPatternException(
            "The number of columns to extract from the leading edge is bigger than the entire"
            "ci ci_region"
        )
