import numpy as np


class AbstractStructure(np.ndarray):
    def __new__(cls, structure, mask, store_in_1d=True, *args, **kwargs):
        """A grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of an \
        unmasked pixel. The positive y-axis is upwards and poitive x-axis to the right.

        A *Grid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_unmasked_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixel index and second element the y or x arc -econd coordinates. For example:

        - grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the grid)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the grid)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        The mask pixel index's will come out like this (and the direction of arc-second coordinates is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        |x|x|x|x|x|x|x|x|x|x|  ^   grid[0] = [ 1.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[1] = [ 1.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[2] = [ 0.5, -1.5]
        |x|x|x|x|0|1|x|x|x|x| +ve  grid[3] = [ 0.5, -0.5]
        |x|x|x|2|3|4|5|x|x|x|  y   grid[4] = [ 0.5,  0.5]
        |x|x|x|6|7|8|9|x|x|x| -ve  grid[5] = [ 0.5,  1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[6] = [-0.5, -1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[7] = [-0.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x| \/   grid[8] = [-0.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|      grid[9] = [-0.5,  1.5]

        A sub-grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of each \
        sub-pixel of an unmasked pixel (e.g. the pixels of a grid). The positive y-axis is upwards and poitive \
        x-axis to the right, and this convention is followed for the sub-pixels in each unmasked pixel.

        A *ScaledSubGrid* is ordered such that pixels begin from the first (top-left) sub-pixel in the first unmasked pixel. \
        Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel. Therefore, \
        the sub-grid is an ndarray of shape [total_unmasked_pixels*(sub_grid_shape)**2, 2]. For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Like the grid, the indexing of each sub-pixel goes from \
        the top-left corner. In contrast to the grid above, our illustration below restricts the mask to just \
        2 pixels, to keep the illustration brief.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        Our grid looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

        |x|x|x|x|x|x|x|x|x|x|  ^
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |                        y     x
        |x|x|x|x|x|x|x|x|x|x| +ve  grid[0] = [0.5,  -1.5]
        |x|x|x|0|1|x|x|x|x|x|  y   grid[1] = [0.5,  -0.5]
        |x|x|x|x|x|x|x|x|x|x| -ve
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| \/
        |x|x|x|x|x|x|x|x|x|x|

        However, we now go to each unmasked pixel and derive a sub-pixel grid for it. For example, for pixel 0,
        if *sub_size=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):
                                y      x
               grid[0] = [0.66, -1.66]
        |0|1|  grid[1] = [0.66, -1.33]
        |2|3|  grid[2] = [0.33, -1.66]
               grid[3] = [0.33, -1.33]

        Now, we'd normally sub-grid all pixels using the same *sub_size*, but for this illustration lets
        pretend we used a sub_size of 3x3 for pixel 1:

                                  y      x
                 grid[0] = [0.75, -0.75]
                 grid[1] = [0.75, -0.5]
                 grid[2] = [0.75, -0.25]
        |0|1|2|  grid[3] = [0.5,  -0.75]
        |3|4|5|  grid[4] = [0.5,  -0.5]
        |6|7|8|  grid[5] = [0.5,  -0.25]
                 grid[6] = [0.25, -0.75]
                 grid[7] = [0.25, -0.5]
                 grid[8] = [0.25, -0.25]

        """
        obj = structure.view(cls)
        obj.mask = mask
        obj.store_in_1d = store_in_1d
        return obj

    def __array_finalize__(self, obj):

        if isinstance(obj, AbstractStructure):
            if hasattr(obj, "mask"):
                self.mask = obj.mask

            if hasattr(obj, "store_in_1d"):
                self.store_in_1d = obj.store_in_1d

    @property
    def shape_1d(self):
        return self.mask.shape_1d

    @property
    def sub_shape_1d(self):
        return self.mask.sub_shape_1d

    @property
    def shape_2d(self):
        return self.mask.shape

    @property
    def sub_shape_2d(self):
        return self.mask.sub_shape_2d

    @property
    def pixel_scales(self):
        return self.mask.pixel_scales

    @property
    def pixel_scale(self):
        return self.mask.pixel_scale

    @property
    def origin(self):
        return self.mask.origin

    @property
    def sub_size(self):
        return self.mask.sub_size

    @property
    def regions(self):
        return self.mask.regions

    @property
    def geometry(self):
        return self.mask.geometry

    @property
    def unmasked_grid(self):
        return self.mask.geometry.unmasked_grid

    @property
    def total_pixels(self):
        return self.shape[0]

    @property
    def binned_pixel_scales_from_bin_up_factor(self):
        return self.mask.binned_pixel_scales_from_bin_up_factor
