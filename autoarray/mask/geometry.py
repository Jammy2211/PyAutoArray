import numpy as np

from autoarray.structures import arrays, grids
from autoarray.util import array_util, grid_util


class Geometry:
    def __init__(self, mask):

        self.mask = mask

    @property
    def regions(self):
        return self.mask.regions

    @property
    def central_pixel_coordinates(self):
        return (
            float(self.mask.shape_2d[0] - 1) / 2,
            float(self.mask.shape_2d[1] - 1) / 2,
        )

    @property
    def central_scaled_coordinates(self):
        return (
            self.central_pixel_coordinates[0]
            + (self.origin[0] / self.mask.pixel_scales[0]),
            self.central_pixel_coordinates[1]
            - (self.origin[1] / self.mask.pixel_scales[1]),
        )

    @property
    def origin(self):
        return self.mask.origin

    def pixel_coordinates_from_scaled_coordinates(self, scaled_coordinates):
        return (
            int(
                (
                    (-scaled_coordinates[0] + self.mask.origin[0])
                    / self.mask.pixel_scales[0]
                )
                + self.central_pixel_coordinates[0]
                + 0.5
            ),
            int(
                (
                    (scaled_coordinates[1] - self.mask.origin[1])
                    / self.mask.pixel_scales[1]
                )
                + self.central_pixel_coordinates[1]
                + 0.5
            ),
        )

    def scaled_coordinates_from_pixel_coordinates(self, pixel_coordinates):
        return (
            self.mask.pixel_scales[0]
            * -(pixel_coordinates[0] - self.central_scaled_coordinates[0]),
            self.mask.pixel_scales[1]
            * (pixel_coordinates[1] - self.central_scaled_coordinates[1]),
        )

    @property
    @array_util.Memoizer()
    def mask_centre(self):
        return grid_util.grid_centre_from(grid_1d=self.masked_grid_sub_1)

    @property
    def shape_2d_scaled(self):
        return (
            float(self.mask.pixel_scales[0] * self.mask.shape[0]),
            float(self.mask.pixel_scales[1] * self.mask.shape[1]),
        )

    @property
    def scaled_maxima(self):
        return (
            (self.shape_2d_scaled[0] / 2.0) + self.mask.origin[0],
            (self.shape_2d_scaled[1] / 2.0) + self.mask.origin[1],
        )

    @property
    def scaled_minima(self):
        return (
            (-(self.shape_2d_scaled[0] / 2.0)) + self.mask.origin[0],
            (-(self.shape_2d_scaled[1] / 2.0)) + self.mask.origin[1],
        )

    @property
    def extent(self):
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing an image-grid"""
        return np.linspace(self.scaled_minima[0], self.scaled_maxima[0], 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing an image-grid"""
        return np.linspace(self.scaled_minima[1], self.scaled_maxima[1], 4)

    @property
    def unmasked_grid_sub_1(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=1,
            origin=self.mask.origin,
        )

        return grids.Grid(
            grid=grid_1d,
            mask=self.mask.regions.unmasked_mask.mask_sub_1,
            store_in_1d=True,
        )

    @property
    def masked_grid_sub_1(self):

        grid_1d = grid_util.grid_1d_via_mask_from(
            mask=self.mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=1,
            origin=self.mask.origin,
        )
        return grids.Grid(grid=grid_1d, mask=self.mask.mask_sub_1, store_in_1d=True)

    @property
    def edge_grid_sub_1(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        edge_grid_1d = self.masked_grid_sub_1[self.regions._edge_1d_indexes]
        return grids.Grid(
            grid=edge_grid_1d, mask=self.regions.edge_mask.mask_sub_1, store_in_1d=True
        )

    @property
    def border_grid_sub_1(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        border_grid_1d = self.masked_grid_sub_1[self.regions._border_1d_indexes]
        return grids.Grid(
            grid=border_grid_1d,
            mask=self.regions.border_mask.mask_sub_1,
            store_in_1d=True,
        )

    def grid_pixels_from_grid_scaled_1d(self, grid_scaled_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        highest y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: ndarray
            A grid of (y,x) coordinates in arc seconds.
        """
        grid_pixels_1d = grid_util.grid_pixels_1d_from(
            grid_scaled_1d=grid_scaled_1d,
            shape_2d=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
        return grids.Grid(
            grid=grid_pixels_1d, mask=self.mask.mask_sub_1, store_in_1d=True
        )

    def grid_pixel_centres_from_grid_scaled_1d(self, grid_scaled_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        grid_pixel_centres_1d = grid_util.grid_pixel_centres_1d_from(
            grid_scaled_1d=grid_scaled_1d,
            shape_2d=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        ).astype("int")

        return grids.Grid(
            grid=grid_pixel_centres_1d,
            mask=self.regions.edge_mask.mask_sub_1,
            store_in_1d=True,
        )

    def grid_pixel_indexes_from_grid_scaled_1d(self, grid_scaled_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are \
        returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
        downwards.

        For example:

        The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
        The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
        The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        grid_pixel_indexes_1d = grid_util.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid_scaled_1d,
            shape_2d=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        ).astype("int")

        return arrays.Array(
            array=grid_pixel_indexes_1d,
            mask=self.regions.edge_mask.mask_sub_1,
            store_in_1d=True,
        )

    def grid_scaled_from_grid_pixels_1d(self, grid_pixels_1d):
        """Convert a grid of (y,x) pixel coordinates to a grid of (y,x) arc second values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels_1d : ndarray
            The grid of (y,x) coordinates in pixels.
        """
        grid_scaled_1d = grid_util.grid_scaled_1d_from(
            grid_pixels_1d=grid_pixels_1d,
            shape_2d=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
        return grids.Grid(
            grid=grid_scaled_1d,
            mask=self.regions.edge_mask.mask_sub_1,
            store_in_1d=True,
        )

    @property
    def sub_size(self):
        return self.mask.sub_size

    def grid_scaled_from_grid_pixels_1d_for_marching_squares(
        self, grid_pixels_1d, shape_2d
    ):

        grid_scaled_1d = grid_util.grid_scaled_1d_from(
            grid_pixels_1d=grid_pixels_1d,
            shape_2d=shape_2d,
            pixel_scales=(
                self.mask.pixel_scales[0] / self.mask.sub_size,
                self.mask.pixel_scales[1] / self.mask.sub_size,
            ),
            origin=self.mask.origin,
        )

        grid_scaled_1d[:, 0] -= self.mask.pixel_scales[0] / (2.0 * self.mask.sub_size)
        grid_scaled_1d[:, 1] += self.mask.pixel_scales[1] / (2.0 * self.mask.sub_size)

        return grids.Grid(
            grid=grid_scaled_1d,
            mask=self.regions.edge_mask.mask_sub_1,
            store_in_1d=True,
        )

    @property
    def masked_grid(self):
        sub_grid_1d = grid_util.grid_1d_via_mask_from(
            mask=self.mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )
        return grids.Grid(
            grid=sub_grid_1d, mask=self.regions.edge_mask.mask_sub_1, store_in_1d=True
        )

    @property
    def border_grid_1d(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return self.masked_grid[self.mask.regions._sub_border_1d_indexes]

    @property
    def _zoom_centre(self):

        extraction_grid_1d = self.mask.geometry.grid_pixels_from_grid_scaled_1d(
            grid_scaled_1d=self.masked_grid_sub_1.in_1d
        )
        y_pixels_max = np.max(extraction_grid_1d[:, 0])
        y_pixels_min = np.min(extraction_grid_1d[:, 0])
        x_pixels_max = np.max(extraction_grid_1d[:, 1])
        x_pixels_min = np.min(extraction_grid_1d[:, 1])

        return (
            ((y_pixels_max + y_pixels_min - 1.0) / 2.0),
            ((x_pixels_max + x_pixels_min - 1.0) / 2.0),
        )

    @property
    def _zoom_offset_pixels(self):

        if self.mask.pixel_scales is None:
            return self.central_pixel_coordinates

        return (
            self._zoom_centre[0] - self.central_pixel_coordinates[0],
            self._zoom_centre[1] - self.central_pixel_coordinates[1],
        )

    @property
    def _zoom_offset_scaled(self):

        return (
            -self.mask.pixel_scales[0] * self._zoom_offset_pixels[0],
            self.mask.pixel_scales[1] * self._zoom_offset_pixels[1],
        )

    @property
    def _zoom_region(self):
        """The zoomed rectangular region corresponding to the square encompassing all unmasked values. This zoomed
        extraction region is a squuare, even if the mask is rectangular.

        This is used to zoom in on the region of an image that is used in an analysis for visualization."""

        # Have to convert mask to bool for invert function to work.
        where = np.array(np.where(np.invert(self.mask.astype("bool"))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        ylength = y1 - y0
        xlength = x1 - x0

        if ylength > xlength:
            length_difference = ylength - xlength
            x1 += int(length_difference / 2.0)
            x0 -= int(length_difference / 2.0)
        elif xlength > ylength:
            length_difference = xlength - ylength
            y1 += int(length_difference / 2.0)
            y0 -= int(length_difference / 2.0)

        return [y0, y1 + 1, x0, x1 + 1]
