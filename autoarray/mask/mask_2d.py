import logging
import copy
import numpy as np

from autoarray import exc
from autoarray.mask import abstract_mask, mask_2d_util
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d


logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractMask2D(abstract_mask.AbstractMask):

    # noinspection PyUnusedLocal
    def __new__(
        cls,
        mask: np.ndarray,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        *args,
        **kwargs
    ):
        """
        A 2D mask, representing a uniform rectangular grid of neighboring rectangular pixels.

        When applied to 2D data it extracts or masks the unmasked image pixels corresponding to mask entries that are
        `False` or 0).

        The mask defines the geometry of the 2D uniform grid of pixels for the 2D data structure it is paired with,
        for example the grid's ``pixel scales`` (y,x) ``origin``. The 2D uniform grid may also be sub-gridded,
        whereby every pixel is sub-divided into a uniform grid of sub-pixels which are all used to perform
        calculations more accurate.

        The mask includes tols to map the 2D data structure between 2D representations (that include all  data-points
        irrespective of if they are masked or not) and 1D data structures (that only contain the unmasked data).

        Parameters
        ----------
        mask: np.ndarray
            The ``ndarray`` of shape [total_y_pixels, total_x_pixels] containing the ``bool``'s representing the
            ``mask``, where `False` signifies an entry is unmasked and used in calculations.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        obj = abstract_mask.AbstractMask.__new__(
            cls=cls,
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )
        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj=obj)

        if isinstance(obj, AbstractMask2D):
            pass
        else:
            self.origin = (0.0, 0.0)

    @property
    def shape_native(self):
        return self.shape

    @property
    def sub_shape_native(self):
        try:
            return (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)
        except AttributeError:
            print("bleh")

    @property
    def sub_mask(self):

        sub_shape = (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

        return mask_2d_util.mask_2d_via_shape_native_and_native_for_slim(
            shape_native=sub_shape,
            native_for_slim=self._sub_mask_index_for_sub_mask_1d_index,
        ).astype("bool")

    def rescaled_mask_from_rescale_factor(self, rescale_factor):

        rescaled_mask = mask_2d_util.rescaled_mask_2d_from(
            mask_2d=self, rescale_factor=rescale_factor
        )

        return Mask2D(
            mask=rescaled_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    @property
    def mask_sub_1(self):
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of ``sub_size`` `.
        """
        return Mask2D(
            mask=self, sub_size=1, pixel_scales=self.pixel_scales, origin=self.origin
        )

    def resized_mask_from_new_shape(self, new_shape):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """

        mask = copy.deepcopy(self)

        resized_mask = array_2d_util.resized_array_2d_from_array_2d(
            array_2d=mask, resized_shape=new_shape
        ).astype("bool")

        return Mask2D(
            mask=resized_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def trimmed_array_from_padded_array_and_image_shape(
        self, padded_array, image_shape
    ):
        """Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array : np.ndarray
            A 1D array of values which were computed using a padded grid
        """

        pad_size_0 = self.shape[0] - image_shape[0]
        pad_size_1 = self.shape[1] - image_shape[1]
        trimmed_array = padded_array.native_binned[
            pad_size_0 // 2 : self.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.shape[1] - pad_size_1 // 2,
        ]
        return array_2d.Array2D.manual(
            array=trimmed_array,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

    def unmasked_blurred_array_from_padded_array_psf_and_image_shape(
        self, padded_array, psf, image_shape
    ):
        """For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens dataset's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : aa.Kernel2D
            The PSF of the image used for convolution.
        unmasked_image_1d : np.ndarray
            The 1D unmasked image which is blurred.
        """

        blurred_image = psf.convolved_array_from_array(array=padded_array)

        return self.trimmed_array_from_padded_array_and_image_shape(
            padded_array=blurred_image, image_shape=image_shape
        )

    def output_to_fits(self, file_path, overwrite=False):
        """
        Write the 2D Mask to a .fits file.

        Before outputting a NumPy array, the array may be flipped upside-down using np.flipud depending on the project
        config files. This is for Astronomy projects so that structures appear the same orientation as ``.fits`` files
        loaded in DS9.

        Parameters
        ----------
        file_path : str
            The full path of the file that is output, including the file name and ``.fits`` extension.
        overwrite : bool
            If `True` and a file already exists with the input file_path the .fits file is overwritten. If `False`, an
            error is raised.

        Returns
        -------
        None

        Examples
        --------
        mask = Mask2D(mask=np.full(shape=(5,5), fill_value=False))
        mask.output_to_fits(file_path='/path/to/file/filename.fits', overwrite=True)
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def shape_native_scaled(self):
        return (
            float(self.pixel_scales[0] * self.shape[0]),
            float(self.pixel_scales[1] * self.shape[1]),
        )

    @property
    def central_pixel_coordinates(self):
        return geometry_util.central_pixel_coordinates_2d_from(
            shape_native=self.shape_native
        )

    @property
    def central_scaled_coordinates(self):

        return geometry_util.central_scaled_coordinate_2d_from(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def pixel_coordinates_2d_from(self, scaled_coordinates_2d):

        return geometry_util.pixel_coordinates_2d_from(
            scaled_coordinates_2d=scaled_coordinates_2d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def scaled_coordinates_2d_from(self, pixel_coordinates_2d):

        return geometry_util.scaled_coordinates_2d_from(
            pixel_coordinates_2d=pixel_coordinates_2d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    @property
    @array_2d_util.Memoizer()
    def mask_centre(self):
        return grid_2d_util.grid_2d_centre_from(grid_2d_slim=self.masked_grid_sub_1)

    @property
    def scaled_maxima(self):
        return (
            (self.shape_native_scaled[0] / 2.0) + self.origin[0],
            (self.shape_native_scaled[1] / 2.0) + self.origin[1],
        )

    @property
    def scaled_minima(self):
        return (
            (-(self.shape_native_scaled[0] / 2.0)) + self.origin[0],
            (-(self.shape_native_scaled[1] / 2.0)) + self.origin[1],
        )

    @property
    def extent(self):
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

    @property
    def edge_buffed_mask(self):
        edge_buffed_mask = mask_2d_util.buffed_mask_2d_from(mask_2d=self).astype("bool")
        return Mask2D(
            mask=edge_buffed_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    @property
    def unmasked_grid_sub_1(self):
        """ The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in scaled units.
        """
        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

        return grid_2d.Grid2D(
            grid=grid_slim, mask=self.unmasked_mask.mask_sub_1, store_slim=True
        )

    @property
    def masked_grid(self):
        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )
        return grid_2d.Grid2D(
            grid=sub_grid_1d, mask=self.edge_mask.mask_sub_1, store_slim=True
        )

    @property
    def masked_grid_sub_1(self):

        grid_slim = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin
        )
        return grid_2d.Grid2D(grid=grid_slim, mask=self.mask_sub_1, store_slim=True)

    @property
    def edge_grid_sub_1(self):
        """
        The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        edge_grid_1d = self.masked_grid_sub_1[self._edge_1d_indexes]
        return grid_2d.Grid2D(
            grid=edge_grid_1d, mask=self.edge_mask.mask_sub_1, store_slim=True
        )

    @property
    def border_grid_1d(self):
        """
        The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        return self.masked_grid[self._sub_border_flat_indexes]

    @property
    def border_grid_sub_1(self):
        """
        The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        border_grid_1d = self.masked_grid_sub_1[self._border_1d_indexes]
        return grid_2d.Grid2D(
            grid=border_grid_1d, mask=self.border_mask.mask_sub_1, store_slim=True
        )

    def grid_pixels_from_grid_scaled_1d(self, grid_scaled_1d):
        """
        Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            A grid of (y,x) coordinates in scaled units.
        """
        grid_pixels_1d = grid_2d_util.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return grid_2d.Grid2D(
            grid=grid_pixels_1d, mask=self.mask_sub_1, store_slim=True
        )

    def grid_pixel_centres_from_grid_scaled_1d(self, grid_scaled_1d):
        """Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            The grid of (y,x) coordinates in scaled units.
        """
        grid_pixel_centres_1d = grid_2d_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

        return grid_2d.Grid2D(
            grid=grid_pixel_centres_1d, mask=self.edge_mask.mask_sub_1, store_slim=True
        )

    def grid_pixel_indexes_from_grid_scaled_1d(self, grid_scaled_1d):
        """Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are \
        returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
        downwards.

        For example:

        The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
        The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
        The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            The grid of (y,x) coordinates in scaled units.
        """
        grid_pixel_indexes_1d = grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

        return array_2d.Array2D(
            array=grid_pixel_indexes_1d, mask=self.edge_mask.mask_sub_1, store_slim=True
        )

    def grid_scaled_from_grid_pixels_1d(self, grid_pixels_1d):
        """Convert a grid of (y,x) pixel coordinates to a grid of (y,x) scaled values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels_1d : np.ndarray
            The grid of (y,x) coordinates in pixels.
        """
        grid_scaled_1d = grid_2d_util.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels_1d,
            shape_native=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return grid_2d.Grid2D(
            grid=grid_scaled_1d, mask=self.edge_mask.mask_sub_1, store_slim=True
        )

    def grid_scaled_from_grid_pixels_1d_for_marching_squares(
        self, grid_pixels_1d, shape_native
    ):

        grid_scaled_1d = grid_2d_util.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels_1d,
            shape_native=shape_native,
            pixel_scales=(
                self.pixel_scales[0] / self.sub_size,
                self.pixel_scales[1] / self.sub_size,
            ),
            origin=self.origin,
        )

        grid_scaled_1d[:, 0] -= self.pixel_scales[0] / (2.0 * self.sub_size)
        grid_scaled_1d[:, 1] += self.pixel_scales[1] / (2.0 * self.sub_size)

        return grid_2d.Grid2D(
            grid=grid_scaled_1d, mask=self.edge_mask.mask_sub_1, store_slim=True
        )

    @property
    def _sub_native_index_for_sub_slim_index(self):
        """A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates."""
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self, sub_size=1
        ).astype("int")

    @property
    def _edge_1d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a `True` value).
        """
        return mask_2d_util.edge_1d_indexes_from(mask_2d=self).astype("int")

    @property
    def _edge_2d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a `True` value).
        """
        return self._sub_native_index_for_sub_slim_index[self._edge_1d_indexes].astype(
            "int"
        )

    @property
    def _border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        return mask_2d_util.border_slim_indexes_from(mask_2d=self).astype("int")

    @property
    def _border_2d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        return self._sub_native_index_for_sub_slim_index[
            self._border_1d_indexes
        ].astype("int")

    @property
    def _sub_border_flat_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        return mask_2d_util.sub_border_pixel_slim_indexes_from(
            mask_2d=self, sub_size=self.sub_size
        ).astype("int")

    @array_2d_util.Memoizer()
    def blurring_mask_from_kernel_shape(self, kernel_shape_native):
        """
        Returns a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid2D.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        kernel_shape_native : (int, int)
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if kernel_shape_native[0] % 2 == 0 or kernel_shape_native[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_2d_util.blurring_mask_2d_from(
            mask_2d=self, kernel_shape_native=kernel_shape_native
        )

        return Mask2D(
            mask=blurring_mask,
            sub_size=1,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def unmasked_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        return Mask2D.unmasked(
            shape_native=self.shape_native,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def edge_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        mask = np.full(fill_value=True, shape=self.shape)
        mask[self._edge_2d_indexes[:, 0], self._edge_2d_indexes[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def border_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        mask = np.full(fill_value=True, shape=self.shape)
        mask[self._border_2d_indexes[:, 0], self._border_2d_indexes[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def _sub_mask_index_for_sub_mask_1d_index(self):
        """A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates."""
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self, sub_size=self.sub_size
        ).astype("int")

    @property
    @array_2d_util.Memoizer()
    def _slim_index_for_sub_slim_index(self):
        """The util between every sub-pixel and its host pixel.

        For example:

        - sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd pixel.
        - sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th pixel.
        """
        return mask_2d_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=self, sub_size=self.sub_size
        ).astype("int")

    @property
    def zoom_centre(self):

        extraction_grid_1d = self.grid_pixels_from_grid_scaled_1d(
            grid_scaled_1d=self.masked_grid_sub_1.slim
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
    def zoom_offset_pixels(self):

        if self.pixel_scales is None:
            return self.central_pixel_coordinates

        return (
            self.zoom_centre[0] - self.central_pixel_coordinates[0],
            self.zoom_centre[1] - self.central_pixel_coordinates[1],
        )

    @property
    def zoom_offset_scaled(self):

        return (
            -self.pixel_scales[0] * self.zoom_offset_pixels[0],
            self.pixel_scales[1] * self.zoom_offset_pixels[1],
        )

    @property
    def zoom_region(self):
        """The zoomed rectangular region corresponding to the square encompassing all unmasked values. This zoomed
        extraction region is a squuare, even if the mask is rectangular.

        This is used to zoom in on the region of an image that is used in an analysis for visualization."""

        # Have to convert mask to bool for invert function to work.
        where = np.array(np.where(np.invert(self.astype("bool"))))
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

    @property
    def zoom_shape_native(self):
        region = self.zoom_region
        return (region[1] - region[0], region[3] - region[2])

    @property
    def zoom_mask_unmasked(self):
        """ The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in scaled units.
        """

        return Mask2D.unmasked(
            shape_native=self.zoom_shape_native,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.zoom_offset_scaled,
        )


class Mask2D(AbstractMask2D):
    @classmethod
    def manual(
        cls,
        mask: np.ndarray or list,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see ``AbstractMask2D.__new__``) by inputting the array values in 2D, for example:

        mask=np.array([[False, False],
                       [True, False]])

        mask=[[False, False],
               [True, False]]

        Parameters
        ----------
        mask : np.ndarray or list
            The ``bool`` values of the mask input as an ``np.ndarray`` of shape [total_y_pixels, total_x_pixels] or a
            list of lists.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return cls(
            mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def unmasked(
        cls,
        shape_native: (int, int),
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """Create a mask where all pixels are `False` and therefore unmasked.

        Parameters
        ----------
        shape_native : (int, int)
            The 2D shape of the mask that is created.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        return cls.manual(
            mask=np.full(shape=shape_native, fill_value=False),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular(
        cls,
        shape_native: (int, int),
        radius: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within a circle of input radius.

        The ``radius`` and ``centre`` are both input in scaled units.

        Parameters
        ----------
        shape_native : (int, int)
            The (y,x) shape of the mask in units of pixels.
        radius : float
            The radius in scaled units of the circle within which pixels are `False` and unmasked.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the circle used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_circular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            radius=radius,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular_annular(
        cls,
        shape_native: (int, int),
        inner_radius: float,
        outer_radius: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an annulus of input
        inner radius and outer radius.

        The ``inner_radius``, ``outer_radius`` and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_native : (int, int)
            The (y,x) shape of the mask in units of pixels.
        inner_radius : float
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius : float
            The outer radius in scaled units of the annulus within which pixels are `False` and unmasked.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the annulus used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_circular_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular_anti_annular(
        cls,
        shape_native: (int, int),
        inner_radius: float,
        outer_radius: float,
        outer_radius_2: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an inner circle and second
        outer circle, forming an inverse annulus.

        The ``inner_radius``, ``outer_radius``, ``outer_radius_2`` and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_native : (int, int)
            The (y,x) shape of the mask in units of pixels.
        inner_radius : float
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius : float
            The first outer radius in scaled units of the annulus within which pixels are `True` and masked.
        outer_radius_2 : float
            The second outer radius in scaled units of the annulus within which pixels are `False` and unmasked and
            outside of which all entries are `True` and masked.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the anti-annulus used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_circular_anti_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            outer_radius_2_scaled=outer_radius_2,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def elliptical(
        cls,
        shape_native: (int, int),
        major_axis_radius: float,
        axis_ratio: float,
        phi: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an ellipse.
        
        The ``major_axis_radius``, and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_native : (int, int)
            The (y,x) shape of the mask in units of pixels.
        major_axis_radius : float
            The major-axis in scaled units of the ellipse within which pixels are unmasked.
        axis_ratio : float
            The axis-ratio of the ellipse within which pixels are unmasked.
        phi : float
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive \
             x-axis).
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centred of the ellipse used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_elliptical_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            major_axis_radius=major_axis_radius,
            axis_ratio=axis_ratio,
            phi=phi,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def elliptical_annular(
        cls,
        shape_native: (int, int),
        inner_major_axis_radius: float,
        inner_axis_ratio: float,
        inner_phi: float,
        outer_major_axis_radius: float,
        outer_axis_ratio: float,
        outer_phi: float,
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        centre: (float, float) = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an elliptical annulus of input
       inner and outer scaled major-axis and centre.

        The ``outer_major_axis_radius``, ``inner_major_axis_radius`` and ``centre`` are all input in scaled units.

        Parameters
        ----------
        shape_native (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        inner_major_axis_radius : float
            The major-axis in scaled units of the inner ellipse within which pixels are masked.
        inner_axis_ratio : float
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi : float
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the \
            positive x-axis).
        outer_major_axis_radius : float
            The major-axis in scaled units of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio : float
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi : float
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the \
            positive x-axis).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        centre: (float, float)
            The (y,x) scaled units centre of the elliptical annuli used to mask pixels.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = mask_2d_util.mask_2d_elliptical_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_major_axis_radius=inner_major_axis_radius,
            inner_axis_ratio=inner_axis_ratio,
            inner_phi=inner_phi,
            outer_major_axis_radius=outer_major_axis_radius,
            outer_axis_ratio=outer_axis_ratio,
            outer_phi=outer_phi,
            centre=centre,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_pixel_coordinates(
        cls,
        shape_native: (int, int),
        pixel_coordinates: [[int, int]],
        pixel_scales: (float, float),
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        buffer: int = 0,
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are defined from an input list of list of
        pixel coordinates.

        These may be buffed via an input ``buffer``, whereby all entries in all 8 neighboring directions by this
        amount.

        Parameters
        ----------
        shape_native (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_coordinates : [[int, int]]
            The input lists of 2D pixel coordinates where `False` entries are created.
        pixel_scales : (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        buffer : int
            All input ``pixel_coordinates`` are buffed with `False` entries in all 8 neighboring directions by this
            amount.
        invert : bool
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        mask = mask_2d_util.mask_2d_via_pixel_coordinates_from(
            shape_native=shape_native,
            pixel_coordinates=pixel_coordinates,
            buffer=buffer,
        )

        return cls.manual(
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: str,
        pixel_scales: (float, float),
        hdu: int = 0,
        sub_size: int = 1,
        origin: (float, float) = (0.0, 0.0),
        resized_mask_shape: (int, int) = None,
    ) -> "Mask2D":
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = cls(
            array_2d_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.resized_mask_from_new_shape(new_shape=resized_mask_shape)

        return mask
