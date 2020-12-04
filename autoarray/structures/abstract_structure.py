from os import path
import numpy as np
import pickle


class AbstractStructure(np.ndarray):
    def __array_finalize__(self, obj):

        if isinstance(obj, AbstractStructure):
            if hasattr(obj, "mask"):
                self.mask = obj.mask

            if hasattr(obj, "store_in_1d"):
                self.store_in_1d = obj.store_in_1d

    def _new_structure(self, grid, mask, store_in_1d):
        """Conveninence method for creating a new instance of the Grid class from this grid.

        This method is over-written by other grids (e.g. GridIterate) such that the in_1d and in_2d methods return
        instances of that Grid's type.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        raise NotImplementedError()

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
        return self.mask.geometry.unmasked_grid_sub_1

    @property
    def total_pixels(self):
        return self.shape[0]

    @property
    def binned_pixel_scales_from_bin_up_factor(self):
        return self.mask.binned_pixel_scales_from_bin_up_factor

    @classmethod
    def load(cls, file_path, filename):
        with open(path.join(file_path, f"{filename}.pickle"), "rb") as f:
            return pickle.load(f)

    def save(self, file_path, filename):
        """
        Save the tracer by serializing it with pickle.
        """
        with open(path.join(file_path, f"{filename}.pickle"), "wb") as f:
            pickle.dump(self, f)

    def resized_from(self, new_shape):
        raise NotImplementedError

    def padded_before_convolution_from(self, kernel_shape):
        raise NotImplementedError

    def trimmed_after_convolution_from(self, kernel_shape):
        raise NotImplementedError

    def binned_up_from(self, bin_up_factor, method):
        raise NotImplementedError

    def output_to_fits(self, file_path, overwrite):
        raise NotImplementedError
