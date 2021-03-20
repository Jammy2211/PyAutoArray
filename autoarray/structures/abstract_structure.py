from os import path
import numpy as np
import pickle


class AbstractStructure(np.ndarray):
    def __array_finalize__(self, obj):

        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "store_slim"):
            self.store_slim = obj.store_slim

    def __reduce__(self):

        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()

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

        super().__setstate__(state[0:-1])

    def _new_structure(self, grid, mask, store_slim):
        """Conveninence method for creating a new instance of the Grid2D class from this grid.

        This method is over-written by other grids (e.g. Grid2DIterate) such that the slim and native methods return
        instances of that Grid2D's type.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        raise NotImplementedError()

    @property
    def slim(self):
        raise NotImplementedError

    @property
    def native(self):
        raise NotImplementedError

    @property
    def shape_slim(self):
        return self.mask.shape_slim

    @property
    def sub_shape_slim(self):
        return self.mask.sub_shape_slim

    @property
    def shape_native(self):
        return self.mask.shape

    @property
    def sub_shape_native(self):
        return self.mask.sub_shape_native

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
    def unmasked_grid(self):
        return self.mask.unmasked_grid_sub_1

    @property
    def total_pixels(self):
        return self.shape[0]

    def resized_from(self, new_shape):
        raise NotImplementedError

    def padded_before_convolution_from(self, kernel_shape):
        raise NotImplementedError

    def trimmed_after_convolution_from(self, kernel_shape):
        raise NotImplementedError

    def structure_from_result(self, result: np.ndarray):
        raise NotImplementedError

    def structure_list_from_result_list(self, result_list: list):
        raise NotImplementedError

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

    def output_to_fits(self, file_path, overwrite):
        raise NotImplementedError


class AbstractStructure1D(AbstractStructure):

    pass


class AbstractStructure2D(AbstractStructure):
    @property
    def native(self):
        raise NotImplementedError

    @property
    def shape_native(self):
        return self.mask.shape

    @property
    def sub_shape_native(self):
        return self.mask.sub_shape_native
