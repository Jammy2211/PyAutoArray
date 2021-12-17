import numpy as np

from autoconf import cached_property

from autoarray.numba_util import profile_func


class LinearObj:

    # def __init__(self, sub_slim_shape, sub_size):
    #
    #     self.sub_slim_shpe = sub_slim_shape
    #     self.sub_size = sub_size

    @property
    def pixels(self) -> int:
        return 1

    @property
    def mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError


class UniqueMappings:
    def __init__(self, data_to_pix_unique, data_weights, pix_lengths):

        self.data_to_pix_unique = data_to_pix_unique.astype("int")
        self.data_weights = data_weights
        self.pix_lengths = pix_lengths.astype("int")
