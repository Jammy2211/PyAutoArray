import numpy as np
from typing import Optional, Dict

from autoconf import cached_property

from autoarray.numba_util import profile_func


class UniqueMappings:
    def __init__(self, data_to_pix_unique, data_weights, pix_lengths):

        self.data_to_pix_unique = data_to_pix_unique.astype("int")
        self.data_weights = data_weights
        self.pix_lengths = pix_lengths.astype("int")


class LinearObj:
    @property
    def pixels(self) -> int:
        raise NotImplementedError

    @property
    def mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    @profile_func
    def data_unique_mappings(self):
        raise NotImplementedError


class LinearObjFunc(LinearObj):
    def __init__(
        self, sub_slim_shape: int, sub_size: int, profiling_dict: Optional[Dict] = None
    ):

        self.sub_slim_shape = sub_slim_shape
        self.sub_size = sub_size

        self.profiling_dict = profiling_dict

    @property
    def slim_shape(self) -> int:
        return int(self.sub_slim_shape / self.sub_size ** 2)

    @property
    def pixels(self) -> int:
        return 1

    @property
    def mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError

    # TODO : perma store in memory and pass via lazy instantiate somehwere for model fit.

    @cached_property
    @profile_func
    def data_unique_mappings(self):
        """
        The w_tilde formalism requires us to compute an array that gives the unique mappings between the sub-pixels of
        every image pixel to their corresponding pixelization pixels.
        """

        data_to_pix_unique = -1.0 * np.ones(
            shape=(self.slim_shape, self.sub_size ** 2)
        ).astype("int")
        data_weights = np.zeros(shape=(self.slim_shape, self.sub_size ** 2))
        pix_lengths = np.ones(shape=self.slim_shape).astype("int")

        data_to_pix_unique[:, 0] = 0
        data_weights[:, 0] = 1.0

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )
