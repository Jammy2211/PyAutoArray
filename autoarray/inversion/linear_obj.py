import numpy as np
from typing import Optional, Dict

from autoconf import cached_property

from autoarray.numba_util import profile_func


class UniqueMappings:
    def __init__(self, data_to_pix_unique, data_weights, pix_lengths):
        """
        Packages the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding pixelization pixels (e.g. `pixelization_grid`).

        The following quantities are packaged in this class as ndarray:

        - `data_to_pix_unique`: the unique mapping of every data pixel's grouped sub-pixels to pixelization pixels.
        - `data_weights`: the weights of each data pixel's grouped sub-pixels to pixelization pixels (e.g. determined
        via their sub-size fractional mappings and interpolation weights).
        - `pix_lengths`: the number of unique pixelization pixels each data pixel's grouped sub-pixels map too.

        The need to store separately the mappings and pixelization lengths is so that they be easily iterated over when
        perform calculations for efficiency.

        See the mapper properties `data_unique_mappings()` for a description of the use of this object in mappers.

        Parameters
        ----------
        data_to_pix_unique
            The unique mapping of every data pixel's grouped sub-pixels to pixelization pixels.
        data_weights
            The weights of each data pixel's grouped sub-pixels to pixelization pixels
        pix_lengths
            The number of unique pixelization pixels each data pixel's grouped sub-pixels map too.
        """
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
        Returns the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding pixelization pixels (e.g. `pixelization_grid`).

        To perform an `Inversion` efficiently the linear algebra can bypass the calculation of a `mapping_matrix` and
        instead use the w-tilde formalism, which requires these unique mappings for efficient computation. For
        convenience, these mappings and associated metadata are packaged into the class `UniqueMappings`.

        For a `LinearObjFunc` every data pixel's group of sub-pixels maps directly to the linear function.
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
