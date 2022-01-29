import numpy as np
from typing import Optional, Dict

from autoconf import cached_property

from autoarray.type import Grid1D2DLike
from autoarray.numba_util import profile_func


class UniqueMappings:
    def __init__(
        self,
        data_to_pix_unique: np.ndarray,
        data_weights: np.ndarray,
        pix_lengths: np.ndarray,
    ):
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

    @property
    def blurred_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        The `LinearEqn` object takes the `mapping_matrix` of each linear object and combines it with the `Convolver`
        operator to perform a 2D convolution and compute the `blurred_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `blurred_mapping_matrix` output this
        property automatically used instead.

        This is used for linear objects where properly performing the 2D convolution within only the `LinearEqn`
        object is not possible. For example, images may have flux outside the masked region which is blurred into the
        masked region which is linear solved for. This flux is outside the region that defines the `mapping_matrix and
        thus this override is required to properly incorporate it.

        Returns
        -------
        A blurred mapping matrix of dimensions (total_mask_pixels, 1) which overrides the mapping matrix calculations
        performed in the linear equation solvers.
        """
        return None

    @cached_property
    @profile_func
    def data_unique_mappings(self):
        raise NotImplementedError


class LinearObjFunc(LinearObj):
    def __init__(self, grid: Grid1D2DLike, profiling_dict: Optional[Dict] = None):
        """
        An object represented by one or more analytic functions, the solution of which can be solved for linearly via an 
        inversion.
        
        By overwriting the `mapping_matrix` function with a method that fills in its value with the solution of the
        analytic function, this is then passed through the `inversion` package to perform the linear inversion. The
        API is identical to `Mapper` objects such that linear functions can easily be combined with mappers.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile` 
        objects, which describe the surface brightness of a galaxy as a function. This function can either be assigned
        an overall intensity (e.g. the normalization) which describes how bright it is. Using the `LinearObjFunc` the 
        intensity can be solved for linearly instead.
        
        Parameters
        ----------
        grid
            The grid of data points representing the data that is fitted and therefore where the analytic function
            is evaluated.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        self.grid = grid

        self.profiling_dict = profiling_dict

    @property
    def sub_size(self):
        return self.grid.sub_size

    @property
    def shape_slim(self) -> int:
        return self.grid.mask.shape_slim

    @property
    def sub_shape_slim(self):
        return self.grid.mask.sub_shape_slim

    @property
    def pixels(self) -> int:
        return 1

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
            shape=(self.shape_slim, self.sub_size ** 2)
        ).astype("int")
        data_weights = np.zeros(shape=(self.shape_slim, self.sub_size ** 2))
        pix_lengths = np.ones(shape=self.shape_slim).astype("int")

        data_to_pix_unique[:, 0] = 0
        data_weights[:, 0] = 1.0

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )

    @property
    def mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError
