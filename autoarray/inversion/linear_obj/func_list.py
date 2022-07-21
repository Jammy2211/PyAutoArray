import numpy as np
from typing import Optional, Dict

from autoconf import cached_property

from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.unique_mappings import UniqueMappings
from autoarray.type import Grid1D2DLike

from autoarray.numba_util import profile_func


class AbstractLinearObjFuncList(LinearObj):
    def __init__(self, grid: Grid1D2DLike, profiling_dict: Optional[Dict] = None):
        """
        An object represented by one or more analytic functions, the solution of which can be solved for linearly via an
        inversion.

        By overwriting the `mapping_matrix` function with a method that fills in its value with the solution of the
        analytic function, this is then passed through the `inversion` package to perform the linear inversion. The
        API is identical to `Mapper` objects such that linear functions can easily be combined with mappers.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile`
        objects, which describe the surface brightness of a galaxy as a function. This function can either be assigned
        an overall intensity (e.g. the normalization) which describes how bright it is. Using the `LinearObjFuncList` the
        intensity can be solved for linearly instead.

        Parameters
        ----------
        grid
            The grid of data points representing the data that is fitted and therefore where the analytic function
            is evaluated.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        super().__init__(profiling_dict=profiling_dict)

        self.grid = grid

    # TODO : perma store in memory and pass via lazy instantiate somehwere for model fit.

    @property
    def pixels(self):
        return 1

    @cached_property
    @profile_func
    def unique_mappings(self):
        """
        Returns the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding pixelization pixels (e.g. `pixelization_grid`).

        To perform an `Inversion` efficiently the linear algebra can bypass the calculation of a `mapping_matrix` and
        instead use the w-tilde formalism, which requires these unique mappings for efficient computation. For
        convenience, these mappings and associated metadata are packaged into the class `UniqueMappings`.

        For a `LinearObjFuncList` every data pixel's group of sub-pixels maps directly to the linear function.
        """

        sub_size = self.grid.sub_size
        shape_slim = self.grid.mask.shape_slim

        data_to_pix_unique = -1.0 * np.ones(shape=(shape_slim, sub_size ** 2)).astype(
            "int"
        )
        data_weights = np.zeros(shape=(shape_slim, sub_size ** 2))
        pix_lengths = np.ones(shape=shape_slim).astype("int")

        data_to_pix_unique[:, 0] = 0
        data_weights[:, 0] = 1.0

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )


class LinearObjFuncList(AbstractLinearObjFuncList):
    def __init__(self, grid: Grid1D2DLike, profiling_dict: Optional[Dict] = None):
        """
        An object represented by one or more analytic functions, the solution of which can be solved for linearly via an 
        inversion.
        
        By overwriting the `mapping_matrix` function with a method that fills in its value with the solution of the
        analytic function, this is then passed through the `inversion` package to perform the linear inversion. The
        API is identical to `Mapper` objects such that linear functions can easily be combined with mappers.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile` 
        objects, which describe the surface brightness of a galaxy as a function. This function can either be assigned
        an overall intensity (e.g. the normalization) which describes how bright it is. Using the `LinearObjFuncList` the
        intensity can be solved for linearly instead.
        
        Parameters
        ----------
        grid
            The grid of data points representing the data that is fitted and therefore where the analytic function
            is evaluated.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        super().__init__(grid=grid, profiling_dict=profiling_dict)
