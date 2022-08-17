import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.numba_util import profile_func


class LinearObj:
    def __init__(
        self,
        regularization: Optional[AbstractRegularization],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        A linear object which reconstructs a dataset based on mapping between the data points of that dataset and
        components of the linear object. For example, the linear obj could map to the data via analytic functions
        or discrete pixels on a mesh.

        The values of linear object are computed via a regularized linear matrix inversion, which infers a solution
        which best fits the data given its noise-map (by minimizing a chi-squared term) whilst accounting for smoothing
        due to regularizaiton.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile`
        objects, which describe the surface brightness of a galaxy as a function. This function can either be assigned
        an overall intensity (e.g. the normalization) which describes how bright it is. Using a `LinearObj` the
        intensity can be solved for linearly instead.

        Parameters
        ----------
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        self.regularization = regularization
        self.profiling_dict = profiling_dict

    @property
    def parameters(self) -> int:
        """
        The total number of parameters used to reconstruct the data.

        For example for the following linear objects:

        - `AbstractLinearObjFuncList` the number of analytic functions.
        - `Mapper` the number of parameters in the mesh used to reconstruct the data.

        Returns
        -------
        The number of parameters used to reconstruct the data.
        """
        raise NotImplementedError

    @property
    def neighbors(self) -> Neighbors:
        """
        An object describing how the different components in the linear object neighbor one another.

        For example for the following linear objects:

        - `AbstractLinearObjFuncList` whether certain analytic functions reconstruct nearby components to one another.
        - `Mapper` how the parameters on the mesh used to reconstruct the data neighbor one another.

        Returns
        -------
        An object describing how the parameters of the linear object neighbor one another.
        """
        raise NotImplementedError

    @cached_property
    @profile_func
    def unique_mappings(self):
        raise NotImplementedError

    @property
    def mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def operated_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        The `LinearEqn` object takes the `mapping_matrix` of each linear object and combines it with the `Convolver`
        operator to perform a 2D convolution and compute the `operated_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `operated_mapping_matrix` output this
        property automatically used instead.

        This is used for linear objects where properly performing the 2D convolution within only the `LinearEqn`
        object is not possible. For example, images may have flux outside the masked region which is blurred into the
        masked region which is linear solved for. This flux is outside the region that defines the `mapping_matrix` and
        thus this override is required to properly incorporate it.

        Returns
        -------
        A blurred mapping matrix of dimensions (total_mask_pixels, 1) which overrides the mapping matrix calculations
        performed in the linear equation solvers.
        """
        return None

    @property
    def regularization_matrix(self) -> np.ndarray:

        if self.regularization is None:

            return np.zeros((self.parameters, self.parameters))

        return self.regularization.regularization_matrix_from(linear_obj=self)
