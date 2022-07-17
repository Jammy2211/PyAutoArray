import numpy as np
from typing import Optional

from autoconf import cached_property

from autoarray.numba_util import profile_func


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
        masked region which is linear solved for. This flux is outside the region that defines the `mapping_matrix` and
        thus this override is required to properly incorporate it.

        Returns
        -------
        A blurred mapping matrix of dimensions (total_mask_pixels, 1) which overrides the mapping matrix calculations
        performed in the linear equation solvers.
        """
        return None

    @property
    def transformed_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        The `LinearEqn` object takes the `mapping_matrix` of each linear object and combines it with the `Transformer`
        operator to perform a Fourier transform and compute the `transformed_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `blurred_mapping_matrix` output this
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

    @cached_property
    @profile_func
    def data_unique_mappings(self):
        raise NotImplementedError
