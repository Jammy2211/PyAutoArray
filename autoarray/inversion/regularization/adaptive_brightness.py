import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class AdaptiveBrightness(AbstractRegularization):
    def __init__(
        self,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
    ):
        """
        An adaptive regularization scheme (regularization is described in the `Regularization` class above).

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is 
        applied when each set of pixel neighbors are regularized with one another. The motivation of this is that 
        different regions of a pixelization's mesh require different levels of regularization (e.g., high smoothing where the 
        no signal is present and less smoothing where it is, see (Nightingale, Dye and Massey 2018)).

        Unlike the instance regularization_matrix scheme, neighboring pixels must now be regularized with one another 
        in both directions (e.g. if pixel 0 regularizes pixel 1, pixel 1 must also regularize pixel 0). For example:

        B = [-1, 1]  [0->1]
            [-1, -1]  1 now also regularizes 0

        For a instance regularization coefficient this would NOT produce a positive-definite matrix. However, for
        the weighted scheme, it does!

        The regularize weight_list change the B matrix as shown below - we simply multiply each pixel's effective 
        regularization weight by each row of B it has a -1 in, so:

        regularization_weights = [1, 2, 3, 4]

        B = [-1, 1, 0 ,0] # [0->1]
            [0, -2, 2 ,0] # [1->2]
            [0, 0, -3 ,3] # [2->3]
            [4, 0, 0 ,-4] # [3->0]

        If our -1's werent down the diagonal this would look like:

        B = [4, 0, 0 ,-4] # [3->0]
            [0, -2, 2 ,0] # [1->2]
            [-1, 1, 0 ,0] # [0->1]
            [0, 0, -3 ,3] # [2->3] This is valid!

        Parameters
        ----------
        coefficients
            The regularization coefficients which controls the degree of smoothing of the inversion reconstruction in 
            high and low signal regions of the reconstruction.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to 
            low signal regions.
        """

        super().__init__()

        self.inner_coefficient = inner_coefficient
        self.outer_coefficient = outer_coefficient
        self.signal_scale = signal_scale

    def regularization_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``AdaptiveBrightness``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        pixel_signals = linear_obj.pixel_signals_from(signal_scale=self.signal_scale)

        return regularization_util.adaptive_regularization_weights_from(
            inner_coefficient=self.inner_coefficient,
            outer_coefficient=self.outer_coefficient,
            pixel_signals=pixel_signals,
        )

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization matrix of this regularization scheme.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        regularization_weights = self.regularization_weights_from(linear_obj=linear_obj)

        return regularization_util.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=linear_obj.source_mesh_grid.neighbors,
            neighbors_sizes=linear_obj.source_mesh_grid.neighbors.sizes,
        )
