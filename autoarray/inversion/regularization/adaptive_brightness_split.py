from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.adaptive_brightness import AdaptiveBrightness

from autoarray.inversion.regularization import regularization_util


class AdaptiveBrightnessSplit(AdaptiveBrightness):
    def __init__(
        self,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
    ):
        """
        An adaptive regularization scheme which splits every source pixel into a cross of four regularization points
        (regularization is described in the `Regularization` class above) and interpolates to these points in order
        to apply smoothing on the solution of an `Inversion`.

        The size of this cross is determined via the size of the source-pixel, for example if the source pixel is a
        Voronoi pixel the area of the pixel is computed and the distance of each point of the cross is given by
        the area times 0.5.

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

        super().__init__(
            inner_coefficient=inner_coefficient,
            outer_coefficient=outer_coefficient,
            signal_scale=signal_scale,
        )

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:

        regularization_weights = self.regularization_weights_from(linear_obj=linear_obj)

        pix_sub_weights_split_cross = linear_obj.pix_sub_weights_split_cross

        (
            splitted_mappings,
            splitted_sizes,
            splitted_weights,
        ) = regularization_util.reg_split_from(
            splitted_mappings=pix_sub_weights_split_cross.mappings,
            splitted_sizes=pix_sub_weights_split_cross.sizes,
            splitted_weights=pix_sub_weights_split_cross.weights,
        )

        return regularization_util.pixel_splitted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )
