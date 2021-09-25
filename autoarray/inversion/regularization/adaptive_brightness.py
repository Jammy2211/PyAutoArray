from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class AdaptiveBrightness(AbstractRegularization):
    def __init__(self, inner_coefficient=1.0, outer_coefficient=1.0, signal_scale=1.0):
        """ A instance-regularization scheme (regularization is described in the `Regularization` class above).

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is \
        applied when each set of pixel neighbors are regularized with one another. The motivation of this is that \
        different regions of a pixelization require different levels of regularization (e.g., high smoothing where the \
        no signal is present and less smoothing where it is, see (Nightingale, Dye and Massey 2018)).

        Unlike the instance regularization_matrix scheme, neighboring pixels must now be regularized with one another \
        in both directions (e.g. if pixel 0 regularizes pixel 1, pixel 1 must also regularize pixel 0). For example:

        B = [-1, 1]  [0->1]
            [-1, -1]  1 now also regularizes 0

        For a instance regularization coefficient this would NOT produce a positive-definite matrix. However, for
        the weighted scheme, it does!

        The regularize weight_list change the B matrix as shown below - we simply multiply each pixel's effective \
        regularization weight by each row of B it has a -1 in, so:

        regularization_weight_list = [1, 2, 3, 4]

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
        -----------
        coefficients
            The regularization coefficients which controls the degree of smoothing of the inversion reconstruction in \
            high and low signal regions of the reconstruction.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to \
            low signal regions.
        """

        super().__init__()

        self.inner_coefficient = inner_coefficient
        self.outer_coefficient = outer_coefficient
        self.signal_scale = signal_scale

    def regularization_weight_list_from_mapper(self, mapper):
        pixel_signals = mapper.pixel_signals_from(signal_scale=self.signal_scale)

        return regularization_util.adaptive_regularization_weight_list_from(
            inner_coefficient=self.inner_coefficient,
            outer_coefficient=self.outer_coefficient,
            pixel_signals=pixel_signals,
        )

    def regularization_matrix_from_mapper(self, mapper):
        regularization_weight_list = self.regularization_weight_list_from_mapper(
            mapper=mapper
        )

        return regularization_util.weighted_regularization_matrix_from(
            regularization_weight_list=regularization_weight_list,
            pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=mapper.source_pixelization_grid.pixel_neighbors.sizes,
        )
