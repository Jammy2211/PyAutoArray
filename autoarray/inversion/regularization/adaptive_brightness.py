import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util

from autofit import exc


class AdaptiveBrightness(AbstractRegularization):
    def __init__(self, inner_coefficient=1.0, outer_coefficient=1.0, signal_scale=1.0):
        """
        An adaptive regularization scheme (regularization is described in the `Regularization` class above).

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

    def regularization_weights_from(self, mapper) -> np.ndarray:
        pixel_signals = mapper.pixel_signals_from(signal_scale=self.signal_scale)

        return regularization_util.adaptive_regularization_weights_from(
            inner_coefficient=self.inner_coefficient,
            outer_coefficient=self.outer_coefficient,
            pixel_signals=pixel_signals,
        )

    def regularization_matrix_from(self, mapper) -> np.ndarray:

        regularization_weights = self.regularization_weights_from(mapper=mapper)

        return regularization_util.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=mapper.source_pixelization_grid.pixel_neighbors.sizes,
        )


class AdaptiveBrightnessSplit(AdaptiveBrightness):
    def __init__(
        self,
        inner_coefficient=1.0,
        outer_coefficient=1.0,
        signal_scale=1.0,
        if_interpolated=False,
    ):
        """
        An adaptive regularization scheme which splits every source pixel into a cross of four regularization points
        (regularization is described in the `Regularization` class above) and interpolates to these points in order
        to apply smoothing on the solution of an `Inversion`.

        The size of this cross is determined via the size of the source-pixel, for example if the source pixel is a
        Voronoi pixel the area of the pixel is computed and the distance of each point of the cross is given by
        the area times 0.5.

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
        self.if_interpolated = if_interpolated

    def regularization_matrix_from(self, mapper) -> np.ndarray:

        regularization_weights = self.regularization_weights_from(mapper=mapper)
        (
            splitted_mappings,
            splitted_sizes,
            splitted_weights,
        ) = mapper.splitted_pixelization_mappings_sizes_and_weights
        max_j = np.shape(splitted_weights)[1] - 1
        # The maximum neighbor index for a grid, if the maximum number of neighbors is 100, then it should be 99.

        # Usually, there should not be such a case where a grid can have over 100 (somtimes even 200) neighbours,
        # but when running the codes, this indeed happens.
        # From my experience, it only happens in the initializing phase (generating initializing points), when very weird lens mass
        # distribution can be considered. So, when coming across such extreme cases, we throw out a "exc.FitException" error to do a resample.
        # We should keep an eye on this.
        splitted_weights *= -1.0

        for i in range(len(splitted_mappings)):

            pixel_index = i // 4

            flag = 0

            for j in range(splitted_sizes[i]):

                if splitted_mappings[i][j] == pixel_index:
                    splitted_weights[i][j] += 1.0
                    flag = 1

                if j >= max_j:
                    raise exc.FitException("neighbours exceeds!")

            if flag == 0:
                splitted_mappings[i][j + 1] = pixel_index
                splitted_sizes[i] += 1
                splitted_weights[i][j + 1] = 1.0

        return regularization_util.weighted_pixel_splitted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )
