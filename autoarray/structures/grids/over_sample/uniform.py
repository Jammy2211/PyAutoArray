
class OverSampleUniform:
    def __init__(
        self,
        sub_size : int = 1
    ):
        """
        Over samples grid calculations using an iterative sub-grid that increases the sampling until a threshold
        accuracy is met.

        When a 2D grid of (y,x) coordinates is input into a function, the result is evaluated at every coordinate
        on the grid. When the grid is paired to a 2D image (e.g. an `Array2D`) the solution needs to approximate
        the 2D integral of that function in each pixel. Over sample objects define how this over-sampling is performed.

        This object iteratively recomputes the analytic function at increasing sub-grid resolutions until an input
        fractional accuracy is reached. The sub-grid is increase in each pixel, therefore it will gradually better
        approximate the 2D integral after each iteration.

        Iteration is performed on a per pixel basis, such that the sub-grid size will stop at lower values
        in pixels where the fractional accuracy is met quickly. It will only go to high values where high sampling is
        required to meet the accuracy. This ensures the function is evaluated accurately in a computationally
        efficient manner.

        Parameters
        ----------
        fractional_accuracy
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub size to the value computed using the previous sub_size. The fractional
            accuracy does not depend on the units or magnitude of the function being evaluated.
        relative_accuracy
            The relative accuracy the function evaluted must meet to be accepted, where this value is the absolute
            difference of the values computed using the higher sub size and lower sub size grids. The relative
            accuracy depends on the units / magnitude of the function being evaluated.
        sub_steps
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        """

        self.sub_size = sub_size