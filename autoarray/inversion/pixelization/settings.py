import copy


class SettingsPixelization:
    def __init__(self, use_border: bool = True, seed: int = 0):
        """
        Settings which control how a pixelization is performed.

        Parameters
        ----------
        use_border
            If `True`, all coordinates of both `source` mesh grids have pixels outside their border relocated to
            their edge (see `relocated_grid_from()`).
        seed
            A fixed value for the KMeans seed that dictates the pixelization that is derived for
            a `VoronoiBrightnessImage` pixelization.
        """
        self.use_border = use_border
        self.seed = seed
