import copy


from autoconf.dictable import Dictable


class SettingsPixelization(Dictable):
    def __init__(
        self, use_border: bool = True, is_stochastic: bool = False, kmeans_seed: int = 0
    ):
        """
        Settings which control how a pixelization is performed.

        Parameters
        ----------
        use_border
            If `True`, all coordinates of both `source` mesh grids have pixels outside their border relocated to
            their edge (see `relocated_grid_from()`).
        is_stochastic
            Cetrain pixelizations can create different discretizations using the same parameters / inputs, by changing
            their random seed (e.g. changing the KMeans seed of the `VoronoiBrightnessImage` pixelization). If `True`,
            this random seed changes for every function call, creating completely stochastic pixelizations every time.
        kmeans_seed
            A fixed value for the KMeans seed that dictates the pixelization that is derived for
            a `VoronoiBrightnessImage` pixelization.
        """
        self.use_border = use_border
        self.is_stochastic = is_stochastic
        self.kmeans_seed = kmeans_seed

    def settings_with_is_stochastic_true(self):
        """
        Returns a `SettingsPixelization` object with the same attributes but where `is_stochastic` is `True`.
        """
        settings = copy.copy(self)
        settings.is_stochastic = True
        return settings
