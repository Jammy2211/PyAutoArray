import copy


class SettingsPixelization:
    def __init__(
        self,
        use_border: bool = True,
        pixel_limit: int = None,
        is_stochastic: bool = False,
        kmeans_seed: int = 0,
    ):

        self.use_border = use_border
        self.pixel_limit = pixel_limit
        self.is_stochastic = is_stochastic
        self.kmeans_seed = kmeans_seed

    def settings_with_is_stochastic_true(self):
        settings = copy.copy(self)
        settings.is_stochastic = True
        return settings
