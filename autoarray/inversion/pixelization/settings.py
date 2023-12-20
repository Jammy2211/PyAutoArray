import copy


class SettingsPixelization:
    def __init__(self, use_border: bool = True):
        """
        Settings which control how a pixelization is performed.

        Parameters
        ----------
        use_border
            If `True`, all coordinates of both `source` mesh grids have pixels outside their border relocated to
            their edge (see `relocated_grid_from()`).
        """
        self.use_border = use_border
