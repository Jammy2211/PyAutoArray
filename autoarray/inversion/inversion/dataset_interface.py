class DatasetInterface:
    def __init__(
        self,
        data,
        noise_map,
        convolver=None,
        transformer=None,
        w_tilde=None,
        grid=None,
        grid_pixelization=None,
        blurring_grid=None,
    ):
        self.data = data
        self.noise_map = noise_map
        self.convolver = convolver
        self.w_tilde = w_tilde
        self.grid = grid
        self.grid_pixelization = grid_pixelization
        self.blurring_grid = blurring_grid
