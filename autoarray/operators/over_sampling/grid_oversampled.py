class Grid2DOverSampled:
    def __init__(self, grid, over_sampler, pixels_in_mask):
        self.grid = grid
        self.over_sampler = over_sampler
        self.pixels_in_mask = pixels_in_mask

    @property
    def mask(self):
        return self.grid.mask
