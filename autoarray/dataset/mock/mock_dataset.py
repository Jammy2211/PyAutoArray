from autoarray.structures.grids.uniform_2d import Grid2D


class MockDataset:
    def __init__(self, grid_pixelization=None, psf=None, mask=None):
        self.grid_pixelization = grid_pixelization or Grid2D.no_mask(
            values=[[[1.0, 1.0]]], pixel_scales=1.0
        )
        self.psf = psf
        self.mask = mask
