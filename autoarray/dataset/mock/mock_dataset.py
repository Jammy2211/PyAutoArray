class MockDataset:
    def __init__(self, grid_pixelization=None, psf=None, mask=None):

        self.grid_pixelization = grid_pixelization
        self.psf = psf
        self.mask = mask
