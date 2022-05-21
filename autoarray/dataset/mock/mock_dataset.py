class MockDataset:
    def __init__(self, grid_pixelized=None, psf=None, mask=None):

        self.grid_pixelized = grid_pixelized
        self.psf = psf
        self.mask = mask
