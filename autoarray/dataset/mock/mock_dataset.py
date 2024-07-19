from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.dataset.grids import GridsInterface


class MockDataset:
    def __init__(self, grids=None, psf=None, mask=None):
        if grids is None:
            self.grids = GridsInterface(
                uniform=None,
                non_uniform=None,
                pixelization=Grid2D.no_mask(values=[[[1.0, 1.0]]], pixel_scales=1.0),
                blurring=None,
                border_relocator=None,
            )

        self.psf = psf
        self.mask = mask
