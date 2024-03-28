from typing import Optional

from autoarray.structures.grids.uniform_2d import Grid2D


class AbstractSettingsDataset:
    def __init__(
        self,
        grid: Optional[Grid2D] = None,
        grid_pixelization: Optional[Grid2D] = None,
    ):
        """
        An abstract dataset's settings, wcontaining quantities used for fitting the dataset like the grids of (y,x)
        coordinates.

        The dataset ettings control the following:

        - `grid`: A grids of (y,x) coordinates which align with the image pixels, whereby each coordinate corresponds to
        the centre of an image pixel. This may be used in fits to calculate the model image of the imaging data.

        - `grid_pixelization`: A grid of (y,x) coordinates which align with the pixels of a pixelization. This grid
        is specifically used for pixelizations computed via the `invserion` module, which often use different
        oversampling and sub-size values to the grid above.

        In the project PyAutoGalaxy the imaging data grids are used to compute the images of galaxies via their light
        profiles. In PyAutoLens, the grids are used for ray tracing lensing calculations associated with a mass profile.

        Parameters
        ----------
        grid
            The grid used to perform calculations not associated with a pixelization. In PyAutoGalaxy and
            PyAutoLens this is light profile calculations.
        grid_pixelization
            The grid used to perform calculations associated with a pixelization, which is therefore passed into
            the calculations performed in the `inversion` module.
        """

        self.grid = grid
        self.grid_pixelization = grid_pixelization

