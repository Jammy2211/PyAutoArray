from typing import Optional

from autoarray.dataset.abstract.settings import AbstractSettingsDataset
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.operators.transformer import TransformerNUFFT


class SettingsInterferometer(AbstractSettingsDataset):
    def __init__(
        self,
        grid: Optional[Grid2D] = None,
        grid_pixelization: Optional[Grid2D] = None,
        transformer_class=TransformerNUFFT,
    ):
        """
        An interferometer dataset's settings, containing quantities used for fitting the dataset like the grids
        of (y,x) coordinates.

        The interferometer dataset is described in the `Interferometer` class, but in brief it is a dataset which
        contains the visibilities data, noise-map, Fourier transformer, etc.

        It also contains the `uv_wavelengths` and `real_space_mask` which define how a real space image is Fourier
        transformed to the uv-plane. The grids contained in the settings are aligned with this mask.

        The interferometer settings control the following:

        - `grid`: A grids of (y,x) coordinates which align with the real-space mask's pixels, whereby each coordinate
        corresponds to the centre of an image pixel. This may be used in fits to calculate the model image of the
        interferometer data.

        - `grid_pixelization`: A grid of (y,x) coordinates which align with the pixels of a pixelization. This grid
        is specifically used for pixelizations computed via the `invserion` module, which often use different
        oversampling and sub-size values to the grid above.

        - `transformer_class`: The class of the Fourier Transform which maps images from real space to Fourier space
        visibilities and the uv-plane.

        In the project PyAutoGalaxy the interferometer data grids are used to compute the images of galaxies via their light
        profiles. In PyAutoLens, the grids are used for ray tracing lensing calculations associated with a mass profile.

        Parameters
        ----------
        grid
            The grid used to perform calculations not associated with a pixelization. In PyAutoGalaxy and
            PyAutoLens this is light profile calculations.
        grid_pixelization
            The grid used to perform calculations associated with a pixelization, which is therefore passed into
            the calculations performed in the `inversion` module.
        transformer_class
            The class of the Fourier Transform which maps images from real space to Fourier space visibilities and
            the uv-plane.
        """

        super().__init__(
            grid=grid,
            grid_pixelization=grid_pixelization,
        )

        self.transformer_class = transformer_class
