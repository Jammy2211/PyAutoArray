import logging
from typing import Union

from autoarray.structures.arrays.uniform_2d import Array2D

logger = logging.getLogger(__name__)


class OverSamplingDataset:
    def __init__(
        self,
        lp: Union[int, Array2D] = 4,
        pixelization: Union[int, Array2D] = 4,
    ):
        """
        Customizes how over sampling calculations are performed using the grids of the data.

        If a grid is uniform and the centre of each point on the grid is the centre of a 2D pixel, evaluating
        the value of a function on the grid requires a 2D line integral to compute it precisely. This can be
        computationally expensive and difficult to implement.

        Over sampling is a numerical technique where the function is evaluated on a sub-grid within each grid pixel
        which is higher resolution than the grid itself. This approximates more closely the value of the function
        within a 2D line intergral of the values in the square pixel that the grid is centred.

        For example, in PyAutoGalaxy and PyAutoLens the light profiles and galaxies are evaluated in order to determine
        how much light falls in each pixel. This uses over sampling and therefore a higher resolution grid than the
        image data to ensure the calculation is accurate.

        This class controls how over sampling is performed for 3 different types of grids:

        - `grid`: A grids of (y,x) coordinates which aligns with the centre of every image pixel of the image data.

        - `grids.pixelization`: A grid of (y,x) coordinates which again align with the centre of every image pixel of
        the image data. This grid is used specifically for pixelizations computed via the `inversion` module, which
        can benefit from using different oversampling schemes than the normal grid.

        Different calculations typically benefit from different over sampling, which this class enables
        the customization of.

        Parameters
        ----------
        lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid so as to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        """
        self.lp = lp
        self.pixelization = pixelization
