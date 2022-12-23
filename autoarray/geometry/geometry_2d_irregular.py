from typing import Tuple

from autoarray.geometry.abstract_2d import AbstractGeometry2D


class Geometry2DIrregular(AbstractGeometry2D):
    def __init__(
        self,
        shape_native_scaled: Tuple[float, float],
        scaled_maxima: Tuple[float, float],
        scaled_minima: Tuple[float, float],
    ):
        """
        A 2D geometry, representing an irregular grid of (y,x) coordinates.

        This class is used for defining the extent of the irregular grid when visualizing it.

        Parameters
        ----------
        shape_native_scaled
            The 2D scaled shape of the geometry defining the full extent of this object.
        scaled_maxima
            The maximum (y,x) scaled coordinates of the 2D geometry.
        scaled_minima
            The minimum (y,x) scaled coordinates of the 2D geometry.
        """
        self.shape_native_scaled = shape_native_scaled
        self.scaled_maxima = scaled_maxima
        self.scaled_minima = scaled_minima

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """
        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module, which is why the x and y coordinates are swapped compared to
        the normal PyAutoArray convention.
        """
        return (
            self.scaled_minima[1],
            self.scaled_maxima[1],
            self.scaled_minima[0],
            self.scaled_maxima[0],
        )
