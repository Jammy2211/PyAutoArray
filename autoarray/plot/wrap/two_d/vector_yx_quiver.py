import matplotlib.pyplot as plt

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular


class VectorYXQuiver(AbstractMatWrap2D):
    """
    Plots a `VectorField` data structure. A vector field is a set of 2D vectors on a grid of 2d (y,x) coordinates.
    These are plotted as arrows representing the (y,x) components of each vector at each (y,x) coordinate of it
    grid.

    This object wraps the following Matplotlib method:

    https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
    """

    def quiver_vectors(self, vectors: VectorYX2DIrregular):
        """
        Plot a vector field using the matplotlib method `plt.quiver` such that each vector appears as an arrow whose
        direction depends on the y and x magnitudes of the vector.

        Parameters
        ----------
        vectors : VectorYX2DIrregular
            The vector field that is plotted using `plt.quiver`.
        """
        plt.quiver(
            vectors.grid[:, 1],
            vectors.grid[:, 0],
            vectors[:, 1],
            vectors[:, 0],
            **self.config_dict,
        )