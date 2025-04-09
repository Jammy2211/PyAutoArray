import matplotlib.pyplot as plt

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D


class ArrayOverlay(AbstractMatWrap2D):
    """
    Overlays an `Array2D` data structure over a figure.

    This object wraps the following Matplotlib method:

    - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

    This uses the `Units` and coordinate system of the `Array2D` to overlay it on on the coordinate system of the
    figure that is plotted.
    """

    def overlay_array(self, array, figure):
        aspect = figure.aspect_from(shape_native=array.shape_native)
        extent = array.extent_of_zoomed_array(buffer=0)

        plt.imshow(X=array.native, aspect=aspect, extent=extent, **self.config_dict)
