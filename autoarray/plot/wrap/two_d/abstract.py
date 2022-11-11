from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.wrap.base.abstract import AbstractMatWrap

class AbstractMatWrap2D(AbstractMatWrap):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array2D` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatWrap`.
    """

    @property
    def config_folder(self):
        return "mat_wrap_2d"