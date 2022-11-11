from typing import Optional

from autoconf import conf


class AbstractInclude:
    def __init__(self, origin: Optional[bool] = None, mask: Optional[bool] = None):
        """
        Sets which `Visuals` are included on a figure that is plotted using a `Plotter`.

        The `Include` object is used to extract the visuals of the plotted data structure (e.g. `Array2D`, `Grid2D`) so
        they can be used in plot functions. Only visuals with a `True` entry in the `Include` object are extracted and t
        plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin
            If `True`, the `origin` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mask
            if `True`, the `mask` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        """

        self._origin = origin
        self._mask = mask

    def load(self, value, name):
        if value is True:
            return True
        elif value is False:
            return False
        elif value is None:
            return conf.instance["visualize"]["include"][self.section][name]

    @property
    def section(self):
        raise NotImplementedError

    @property
    def origin(self):
        return self.load(value=self._origin, name="origin")

    @property
    def mask(self):
        return self.load(value=self._mask, name="mask")
