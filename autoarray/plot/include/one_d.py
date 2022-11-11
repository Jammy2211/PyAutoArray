from typing import Optional

from autoarray.plot.include.abstract import AbstractInclude


class Include1D(AbstractInclude):
    def __init__(self, origin: Optional[bool] = None, mask: Optional[bool] = None):
        """
        Sets which `Visuals1D` are included on a figure plotting 1D data that is plotted using a `Plotter1D`.

        The `Include` object is used to extract the visuals of the plotted 1D data structures so they can be used in
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin
            If `True`, the `origin` of the plotted data structure (e.g. `Line`)  is included on the figure.
        mask
            if `True`, the `mask` of the plotted data structure (e.g. `Line`)  is included on the figure.
        """
        super().__init__(origin=origin, mask=mask)

    @property
    def section(self):
        return "include_1d"
