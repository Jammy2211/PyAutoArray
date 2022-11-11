from typing import Optional, Union

from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.visuals.two_d import Visuals2D

class AbstractGetVisuals:
    def __init__(
        self, include: Union[Include1D, Include2D], visuals: Union[Visuals1D, Visuals2D]
    ):
        """
        Class which gets attributes and adds them to a `Visuals` objects, such that they are plotted on figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include` object. If this entry is `False`, the `GetVisuals.get` method returns a None and the attribute
        is omitted from the plot.

        The `GetVisuals` class adds new visuals to a pre-existing `Visuals` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals` class.
        """
        self.include = include
        self.visuals = visuals

    def get(self, name: str, value, include_name: Optional[str] = None):
        """
        Get an attribute for plotting in a `Visuals1D` object based on the following criteria:

        1) If `visuals_1d` already has a value for the attribute this is returned, over-riding the input `value` of
        that attribute.

        2) If `visuals_1d` do not contain the attribute, the input `value` is returned provided its corresponding
        entry in the `Include1D` class is `True`.

        3) If the `Include1D` entry is `False` a None is returned and the attribute is therefore not plotted.

        Parameters
        ----------
        name
            The name of the attribute which is to be extracted.
        value
            The `value` of the attribute, which is used when criteria 2 above is met.

        Returns
        -------
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        if include_name is None:
            include_name = name

        if getattr(self.visuals, name) is not None:
            return getattr(self.visuals, name)
        else:
            if getattr(self.include, include_name):
                return value