from autoarray.plot.get_visuals.abstract import AbstractGetVisuals
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.structures.arrays.uniform_1d import Array1D


class GetVisuals1D(AbstractGetVisuals):
    def __init__(self, visuals: Visuals1D):
        """
        Class which gets 1D attributes and adds them to a `Visuals1D` objects, such that they are plotted on 1D figures.

        The `GetVisuals1D` class adds new visuals to a pre-existing `Visuals1D` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals1D` class.
        """
        super().__init__(visuals=visuals)

    def via_array_1d_from(self, array_1d: Array1D) -> Visuals1D:
        """
        From an `Array1D` get its attributes that can be plotted and return them in a `Visuals1D` object.

        From an `Array1D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 1D array's coordinate system.
        - mask: the mask of the 1D array.

        Parameters
        ----------
        array
            The 1D array whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        return self.visuals + self.visuals.__class__(
            origin=self.get("origin", array_1d.origin),
            mask=self.get("mask", array_1d.mask),
        )
