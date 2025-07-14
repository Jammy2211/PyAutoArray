from abc import ABC


class AbstractVisuals(ABC):
    def __add__(self, other):
        """
        Adds two `Visuals` classes together.

        When we perform plotting, the `Include` class is used to create additional `Visuals` class from the data
        structures that are plotted, for example:

        mask = Mask2D.circular(shape_native=(100, 100), pixel_scales=0.1, radius=3.0)
        array = Array2D.ones(shape_native=(100, 100), pixel_scales=0.1)
        masked_array = al.Array2D(values=array, mask=mask)
        array_plotter = aplt.Array2DPlotter(array=masked_array)
        array_plotter.figure()

        If the user did not manually input a `Visuals2D` object, the one created in `function_array` is the one used to
        plot the image

        However, if the user specifies their own `Visuals2D` object and passed it to the plotter, e.g.:

        visuals_2d = Visuals2D(origin=(0.0, 0.0))
        array_plotter = aplt.Array2DPlotter(array=masked_array)

        We now wish for the `Plotter` to plot the `origin` in the user's input `Visuals2D` object. To achieve this,
        one `Visuals2D` object is created: (i) the user's input instance (with an origin).

        This `__add__` override means we can add the two together to make the final `Visuals2D` object that is
        plotted on the figure containing both the `origin` and `Mask2D`.:

        visuals_2d = visuals_2d_via_user + visuals_2d_via_include

        The ordering of the addition has been specifically chosen to ensure that the `visuals_2d_via_user` does not
        retain the attributes that are added to it by the `visuals_2d_via_include`. This ensures that if multiple plots
        are made, the same `visuals_2d_via_user` is used for every plot. If this were not the case, it would
        permanently inherit attributes from the `Visuals` from the `Include` method and plot them on all figures.
        """

        for attr, value in self.__dict__.items():
            try:
                if other.__dict__[attr] is None and self.__dict__[attr] is not None:
                    other.__dict__[attr] = self.__dict__[attr]
            except KeyError:
                pass

        return other
