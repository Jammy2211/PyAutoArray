from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

import copy
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union

from autoarray.plot.wrap import base as wb


class AbstractMatPlot:
    def __init__(
        self,
        units: Optional[wb.Units] = None,
        figure: Optional[wb.Figure] = None,
        axis: Optional[wb.Axis] = None,
        cmap: Optional[wb.Cmap] = None,
        colorbar: Optional[wb.Colorbar] = None,
        colorbar_tickparams: Optional[wb.ColorbarTickParams] = None,
        tickparams: Optional[wb.TickParams] = None,
        yticks: Optional[wb.YTicks] = None,
        xticks: Optional[wb.XTicks] = None,
        title: Optional[wb.Title] = None,
        ylabel: Optional[wb.YLabel] = None,
        xlabel: Optional[wb.XLabel] = None,
        text: Optional[Union[wb.Text, List[wb.Text]]] = None,
        annotate: Optional[Union[wb.Annotate, List[wb.Annotate]]] = None,
        legend: Optional[wb.Legend] = None,
        output: Optional[wb.Output] = None,
    ):
        """
        Visualizes data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `wrap_base` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following data structures can be plotted using the following matplotlib functions:

        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `MapperVoronoiNoInterp`: using `plt.fill`.

        Parameters
        ----------
        units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`
        axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams
            Customizes the appearances of the y and x ticks on the plot (e.g. the fontsize) using `plt.tick_params`.
        yticks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        text
            Sets any text on the figure and customizes its appearance using `plt.text`.
        annotate
            Sets any annotations on the figure and customizes its appearance using `plt.annotate`.
        legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        """

        self.units = units or wb.Units(is_default=True)
        self.figure = figure or wb.Figure(is_default=True)
        self.axis = axis or wb.Axis(is_default=True)

        self.cmap = cmap or wb.Cmap(is_default=True)

        self.colorbar = colorbar or wb.Colorbar(is_default=True)
        self.colorbar_tickparams = colorbar_tickparams or wb.ColorbarTickParams(
            is_default=True
        )

        self.tickparams = tickparams or wb.TickParams(is_default=True)
        self.yticks = yticks or wb.YTicks(is_default=True)
        self.xticks = xticks or wb.XTicks(is_default=True)

        self.title = title or wb.Title(is_default=True)
        self.ylabel = ylabel or wb.YLabel(is_default=True)
        self.xlabel = xlabel or wb.XLabel(is_default=True)

        self.text = text or wb.Text(is_default=True)
        self.annotate = annotate or wb.Annotate(is_default=True)
        self.legend = legend or wb.Legend(is_default=True)
        self.output = output or wb.Output(is_default=True)

        self.number_subplots = None
        self.subplot_shape = None
        self.subplot_index = None

    def __add__(self, other):
        """
        Adds two `MatPlot` classes together.

        A `MatPlot` class contains many of the `MatWrap` objects which customize matplotlib figures. One
        may have a standard `MatPlot` object, which customizes many figures in the same way, for example:

        mat_plot_2d_base = aplt.MatPlot2D(
            yticks=aplt.YTicks(fontsize=18),
            xticks=aplt.XTicks(fontsize=18),
            ylabel=aplt.YLabel(label=""),
            xlabel=aplt.XLabel(label=""),
        )

        However, one may require many unique `MatPlot` objects for a number of different figures, which all use
        these settings. These can be created by creating the unique `MatPlot` objects and adding the above object
        to each:

        mat_plot_2d = aplt.MatPlot2D(
            title=aplt.Title(label="Example Figure 1"),
        )

        mat_plot_2d = mat_plot_2d + mat_plot_2d_base

        mat_plot_2d = aplt.MatPlot2D(
            title=aplt.Title(label="Example Figure 2"),
        )

        mat_plot_2d = mat_plot_2d + mat_plot_2d_base
        """

        other = copy.deepcopy(other)

        for attr, value in self.__dict__.items():

            try:
                if value.kwargs.get("is_default") is not True:
                    other.__dict__[attr] = value
            except AttributeError:
                pass

        return other

    def set_for_subplot(self, is_for_subplot: bool):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The [subplot] section of the config file of every `MatWrap` object is used instead of [figure].
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """
        self.is_for_subplot = is_for_subplot
        self.output.bypass = is_for_subplot

        for attr, value in self.__dict__.items():
            if hasattr(value, "is_for_subplot"):
                value.is_for_subplot = is_for_subplot

    def get_subplot_rows_columns(self, number_subplots):
        """
        Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are
        going to be plotted.

        Parameters
        ----------
        number_subplots
            The number of subplots that are to be plotted in the figure.
        """

        if self.subplot_shape is not None:
            return self.subplot_shape

        if number_subplots <= 2:
            return 1, 2
        elif number_subplots <= 4:
            return 2, 2
        elif number_subplots <= 6:
            return 2, 3
        elif number_subplots <= 9:
            return 3, 3
        elif number_subplots <= 12:
            return 3, 4
        elif number_subplots <= 16:
            return 4, 4
        elif number_subplots <= 20:
            return 4, 5
        else:
            return 6, 6

    def setup_subplot(
        self,
        aspect: Optional[Tuple[float, float]] = None,
        subplot_rows_columns: Tuple[int, int] = None,
    ):
        """
        Setup a new figure to be plotted on a subplot, which is used by a `Plotter` when plotting multiple images
        on a subplot.

        Every time a new figure is plotted on the subplot, the counter `subplot_index` increases by 1.

        The shape of the subplot is determined by the number of figures on the subplot.

        The aspect ratio of the subplot can be customized based on the size of the figures.

        Every time

        Parameters
        ----------
        aspect
            The aspect ratio of the overall subplot.
        subplot_rows_columns
            The number of rows and columns in the subplot.
        """
        if subplot_rows_columns is None:
            rows, columns = self.get_subplot_rows_columns(
                number_subplots=self.number_subplots
            )
        else:
            rows = subplot_rows_columns[0]
            columns = subplot_rows_columns[1]

        if aspect is None:
            ax = plt.subplot(rows, columns, self.subplot_index)
        else:
            ax = plt.subplot(rows, columns, self.subplot_index, aspect=float(aspect))

        self.subplot_index += 1

        return ax
