from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt
import numpy as np
import itertools

from autoarray.plotters import plotters
from autoarray.util import array_util


class GridPlotter(plotters.Plotter):
    def __init__(
        self,
        is_sub_plotter=False,
        use_scaled_units=None,
        unit_conversion_factor=None,
        figsize=None,
        aspect=None,
        cmap=None,
        norm=None,
        norm_min=None,
        norm_max=None,
        linthresh=None,
        linscale=None,
        cb_ticksize=None,
        cb_fraction=None,
        cb_pad=None,
        cb_tick_values=None,
        cb_tick_labels=None,
        grid_pointsize=5,
        grid_pointcolor="k",
        ticks=plotters.Ticks(),
        labels=plotters.Labels(),
        output=plotters.Output(),
    ):

        super(GridPlotter, self).__init__(
            is_sub_plotter=is_sub_plotter,
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            grid_pointsize=grid_pointsize,
            ticks=ticks,
            labels=labels,
            output=output,
        )

        self.grid_pointcolor = grid_pointcolor

    def plot_grid(
        self,
        grid,
        colors=None,
        axis_limits=None,
        points=None,
        lines=None,
        symmetric_around_centre=True,
        bypass_limits=False,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotters of points.

        Parameters
        -----------
        grid : data_type.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        points : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        as_subplot : bool
            Whether the grid is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        label_yunits : str
            The label of the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        pointsize : int
            The size of the points plotted on the grid.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
        """

        self.setup_figure()

        if colors is not None:

            plt.cm.get_cmap(self.cmap)

        plt.scatter(
            y=np.asarray(grid[:, 0]),
            x=np.asarray(grid[:, 1]),
            c=colors,
            s=self.grid_pointsize,
            marker=".",
            cmap=self.cmap,
        )

        if colors is not None:

            self.set_colorbar()

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        if not bypass_limits:

            self.set_axis_limits(
                axis_limits=axis_limits,
                grid=grid,
                symmetric_around_centre=symmetric_around_centre,
            )

        self.ticks.set_yticks(
            array=None,
            extent=grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            symmetric_around_centre=symmetric_around_centre,
        )
        self.ticks.set_xticks(
            array=None,
            extent=grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            symmetric_around_centre=symmetric_around_centre,
        )

        self.plot_points(grid=grid, points=points)
        self.plot_lines(line_lists=lines)

        self.output.to_figure(structure=grid, is_sub_plotter=self.is_sub_plotter)
        self.close_figure()

    def set_axis_limits(self, axis_limits, grid, symmetric_around_centre):
        """Set the axis limits of the figure the grid is plotted on.

        Parameters
        -----------
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        """
        if axis_limits is not None:
            plt.axis(axis_limits)
        elif symmetric_around_centre:
            ymin = np.min(grid[:, 0])
            ymax = np.max(grid[:, 0])
            xmin = np.min(grid[:, 1])
            xmax = np.max(grid[:, 1])
            x = np.max([np.abs(xmin), np.abs(xmax)])
            y = np.max([np.abs(ymin), np.abs(ymax)])
            axis_limits = [-x, x, -y, y]
            plt.axis(axis_limits)

    def plot_points(self, grid, points):
        """Plot a subset of points in a different color, to highlight a specifc region of the grid (e.g. how certain \
        pixels map between different planes).

        Parameters
        -----------
        grid : ndarray or data_type.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        points : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        pointcolor : str or None
            The color the points should be plotted. If None, the points are iterated through a cycle of colors.
        """
        if points is not None:

            if self.grid_pointcolor is None:

                point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
                for point_set in points:
                    plt.scatter(
                        y=np.asarray(grid[point_set, 0]),
                        x=np.asarray(grid[point_set, 1]),
                        s=8,
                        color=next(point_colors),
                    )

            else:

                for point_set in points:
                    plt.scatter(
                        y=np.asarray(grid[point_set, 0]),
                        x=np.asarray(grid[point_set, 1]),
                        s=8,
                        color=self.grid_pointcolor,
                    )
