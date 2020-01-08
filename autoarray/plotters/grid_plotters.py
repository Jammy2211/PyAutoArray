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
        titlesize=None,
        xlabelsize=None,
        ylabelsize=None,
        xyticksize=None,
        grid_pointsize=5,
        grid_pointcolor="k",
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        label_yticks=None,
        label_xticks=None,
        output_path=None,
        output_format="show",
        output_filename=None,
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
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            grid_pointsize=grid_pointsize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=label_yticks,
            label_xticks=label_xticks,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

        self.grid_pointcolor = grid_pointcolor

    def plotter_as_sub_plotter(self,):

        return GridPlotter(
            is_sub_plotter=True,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            grid_pointsize=self.grid_pointsize,
            label_title=self.label_title,
            label_yunits=self.label_yunits,
            label_xunits=self.label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=self.output_filename,
        )

    def plotter_with_new_labels_and_filename(
        self,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        output_filename=None,
        unit_conversion_factor=None,
    ):

        label_title = self.label_title if label_title is None else label_title
        label_yunits = self.label_yunits if label_yunits is None else label_yunits
        label_xunits = self.label_xunits if label_xunits is None else label_xunits
        output_filename = (
            self.output_filename if output_filename is None else output_filename
        )
        unit_conversion_factor = (
            self.unit_conversion_factor
            if unit_conversion_factor is None
            else unit_conversion_factor
        )

        return GridPlotter(
            is_sub_plotter=self.is_sub_plotter,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            grid_pointsize=self.grid_pointsize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=output_filename,
        )

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
        xlabelsize : int
            The fontsize of the x axes label.
        ylabelsize : int
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

        self.set_title()
        self.set_yx_labels_and_ticksize()

        if not bypass_limits:

            self.set_axis_limits(
                axis_limits=axis_limits,
                grid=grid,
                symmetric_around_centre=symmetric_around_centre,
            )

        self.set_yxticks(
            array=None,
            extent=grid.extent,
            symmetric_around_centre=symmetric_around_centre,
        )

        self.plot_points(grid=grid, points=points)
        self.plot_lines(line_lists=lines)

        plt.tick_params(labelsize=self.xyticksize)
        self.output_figure(array=None)
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
            print(grid)
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

    def output_figure(self, array):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        array : ndarray
            The 2D array of image to be output, required for outputting the image as a fits file.
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
            be output instead using the *output_subplot_array* function.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'
        """
        if not self.is_sub_plotter:

            if self.output_format is "show":
                plt.show()
            elif self.output_format is "png":
                plt.savefig(
                    self.output_path + self.output_filename + ".png",
                    bbox_inches="tight",
                )
            elif self.output_format is "fits":
                array_util.numpy_array_1d_to_fits(
                    array_1d=array,
                    file_path=self.output_path + self.output_filename + ".fits",
                    overwrite=True,
                )
