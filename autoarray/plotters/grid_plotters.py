from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt
import numpy as np
import itertools

from autoarray.structures import grids
from autoarray.util import array_util, plotter_util


def plot_grid(
    grid,
    colors=None,
    axis_limits=None,
    points=None,
    lines=None,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label_y="scaled",
    unit_label_x="scaled",
    figsize=(12, 8),
    pointsize=5,
    pointcolor="k",
    xyticksize=16,
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Grid",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    symmetric_around_centre=True,
    bypass_limits=False,
    output_path=None,
    output_format="show",
    output_filename="grid",
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
    unit_label_y : str
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

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    if colors is not None:

        plt.cm.get_cmap(cmap)

    plt.scatter(
        y=np.asarray(grid[:, 0]),
        x=np.asarray(grid[:, 1]),
        c=colors,
        s=pointsize,
        marker=".",
        cmap=cmap,
    )

    if colors is not None:

        plotter_util.set_colorbar(
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
        )

    plotter_util.set_title(title=title, titlesize=titlesize)
    plotter_util.set_yx_labels_and_ticksize(
        unit_label_y=unit_label_y,
        unit_label_x=unit_label_x,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
    )

    if not bypass_limits:

        set_axis_limits(
            axis_limits=axis_limits,
            grid=grid,
            symmetric_around_centre=symmetric_around_centre,
        )

    plotter_util.set_yxticks(
        array=None,
        extent=grid.extent,
        use_scaled_units=True,
        unit_conversion_factor=unit_conversion_factor,
        xticks_manual=None,
        yticks_manual=None,
        symmetric_around_centre=symmetric_around_centre,
    )

    plot_points(grid=grid, points=points, pointcolor=pointcolor)
    plotter_util.plot_lines(line_lists=lines)

    plt.tick_params(labelsize=xyticksize)
    output_figure(
        array=None,
        as_subplot=as_subplot,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plotter_util.close_figure(as_subplot=as_subplot)


def set_axis_limits(axis_limits, grid, symmetric_around_centre):
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


def plot_points(grid, points, pointcolor):
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

        if pointcolor is None:

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
                    color=pointcolor,
                )


def output_figure(array, as_subplot, output_path, output_filename, output_format):
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
    if not as_subplot:

        if output_format is "show":
            plt.show()
        elif output_format is "png":
            plt.savefig(output_path + output_filename + ".png", bbox_inches="tight")
        elif output_format is "fits":
            array_util.numpy_array_1d_to_fits(
                array_1d=array,
                file_path=output_path + output_filename + ".fits",
                overwrite=True,
            )
