from autoarray import exc
from autoarray.util import plotter_util

import matplotlib.pyplot as plt
import numpy as np


def plot_line(
    y,
    x,
    as_subplot=False,
    label=None,
    plot_axis_type="semilogy",
    vertical_lines=None,
    vertical_line_labels=None,
    unit_label_y="Quantity",
    unit_label_x="scaled",
    unit_conversion_factor=None,
    figsize=(7, 7),
    plot_legend=False,
    title="Quantity vs Radius",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    pointsize=20,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="quantity_vs_radius",
):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    plotter_util.set_title(title=title, titlesize=titlesize)

    if y is None:
        return

    if x is None:
        x = np.arange(len(y))

    plot_y_vs_x(
        y=y, x=x, plot_axis_type=plot_axis_type, label=label, pointsize=pointsize
    )

    set_xy_labels_and_ticksize(
        unit_label_y=unit_label_y,
        unit_label_x=unit_label_x,
        ylabelsize=ylabelsize,
        xlabelsize=xlabelsize,
        xyticksize=xyticksize,
    )

    plot_vertical_lines(
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
        unit_conversion_factor=unit_conversion_factor,
    )

    set_legend(plot_legend=plot_legend, legend_fontsize=legend_fontsize)

    set_xticks(
        extent=[np.min(x), np.max(x)],
        unit_conversion_factor=unit_conversion_factor,
        xticks_manual=None,
    )

    if output_format is not "fits":

        plotter_util.output_figure(
            array=None,
            as_subplot=as_subplot,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
        )

    plotter_util.close_figure(as_subplot=as_subplot)


def plot_y_vs_x(y, x, plot_axis_type, label, pointsize):

    if plot_axis_type is "linear":
        plt.plot(x, y, label=label)
    elif plot_axis_type is "semilogy":
        plt.semilogy(x, y, label=label)
    elif plot_axis_type is "loglog":
        plt.loglog(x, y, label=label)
    elif plot_axis_type is "scatter":
        plt.scatter(x, y, label=label, s=pointsize)
    else:
        raise exc.PlottingException(
            "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
            "| semilogy | loglog)"
        )


def set_xy_labels_and_ticksize(
    unit_label_x, unit_label_y, xlabelsize, ylabelsize, xyticksize
):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x label is always the distance scale / radius, thus the x-label is either arc-seconds or kpc and depending \
    on the unit_label the figure is plotted in.

    The ylabel is the physical quantity being plotted and is passed as an input parameter.

    Parameters
    -----------
    unit_label_x : str
        The unit_label of the y / x axis of the plots.
    unit_conversion_factor : float
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
    unit_label_y : str
        The y-label of the figure, which is the physical quantity being plotted.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """

    plt.ylabel(ylabel=unit_label_y, fontsize=ylabelsize)
    plt.xlabel("x (" + unit_label_x + ")", fontsize=xlabelsize)
    plt.tick_params(labelsize=xyticksize)


def set_xticks(extent, unit_conversion_factor, xticks_manual):
    """Get the extent of the dimensions of the array in the unit_label of the figure (e.g. arc-seconds or kpc).

    This is used to set the extent of the array and thus the y / x axis limits.

    Parameters
    -----------
    array : data_type.array.aa.Scaled
        The 2D array of data_type which is plotted.
    unit_label : str
        The label for the unit_label of the y / x axis of the plots.
    unit_conversion_factor : float
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    """

    xticks = np.round(np.linspace(extent[0], extent[1], 5), 2)

    if xticks_manual is not None:
        xtick_labels = np.asarray([xticks_manual[0], xticks_manual[3]])
    elif unit_conversion_factor is None:
        xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
    elif unit_conversion_factor is not None:
        xtick_labels = (
            np.round(np.linspace(extent[0], extent[1], 5), 2) * unit_conversion_factor
        )
    else:
        raise exc.PlottingException(
            "The y and y ticks cannot be set using the input options."
        )

    plt.xticks(ticks=xticks, labels=xtick_labels)


def plot_vertical_lines(vertical_lines, vertical_line_labels, unit_conversion_factor):

    if vertical_lines is [] or vertical_lines is None:
        return

    for vertical_line, vertical_line_label in zip(vertical_lines, vertical_line_labels):

        if unit_conversion_factor is None:
            x_value_plot = vertical_line
        elif unit_conversion_factor is not None:
            x_value_plot = vertical_line * unit_conversion_factor

        plt.axvline(x=x_value_plot, label=vertical_line_label, linestyle="--")


def set_legend(plot_legend, legend_fontsize):
    if plot_legend:
        plt.legend(fontsize=legend_fontsize)
