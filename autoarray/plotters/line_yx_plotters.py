from autoarray import exc
from autoarray.plotters import plotter_util

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
    unit_label="scaled",
    unit_conversion_factor=None,
    figsize=(7, 7),
    plot_legend=False,
    title="Quantity vs Radius",
    ylabel="Quantity",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="quantity_vs_radius",
):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    plotter_util.set_title(title=title, titlesize=titlesize)

    if x is None:
        x = np.arange(len(y))

    plot_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

    set_xy_labels_and_ticksize(
        unit_label=unit_label,
        ylabel=ylabel,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
    )

    plot_vertical_lines(
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
        unit_conversion_factor=unit_conversion_factor,
    )

    set_legend(plot_legend=plot_legend, legend_fontsize=legend_fontsize)

    plotter_util.output_figure(
        array=None,
        as_subplot=as_subplot,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plotter_util.close_figure(as_subplot=as_subplot)


def plot_y_vs_x(y, x, plot_axis_type, label):

    if plot_axis_type is "linear":
        plt.plot(x, y, label=label)
    elif plot_axis_type is "semilogy":
        plt.semilogy(x, y, label=label)
    elif plot_axis_type is "loglog":
        plt.loglog(x, y, label=label)
    else:
        raise exc.PlottingException(
            "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
            "| semilogy | loglog)"
        )


def set_xy_labels_and_ticksize(
    unit_label, ylabel, xlabelsize, ylabelsize, xyticksize
):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x label is always the distance scale / radius, thus the x-label is either arc-seconds or kpc and depending \
    on the units the figure is plotted in.

    The ylabel is the physical quantity being plotted and is passed as an input parameter.

    Parameters
    -----------
    unit_label : str
        The units of the y / x axis of the plots.
    unit_conversion_factor : float
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the units in kpc.
    ylabel : str
        The y-label of the figure, which is the physical quantity being plotted.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """

    plt.ylabel(ylabel=ylabel, fontsize=ylabelsize)
    plt.xlabel("x (" + unit_label + ")", fontsize=xlabelsize)
    plt.tick_params(labelsize=xyticksize)

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
