from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np
from autoarray import exc
from autoarray.util import array_util


def get_subplot_rows_columns_figsize(number_subplots):
    """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

    Parameters
    -----------
    number_subplots : int
        The number of subplots that are to be plotted in the figure.
    """
    if number_subplots <= 2:
        return 1, 2, (18, 8)
    elif number_subplots <= 4:
        return 2, 2, (13, 10)
    elif number_subplots <= 6:
        return 2, 3, (18, 12)
    elif number_subplots <= 9:
        return 3, 3, (25, 20)
    elif number_subplots <= 12:
        return 3, 4, (25, 20)
    elif number_subplots <= 16:
        return 4, 4, (25, 20)
    elif number_subplots <= 20:
        return 4, 5, (25, 20)
    else:
        return 6, 6, (25, 20)


def setup_figure(figsize, as_subplot):
    """Setup a figure for plotting an image.

    Parameters
    -----------
    figsize : (int, int)
        The size of the figure in (rows, columns).
    as_subplot : bool
        If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
        new figure and so that it can be output using the *output_subplot_array* function.
    """
    if not as_subplot:
        fig = plt.figure(figsize=figsize)
        return fig


def set_title(title, titlesize):
    """Set the title and title size of the figure.

    Parameters
    -----------
    title : str
        The text of the title.
    titlesize : int
        The size of of the title of the figure.
    """
    plt.title(title, fontsize=titlesize)


def set_yx_labels_and_ticksize(
    unit_label_y, unit_label_x, xlabelsize, ylabelsize, xyticksize
):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
    unit_label the figure is plotted in.

    Parameters
    -----------
    unit_label : str
        The label for the unit_label of the y / x axis of the plots.
    unit_conversion_factor : float
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """

    plt.ylabel("y (" + unit_label_y + ")", fontsize=ylabelsize)
    plt.xlabel("x (" + unit_label_x + ")", fontsize=xlabelsize)

    plt.tick_params(labelsize=xyticksize)


def set_yxticks(
    array,
    extent,
    use_scaled_units,
    unit_conversion_factor,
    xticks_manual,
    yticks_manual,
    symmetric_around_centre=False,
):
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

    if symmetric_around_centre:
        return

    yticks = np.linspace(extent[2], extent[3], 5)
    xticks = np.linspace(extent[0], extent[1], 5)

    if xticks_manual is not None and yticks_manual is not None:
        ytick_labels = np.asarray([yticks_manual[0], yticks_manual[3]])
        xtick_labels = np.asarray([xticks_manual[0], xticks_manual[3]])
    elif not use_scaled_units:
        ytick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        xtick_labels = np.linspace(0, array.shape_2d[1], 5).astype("int")
    elif use_scaled_units and unit_conversion_factor is None:
        ytick_labels = np.round(np.linspace(extent[2], extent[3], 5), 2)
        xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
    elif use_scaled_units and unit_conversion_factor is not None:
        ytick_labels = np.round(
            np.linspace(
                extent[2] * unit_conversion_factor,
                extent[3] * unit_conversion_factor,
                5,
            ),
            2,
        )
        xtick_labels = np.round(
            np.linspace(
                extent[0] * unit_conversion_factor,
                extent[1] * unit_conversion_factor,
                5,
            ),
            2,
        )
    else:
        raise exc.PlottingException(
            "The y and y ticks cannot be set using the input options."
        )

    plt.yticks(ticks=yticks, labels=ytick_labels)
    plt.xticks(ticks=xticks, labels=xtick_labels)


def set_colorbar(cb_ticksize, cb_fraction, cb_pad, cb_tick_values, cb_tick_labels):
    """Setup the colorbar of the figure, specifically its ticksize and the size is appears relative to the figure.

    Parameters
    -----------
    cb_ticksize : int
        The size of the tick labels on the colorbar.
    cb_fraction : float
        The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
    cb_pad : float
        Pads the color bar in the figure, which resizes the colorbar relative to the figure.
    cb_tick_values : [float]
        Manually specified values of where the colorbar tick labels appear on the colorbar.
    cb_tick_labels : [float]
        Manually specified labels of the color bar tick labels, which appear where specified by cb_tick_values.
    """

    if cb_tick_values is None and cb_tick_labels is None:
        cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad)
    elif cb_tick_values is not None and cb_tick_labels is not None:
        cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad, ticks=cb_tick_values)
        cb.ax.set_yticklabels(cb_tick_labels)
    else:
        raise exc.PlottingException(
            "Only 1 entry of cb_tick_values or cb_tick_labels was input. You must either supply"
            "both the values and labels, or neither."
        )

    cb.ax.tick_params(labelsize=cb_ticksize)


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
            array_util.numpy_array_2d_to_fits(
                array_2d=array,
                file_path=output_path + output_filename + ".fits",
                overwrite=True,
            )


def output_subplot_array(output_path, output_filename, output_format):
    """Output a figure which consists of a set of subplot,, either as an image on the screen or to the hard-disk as a \
    .png file.

    Parameters
    -----------
    output_path : str
        The path on the hard-disk where the figure is output.
    output_filename : str
        The filename of the figure that is output.
    output_format : str
        The format the figue is output:
        'show' - display on computer screen.
        'png' - output to hard-disk as a png.
    """
    if output_format is "show":
        plt.show()
    elif output_format is "png":
        plt.savefig(output_path + output_filename + ".png", bbox_inches="tight")
    elif output_format is "fits":
        raise exc.PlottingException("You cannot output a subplots with format .fits")


def get_mask_from_fit(include_mask, fit):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    include_mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if include_mask:
        return fit.mask
    else:
        return None


def get_real_space_mask_from_fit(include_mask, fit):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    include_mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if include_mask:
        return fit.masked_dataset.mask
    else:
        return None


def plot_lines(line_lists):
    """Plot the liness of the mask or the array on the figure.

    Parameters
    -----------t.
    mask : ndarray of data_type.array.mask.Mask
        The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
    plot_lines : bool
        If a mask is supplied, its liness pixels (e.g. the exterior edge) is plotted if this is *True*.
    unit_label : str
        The unit_label of the y / x axis of the plots.
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
    lines_pointsize : int
        The size of the points plotted to show the liness.
    """
    if line_lists is not None:
        for line_list in line_lists:
            if line_list is not None:
                for line in line_list:
                    if len(line) != 0:
                        plt.plot(line[:, 1], line[:, 0], c="w", lw=2.0, zorder=200)


def close_figure(as_subplot):
    """After plotting and outputting a figure, close the matplotlib figure instance (omit if a subplot).

    Parameters
    -----------
    as_subplot : bool
        Whether the figure is part of subplot, in which case the figure is not closed so that the entire figure can \
        be closed later after output.
    """
    if not as_subplot:
        plt.close()


def radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):
    return (maximum_radius - minimum_radius) / radii_points


def quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):
    return list(
        np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1)
    )


def quantity_and_annuli_radii_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):

    radii_bin_size = radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        radii_points=radii_points,
    )

    quantity_radii = list(
        np.linspace(
            start=minimum_radius + radii_bin_size / 2.0,
            stop=maximum_radius - radii_bin_size / 2.0,
            num=radii_points,
        )
    )
    annuli_radii = list(
        np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1)
    )

    return quantity_radii, annuli_radii
