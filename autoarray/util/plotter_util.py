from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np
from autoarray import exc
from autoarray.util import array_util










def output_figure(array, as_subplot, outputs):
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

        if outputs.format is "show":
            plt.show()
        elif outputs.format is "png":
            plt.savefig(outputs.path + outputs.filename + ".png", bbox_inches="tight")
        elif outputs.format is "fits":
            array_util.numpy_array_2d_to_fits(
                array_2d=array,
                file_path=outputs.path + outputs.filename + ".fits",
                overwrite=True,
            )


def output_subplot_array(outputs):
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
    if outputs.format is "show":
        plt.show()
    elif outputs.format is "png":
        plt.savefig(outputs.path + outputs.filename + ".png", bbox_inches="tight")
    elif outputs.format is "fits":
        raise exc.PlottingException("You cannot output a subplots with format .fits")


def get_mask_from_fit(mask, fit):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if mask:
        return fit.mask
    else:
        return None


def get_real_space_mask_from_fit(mask, fit):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if mask:
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
