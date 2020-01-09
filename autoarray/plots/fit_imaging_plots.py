import matplotlib
from autoarray import conf

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plots import inversion_plots
from autoarray.plotters import plotters, array_plotters


def subplot(
    fit,
    grid=None,
    points=None,
    lines=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="fit_imaging"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    image(
        fit=fit, grid=grid, include=include, points=points, array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 2)

    signal_to_noise_map(fit=fit, include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 3)

    model_image(fit=fit, include=include, lines=lines, array_plotter=array_plotter)

    plt.subplot(rows, columns, 4)

    residual_map(fit=fit, include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 5)

    normalized_residual_map(fit=fit, include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 6)

    chi_squared_map(fit=fit, include=include, array_plotter=array_plotter)

    array_plotter.output.to_figure(structure=None, is_sub_plotter=False)

    plt.close()


def individuals(
    fit,
    lines=None,
    grid=None,
    points=None,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_inversion_reconstruction=False,
    plot_inversion_errors=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
    plot_inversion_interpolated_reconstruction=False,
    plot_inversion_interpolated_errors=False,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    if plot_image:

        image(
            fit=fit,
            include=include,
            points=points,
            grid=grid,
            array_plotter=array_plotter,
        )

    if plot_noise_map:

        noise_map(fit=fit, include=include, array_plotter=array_plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(fit=fit, include=include, array_plotter=array_plotter)

    if plot_model_image:

        model_image(fit=fit, include=include, lines=lines, array_plotter=array_plotter)

    if plot_residual_map:

        residual_map(fit=fit, include=include, array_plotter=array_plotter)

    if plot_normalized_residual_map:

        normalized_residual_map(fit=fit, include=include, array_plotter=array_plotter)

    if plot_chi_squared_map:

        chi_squared_map(fit=fit, include=include, array_plotter=array_plotter)

    if fit.total_inversions == 1:

        inversion_plots.individuals(
            inversion=fit.inversion,
            lines=lines,
            plot_inversion_reconstruction=plot_inversion_reconstruction,
            plot_inversion_errors=plot_inversion_errors,
            plot_inversion_residual_map=plot_inversion_residual_map,
            plot_inversion_normalized_residual_map=plot_inversion_normalized_residual_map,
            plot_inversion_chi_squared_map=plot_inversion_chi_squared_map,
            plot_inversion_regularization_weight_map=plot_inversion_regularization_weight_map,
            plot_inversion_interpolated_reconstruction=plot_inversion_interpolated_reconstruction,
            plot_inversion_interpolated_errors=plot_inversion_interpolated_errors,
            array_plotter=array_plotter,
        )


@plotters.set_labels
def image(
    fit,
    points=None,
    grid=None,
    lines=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=fit.data,
        grid=grid,
        mask=include.mask_from_fit(fit=fit),
        lines=lines,
        points=points,
    )


@plotters.set_labels
def noise_map(
    fit,
    points=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=fit.noise_map, mask=include.mask_from_fit(fit=fit), points=points
    )


@plotters.set_labels
def signal_to_noise_map(
    fit,
    points=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which include the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=fit.signal_to_noise_map,
        mask=include.mask_from_fit(fit=fit),
        points=points,
    )


@plotters.set_labels
def model_image(
    fit,
    lines=None,
    points=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the model image of a fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """
    array_plotter.plot_array(
        array=fit.model_data,
        mask=include.mask_from_fit(fit=fit),
        lines=lines,
        points=points,
    )


@plotters.set_labels
def residual_map(
    fit,
    points=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    array_plotter.plot_array(
        array=fit.residual_map, mask=include.mask_from_fit(fit=fit), points=points
    )


@plotters.set_labels
def normalized_residual_map(
    fit,
    points=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, normalized_residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    array_plotter.plot_array(
        array=fit.normalized_residual_map,
        mask=include.mask_from_fit(fit=fit),
        points=points,
    )


@plotters.set_labels
def chi_squared_map(
    fit,
    points=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the chi-squared map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    array_plotter.plot_array(
        array=fit.chi_squared_map, mask=include.mask_from_fit(fit=fit), points=points
    )
