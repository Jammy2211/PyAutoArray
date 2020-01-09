from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters, array_plotters, grid_plotters, line_plotters
from autoarray.plots import inversion_plots



@plotters.set_labels
def subplot(
    fit,
include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    grid_plotter = grid_plotter.plotter_as_sub_plotter()
    line_plotter = line_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="fit_interferometer"
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

    residual_map_vs_uv_distances(fit=fit, include=include, line_plotter=line_plotter)

    plt.subplot(rows, columns, 2)

    normalized_residual_map_vs_uv_distances(fit=fit, include=include, line_plotter=line_plotter)

    plt.subplot(rows, columns, 3)

    chi_squared_map_vs_uv_distances(fit=fit, include=include, line_plotter=line_plotter)

    plt.subplot(rows, columns, 4)

    residual_map_vs_uv_distances(fit=fit, plot_real=False, include=include, line_plotter=line_plotter)

    plt.subplot(rows, columns, 5)

    normalized_residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, line_plotter=line_plotter
    )

    plt.subplot(rows, columns, 6)

    chi_squared_map_vs_uv_distances(fit=fit, plot_real=False, include=include, line_plotter=line_plotter)

    array_plotter.output_subplot_array()

    plt.close()



def individuals(
    fit,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
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
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
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

    if plot_visibilities:

        visibilities(fit=fit, include=include, grid_plotter=grid_plotter)

    if plot_noise_map:

        noise_map(fit=fit, include=include, grid_plotter=grid_plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(fit=fit, include=include, grid_plotter=grid_plotter)

    if plot_model_visibilities:

        model_visibilities(fit=fit, include=include, grid_plotter=grid_plotter)

    if plot_residual_map:

        residual_map_vs_uv_distances(fit=fit, plot_real=True, include=include, line_plotter=line_plotter)

        residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, line_plotter=line_plotter
        )

    if plot_normalized_residual_map:

        normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, line_plotter=line_plotter
        )

        normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, line_plotter=line_plotter
        )

    if plot_chi_squared_map:

        chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, line_plotter=line_plotter
        )

        chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, line_plotter=line_plotter
        )

    if fit.total_inversions == 1:

        inversion_plots.individuals(
            inversion=fit.inversion,
            plot_inversion_reconstruction=plot_inversion_reconstruction,
            plot_inversion_errors=plot_inversion_errors,
            plot_inversion_residual_map=plot_inversion_residual_map,
            plot_inversion_normalized_residual_map=plot_inversion_normalized_residual_map,
            plot_inversion_chi_squared_map=plot_inversion_chi_squared_map,
            plot_inversion_regularization_weight_map=plot_inversion_regularization_weight_map,
            plot_inversion_interpolated_reconstruction=plot_inversion_interpolated_reconstruction,
            plot_inversion_interpolated_errors=plot_inversion_interpolated_errors,
            include=include,
            array_plotter=array_plotter,
            grid_plotter=grid_plotter,
            line_plotter=line_plotter,
        )



@plotters.set_labels
def visibilities(fit, include=plotters.Include(), grid_plotter=grid_plotters.GridPlotter()):
    """Plot the visibilities of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotter.plot_grid(grid=fit.visibilities)



@plotters.set_labels
def noise_map(fit, include=plotters.Include(), grid_plotter=grid_plotters.GridPlotter()):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotter.plot_grid(grid=fit.visibilities, colors=fit.noise_map[:, 0])



@plotters.set_labels
def signal_to_noise_map(fit, include=plotters.Include(), grid_plotter=grid_plotters.GridPlotter()):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which include the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotter.plot_grid(grid=fit.visibilities, colors=fit.signal_to_noise_map[:, 0])



@plotters.set_labels
def model_visibilities(fit, include=plotters.Include(), grid_plotter=grid_plotters.GridPlotter()):
    """Plot the model visibilities of a fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the model visibilities is plotted.
    """
    grid_plotter.plot_grid(grid=fit.visibilities)



@plotters.set_labels
def residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=plotters.Include(),
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = fit.residual_map[:, 0]
        line_plotter = line_plotter.plotter_with_new_labels(
            labels=plotters.Labels(title=line_plotter.labels.title + " Real"),
        )
        line_plotter = line_plotter.plotter_with_new_output_filename(
            output_filename=line_plotter.output.filename + "_real",
        )
    else:
        y = fit.residual_map[:, 1]
        line_plotter = line_plotter.plotter_with_new_labels(
            labels=plotters.Labels(title=line_plotter.labels.title + " Imag"),
        )
        line_plotter = line_plotter.plotter_with_new_output_filename(
            output_filename=line_plotter.output.filename + "_imag",
        )

    line_plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )



@plotters.set_labels
def normalized_residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=plotters.Include(),
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = fit.residual_map[:, 0]
        line_plotter = line_plotter.plotter_with_new_labels(
            labels=plotters.Labels(title=line_plotter.labels.title + " Real"),
        )
        line_plotter = line_plotter.plotter_with_new_output_filename(
            output_filename=line_plotter.output.filename + "_real",
        )
    else:
        y = fit.residual_map[:, 1]
        line_plotter = line_plotter.plotter_with_new_labels(
            labels=plotters.Labels(title=line_plotter.labels.title + " Imag"),
        )
        line_plotter = line_plotter.plotter_with_new_output_filename(
            output_filename=line_plotter.output.filename + "_imag",
        )

    line_plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )



@plotters.set_labels
def chi_squared_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=plotters.Include(),
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = fit.residual_map[:, 0]
        line_plotter = line_plotter.plotter_with_new_labels(
            labels=plotters.Labels(title=line_plotter.labels.title + " Real"),
        )
        line_plotter = line_plotter.plotter_with_new_output_filename(
            output_filename=line_plotter.output.filename + "_real",
        )
    else:
        y = fit.residual_map[:, 1]
        line_plotter = line_plotter.plotter_with_new_labels(
            labels=plotters.Labels(title=line_plotter.labels.title + " Imag"),
        )
        line_plotter = line_plotter.plotter_with_new_output_filename(
            output_filename=line_plotter.output.filename + "_imag",
        )

    line_plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )
