from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters, grid_plotters, line_plotters
from autoarray.plots import inversion_plots
from autoarray.util import plotter_util

@plotters.set_includes
@plotters.set_labels
def subplot(
    fit,
    unit_conversion_factor=None,
    figsize=None,
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    output_path=None,
    output_filename="fit",
    output_format="show",
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    residual_map_vs_uv_distances(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    normalized_residual_map_vs_uv_distances(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    chi_squared_map_vs_uv_distances(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4)

    residual_map_vs_uv_distances(
        fit=fit,
        plot_real=False,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 5)

    normalized_residual_map_vs_uv_distances(
        fit=fit,
        plot_real=False,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 6)

    chi_squared_map_vs_uv_distances(
        fit=fit,
        plot_real=False,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()

@plotters.set_includes
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
    unit_conversion_factor=None,
    unit_label="arcsec",
    output_path=None,
    output_format="show",
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

        visibilities(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:

        noise_map(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_model_visibilities:

        model_visibilities(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_residual_map:

        residual_map_vs_uv_distances(
            fit=fit,
            plot_real=True,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

        residual_map_vs_uv_distances(
            fit=fit,
            plot_real=False,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_normalized_residual_map:

        normalized_residual_map_vs_uv_distances(
            fit=fit,
            plot_real=True,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

        normalized_residual_map_vs_uv_distances(
            fit=fit,
            plot_real=False,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_chi_squared_map:

        chi_squared_map_vs_uv_distances(
            fit=fit,
            plot_real=True,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

        chi_squared_map_vs_uv_distances(
            fit=fit,
            plot_real=False,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
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
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            output_path=output_path,
            output_format=output_format,
        )

@plotters.set_includes
@plotters.set_labels
def visibilities(
    fit,
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the visibilities of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotter.plot_grid(
        grid=fit.visibilities,
    )

@plotters.set_includes
@plotters.set_labels
def noise_map(
    fit,
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotter.plot_grid(
        grid=fit.visibilities,
        colors=fit.noise_map[:, 0],
    )

@plotters.set_includes
@plotters.set_labels
def signal_to_noise_map(
    fit,
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which includes the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotter.plot_grid(
        grid=fit.visibilities,
        colors=fit.signal_to_noise_map[:, 0],
    )

@plotters.set_includes
@plotters.set_labels
def model_visibilities(
    fit,
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the model visibilities of a fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the model visibilities is plotted.
    """
    grid_plotter.plot_grid(
        grid=fit.visibilities,
    )

@plotters.set_includes
@plotters.set_labels
def residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits)

    if plot_real:
        y = fit.residual_map[:, 0]
        line_plotter = line_plotter.plotter_with_new_labels_and_filename(
            label_title=line_plotter.label_title + " Real",
        output_filename=line_plotter.output_filename + "_real")
    else:
        y = fit.residual_map[:, 1]
        line_plotter = line_plotter.plotter_with_new_labels_and_filename(
            label_title=line_plotter.label_title + " Imag",
        output_filename=line_plotter.output_filename + "_imag")

    line_plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )

@plotters.set_includes
@plotters.set_labels
def normalized_residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits)

    if plot_real:
        y = fit.residual_map[:, 0]
        line_plotter = line_plotter.plotter_with_new_labels_and_filename(
            label_title=line_plotter.label_title + " Real",
        output_filename=line_plotter.output_filename + "_real")
    else:
        y = fit.residual_map[:, 1]
        line_plotter = line_plotter.plotter_with_new_labels_and_filename(
            label_title=line_plotter.label_title + " Imag",
        output_filename=line_plotter.output_filename + "_imag")

    line_plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )

@plotters.set_includes
@plotters.set_labels
def chi_squared_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits)

    if plot_real:
        y = fit.residual_map[:, 0]
        line_plotter = line_plotter.plotter_with_new_labels_and_filename(
            label_title=line_plotter.label_title + " Real",
        output_filename=line_plotter.output_filename + "_real")
    else:
        y = fit.residual_map[:, 1]
        line_plotter = line_plotter.plotter_with_new_labels_and_filename(
            label_title=line_plotter.label_title + " Imag",
        output_filename=line_plotter.output_filename + "_imag")

    line_plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )
