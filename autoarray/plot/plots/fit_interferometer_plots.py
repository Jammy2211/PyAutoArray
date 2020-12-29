from autoarray.plot.mat_wrap import mat_decorators
import numpy as np


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_plotter_2d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_fit_interferometer(fit, include_2d=None, plotter_2d=None):

    number_subplots = 6

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    residual_map_vs_uv_distances(fit=fit, include_2d=include_2d, plotter_2d=plotter_2d)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    normalized_residual_map_vs_uv_distances(
        fit=fit, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    chi_squared_map_vs_uv_distances(
        fit=fit, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    normalized_residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    chi_squared_map_vs_uv_distances(
        fit=fit, plot_real=False, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individuals(
    fit,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    include_2d=None,
    plotter_2d=None,
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

        visibilities(fit=fit, include_2d=include_2d, plotter_2d=plotter_2d)

    if plot_noise_map:

        noise_map(fit=fit, include_2d=include_2d, plotter_2d=plotter_2d)

    if plot_signal_to_noise_map:

        signal_to_noise_map(fit=fit, include_2d=include_2d, plotter_2d=plotter_2d)

    if plot_model_visibilities:

        model_visibilities(fit=fit, include_2d=include_2d, plotter_2d=plotter_2d)

    if plot_residual_map:

        residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include_2d=include_2d, plotter_2d=plotter_2d
        )

        residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_normalized_residual_map:

        normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include_2d=include_2d, plotter_2d=plotter_2d
        )

        normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_chi_squared_map:

        chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=True, include_2d=include_2d, plotter_2d=plotter_2d
        )

        chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=False, include_2d=include_2d, plotter_2d=plotter_2d
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def visibilities(fit, include_2d=None, plotter_2d=None):
    """Plot the visibilities of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter_2d._plot_grid(grid=fit.visibilities.in_grid)


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def noise_map(fit, include_2d=None, plotter_2d=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter_2d._plot_grid(
        grid=fit.visibilities.in_grid, color_array=np.real(fit.noise_map)
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def signal_to_noise_map(fit, include_2d=None, plotter_2d=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter_2d._plot_grid(
        grid=fit.visibilities.in_grid, color_array=fit.signal_to_noise_map.real
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def model_visibilities(fit, include_2d=None, plotter_2d=None):
    """Plot the model visibilities of a fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the model visibilities is plotted.
    """
    plotter_2d._plot_grid(grid=fit.visibilities.in_grid)


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include_2d=None,
    plotter_2d=None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = np.real(fit.residual_map)
        plotter_2d = plotter_2d.plotter_with_new_labels(
            title_label=f"{plotter_2d.title.kwargs['label']} Real"
        )
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=plotter_2d.output.filename + "_real"
        )
    else:
        y = np.imag(fit.residual_map)
        plotter_2d = plotter_2d.plotter_with_new_labels(
            title_label=f"{plotter_2d.title.kwargs['label']} Imag"
        )
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=plotter_2d.output.filename + "_imag"
        )

    plotter_2d._plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def normalized_residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include_2d=None,
    plotter_2d=None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = np.real(fit.residual_map)
        plotter_2d = plotter_2d.plotter_with_new_labels(
            title_label=f"{plotter_2d.title.kwargs['label']} Real"
        )
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=plotter_2d.output.filename + "_real"
        )
    else:
        y = np.imag(fit.residual_map)
        plotter_2d = plotter_2d.plotter_with_new_labels(
            title_label=f"{plotter_2d.title.kwargs['label']} Imag"
        )
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=plotter_2d.output.filename + "_imag"
        )

    plotter_2d._plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def chi_squared_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include_2d=None,
    plotter_2d=None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = np.real(fit.residual_map)
        plotter_2d = plotter_2d.plotter_with_new_labels(
            title_label=f"{plotter_2d.title.kwargs['label']} Real"
        )
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=plotter_2d.output.filename + "_real"
        )
    else:
        y = np.imag(fit.residual_map)
        plotter_2d = plotter_2d.plotter_with_new_labels(
            title_label=f"{plotter_2d.title.kwargs['label']} Imag"
        )
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=plotter_2d.output.filename + "_imag"
        )

    plotter_2d._plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )
