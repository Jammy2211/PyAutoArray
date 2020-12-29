from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
from autoarray.plot.plots import structure_plots
from autoarray.fit import fit as f
import typing
import numpy as np


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_plotter_1d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_fit_interferometer(
    fit: f.FitInterferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
):

    number_subplots = 6

    plotter_1d.open_subplot_figure(number_subplots=number_subplots)

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    residual_map_vs_uv_distances(
        fit=fit, visuals_1d=visuals_1d, include_1d=include_1d, plotter_1d=plotter_1d
    )

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    normalized_residual_map_vs_uv_distances(
        fit=fit, visuals_1d=visuals_1d, include_1d=include_1d, plotter_1d=plotter_1d
    )

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    chi_squared_map_vs_uv_distances(
        fit=fit, visuals_1d=visuals_1d, include_1d=include_1d, plotter_1d=plotter_1d
    )

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    residual_map_vs_uv_distances(
        fit=fit,
        plot_real=False,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
    )

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    normalized_residual_map_vs_uv_distances(
        fit=fit,
        plot_real=False,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
    )

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    chi_squared_map_vs_uv_distances(
        fit=fit,
        plot_real=False,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
    )

    plotter_1d.output.subplot_to_figure()

    plotter_1d.figure.close()


def individuals(
    fit: f.FitInterferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
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
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_noise_map:

        noise_map(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_model_visibilities:

        model_visibilities(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_residual_map:

        residual_map_vs_uv_distances(
            fit=fit,
            plot_real=True,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

        residual_map_vs_uv_distances(
            fit=fit,
            plot_real=False,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

    if plot_normalized_residual_map:

        normalized_residual_map_vs_uv_distances(
            fit=fit,
            plot_real=True,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

        normalized_residual_map_vs_uv_distances(
            fit=fit,
            plot_real=False,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

    if plot_chi_squared_map:

        chi_squared_map_vs_uv_distances(
            fit=fit,
            plot_real=True,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

        chi_squared_map_vs_uv_distances(
            fit=fit,
            plot_real=False,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def visibilities(
    fit: f.FitInterferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the visibilities of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_grid(
        grid=fit.visibilities.in_grid,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def noise_map(
    fit: f.FitInterferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_grid(
        grid=fit.visibilities.in_grid,
        color_array=np.real(fit.noise_map),
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def signal_to_noise_map(
    fit: f.FitInterferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_grid(
        grid=fit.visibilities.in_grid,
        color_array=fit.signal_to_noise_map.real,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def model_visibilities(
    fit: f.FitInterferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the model visibilities of a fit.

    Set *autolens.datas.grid.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the model visibilities is plotted.
    """
    structure_plots.plot_grid(
        grid=fit.visibilities.in_grid,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def residual_map_vs_uv_distances(
    fit: f.FitInterferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotter_1d.plotter_1d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = np.real(fit.residual_map)
        plotter_1d = plotter_1d.plotter_with_new_labels(
            title_label=f"{plotter_1d.title.kwargs['label']} Real"
        )
        plotter_1d = plotter_1d.plotter_with_new_output(
            filename=plotter_1d.output.filename + "_real"
        )
    else:
        y = np.imag(fit.residual_map)
        plotter_1d = plotter_1d.plotter_with_new_labels(
            title_label=f"{plotter_1d.title.kwargs['label']} Imag"
        )
        plotter_1d = plotter_1d.plotter_with_new_output(
            filename=plotter_1d.output.filename + "_imag"
        )

    structure_plots.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
        plot_axis_type="scatter",
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def normalized_residual_map_vs_uv_distances(
    fit: f.FitInterferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotter_1d.plotter_1d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = np.real(fit.residual_map)
        plotter_1d = plotter_1d.plotter_with_new_labels(
            title_label=f"{plotter_1d.title.kwargs['label']} Real"
        )
        plotter_1d = plotter_1d.plotter_with_new_output(
            filename=plotter_1d.output.filename + "_real"
        )
    else:
        y = np.imag(fit.residual_map)
        plotter_1d = plotter_1d.plotter_with_new_labels(
            title_label=f"{plotter_1d.title.kwargs['label']} Imag"
        )
        plotter_1d = plotter_1d.plotter_with_new_output(
            filename=plotter_1d.output.filename + "_imag"
        )

    structure_plots.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
        plot_axis_type="scatter",
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def chi_squared_map_vs_uv_distances(
    fit: f.FitInterferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotter_1d.plotter_1d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = np.real(fit.residual_map)
        plotter_1d = plotter_1d.plotter_with_new_labels(
            title_label=f"{plotter_1d.title.kwargs['label']} Real"
        )
        plotter_1d = plotter_1d.plotter_with_new_output(
            filename=plotter_1d.output.filename + "_real"
        )
    else:
        y = np.imag(fit.residual_map)
        plotter_1d = plotter_1d.plotter_with_new_labels(
            title_label=f"{plotter_1d.title.kwargs['label']} Imag"
        )
        plotter_1d = plotter_1d.plotter_with_new_output(
            filename=plotter_1d.output.filename + "_imag"
        )

    structure_plots.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
        plot_axis_type="scatter",
    )
