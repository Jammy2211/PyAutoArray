from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
from autoarray.plot.plots import structure_plots
from autoarray.fit import fit as f
import typing


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_plotter_2d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_fit_imaging(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):

    number_subplots = 6

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    signal_to_noise_map(
        fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    model_image(
        fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    residual_map(
        fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    normalized_residual_map(
        fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    chi_squared_map(
        fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individuals(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
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

    if plot_image:

        image(
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

    if plot_model_image:

        model_image(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_residual_map:

        residual_map(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_normalized_residual_map:

        normalized_residual_map(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_chi_squared_map:

        chi_squared_map(
            fit=fit, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def image(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_array(
        array=fit.data,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def noise_map(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_array(
        array=fit.noise_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def signal_to_noise_map(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_array(
        array=fit.signal_to_noise_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def model_image(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the model image of a fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """

    structure_plots.plot_array(
        array=fit.model_data,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def residual_map(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    structure_plots.plot_array(
        array=fit.residual_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def normalized_residual_map(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, normalized_residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    structure_plots.plot_array(
        array=fit.normalized_residual_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def chi_squared_map(
    fit: f.FitImaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the chi-squared-map of a lens fit.

    Set *autolens.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    structure_plots.plot_array(
        array=fit.chi_squared_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )
