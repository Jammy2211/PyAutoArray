from autoarray.plot import plotters


@plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_imaging(
    fit, grid=None, positions=None, lines=None, include=None, sub_plotter=None
):

    number_subplots = 6

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(fit=fit, grid=grid, positions=positions, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    signal_to_noise_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    model_image(fit=fit, lines=lines, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    normalized_residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    chi_squared_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals(
    fit,
    lines=None,
    grid=None,
    positions=None,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    include=None,
    plotter=None,
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

        image(fit=fit, include=include, positions=positions, grid=grid, plotter=plotter)

    if plot_noise_map:

        noise_map(fit=fit, include=include, plotter=plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(fit=fit, include=include, plotter=plotter)

    if plot_model_image:

        model_image(fit=fit, include=include, lines=lines, plotter=plotter)

    if plot_residual_map:

        residual_map(fit=fit, include=include, plotter=plotter)

    if plot_normalized_residual_map:

        normalized_residual_map(fit=fit, include=include, plotter=plotter)

    if plot_chi_squared_map:

        chi_squared_map(fit=fit, include=include, plotter=plotter)


@plotters.set_include_and_plotter
@plotters.set_labels
def image(fit, positions=None, grid=None, lines=None, include=None, plotter=None):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_array(
        array=fit.data,
        grid=grid,
        mask=include.mask_from_fit(fit=fit),
        lines=lines,
        positions=positions,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def noise_map(fit, positions=None, include=None, plotter=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_array(
        array=fit.noise_map, mask=include.mask_from_fit(fit=fit), positions=positions
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map(fit, positions=None, include=None, plotter=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which include the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_array(
        array=fit.signal_to_noise_map,
        mask=include.mask_from_fit(fit=fit),
        positions=positions,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def model_image(fit, lines=None, positions=None, include=None, plotter=None):
    """Plot the model image of a fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """

    plotter.plot_array(
        array=fit.model_data,
        mask=include.mask_from_fit(fit=fit),
        lines=lines,
        positions=positions,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def residual_map(fit, positions=None, include=None, plotter=None):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    plotter.plot_array(
        array=fit.residual_map, mask=include.mask_from_fit(fit=fit), positions=positions
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def normalized_residual_map(fit, positions=None, include=None, plotter=None):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, normalized_residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    plotter.plot_array(
        array=fit.normalized_residual_map,
        mask=include.mask_from_fit(fit=fit),
        positions=positions,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def chi_squared_map(fit, positions=None, include=None, plotter=None):
    """Plot the chi-squared-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    plotter.plot_array(
        array=fit.chi_squared_map,
        mask=include.mask_from_fit(fit=fit),
        positions=positions,
    )
