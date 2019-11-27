from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import grid_plotters, inversion_plotters
from autoarray.util import plotter_util


def subplot(
    fit,
    unit_conversion_factor=None,
    unit_label="arcsec",
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
    grid_pointsize=1,
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

    visibilities(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        pointsize=grid_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    signal_to_noise_map(
        fit=fit,
        as_subplot=True,
        unit_label=unit_label,
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

    model_visibilities(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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

    residual_map(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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

    normalized_residual_map(
        fit=fit,
        as_subplot=True,
        unit_label=unit_label,
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

    chi_squared_map(
        fit=fit,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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


def individuals(
    fit,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
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

        residual_map(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_normalized_residual_map:

        normalized_residual_map(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_chi_squared_map:

        chi_squared_map(
            fit=fit,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_residual_map:

        if fit.total_inversions == 1:

            inversion_plotters.residual_map(
                inversion=fit.inversion,
                include_grid=True,
                unit_label=unit_label,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_inversion_normalized_residual_map:

        if fit.total_inversions == 1:

            inversion_plotters.normalized_residual_map(
                inversion=fit.inversion,
                include_grid=True,
                unit_label=unit_label,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_inversion_chi_squared_map:

        if fit.total_inversions == 1:

            inversion_plotters.chi_squared_map(
                inversion=fit.inversion,
                include_grid=True,
                unit_conversion_factor=unit_conversion_factor,
                unit_label=unit_label,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_inversion_regularization_weight_map:

        if fit.total_inversions == 1:

            inversion_plotters.regularization_weights(
                inversion=fit.inversion,
                include_grid=True,
                unit_conversion_factor=unit_conversion_factor,
                unit_label=unit_label,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )


def visibilities(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="fit_visibilities",
):
    """Plot the visibilities of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        pointsize=pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def noise_map(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_noise_map",
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        colors=fit.noise_map[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def signal_to_noise_map(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Signal-to-Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_signal_to_noise_map",
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which includes the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    include_origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        colors=fit.signal_to_noise_map[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def model_visibilities(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Model Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_model_visibilities",
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
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        colors=fit.model_visibilities[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def residual_map(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Residuals",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_residual_map",
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
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        colors=fit.residual_map[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def normalized_residual_map(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Normalized Residuals",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_normalized_residual_map",
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model visibilities, normalized_residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        colors=fit.normalized_residual_map[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def chi_squared_map(
    fit,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Chi-Squareds",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_chi_squared_map",
):
    """Plot the chi-squared map of a lens fit.

    Set *autolens.datas.grid.plotters.grid_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    grid_plotters.plot_grid(
        grid=fit.visibilities,
        colors=fit.chi_squared_map[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )