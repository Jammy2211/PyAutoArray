import matplotlib
from autoarray import conf

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plots import inversion_plots
from autoarray.plotters import plotters, array_plotters
from autoarray.util import plotter_util

@plotters.set_includes
def subplot(
    fit,
    mask=True,
    grid=None,
    points=None,
    lines=None,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="arcsec",
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_filename="fit",
    output_format="show",
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    image(
        fit=fit,
        grid=grid,
        include_mask=include_mask,
        points=points,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        grid_pointsize=grid_pointsize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    signal_to_noise_map(
        fit=fit,
        include_mask=include_mask,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    model_image(
        fit=fit,
        include_mask=include_mask,
        lines=lines,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
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
        include_mask=include_mask,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
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
        include_mask=include_mask,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
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
        include_mask=include_mask,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
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
    include_mask=None,
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
            include_mask=include_mask,
            points=points,
            grid=grid,
            array_plotter=array_plotter,
        )

    if plot_noise_map:

        noise_map(
            fit=fit,
            include_mask=include_mask,
            array_plotter=array_plotter,
        )

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            fit=fit,
            include_mask=include_mask,
            array_plotter=array_plotter,
        )

    if plot_model_image:

        model_image(
            fit=fit,
            include_mask=include_mask,
            lines=lines,
            array_plotter=array_plotter,
        )

    if plot_residual_map:

        residual_map(
            fit=fit,
            include_mask=include_mask,
            array_plotter=array_plotter,
        )

    if plot_normalized_residual_map:

        normalized_residual_map(
            fit=fit,
            include_mask=include_mask,
            array_plotter=array_plotter,
        )

    if plot_chi_squared_map:

        chi_squared_map(
            fit=fit,
            include_mask=include_mask,
            array_plotter=array_plotter,
        )

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

@plotters.set_includes
@plotters.set_labels
def image(
    fit,
    include_mask=None,
    points=None,
    grid=None,
    lines=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=fit.data,
        grid=grid,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        lines=lines,
        points=points,
    )

@plotters.set_includes
@plotters.set_labels
def noise_map(
    fit,
    include_mask=None,
    points=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=fit.noise_map,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        points=points,
    )

@plotters.set_includes
@plotters.set_labels
def signal_to_noise_map(
    fit,
    include_mask=None,
    points=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which includes the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=fit.signal_to_noise_map,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        points=points,
    )

@plotters.set_includes
@plotters.set_labels
def model_image(
    fit,
    include_mask=None,
    lines=None,
    points=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the model image of a fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """
    array_plotter.plot_array(
        array=fit.model_data,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        lines=lines,
        points=points,
    )

@plotters.set_includes
@plotters.set_labels
def residual_map(
    fit,
    include_mask=None,
    points=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    array_plotter.plot_array(
        array=fit.residual_map,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        points=points,
    )

@plotters.set_includes
@plotters.set_labels
def normalized_residual_map(
    fit,
    include_mask=None,
    points=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, normalized_residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    array_plotter.plot_array(
        array=fit.normalized_residual_map,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        points=points,
    )

@plotters.set_includes
@plotters.set_labels
def chi_squared_map(
    fit,
    include_mask=None,
    points=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the chi-squared map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    array_plotter.plot_array(
        array=fit.chi_squared_map,
        mask=plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask),
        points=points,
    )
