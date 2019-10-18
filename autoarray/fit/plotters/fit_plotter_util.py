from autoarray.plotters import array_plotters


def plot_image(
    fit,
    mask_overlay=None,
    positions=None,
    grid=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    grid_pointsize=1,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_image",
):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotters.plot_array(
        array=fit.data,
        grid=grid,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        grid_pointsize=grid_pointsize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_noise_map(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_noise_map",
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotters.plot_array(
        array=fit.noise_map,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_signal_to_noise_map(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Signal-to-Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_signal_to_noise_map",
):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which includes the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    plot_origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotters.plot_array(
        array=fit.signal_to_noise_map,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_model_image(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Model Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_model_image",
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
    array_plotters.plot_array(
        array=fit.model_data,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_residual_map(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Residuals",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_residual_map",
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
    array_plotters.plot_array(
        array=fit.residual_map,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_normalized_residual_map(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Normalized Residuals",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_normalized_residual_map",
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
    array_plotters.plot_array(
        array=fit.normalized_residual_map,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_chi_squared_map(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
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
    title="Fit Chi-Squareds",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_chi_squared_map",
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
    array_plotters.plot_array(
        array=fit.chi_squared_map,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )