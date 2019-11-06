from autoarray.plotters import grid_plotters


def visibilities(
    fit,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
    units="arcsec",
    kpc_per_arcsec=None,
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
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
