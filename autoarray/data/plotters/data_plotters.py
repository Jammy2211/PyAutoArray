from autoarray import conf
import matplotlib

backend = conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)

import autoarray as aa

def plot_image(
    image,
    plot_origin=True,
    grid=None,
    mask_overlay=None,
    should_plot_border=False,
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
    title="Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=30,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="image",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    """

    aa.plot_array(
        array=image,
        should_plot_origin=plot_origin,
        grid=grid,
    mask_overlay=mask_overlay,
        should_plot_border=should_plot_border,
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
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_noise_map(
    noise_map,
    plot_origin=True,
    mask_overlay=None,
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
    title="Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="noise_map",
):
    """Plot the noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : ScaledSquarePixelArray
        The noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    aa.plot_array(
        array=noise_map,
        should_plot_origin=plot_origin,
    mask_overlay=mask_overlay,
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
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_psf(
    psf,
    plot_origin=True,
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
    title="PSF",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="psf",
):
    """Plot the PSF of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : ScaledSquarePixelArray
        The psf of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    aa.plot_array(
        array=psf,
        should_plot_origin=plot_origin,
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
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_signal_to_noise_map(
    signal_to_noise_map,
    plot_origin=True,
    mask_overlay=None,
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
    title="Signal-To-Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="signal_to_noise_map",
):
    """Plot the signal-to-noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : ScaledSquarePixelArray
        The signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    aa.plot_array(
        array=signal_to_noise_map,
        should_plot_origin=plot_origin,
    mask_overlay=mask_overlay,
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
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_absolute_signal_to_noise_map(
    absolute_signal_to_noise_map,
    plot_origin=True,
    mask_overlay=None,
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
    title="Absolute Signal-To-Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="absolute_signal_to_noise_map",
):
    """Plot the absolute signal-to-noise map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    absolute_signal_to_noise_map : ScaledSquarePixelArray
        The absolute signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    aa.plot_array(
        array=absolute_signal_to_noise_map,
        should_plot_origin=plot_origin,
    mask_overlay=mask_overlay,
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
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_potential_chi_squared_map(
    potential_chi_squared_map,
    plot_origin=True,
    mask_overlay=None,
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
    title="Potential Chi-Squared Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="potential_chi_squared_map",
):
    """Plot the signal-to-noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    potential_chi_squared_map : ScaledSquarePixelArray
        The signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    aa.plot_array(
        array=potential_chi_squared_map,
        should_plot_origin=plot_origin,
    mask_overlay=mask_overlay,
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
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_visibilities(
    visibilities,
    noise_map=None,
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
    title="Visibilities",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="visibilities",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    """

    aa.plot_grid(
        grid=visibilities,
        colors=noise_map,
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


def plot_u_wavelengths(
    uv_wavelengths,
    as_subplot=False,
    label="Wavelengths",
    units="",
    kpc_per_arcsec=None,
    figsize=(14, 7),
    plot_axis_type="linear",
    ylabel="U-Wavelength",
    title="U-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="u_wavelengths",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    """

    aa.plot_line(
        y=uv_wavelengths[:, 0],
        x=None,
        as_subplot=as_subplot,
        label=label,
        plot_axis_type=plot_axis_type,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        title=title,
        ylabel=ylabel,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_v_wavelengths(
    uv_wavelengths,
    as_subplot=False,
    label="Wavelengths",
    units="",
    kpc_per_arcsec=None,
    figsize=(14, 7),
    plot_axis_type="linear",
    ylabel="V-Wavelength",
    title="V-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="v_wavelengths",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    """

    aa.plot_line(
        y=uv_wavelengths[:, 1],
        x=None,
        as_subplot=as_subplot,
        label=label,
        plot_axis_type=plot_axis_type,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        title=title,
        ylabel=ylabel,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_primary_beam(
    primary_beam,
    plot_origin=True,
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
    title="Primary Beam",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="primary_beam",
):
    """Plot the PSF of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : ScaledSquarePixelArray
        The primary_beam of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    aa.plot_array(
        array=primary_beam,
        should_plot_origin=plot_origin,
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
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
