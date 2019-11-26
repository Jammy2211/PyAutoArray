from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import array_plotters
from autoarray.util import plotter_util


def subplot(
    imaging,
    include_origin=True,
    mask=None,
    include_border=False,
    positions=None,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    position_pointsize=30,
    grid_pointsize=1,
    output_path=None,
    output_filename="imaging",
    output_format="show",
):
    """Plot the imaging data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    image(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        include_border=include_border,
        positions=positions,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    noise_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    psf(
        imaging=imaging,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4)

    signal_to_noise_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 5)

    absolute_signal_to_noise_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 6)

    potential_chi_squared_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individual(
    imaging,
    include_origin=True,
    mask=None,
    positions=None,
    unit_label="scaled",
    unit_conversion_factor=None,
    plot_image=False,
    plot_noise_map=False,
    plot_psf=False,
    plot_signal_to_noise_map=False,
    plot_absolute_signal_to_noise_map=False,
    plot_potential_chi_squared_map=False,
    output_path=None,
    output_format="png",
):
    """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_image:

        image(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            positions=positions,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:

        noise_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_psf:

        psf(
            imaging=imaging,
            include_origin=include_origin,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_absolute_signal_to_noise_map:

        absolute_signal_to_noise_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_potential_chi_squared_map:

        potential_chi_squared_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )


def image(
    imaging,
    include_origin=True,
    grid=None,
    mask=None,
    include_border=False,
    positions=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    title="Imaging Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=30,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="imaging_image",
):
    """Plot the observed data_type of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """
    array_plotters.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        include_border=include_border,
        points=positions,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def noise_map(
    imaging,
    include_origin=True,
    mask=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    title="Imaging Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="imaging_noise_map",
):
    """Plot the noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=imaging.noise_map,
        include_origin=include_origin,
        mask=mask,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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


def psf(
    imaging,
    include_origin=True,
    as_subplot=False,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    title="Imaging PSF",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="imaging_psf",
):
    """Plot the PSF of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=imaging.psf,
        include_origin=include_origin,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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


def signal_to_noise_map(
    imaging,
    include_origin=True,
    mask=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    title="Imaging Signal-To-Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="imaging_signal_to_noise_map",
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=imaging.signal_to_noise_map,
        include_origin=include_origin,
        mask=mask,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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


def absolute_signal_to_noise_map(
    imaging,
    include_origin=True,
    mask=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    title="Imaging Absolute Signal-To-Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="imaging_absolute_signal_to_noise_map",
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=imaging.absolute_signal_to_noise_map,
        include_origin=include_origin,
        mask=mask,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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


def potential_chi_squared_map(
    imaging,
    include_origin=True,
    mask=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
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
    title="Imaging Potential Chi-Squared Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="imaging_potential_chi_squared_map",
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=imaging.potential_chi_squared_map,
        include_origin=include_origin,
        mask=mask,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
