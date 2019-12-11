from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import array_plotters, grid_plotters, line_yx_plotters
from autoarray.util import plotter_util
from autoarray.structures import grids
import numpy as np


def subplot(
    interferometer,
    unit_conversion_factor=None,
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
    output_path=None,
    output_filename="interferometer",
    output_format="show",
):
    """Plot the interferometer data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
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
        number_subplots=4
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    uv_wavelengths(
        interferometer=interferometer,
        as_subplot=True,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 2)

    visibilities(
        interferometer=interferometer,
        as_subplot=True,
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
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 3)

    amplitudes_vs_uv_distances(
        interferometer=interferometer,
        as_subplot=True,
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
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 4)

    phases_vs_uv_distances(
        interferometer=interferometer,
        as_subplot=True,
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
        output_format=output_format,
        output_filename=output_filename,
    )

    # plt.subplot(rows, columns, 3)
    #
    # primary_beam(
    #     interferometer=interferometer,
    #     include_origin=include_origin,
    #     as_subplot=True,
    #     unit_label=unit_label,
    #     kpc_per_arcsec=kpc_per_arcsec,
    #     figsize=figsize,
    #     aspect=aspect,
    #     cmap=cmap,
    #     norm=norm,
    #     norm_min=norm_min,
    #     norm_max=norm_max,
    #     linthresh=linthresh,
    #     linscale=linscale,
    #     cb_ticksize=cb_ticksize,
    #     cb_fraction=cb_fraction,
    #     cb_pad=cb_pad,
    #     cb_tick_values=cb_tick_values,
    #     cb_tick_labels=cb_tick_labels,
    #     titlesize=titlesize,
    #     xlabelsize=xlabelsize,
    #     ylabelsize=ylabelsize,
    #     xyticksize=xyticksize,
    #     output_path=output_path,
    #     output_format=output_format,
    # )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individual(
    interferometer,
    unit_label="scaled",
    unit_conversion_factor=None,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_u_wavelengths=False,
    plot_v_wavelengths=False,
    plot_uv_wavelengths=False,
    plot_amplitudes_vs_uv_distances=False,
    plot_phases_vs_uv_distances=False,
    plot_primary_beam=False,
    output_path=None,
    output_format="png",
):
    """Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_visibilities:

        visibilities(
            interferometer=interferometer,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:

        noise_map(
            interferometer=interferometer,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_u_wavelengths:

        uv_wavelengths(
            interferometer=interferometer,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_v_wavelengths:

        v_wavelengths(
            interferometer=interferometer,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_uv_wavelengths:

        uv_wavelengths(
            interferometer=interferometer,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_amplitudes_vs_uv_distances:

        amplitudes_vs_uv_distances(
            interferometer=interferometer,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_phases_vs_uv_distances:

        phases_vs_uv_distances(
            interferometer=interferometer,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_primary_beam:

        primary_beam(
            interferometer=interferometer,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )


def visibilities(
    interferometer,
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
    title="Visibilities",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_visibilities",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    grid_plotters.plot_grid(
        grid=interferometer.visibilities,
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_y="imag",
        unit_label_x="real",
        figsize=figsize,
        pointsize=20,
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


def noise_map(
    interferometer,
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
    title="Noise Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_noise_map",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    grid_plotters.plot_grid(
        grid=interferometer.visibilities,
        colors=interferometer.noise_map[:, 0],
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_y=unit_label,
        unit_label_x=unit_label,
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


def u_wavelengths(
    interferometer,
    as_subplot=False,
    label="Wavelengths",
    unit_conversion_factor=None,
    unit_label="",
    figsize=(14, 7),
    plot_axis_type="linear",
    ylabel="U-Wavelength",
    title="U-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_u_wavelengths",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    line_yx_plotters.plot_line(
        y=interferometer.uv_wavelengths[:, 0],
        x=None,
        as_subplot=as_subplot,
        label=label,
        plot_axis_type=plot_axis_type,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_x=unit_label,
        figsize=figsize,
        title=title,
        unit_label_y=ylabel,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def v_wavelengths(
    interferometer,
    as_subplot=False,
    label="Wavelengths",
    unit_conversion_factor=None,
    unit_label="",
    figsize=(14, 7),
    plot_axis_type="linear",
    ylabel="V-Wavelength",
    title="V-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_v_wavelengths",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    line_yx_plotters.plot_line(
        y=interferometer.uv_wavelengths[:, 1],
        x=None,
        as_subplot=as_subplot,
        label=label,
        plot_axis_type=plot_axis_type,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_x=unit_label,
        figsize=figsize,
        title=title,
        unit_label_y=ylabel,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def uv_wavelengths(
    interferometer,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label_y="V-Wavelengths ($\lambda$)",
    unit_label_x="U-Wavelengths ($\lambda$)",
    figsize=(14, 7),
    title="UV-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_uv_wavelengths",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    grid_plotters.plot_grid(
        grid=grids.GridIrregular.manual_yx_1d(
            y=interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
            x=interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
        ),
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_y=unit_label_y,
        unit_label_x=unit_label_x,
        pointsize=20,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        symmetric_around_centre=True,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def amplitudes_vs_uv_distances(
    interferometer,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label_y="amplitude (Jy)",
    unit_label_x=r"UV$_{distance}$ (k$\lambda$)",
    figsize=(14, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Amplitudes vs UV-distances",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_amplitudes_vs_uv_distances",
):
    line_yx_plotters.plot_line(
        y=interferometer.amplitudes,
        x=interferometer.uv_distances / 10 ** 3.0,
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_y=unit_label_y,
        unit_label_x=unit_label_x,
        plot_axis_type="scatter",
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def phases_vs_uv_distances(
    interferometer,
    as_subplot=False,
    unit_conversion_factor=None,
    unit_label_y="phase (deg)",
    unit_label_x=r"UV$_{distance}$ (k$\lambda$)",
    figsize=(14, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Phases vs UV-distances",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_phases_vs_uv_distances",
):

    line_yx_plotters.plot_line(
        y=interferometer.phases,
        x=interferometer.uv_distances / 10 ** 3.0,
        as_subplot=as_subplot,
        unit_conversion_factor=unit_conversion_factor,
        unit_label_y=unit_label_y,
        unit_label_x=unit_label_x,
        plot_axis_type="scatter",
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def primary_beam(
    interferometer,
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
    output_filename="interferometer_primary_beam",
):
    """Plot the PSF of the interferometer data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The interferometer data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=interferometer.primary_beam,
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
