from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters
from autoarray.plotters import array_plotters, grid_plotters, line_plotters
from autoarray.util import plotter_util
from autoarray.structures import grids
import numpy as np

@plotters.set_includes
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
    origin : True
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
    #     origin=origin,
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

@plotters.set_includes
def individual(
    interferometer,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_u_wavelengths=False,
    plot_v_wavelengths=False,
    plot_uv_wavelengths=False,
    plot_amplitudes_vs_uv_distances=False,
    plot_phases_vs_uv_distances=False,
    plot_primary_beam=False,
    array_plotter=array_plotters.ArrayPlotter(),
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_visibilities:

        visibilities(
            interferometer=interferometer,
            grid_plotter=grid_plotter
        )

    if plot_noise_map:

        noise_map(
            interferometer=interferometer,
            grid_plotter=grid_plotter
        )

    if plot_u_wavelengths:

        uv_wavelengths(
            interferometer=interferometer,
            grid_plotter=grid_plotter
        )

    if plot_v_wavelengths:

        v_wavelengths(
            interferometer=interferometer,
            line_plotter=line_plotter
        )

    if plot_uv_wavelengths:

        uv_wavelengths(
            interferometer=interferometer,
            line_plotter=line_plotter
        )

    if plot_amplitudes_vs_uv_distances:

        amplitudes_vs_uv_distances(
            interferometer=interferometer,
            line_plotter=line_plotter
        )

    if plot_phases_vs_uv_distances:

        phases_vs_uv_distances(
            interferometer=interferometer,
            line_plotter=line_plotter
        )

    if plot_primary_beam:

        primary_beam(
            interferometer=interferometer,
            array_plotter=array_plotter
        )

@plotters.set_includes
@plotters.set_labels
def visibilities(
    interferometer,
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    grid_plotter.plot_grid(
        grid=interferometer.visibilities,
    )

@plotters.set_includes
@plotters.set_labels
def noise_map(
    interferometer,
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    grid_plotter.plot_grid(
        grid=interferometer.visibilities,
        colors=interferometer.noise_map[:, 0],
    )

@plotters.set_includes
@plotters.set_labels
def u_wavelengths(
    interferometer,
    label="Wavelengths",
    plot_axis_type="linear",
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    line_plotter.plot_line(
        y=interferometer.uv_wavelengths[:, 0],
        x=None,
        label=label,
        plot_axis_type=plot_axis_type,
    )

@plotters.set_includes
@plotters.set_labels
def v_wavelengths(
    interferometer,
    label="Wavelengths",
    plot_axis_type="linear",
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    line_plotter.plot_line(
        y=interferometer.uv_wavelengths[:, 1],
        x=None,
        label=label,
        plot_axis_type=plot_axis_type,
    )

@plotters.set_includes
@plotters.set_labels
def uv_wavelengths(
    interferometer,
    label_yunits="V-Wavelengths ($\lambda$)",
    label_xunits="U-Wavelengths ($\lambda$)",
    grid_plotter=grid_plotters.GridPlotter(),
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    grid_plotter = grid_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits)

    grid_plotter.plot_grid(
        grid=grids.GridIrregular.manual_yx_1d(
            y=interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
            x=interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
        ),
        symmetric_around_centre=True,
    )

@plotters.set_includes
@plotters.set_labels
def amplitudes_vs_uv_distances(
    interferometer,
    label_yunits="amplitude (Jy)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    line_plotter=line_plotters.LinePlotter(),
):

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits)

    line_plotter.plot_line(
        y=interferometer.amplitudes,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )

@plotters.set_includes
@plotters.set_labels
def phases_vs_uv_distances(
    interferometer,
    label_yunits="phase (deg)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    line_plotter=line_plotters.LinePlotter(),
):

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits)

    line_plotter.plot_line(
        y=interferometer.phases,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )

@plotters.set_includes
@plotters.set_labels
def primary_beam(
    interferometer,
    include_origin=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the PSF of the interferometer data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The interferometer data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=interferometer.primary_beam,
        include_origin=include_origin,
    )
