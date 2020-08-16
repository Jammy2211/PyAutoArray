from autoarray.plot import plotters
from autoarray.structures import grids


@plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_interferometer(interferometer, include=None, sub_plotter=None):
    """Plot the interferometer data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    number_subplots = 4

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    uv_wavelengths(interferometer=interferometer, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    visibilities(interferometer=interferometer, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    amplitudes_vs_uv_distances(
        interferometer=interferometer, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    phases_vs_uv_distances(
        interferometer=interferometer, include=include, plotter=sub_plotter
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individual(
    interferometer,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_u_wavelengths=False,
    plot_v_wavelengths=False,
    plot_uv_wavelengths=False,
    plot_amplitudes_vs_uv_distances=False,
    plot_phases_vs_uv_distances=False,
    include=None,
    plotter=None,
):
    """Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_visibilities:

        visibilities(interferometer=interferometer, include=include, plotter=plotter)

    if plot_noise_map:

        noise_map(interferometer=interferometer, include=include, plotter=plotter)

    if plot_u_wavelengths:

        uv_wavelengths(interferometer=interferometer, include=include, plotter=plotter)

    if plot_v_wavelengths:

        v_wavelengths(interferometer=interferometer, include=include, plotter=plotter)

    if plot_uv_wavelengths:

        uv_wavelengths(interferometer=interferometer, include=include, plotter=plotter)

    if plot_amplitudes_vs_uv_distances:

        amplitudes_vs_uv_distances(
            interferometer=interferometer, include=include, plotter=plotter
        )

    if plot_phases_vs_uv_distances:

        phases_vs_uv_distances(
            interferometer=interferometer, include=include, plotter=plotter
        )


@plotters.set_include_and_plotter
@plotters.set_labels
def visibilities(interferometer, include=None, plotter=None):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all input parameters not described below.

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

    plotter.plot_grid(grid=interferometer.visibilities)


@plotters.set_include_and_plotter
@plotters.set_labels
def noise_map(interferometer, include=None, plotter=None):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all input parameters not described below.

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

    plotter.plot_grid(
        grid=interferometer.visibilities, color_array=interferometer.noise_map[:, 0]
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def u_wavelengths(
    interferometer,
    label="Wavelengths",
    plot_axis_type="linear",
    include=None,
    plotter=None,
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all input parameters not described below.

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

    plotter.plot_line(
        y=interferometer.uv_wavelengths[:, 0],
        x=None,
        label=label,
        plot_axis_type=plot_axis_type,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def v_wavelengths(
    interferometer,
    label="Wavelengths",
    plot_axis_type="linear",
    include=None,
    plotter=None,
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all input parameters not described below.

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

    plotter.plot_line(
        y=interferometer.uv_wavelengths[:, 1],
        x=None,
        label=label,
        plot_axis_type=plot_axis_type,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def uv_wavelengths(
    interferometer,
    label_yunits="V-Wavelengths ($\lambda$)",
    label_xunits="U-Wavelengths ($\lambda$)",
    include=None,
    plotter=None,
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all input parameters not described below.

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

    plotter.plot_grid(
        grid=grids.GridCoordinates.from_yx_1d(
            y=interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
            x=interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
        ),
        symmetric_around_centre=True,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def amplitudes_vs_uv_distances(
    interferometer,
    label_yunits="amplitude (Jy)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=None,
    plotter=None,
):

    plotter.plot_line(
        y=interferometer.amplitudes,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def phases_vs_uv_distances(
    interferometer,
    label_yunits="phase (deg)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=None,
    plotter=None,
):

    plotter.plot_line(
        y=interferometer.phases,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )
