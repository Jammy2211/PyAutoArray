from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
import typing
from autoarray.plot.plots import structure_plots
from autoarray.structures import grids
import numpy as np


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_plotter_1d_for_subplot
@mat_decorators.set_plotter_2d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_interferometer(
    interferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
):
    """Plot the interferometer data_type as a sub-plotter_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which include_2d the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
        config file is ignored.
    """

    number_subplots = 4

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    uv_wavelengths(
        interferometer=interferometer,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    visibilities(
        interferometer=interferometer,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    amplitudes_vs_uv_distances(
        interferometer=interferometer,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
    )

    plotter_1d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    phases_vs_uv_distances(
        interferometer=interferometer,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plotter_1d=plotter_1d,
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individual(
    interferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_u_wavelengths=False,
    plot_v_wavelengths=False,
    plot_uv_wavelengths=False,
    plot_amplitudes_vs_uv_distances=False,
    plot_phases_vs_uv_distances=False,
):
    """
    Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which include_2d the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_visibilities:

        visibilities(
            interferometer=interferometer,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_noise_map:

        noise_map(
            interferometer=interferometer,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_u_wavelengths:

        uv_wavelengths(
            interferometer=interferometer,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_v_wavelengths:

        v_wavelengths(
            interferometer=interferometer,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

    if plot_uv_wavelengths:

        uv_wavelengths(
            interferometer=interferometer,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_amplitudes_vs_uv_distances:

        amplitudes_vs_uv_distances(
            interferometer=interferometer,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )

    if plot_phases_vs_uv_distances:

        phases_vs_uv_distances(
            interferometer=interferometer,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            plotter_1d=plotter_1d,
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def visibilities(
    interferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    structure_plots.plot_grid(
        grid=interferometer.visibilities.in_grid,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def noise_map(
    interferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    structure_plots.plot_grid(
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
        grid=interferometer.visibilities.in_grid,
        color_array=interferometer.noise_map.real,
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def u_wavelengths(
    interferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    label="Wavelengths",
    plot_axis_type="linear",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotter_1d.plotter_1d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    structure_plots.plot_line(
        y=interferometer.uv_wavelengths[:, 0],
        x=None,
        plotter_1d=plotter_1d,
        label=label,
        plot_axis_type=plot_axis_type,
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def v_wavelengths(
    interferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    label="Wavelengths",
    plot_axis_type="linear",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotter_1d.plotter_1d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    structure_plots.plot_line(
        y=interferometer.uv_wavelengths[:, 1],
        x=None,
        plotter_1d=plotter_1d,
        label=label,
        plot_axis_type=plot_axis_type,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def uv_wavelengths(
    interferometer,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    label_yunits="V-Wavelengths ($\lambda$)",
    label_xunits="U-Wavelengths ($\lambda$)",
):
    """Plot the observed image of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the dataset.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    structure_plots.plot_grid(
        grid=grids.GridIrregularGrouped.from_yx_1d(
            y=interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
            x=interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
        ),
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
        symmetric_around_centre=True,
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def amplitudes_vs_uv_distances(
    interferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    label_yunits="amplitude (Jy)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):

    structure_plots.plot_line(
        y=interferometer.amplitudes,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
        plotter_1d=plotter_1d,
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def phases_vs_uv_distances(
    interferometer,
    visuals_1d: typing.Optional[vis.Visuals1D] = None,
    include_1d: typing.Optional[inc.Include1D] = None,
    plotter_1d: typing.Optional[p.Plotter1D] = None,
    label_yunits="phase (deg)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):

    structure_plots.plot_line(
        y=interferometer.phases,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
        plotter_1d=plotter_1d,
    )
