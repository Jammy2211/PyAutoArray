from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters
from autoarray.plotters import array_plotters, grid_plotters, line_plotters
from autoarray.structures import grids



def subplot(
    interferometer,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the interferometer data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer : data_type.UVPlaneData
        The interferometer data_type, which include the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    array_plotter = array_plotter.plotter_as_sub_plotter()
    grid_plotter = grid_plotter.plotter_as_sub_plotter()
    line_plotter = line_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_labels_and_filename(
        output_filename="interferometer"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    uv_wavelengths(interferometer=interferometer, include=include, grid_plotter=grid_plotter)

    plt.subplot(rows, columns, 2)

    visibilities(interferometer=interferometer, include=include, grid_plotter=grid_plotter)

    plt.subplot(rows, columns, 3)

    amplitudes_vs_uv_distances(interferometer=interferometer, include=include, line_plotter=line_plotter)

    plt.subplot(rows, columns, 4)

    phases_vs_uv_distances(interferometer=interferometer, include=include, line_plotter=line_plotter)

    array_plotter.output_subplot_array()

    plt.close()



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
    include=plotters.Include(),
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
        The interferometer data_type, which include the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_visibilities:

        visibilities(interferometer=interferometer, include=include, grid_plotter=grid_plotter)

    if plot_noise_map:

        noise_map(interferometer=interferometer, include=include, grid_plotter=grid_plotter)

    if plot_u_wavelengths:

        uv_wavelengths(interferometer=interferometer, include=include, grid_plotter=grid_plotter)

    if plot_v_wavelengths:

        v_wavelengths(interferometer=interferometer, include=include, line_plotter=line_plotter)

    if plot_uv_wavelengths:

        uv_wavelengths(interferometer=interferometer, include=include, line_plotter=line_plotter)

    if plot_amplitudes_vs_uv_distances:

        amplitudes_vs_uv_distances(
            interferometer=interferometer, include=include, line_plotter=line_plotter
        )

    if plot_phases_vs_uv_distances:

        phases_vs_uv_distances(interferometer=interferometer, include=include, line_plotter=line_plotter)

    if plot_primary_beam:

        primary_beam(interferometer=interferometer, include=include, array_plotter=array_plotter)



@plotters.set_labels
def visibilities(interferometer, include=plotters.Include(), grid_plotter=grid_plotters.GridPlotter()):
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

    grid_plotter.plot_grid(grid=interferometer.visibilities)



@plotters.set_labels
def noise_map(interferometer, include=plotters.Include(), grid_plotter=grid_plotters.GridPlotter()):
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
        grid=interferometer.visibilities, colors=interferometer.noise_map[:, 0]
    )



@plotters.set_labels
def u_wavelengths(
    interferometer,
    label="Wavelengths",
    plot_axis_type="linear",
    include=plotters.Include(),
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



@plotters.set_labels
def v_wavelengths(
    interferometer,
    label="Wavelengths",
    plot_axis_type="linear",
include=plotters.Include(),
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



@plotters.set_labels
def uv_wavelengths(
    interferometer,
    label_yunits="V-Wavelengths ($\lambda$)",
    label_xunits="U-Wavelengths ($\lambda$)",
include=plotters.Include(),
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
        label_yunits=label_yunits, label_xunits=label_xunits
    )

    grid_plotter.plot_grid(
        grid=grids.GridIrregular.manual_yx_1d(
            y=interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
            x=interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
        ),
        symmetric_around_centre=True,
    )



@plotters.set_labels
def amplitudes_vs_uv_distances(
    interferometer,
    label_yunits="amplitude (Jy)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
include=plotters.Include(),
    line_plotter=line_plotters.LinePlotter(),
):

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits
    )

    line_plotter.plot_line(
        y=interferometer.amplitudes,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )



@plotters.set_labels
def phases_vs_uv_distances(
    interferometer,
    label_yunits="phase (deg)",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
include=plotters.Include(),
    line_plotter=line_plotters.LinePlotter(),
):

    line_plotter = line_plotter.plotter_with_new_labels_and_filename(
        label_yunits=label_yunits, label_xunits=label_xunits
    )

    line_plotter.plot_line(
        y=interferometer.phases,
        x=interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )



@plotters.set_labels
def primary_beam(
    interferometer, include=plotters.Include(), array_plotter=array_plotters.ArrayPlotter()
):
    """Plot the PSF of the interferometer data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The interferometer data_type, which include the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=interferometer.primary_beam, include_origin=include.origin
    )
