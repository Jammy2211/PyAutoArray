from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters, array_plotters

@plotters.set_includes
def subplot(
    imaging,    
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the imaging data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_labels_and_filename(output_filename="imaging")

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    image(
        imaging=imaging,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        positions=positions,
        array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 2)

    noise_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        positions=positions,
        array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 3)

    psf(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        positions=positions,
        array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 4)

    signal_to_noise_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        positions=positions,
        array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 5)

    absolute_signal_to_noise_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        positions=positions,
        array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 6)

    potential_chi_squared_map(
        imaging=imaging,
        include_origin=include_origin,
        mask=mask,
        positions=positions,
        array_plotter=array_plotter
    )

    array_plotter.output_subplot_array()

    plt.close()

@plotters.set_includes
def individual(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    plot_image=False,
    plot_noise_map=False,
    plot_psf=False,
    plot_signal_to_noise_map=False,
    plot_absolute_signal_to_noise_map=False,
    plot_potential_chi_squared_map=False,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_image:

        image(
            imaging=imaging,
            include_origin=include_origin,
            grid=grid,
            mask=mask,
            positions=positions,
            array_plotter=array_plotter,
        )

    if plot_noise_map:

        noise_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            array_plotter=array_plotter,
        )

    if plot_psf:

        psf(imaging=imaging, include_origin=include_origin, array_plotter=array_plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            array_plotter=array_plotter,
        )

    if plot_absolute_signal_to_noise_map:

        absolute_signal_to_noise_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            array_plotter=array_plotter,
        )

    if plot_potential_chi_squared_map:

        potential_chi_squared_map(
            imaging=imaging,
            include_origin=include_origin,
            mask=mask,
            array_plotter=array_plotter,
        )

@plotters.set_includes
@plotters.set_labels
def image(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the observed data_type of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    array_plotter.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        points=positions,
    )

@plotters.set_includes
@plotters.set_labels
def noise_map(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        points=positions,
    )

@plotters.set_includes
@plotters.set_labels
def psf(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the PSF of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        points=positions,
    )

@plotters.set_includes
@plotters.set_labels
def signal_to_noise_map(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        points=positions,
    )

@plotters.set_includes
@plotters.set_labels
def absolute_signal_to_noise_map(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        points=positions,
    )

@plotters.set_includes
@plotters.set_labels
def potential_chi_squared_map(
    imaging,
    include_origin=None,
    grid=None,
    mask=None,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    array_plotter.plot_array(
        array=imaging.image,
        include_origin=include_origin,
        grid=grid,
        mask=mask,
        points=positions,
    )
