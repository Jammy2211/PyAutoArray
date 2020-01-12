from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters


def subplot(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the imaging data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

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


    plotter = plotter.plotter_with_new_output_filename(
        output_filename="imaging"
    )

    rows, columns, figsize_tool = plotter.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    image(
        imaging=imaging,
        grid=grid,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 2)

    noise_map(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 3)

    psf(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 4)

    signal_to_noise_map(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 5)

    absolute_signal_to_noise_map(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 6)

    potential_chi_squared_map(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    plotter.output.to_figure(structure=None)

    plt.close()


def individual(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    plot_image=False,
    plot_noise_map=False,
    plot_psf=False,
    plot_signal_to_noise_map=False,
    plot_absolute_signal_to_noise_map=False,
    plot_potential_chi_squared_map=False,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

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
            grid=grid,
            mask=mask,
            positions=positions,
            include=include,
            plotter=plotter,
        )

    if plot_noise_map:

        noise_map(
            imaging=imaging, mask=mask, include=include, plotter=plotter
        )

    if plot_psf:

        psf(imaging=imaging, include=include, plotter=plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            imaging=imaging, mask=mask, include=include, plotter=plotter
        )

    if plot_absolute_signal_to_noise_map:

        absolute_signal_to_noise_map(
            imaging=imaging, mask=mask, include=include, plotter=plotter
        )

    if plot_potential_chi_squared_map:

        potential_chi_squared_map(
            imaging=imaging, mask=mask, include=include, plotter=plotter
        )


@plotters.set_labels
def image(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the observed data_type of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

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

    plotter.array.plot(
        array=imaging.image,
        include_origin=include.origin,
        grid=grid,
        mask=mask,
        points=positions,
    )


@plotters.set_labels
def noise_map(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    plotter.array.plot(
        array=imaging.image,
        include_origin=include.origin,
        grid=grid,
        mask=mask,
        points=positions,
    )


@plotters.set_labels
def psf(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the PSF of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    plotter.array.plot(
        array=imaging.image,
        include_origin=include.origin,
        grid=grid,
        mask=mask,
        points=positions,
    )


@plotters.set_labels
def signal_to_noise_map(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    plotter.array.plot(
        array=imaging.image,
        include_origin=include.origin,
        grid=grid,
        mask=mask,
        points=positions,
    )


@plotters.set_labels
def absolute_signal_to_noise_map(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    plotter.array.plot(
        array=imaging.image,
        include_origin=include.origin,
        grid=grid,
        mask=mask,
        points=positions,
    )


@plotters.set_labels
def potential_chi_squared_map(
    imaging,
    grid=None,
    mask=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the signal-to-noise_map-map of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    plotter.array.plot(
        array=imaging.image,
        include_origin=include.origin,
        grid=grid,
        mask=mask,
        points=positions,
    )
