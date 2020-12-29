from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
import typing
from autoarray.plot.plots import structure_plots


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_plotter_2d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_imaging(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the imaging data_type as a sub-plotter_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
        config file is ignored.
    """

    number_subplots = 6

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(
        imaging=imaging,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    noise_map(
        imaging=imaging,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    psf(
        imaging=imaging,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    signal_to_noise_map(
        imaging=imaging,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    inverse_noise_map(
        imaging=imaging,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    potential_chi_squared_map(
        imaging=imaging,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individual(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    plot_image=False,
    plot_noise_map=False,
    plot_psf=False,
    plot_inverse_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_absolute_signal_to_noise_map=False,
    plot_potential_chi_squared_map=False,
):
    """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_image:

        image(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_noise_map:

        noise_map(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_psf:

        psf(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_inverse_noise_map:

        inverse_noise_map(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_signal_to_noise_map:

        signal_to_noise_map(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_absolute_signal_to_noise_map:

        absolute_signal_to_noise_map(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_potential_chi_squared_map:

        potential_chi_squared_map(
            imaging=imaging,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def image(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the observed data_type of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    structure_plots.plot_array(
        array=imaging.image,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def noise_map(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    structure_plots.plot_array(
        array=imaging.noise_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def psf(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the PSF of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    structure_plots.plot_array(
        array=imaging.psf,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def inverse_noise_map(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """

    structure_plots.plot_array(
        array=imaging.inverse_noise_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def signal_to_noise_map(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the signal-to-noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_array(
        array=imaging.signal_to_noise_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def absolute_signal_to_noise_map(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the signal-to-noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_array(
        array=imaging.absolute_signal_to_noise_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def potential_chi_squared_map(
    imaging,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
):
    """Plot the signal-to-noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    structure_plots.plot_array(
        array=imaging.potential_chi_squared_map,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )
