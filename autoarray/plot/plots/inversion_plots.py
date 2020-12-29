from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
from autoarray.plot.plots import structure_plots
import typing


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_plotter_2d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_inversion(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    number_subplots = 6

    aspect_inv = plotter_2d.figure.aspect_for_subplot_from_grid(
        grid=inversion.mapper.source_full_grid
    )

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    reconstructed_image(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=2, aspect=aspect_inv
    )

    reconstruction(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=3, aspect=aspect_inv
    )

    errors(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=4, aspect=aspect_inv
    )

    residual_map(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=5, aspect=aspect_inv
    )

    chi_squared_map(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=6, aspect=aspect_inv
    )

    regularization_weights(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individuals(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    plot_reconstructed_image=False,
    plot_reconstruction=False,
    plot_errors=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_regularization_weight_map=False,
    plot_interpolated_reconstruction=False,
    plot_interpolated_errors=False,
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    if plot_reconstructed_image:

        reconstructed_image(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_reconstruction:

        reconstruction(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_errors:

        errors(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_residual_map:

        residual_map(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_normalized_residual_map:

        normalized_residual_map(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_chi_squared_map:

        chi_squared_map(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_regularization_weight_map:

        regularization_weights(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_interpolated_reconstruction:

        interpolated_reconstruction(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    if plot_interpolated_errors:

        interpolated_errors(
            inversion=inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def reconstructed_image(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
):

    visuals_2d += include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper)

    structure_plots.plot_array(
        array=inversion.mapped_reconstructed_image,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def reconstruction(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
        solution_vector=inversion.reconstruction
    )

    structure_plots.plot_mapper_obj(
        mapper=inversion.mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        source_pixelilzation_values=source_pixelilzation_values,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def errors(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
        solution_vector=inversion.errors
    )

    structure_plots.plot_mapper_obj(
        mapper=inversion.mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        source_pixelilzation_values=source_pixelilzation_values,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def residual_map(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
        solution_vector=inversion.residual_map
    )

    structure_plots.plot_mapper_obj(
        mapper=inversion.mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        source_pixelilzation_values=source_pixelilzation_values,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def normalized_residual_map(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
        solution_vector=inversion.normalized_residual_map
    )

    structure_plots.plot_mapper_obj(
        mapper=inversion.mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        source_pixelilzation_values=source_pixelilzation_values,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def chi_squared_map(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
        solution_vector=inversion.chi_squared_map
    )

    structure_plots.plot_mapper_obj(
        mapper=inversion.mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        source_pixelilzation_values=source_pixelilzation_values,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def regularization_weights(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    regularization_weights = inversion.regularization.regularization_weights_from_mapper(
        mapper=inversion.mapper
    )

    structure_plots.plot_mapper_obj(
        mapper=inversion.mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        source_pixelilzation_values=regularization_weights,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def interpolated_reconstruction(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    structure_plots.plot_array(
        array=inversion.interpolated_reconstructed_data_from_shape_2d(),
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def interpolated_errors(
    inversion,
    plotter_2d: typing.Optional[p.Plotter2D] = None,
    visuals_2d: typing.Optional[vis.Visuals2D] = None,
    include_2d: typing.Optional[inc.Include2D] = None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=inversion.mapper)

    structure_plots.plot_array(
        array=inversion.interpolated_errors_from_shape_2d(),
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )
