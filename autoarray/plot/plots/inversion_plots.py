from autoarray.plot.mat_wrap import mat_decorators


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_plotter_2d_for_subplot
@mat_decorators.set_subplot_filename
def subplot_inversion(
    inversion,
    lines=None,
    image_positions=None,
    source_positions=None,
    grid=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    number_subplots = 6

    ratio = float(
        (
            inversion.mapper.grid.scaled_maxima[1]
            - inversion.mapper.grid.scaled_minima[1]
        )
        / (
            inversion.mapper.grid.scaled_maxima[0]
            - inversion.mapper.grid.scaled_minima[0]
        )
    )

    aspect_inv = plotter_2d.figure.aspect_for_subplot_from_ratio(ratio=ratio)

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    reconstructed_image(
        inversion=inversion,
        lines=lines,
        image_positions=image_positions,
        grid=grid,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=2, aspect=aspect_inv
    )

    reconstruction(
        inversion=inversion,
        source_positions=source_positions,
        lines=lines,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=3, aspect=aspect_inv
    )

    errors(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=4, aspect=aspect_inv
    )

    residual_map(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=5, aspect=aspect_inv
    )

    chi_squared_map(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.setup_subplot(
        number_subplots=number_subplots, subplot_index=6, aspect=aspect_inv
    )

    regularization_weights(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individuals(
    inversion,
    lines=None,
    image_positions=None,
    source_positions=None,
    plot_reconstructed_image=False,
    plot_reconstruction=False,
    plot_errors=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_regularization_weight_map=False,
    plot_interpolated_reconstruction=False,
    plot_interpolated_errors=False,
    include_2d=None,
    plotter_2d=None,
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
            image_positions=image_positions,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_reconstruction:

        reconstruction(
            inversion=inversion,
            source_positions=source_positions,
            lines=lines,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_errors:

        errors(inversion=inversion, include_2d=include_2d, plotter_2d=plotter_2d)

    if plot_residual_map:

        residual_map(inversion=inversion, include_2d=include_2d, plotter_2d=plotter_2d)

    if plot_normalized_residual_map:

        normalized_residual_map(
            inversion=inversion, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_chi_squared_map:

        chi_squared_map(
            inversion=inversion, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_regularization_weight_map:

        regularization_weights(
            inversion=inversion, include_2d=include_2d, plotter_2d=plotter_2d
        )

    if plot_interpolated_reconstruction:

        interpolated_reconstruction(
            inversion=inversion,
            source_positions=source_positions,
            lines=lines,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )

    if plot_interpolated_errors:

        interpolated_errors(
            inversion=inversion,
            lines=lines,
            include_2d=include_2d,
            plotter_2d=plotter_2d,
        )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def reconstructed_image(
    inversion,
    grid=None,
    lines=None,
    image_positions=None,
    include_2d=None,
    plotter_2d=None,
):

    plotter_2d._plot_array(
        array=inversion.mapped_reconstructed_image,
        mask=include_2d.mask_from_grid(grid=inversion.mapper.grid),
        lines=lines,
        positions=image_positions,
        grid=grid,
        include_origin=include_2d.origin,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def reconstruction(
    inversion,
    lines=None,
    source_positions=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.reconstruction
    )

    plotter_2d._plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        lines=lines,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_grid=include_2d.inversion_grid,
        include_border=include_2d.inversion_border,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def errors(
    inversion,
    source_positions=None,
    lines=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.errors
    )

    plotter_2d._plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        lines=lines,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_grid=include_2d.inversion_grid,
        include_border=include_2d.inversion_border,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def residual_map(
    inversion,
    source_positions=None,
    lines=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.residual_map
    )

    plotter_2d._plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        lines=lines,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_grid=include_2d.inversion_grid,
        include_border=include_2d.inversion_border,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def normalized_residual_map(
    inversion,
    source_positions=None,
    lines=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.normalized_residual_map
    )

    plotter_2d._plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        lines=lines,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_grid=include_2d.inversion_grid,
        include_border=include_2d.inversion_border,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def chi_squared_map(
    inversion,
    source_positions=None,
    lines=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.chi_squared_map
    )

    plotter_2d._plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        lines=lines,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_grid=include_2d.inversion_grid,
        include_border=include_2d.inversion_border,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def regularization_weights(
    inversion,
    source_positions=None,
    lines=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    regularization_weights = inversion.regularization.regularization_weights_from_mapper(
        mapper=inversion.mapper
    )

    plotter_2d._plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=regularization_weights,
        lines=lines,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_grid=include_2d.inversion_grid,
        include_border=include_2d.inversion_border,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def interpolated_reconstruction(
    inversion,
    lines=None,
    source_positions=None,
    grid=None,
    include_2d=None,
    plotter_2d=None,
):

    plotter_2d._plot_array(
        array=inversion.interpolated_reconstruction_from_shape_2d(),
        lines=lines,
        positions=source_positions,
        grid=grid,
        include_origin=include_2d.origin,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def interpolated_errors(
    inversion,
    lines=None,
    source_positions=None,
    grid=None,
    include_2d=None,
    plotter_2d=None,
):

    plotter_2d._plot_array(
        array=inversion.interpolated_errors_from_shape_2d(),
        lines=lines,
        positions=source_positions,
        grid=grid,
        include_origin=include_2d.origin,
    )
