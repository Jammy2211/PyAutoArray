import autoarray as aa
import matplotlib

backend = aa.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import mapper_plotters
from autoarray.util import plotter_util
from autoarray.operators.inversion import mappers


def subplot(
    inversion,
    mask=None,
    lines=None,
    positions=None,
    grid=None,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
    figsize=None,
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
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    output_path=None,
    output_format="show",
    output_filename="inversion_subplot",
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if figsize is None:
        figsize = figsize_tool

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

    if aspect is "square":
        aspect_inv = ratio
    elif aspect is "auto":
        aspect_inv = 1.0 / ratio
    elif aspect is "equal":
        aspect_inv = 1.0

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    reconstructed_image(
        inversion=inversion,
        mask=mask,
        lines=lines,
        positions=positions,
        grid=grid,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 2, aspect=float(aspect_inv))

    reconstruction(
        inversion=inversion,
        positions=None,
        lines=lines,
        include_grid=False,
        include_centres=False,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=None,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename=None,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3, aspect=float(aspect_inv))

    errors(
        inversion=inversion,
        positions=None,
        include_grid=False,
        include_centres=False,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=None,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename=None,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4, aspect=float(aspect_inv))

    residual_map(
        inversion=inversion,
        positions=None,
        include_grid=False,
        include_centres=False,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=None,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename=None,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 5, aspect=float(aspect_inv))

    chi_squared_map(
        inversion=inversion,
        positions=None,
        include_grid=False,
        include_centres=False,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=None,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename=None,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 6, aspect=float(aspect_inv))

    regularization_weights(
        inversion=inversion,
        positions=None,
        include_grid=False,
        include_centres=False,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=None,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename=None,
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individuals(
    inversion,
    lines=None,
    plot_inversion_reconstruction=False,
    plot_inversion_errors=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
    plot_inversion_interpolated_reconstruction=False,
    plot_inversion_interpolated_errors=False,
    unit_conversion_factor=None,
    unit_label="scaled",
    output_path=None,
    output_format="show",
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

    if plot_inversion_reconstruction:

        reconstruction(
            inversion=inversion,
            include_grid=True,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            figsize=(20, 20),
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_errors:

        errors(
            inversion=inversion,
            include_grid=True,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            figsize=(20, 20),
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_residual_map:

        residual_map(
            inversion=inversion,
            include_grid=True,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            figsize=(20, 20),
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_normalized_residual_map:

        normalized_residual_map(
            inversion=inversion,
            include_grid=True,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            figsize=(20, 20),
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_chi_squared_map:

        chi_squared_map(
            inversion=inversion,
            include_grid=True,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            figsize=(20, 20),
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_regularization_weight_map:

        regularization_weights(
            inversion=inversion,
            include_grid=True,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            figsize=(20, 20),
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_interpolated_reconstruction:

        interpolated_reconstruction(
            inversion=inversion,
            lines=lines,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_interpolated_errors:

        interpolated_errors(
            inversion=inversion,
            lines=lines,
            unit_conversion_factor=unit_conversion_factor,
            unit_label=unit_label,
            output_path=output_path,
            output_format=output_format,
        )


def reconstructed_image(
    inversion,
    mask=None,
    lines=None,
    positions=None,
    grid=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Reconstructed Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="reconstructed_inversion_image",
):

    aa.plot.array(
        array=inversion.mapped_reconstructed_image,
        mask=mask,
        lines=lines,
        points=positions,
        grid=grid,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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


def reconstruction(
    inversion,
    include_origin=True,
    lines=None,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Inversion Reconstruction",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_reconstruction",
):

    if output_format is "fits":
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.reconstruction,
        include_origin=include_origin,
        lines=lines,
        positions=positions,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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

    plotter_util.close_figure(as_subplot=as_subplot)


def errors(
    inversion,
    include_origin=True,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Inversion Reconstruction Errors",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_errors",
):

    if output_format is "fits":
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.errors,
        include_origin=include_origin,
        positions=positions,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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

    plotter_util.close_figure(as_subplot=as_subplot)


def residual_map(
    inversion,
    include_origin=True,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Reconstructed Pixelization Residual-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_residual_map",
):

    if output_format is "fits":
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.residual_map,
        include_origin=include_origin,
        positions=positions,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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

    plotter_util.close_figure(as_subplot=as_subplot)


def normalized_residual_map(
    inversion,
    include_origin=True,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Reconstructed Pixelization Normalized Residual Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_normalized_residual_map",
):

    if output_format is "fits":
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.normalized_residual_map,
        include_origin=include_origin,
        positions=positions,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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

    plotter_util.close_figure(as_subplot=as_subplot)


def chi_squared_map(
    inversion,
    include_origin=True,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Reconstructed Pixelization Chi-Squared Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_chi_squared_map",
):

    if output_format is "fits":
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.chi_squared_map,
        include_origin=include_origin,
        positions=positions,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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

    plotter_util.close_figure(as_subplot=as_subplot)


def regularization_weights(
    inversion,
    include_origin=True,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Reconstructed Pixelization Regularization Weights",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_regularization_weights",
):

    if output_format is "fits":
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    regularization_weights = inversion.regularization.regularization_weights_from_mapper(
        mapper=inversion.mapper
    )

    plot_values(
        inversion=inversion,
        source_pixel_values=regularization_weights,
        include_origin=include_origin,
        positions=positions,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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

    plotter_util.close_figure(as_subplot=as_subplot)


def plot_values(
    inversion,
    source_pixel_values,
    include_origin=True,
    lines=None,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Reconstructed Pixelization",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="pixelization_source_values",
):

    if isinstance(inversion.mapper, mappers.MapperRectangular):

        reconstructed_pixelization = inversion.mapper.reconstructed_pixelization_from_solution_vector(
            solution_vector=source_pixel_values
        )

        origin = get_origin(
            image=reconstructed_pixelization, include_origin=include_origin
        )

        aa.plot.array(
            array=reconstructed_pixelization,
            include_origin=origin,
            lines=lines,
            points=positions,
            as_subplot=True,
            use_scaled_units=use_scaled_units,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
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
            output_filename=output_filename,
        )

        mapper_plotters.rectangular_mapper(
            mapper=inversion.mapper,
            include_centres=include_centres,
            include_grid=include_grid,
            include_border=include_border,
            image_pixels=image_pixels,
            source_pixels=source_pixels,
            as_subplot=True,
            use_scaled_units=use_scaled_units,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            title=title,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
        )

        plotter_util.output_figure(
            array=reconstructed_pixelization,
            as_subplot=as_subplot,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
        )

    elif isinstance(inversion.mapper, mappers.MapperVoronoi):

        mapper_plotters.voronoi_mapper(
            mapper=inversion.mapper,
            source_pixel_values=source_pixel_values,
            include_centres=include_centres,
            lines=lines,
            include_grid=include_grid,
            include_border=include_border,
            image_pixels=image_pixels,
            source_pixels=source_pixels,
            as_subplot=True,
            use_scaled_units=use_scaled_units,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            title=title,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
        )

        plotter_util.output_figure(
            array=None,
            as_subplot=as_subplot,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
        )


def interpolated_reconstruction(
    inversion,
    lines=None,
    positions=None,
    grid=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Inversion Interpolated Reconstruction",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_interpolated_reconstruction",
):

    aa.plot.array(
        array=inversion.interpolated_reconstruction_from_shape_2d(),
        lines=lines,
        points=positions,
        grid=grid,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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


def interpolated_errors(
    inversion,
    lines=None,
    positions=None,
    grid=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
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
    title="Inversion Interpolated Errors",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="inversion_interpolated_errors",
):

    aa.plot.array(
        array=inversion.interpolated_errors_from_shape_2d(),
        lines=lines,
        points=positions,
        grid=grid,
        as_subplot=as_subplot,
        use_scaled_units=use_scaled_units,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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


def get_origin(image, include_origin):

    if include_origin:
        return image.origin
    else:
        return None
