from autoarray.plotters import line_yx_plotters


def plot_quantity_as_function_of_radius(
    quantity,
    radii,
    as_subplot=False,
    label=None,
    plot_axis_type="semilogy",
    effective_radius_line=None,
    einstein_radius_line=None,
    unit_label="scaled",
    kpc_per_arcsec=None,
    figsize=(7, 7),
    plot_legend=True,
    title="Quantity vs Radius",
    ylabel="Quantity",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="quantity_vs_radius",
):

    vertical_lines = []
    vertical_line_labels = []

    if effective_radius_line is not None:
        vertical_lines.append(effective_radius_line)
        vertical_line_labels.append("Effective Radius")

    if einstein_radius_line is not None:
        vertical_lines.append(einstein_radius_line)
        vertical_line_labels.append("Einstein Radius")

    line_yx_plotters.plot_line(
        y=quantity,
        x=radii,
        as_subplot=as_subplot,
        label=label,
        plot_axis_type=plot_axis_type,
        unit_label_x=unit_label,
        unit_conversion_factor=kpc_per_arcsec,
        figsize=figsize,
        plot_legend=plot_legend,
        title=title,
        unit_label_y=ylabel,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
