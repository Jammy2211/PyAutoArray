from typing import Optional


from autoarray.plot.array import plot_array
from autoarray.plot.utils import subplots, subplot_save, symmetric_vmin_vmax, hide_unused_axes, conf_subplot_figsize, tight_layout


def subplot_fit_imaging(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "fit",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    residuals_symmetric_cmap: bool = True,
    grid=None,
    positions=None,
    lines=None,
):
    """
    2×3 subplot of ``FitImaging`` components.

    Panels: Data | S/N Map | Model Image | Residual Map | Norm Residual Map | Chi-Squared Map

    Parameters
    ----------
    fit
        A ``FitImaging`` instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation to non-residual panels.
    residuals_symmetric_cmap
        Centre residual / normalised-residual colour scale symmetrically
        around zero.
    grid, positions, lines
        Optional overlays forwarded to every panel.
    """
    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes = axes.flatten()

    plot_array(
        fit.data,
        ax=axes[0],
        title="Data",
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        fit.signal_to_noise_map,
        ax=axes[1],
        title="Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        fit.model_data,
        ax=axes[2],
        title="Model Image",
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )

    if residuals_symmetric_cmap:
        vmin_r, vmax_r = symmetric_vmin_vmax(fit.residual_map)
        vmin_n, vmax_n = symmetric_vmin_vmax(fit.normalized_residual_map)
    else:
        vmin_r = vmax_r = vmin_n = vmax_n = None

    plot_array(
        fit.residual_map,
        ax=axes[3],
        title="Residual Map",
        colormap=colormap,
        use_log10=False,
        vmin=vmin_r,
        vmax=vmax_r,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        fit.normalized_residual_map,
        ax=axes[4],
        title="Normalized Residual Map",
        colormap=colormap,
        use_log10=False,
        vmin=vmin_n,
        vmax=vmax_n,
        cb_unit=r"$\sigma$",
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        fit.chi_squared_map,
        ax=axes[5],
        title="Chi-Squared Map",
        colormap=colormap,
        use_log10=use_log10,
        cb_unit=r"$\chi^2$",
        grid=grid,
        positions=positions,
        lines=lines,
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
