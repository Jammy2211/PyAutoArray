import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.plot.plots.utils import auto_mask_edge, zoom_array, subplot_save


def _plot_array(array, ax, title, colormap, use_log10, vmin=None, vmax=None):
    array = zoom_array(array)
    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None

    plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=auto_mask_edge(array) if hasattr(array, "mask") else None,
        title=title,
        colormap=colormap,
        use_log10=use_log10,
        vmin=vmin,
        vmax=vmax,
    )


def _plot_grid(grid, ax, title, color_array=None):
    plot_grid(
        grid=np.array(grid.array),
        ax=ax,
        color_array=color_array,
        title=title,
    )


def _plot_yx(y, x, ax, title, ylabel="", xlabel="", plot_axis_type="linear"):
    plot_yx(
        y=np.asarray(y),
        x=np.asarray(x) if x is not None else None,
        ax=ax,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        plot_axis_type=plot_axis_type,
    )


def _symmetric_vmin_vmax(array):
    try:
        arr = array.native.array if hasattr(array, "native") else np.asarray(array)
        abs_max = np.nanmax(np.abs(arr))
        return -abs_max, abs_max
    except Exception:
        return None, None


def subplot_fit_interferometer(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_fit",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    residuals_symmetric_cmap: bool = True,
):
    """
    2×3 subplot of ``FitInterferometer`` residuals in UV-plane.

    Panels (real then imaginary): Residual Map | Norm Residual Map | Chi-Squared Map

    Parameters
    ----------
    fit
        A ``FitInterferometer`` instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    residuals_symmetric_cmap
        Not used here (UV-plane residuals are scatter plots); kept for API
        consistency.
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()

    uv = fit.dataset.uv_distances / 10**3.0

    _plot_yx(np.real(fit.residual_map), uv, axes[0], "Residual vs UV-Distance (real)", xlabel="k$\\lambda$", plot_axis_type="scatter")
    _plot_yx(np.real(fit.normalized_residual_map), uv, axes[1], "Norm Residual vs UV-Distance (real)", ylabel="$\\sigma$", xlabel="k$\\lambda$", plot_axis_type="scatter")
    _plot_yx(np.real(fit.chi_squared_map), uv, axes[2], "Chi-Squared vs UV-Distance (real)", ylabel="$\\chi^2$", xlabel="k$\\lambda$", plot_axis_type="scatter")
    _plot_yx(np.imag(fit.residual_map), uv, axes[3], "Residual vs UV-Distance (imag)", xlabel="k$\\lambda$", plot_axis_type="scatter")
    _plot_yx(np.imag(fit.normalized_residual_map), uv, axes[4], "Norm Residual vs UV-Distance (imag)", ylabel="$\\sigma$", xlabel="k$\\lambda$", plot_axis_type="scatter")
    _plot_yx(np.imag(fit.chi_squared_map), uv, axes[5], "Chi-Squared vs UV-Distance (imag)", ylabel="$\\chi^2$", xlabel="k$\\lambda$", plot_axis_type="scatter")

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def subplot_fit_interferometer_dirty_images(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_fit_dirty_images",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    residuals_symmetric_cmap: bool = True,
):
    """
    2×3 subplot of ``FitInterferometer`` dirty-image components.

    Panels: Dirty Image | Dirty S/N Map | Dirty Model Image |
            Dirty Residual Map | Dirty Norm Residual Map | Dirty Chi-Squared Map

    Parameters
    ----------
    fit
        A ``FitInterferometer`` instance.
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
        Centre residual colour scale symmetrically around zero.
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()

    _plot_array(fit.dirty_image, axes[0], "Dirty Image", colormap, use_log10)
    _plot_array(fit.dirty_signal_to_noise_map, axes[1], "Dirty Signal-To-Noise Map", colormap, use_log10)
    _plot_array(fit.dirty_model_image, axes[2], "Dirty Model Image", colormap, use_log10)

    if residuals_symmetric_cmap:
        vmin_r, vmax_r = _symmetric_vmin_vmax(fit.dirty_residual_map)
        vmin_n, vmax_n = _symmetric_vmin_vmax(fit.dirty_normalized_residual_map)
    else:
        vmin_r = vmax_r = vmin_n = vmax_n = None

    _plot_array(fit.dirty_residual_map, axes[3], "Dirty Residual Map", colormap, False, vmin=vmin_r, vmax=vmax_r)
    _plot_array(fit.dirty_normalized_residual_map, axes[4], "Dirty Normalized Residual Map", colormap, False, vmin=vmin_n, vmax=vmax_n)
    _plot_array(fit.dirty_chi_squared_map, axes[5], "Dirty Chi-Squared Map", colormap, use_log10)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
