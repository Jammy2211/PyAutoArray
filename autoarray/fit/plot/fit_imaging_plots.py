import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.utils import (
    auto_mask_edge,
    zoom_array,
    numpy_grid,
    numpy_lines,
    numpy_positions,
    subplot_save,
)


def _plot_fit_array(
    array,
    ax,
    title,
    colormap,
    use_log10,
    vmin=None,
    vmax=None,
    grid=None,
    positions=None,
    lines=None,
):
    if array is None:
        return

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
        grid=numpy_grid(grid),
        positions=numpy_positions(positions),
        lines=numpy_lines(lines),
        title=title,
        colormap=colormap,
        use_log10=use_log10,
        vmin=vmin,
        vmax=vmax,
    )


def _symmetric_vmin_vmax(array):
    """Return (-abs_max, abs_max) for a symmetric colormap."""
    try:
        arr = array.native.array if hasattr(array, "native") else np.asarray(array)
        abs_max = np.nanmax(np.abs(arr))
        return -abs_max, abs_max
    except Exception:
        return None, None


def subplot_fit_imaging(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_fit",
    output_format: str = "png",
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
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()

    _plot_fit_array(fit.data, axes[0], "Data", colormap, use_log10, grid=grid, positions=positions, lines=lines)
    _plot_fit_array(fit.signal_to_noise_map, axes[1], "Signal-To-Noise Map", colormap, use_log10, grid=grid, positions=positions, lines=lines)
    _plot_fit_array(fit.model_data, axes[2], "Model Image", colormap, use_log10, grid=grid, positions=positions, lines=lines)

    if residuals_symmetric_cmap:
        vmin_r, vmax_r = _symmetric_vmin_vmax(fit.residual_map)
        vmin_n, vmax_n = _symmetric_vmin_vmax(fit.normalized_residual_map)
    else:
        vmin_r = vmax_r = vmin_n = vmax_n = None

    _plot_fit_array(fit.residual_map, axes[3], "Residual Map", colormap, False, vmin=vmin_r, vmax=vmax_r, grid=grid, positions=positions, lines=lines)
    _plot_fit_array(fit.normalized_residual_map, axes[4], "Normalized Residual Map", colormap, False, vmin=vmin_n, vmax=vmax_n, grid=grid, positions=positions, lines=lines)
    _plot_fit_array(fit.chi_squared_map, axes[5], "Chi-Squared Map", colormap, use_log10, grid=grid, positions=positions, lines=lines)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
