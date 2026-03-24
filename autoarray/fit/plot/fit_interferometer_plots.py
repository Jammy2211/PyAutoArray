import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.yx import plot_yx
from autoarray.plot.plots.utils import subplot_save, symmetric_vmin_vmax


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

    plot_yx(np.real(fit.residual_map), uv, ax=axes[0], title="Residual vs UV-Distance (real)", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_yx(np.real(fit.normalized_residual_map), uv, ax=axes[1], title="Norm Residual vs UV-Distance (real)", ylabel="$\\sigma$", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_yx(np.real(fit.chi_squared_map), uv, ax=axes[2], title="Chi-Squared vs UV-Distance (real)", ylabel="$\\chi^2$", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_yx(np.imag(fit.residual_map), uv, ax=axes[3], title="Residual vs UV-Distance (imag)", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_yx(np.imag(fit.normalized_residual_map), uv, ax=axes[4], title="Norm Residual vs UV-Distance (imag)", ylabel="$\\sigma$", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_yx(np.imag(fit.chi_squared_map), uv, ax=axes[5], title="Chi-Squared vs UV-Distance (imag)", ylabel="$\\chi^2$", xlabel="k$\\lambda$", plot_axis_type="scatter")

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

    plot_array(fit.dirty_image, ax=axes[0], title="Dirty Image", colormap=colormap, use_log10=use_log10)
    plot_array(fit.dirty_signal_to_noise_map, ax=axes[1], title="Dirty Signal-To-Noise Map", colormap=colormap, use_log10=use_log10)
    plot_array(fit.dirty_model_image, ax=axes[2], title="Dirty Model Image", colormap=colormap, use_log10=use_log10)

    if residuals_symmetric_cmap:
        vmin_r, vmax_r = symmetric_vmin_vmax(fit.dirty_residual_map)
        vmin_n, vmax_n = symmetric_vmin_vmax(fit.dirty_normalized_residual_map)
    else:
        vmin_r = vmax_r = vmin_n = vmax_n = None

    plot_array(fit.dirty_residual_map, ax=axes[3], title="Dirty Residual Map", colormap=colormap, use_log10=False, vmin=vmin_r, vmax=vmax_r)
    plot_array(fit.dirty_normalized_residual_map, ax=axes[4], title="Dirty Normalized Residual Map", colormap=colormap, use_log10=False, vmin=vmin_n, vmax=vmax_n)
    plot_array(fit.dirty_chi_squared_map, ax=axes[5], title="Dirty Chi-Squared Map", colormap=colormap, use_log10=use_log10)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
