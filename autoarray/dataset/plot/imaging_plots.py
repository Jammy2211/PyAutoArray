from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.array import plot_array
from autoarray.plot.utils import subplot_save


def subplot_imaging_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_dataset",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    grid=None,
    positions=None,
    lines=None,
):
    """
    3×3 subplot of all ``Imaging`` dataset components.

    Panels (row-major):
    0. Data
    1. Data (log10)
    2. Noise-Map
    3. PSF (if present)
    4. PSF log10 (if present)
    5. Signal-To-Noise Map
    6. Over-sample size (light profiles)
    7. Over-sample size (pixelization)

    Parameters
    ----------
    dataset
        An ``Imaging`` dataset instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format, e.g. ``"png"``.
    colormap
        Matplotlib colormap name.  ``None`` uses the package default.
    use_log10
        Apply log10 normalisation to non-log panels.
    grid, positions, lines
        Optional overlays forwarded to every panel.
    """
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    axes = axes.flatten()

    plot_array(
        dataset.data,
        ax=axes[0],
        title="Data",
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        dataset.data,
        ax=axes[1],
        title="Data (log10)",
        colormap=colormap,
        use_log10=True,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        dataset.noise_map,
        ax=axes[2],
        title="Noise-Map",
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )

    if dataset.psf is not None:
        plot_array(
            dataset.psf.kernel,
            ax=axes[3],
            title="Point Spread Function",
            colormap=colormap,
            use_log10=use_log10,
        )
        plot_array(
            dataset.psf.kernel,
            ax=axes[4],
            title="PSF (log10)",
            colormap=colormap,
            use_log10=True,
        )

    plot_array(
        dataset.signal_to_noise_map,
        ax=axes[5],
        title="Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        dataset.grids.over_sample_size_lp,
        ax=axes[6],
        title="Over Sample Size (Light Profiles)",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        dataset.grids.over_sample_size_pixelization,
        ax=axes[7],
        title="Over Sample Size (Pixelization)",
        colormap=colormap,
        use_log10=use_log10,
    )

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
