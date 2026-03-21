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


def _plot_dataset_array(
    array,
    ax,
    title,
    colormap,
    use_log10,
    grid=None,
    positions=None,
    lines=None,
):
    """Internal helper: plot one array component onto *ax*."""
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
    )


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

    _plot_dataset_array(dataset.data, axes[0], "Data", colormap, use_log10, grid, positions, lines)
    _plot_dataset_array(dataset.data, axes[1], "Data (log10)", colormap, True, grid, positions, lines)
    _plot_dataset_array(dataset.noise_map, axes[2], "Noise-Map", colormap, use_log10, grid, positions, lines)

    if dataset.psf is not None:
        _plot_dataset_array(dataset.psf.kernel, axes[3], "Point Spread Function", colormap, use_log10)
        _plot_dataset_array(dataset.psf.kernel, axes[4], "PSF (log10)", colormap, True)

    _plot_dataset_array(dataset.signal_to_noise_map, axes[5], "Signal-To-Noise Map", colormap, use_log10, grid, positions, lines)
    _plot_dataset_array(dataset.grids.over_sample_size_lp, axes[6], "Over Sample Size (Light Profiles)", colormap, use_log10)
    _plot_dataset_array(dataset.grids.over_sample_size_pixelization, axes[7], "Over Sample Size (Pixelization)", colormap, use_log10)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
