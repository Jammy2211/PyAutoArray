import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.array import plot_array
from autoarray.plot.grid import plot_grid
from autoarray.plot.yx import plot_yx
from autoarray.plot.utils import subplot_save
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


def subplot_interferometer_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_dataset",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
):
    """
    2×3 subplot of interferometer dataset components.

    Panels: Visibilities | UV-Wavelengths | Amplitudes vs UV-distances |
            Phases vs UV-distances | Dirty Image | Dirty S/N Map

    Parameters
    ----------
    dataset
        An ``Interferometer`` dataset instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation to image panels.
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()

    plot_grid(dataset.data.in_grid, ax=axes[0], title="Visibilities")
    plot_grid(
        Grid2DIrregular.from_yx_1d(
            y=dataset.uv_wavelengths[:, 1] / 10**3.0,
            x=dataset.uv_wavelengths[:, 0] / 10**3.0,
        ),
        ax=axes[1], title="UV-Wavelengths",
    )
    plot_yx(dataset.amplitudes, dataset.uv_distances / 10**3.0, ax=axes[2],
            title="Amplitudes vs UV-distances", ylabel="Jy", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_yx(dataset.phases, dataset.uv_distances / 10**3.0, ax=axes[3],
            title="Phases vs UV-distances", ylabel="deg", xlabel="k$\\lambda$", plot_axis_type="scatter")
    plot_array(dataset.dirty_image, ax=axes[4], title="Dirty Image", colormap=colormap, use_log10=use_log10)
    plot_array(dataset.dirty_signal_to_noise_map, ax=axes[5], title="Dirty Signal-To-Noise Map", colormap=colormap, use_log10=use_log10)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def subplot_interferometer_dirty_images(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_dirty_images",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
):
    """
    1×3 subplot of dirty image, dirty noise map, and dirty S/N map.

    Parameters
    ----------
    dataset
        An ``Interferometer`` dataset instance.
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
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    plot_array(dataset.dirty_image, ax=axes[0], title="Dirty Image", colormap=colormap, use_log10=use_log10)
    plot_array(dataset.dirty_noise_map, ax=axes[1], title="Dirty Noise Map", colormap=colormap, use_log10=use_log10)
    plot_array(dataset.dirty_signal_to_noise_map, ax=axes[2], title="Dirty Signal-To-Noise Map", colormap=colormap, use_log10=use_log10)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
