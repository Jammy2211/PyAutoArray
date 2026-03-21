import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.plot.plots.utils import auto_mask_edge, zoom_array, subplot_save
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


def _plot_array(array, ax, title, colormap, use_log10, output_path=None, output_filename=None, output_format="png"):
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
        output_path=output_path,
        output_filename=output_filename or "",
        output_format=output_format,
    )


def _plot_grid(grid, ax, title, colormap, color_array=None, output_path=None, output_filename=None, output_format="png"):
    plot_grid(
        grid=np.array(grid.array),
        ax=ax,
        color_array=color_array,
        title=title,
        output_path=output_path,
        output_filename=output_filename or "",
        output_format=output_format,
    )


def _plot_yx(y, x, ax, title, ylabel="", xlabel="", plot_axis_type="linear",
             output_path=None, output_filename=None, output_format="png"):
    plot_yx(
        y=np.asarray(y),
        x=np.asarray(x) if x is not None else None,
        ax=ax,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        plot_axis_type=plot_axis_type,
        output_path=output_path,
        output_filename=output_filename or "",
        output_format=output_format,
    )


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

    _plot_grid(dataset.data.in_grid, axes[0], "Visibilities", colormap)
    _plot_grid(
        Grid2DIrregular.from_yx_1d(
            y=dataset.uv_wavelengths[:, 1] / 10**3.0,
            x=dataset.uv_wavelengths[:, 0] / 10**3.0,
        ),
        axes[1], "UV-Wavelengths", colormap,
    )
    _plot_yx(
        dataset.amplitudes, dataset.uv_distances / 10**3.0,
        axes[2], "Amplitudes vs UV-distances",
        ylabel="Jy", xlabel="k$\\lambda$", plot_axis_type="scatter",
    )
    _plot_yx(
        dataset.phases, dataset.uv_distances / 10**3.0,
        axes[3], "Phases vs UV-distances",
        ylabel="deg", xlabel="k$\\lambda$", plot_axis_type="scatter",
    )
    _plot_array(dataset.dirty_image, axes[4], "Dirty Image", colormap, use_log10)
    _plot_array(dataset.dirty_signal_to_noise_map, axes[5], "Dirty Signal-To-Noise Map", colormap, use_log10)

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

    _plot_array(dataset.dirty_image, axes[0], "Dirty Image", colormap, use_log10)
    _plot_array(dataset.dirty_noise_map, axes[1], "Dirty Noise Map", colormap, use_log10)
    _plot_array(dataset.dirty_signal_to_noise_map, axes[2], "Dirty Signal-To-Noise Map", colormap, use_log10)

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
