from typing import Optional

import matplotlib.pyplot as plt

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
    3×3 subplot of core ``Imaging`` dataset components.

    Panels (row-major):
    0. Data
    1. Data (log10)
    2. Noise-Map
    3. PSF (if present)
    4. PSF (log10, if present)
    5. Signal-To-Noise Map
    6. Over Sample Size (Light Profiles, if present)
    7. Over Sample Size (Pixelization, if present)

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
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0]

    from autoarray.plot.array import plot_array

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
            cb_unit="",
        )
        plot_array(
            dataset.psf.kernel,
            ax=axes[4],
            title="PSF (log10)",
            colormap=colormap,
            use_log10=True,
            cb_unit="",
        )

    plot_array(
        dataset.signal_to_noise_map,
        ax=axes[5],
        title="Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
        cb_unit="",
        grid=grid,
        positions=positions,
        lines=lines,
    )

    over_sample_size_lp = getattr(getattr(dataset, "grids", None), "over_sample_size_lp", None)
    if over_sample_size_lp is not None:
        plot_array(
            over_sample_size_lp,
            ax=axes[6],
            title="Over Sample Size (Light Profiles)",
            colormap=colormap,
            use_log10=use_log10,
            cb_unit="",
        )

    over_sample_size_pix = getattr(getattr(dataset, "grids", None), "over_sample_size_pixelization", None)
    if over_sample_size_pix is not None:
        plot_array(
            over_sample_size_pix,
            ax=axes[7],
            title="Over Sample Size (Pixelization)",
            colormap=colormap,
            use_log10=use_log10,
            cb_unit="",
        )

    from autoarray.plot.utils import hide_unused_axes
    hide_unused_axes(axes)
    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)



def subplot_imaging_dataset_list(
    dataset_list,
    output_path=None,
    output_filename: str = "subplot_dataset_combined",
    output_format="png",
):
    """
    n×3 subplot showing core components for each dataset in a list.

    Each row shows: Data | Noise Map | Signal-To-Noise Map

    Parameters
    ----------
    dataset_list
        List of ``Imaging`` dataset instances.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format string or list, e.g. ``"png"`` or ``["png"]``.
    """
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0]

    from autoarray.plot.array import plot_array

    n = len(dataset_list)
    fig, axes = plt.subplots(n, 3, figsize=(21, 7 * n))
    if n == 1:
        axes = [axes]
    for i, dataset in enumerate(dataset_list):
        plot_array(dataset.data, ax=axes[i][0], title="Data")
        plot_array(dataset.noise_map, ax=axes[i][1], title="Noise Map")
        plot_array(dataset.signal_to_noise_map, ax=axes[i][2], title="Signal-To-Noise Map")
    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
