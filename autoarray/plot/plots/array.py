"""
Standalone function for plotting a 2D array (image) directly with matplotlib.

This replaces the ``MatPlot2D.plot_array`` / ``MatWrap`` system with a plain
function whose defaults are ordinary Python parameter defaults rather than
values loaded from YAML config files.
"""
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from autoarray.plot.plots.utils import apply_extent, conf_figsize, save_figure


def plot_array(
    array: np.ndarray,
    ax: Optional[plt.Axes] = None,
    # --- spatial metadata -------------------------------------------------------
    extent: Optional[Tuple[float, float, float, float]] = None,
    # --- overlays ---------------------------------------------------------------
    mask: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    positions: Optional[List[np.ndarray]] = None,
    lines: Optional[List[np.ndarray]] = None,
    vector_yx: Optional[np.ndarray] = None,
    array_overlay: Optional[np.ndarray] = None,
    # --- cosmetics --------------------------------------------------------------
    title: str = "",
    xlabel: str = 'x (")',
    ylabel: str = 'y (")',
    colormap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_log10: bool = False,
    aspect: str = "auto",
    origin: str = "upper",
    # --- figure control (used only when ax is None) -----------------------------
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    output_filename: str = "array",
    output_format: str = "png",
    structure=None,
) -> None:
    """
    Plot a 2D array (image) using ``plt.imshow``.

    This is the direct-matplotlib replacement for ``MatPlot2D.plot_array``.

    Parameters
    ----------
    array
        2D numpy array of pixel values.
    ax
        Existing matplotlib ``Axes`` to draw onto.  If ``None`` a new figure
        is created and saved / shown according to *output_path*.
    extent
        ``[xmin, xmax, ymin, ymax]`` spatial extent in data coordinates.
        When ``None`` the array pixel indices are used by matplotlib.
    mask
        Array of shape ``(N, 2)`` with ``(y, x)`` coordinates of masked
        pixels to overlay as black dots.
    grid
        Array of shape ``(N, 2)`` with ``(y, x)`` coordinates to scatter.
    positions
        List of ``(N, 2)`` arrays; each is scattered as a distinct group
        of lensed image positions.
    lines
        List of ``(N, 2)`` arrays with ``(y, x)`` columns to plot as lines
        (e.g. critical curves, caustics).
    vector_yx
        Array of shape ``(N, 4)`` — ``(y, x, vy, vx)`` — plotted as quiver
        arrows.
    array_overlay
        A second 2D array rendered on top of *array* with partial alpha.
    title
        Figure title string.
    xlabel, ylabel
        Axis label strings.
    colormap
        Matplotlib colormap name.
    vmin, vmax
        Explicit color scale limits.  When ``None`` the data range is used.
    use_log10
        When ``True`` a ``LogNorm`` is applied.
    aspect
        Passed directly to ``imshow``.
    origin
        Passed directly to ``imshow`` (``"upper"`` or ``"lower"``).
    figsize
        Figure size in inches ``(width, height)``.  Falls back to the value
        in ``visualize/general.yaml`` when ``None``.
    output_path
        Directory to save the figure.  When empty / ``None`` ``plt.show()``
        is called instead.
    output_filename
        Base file name (without extension).
    output_format
        File format, e.g. ``"png"``.
    """
    if array is None or np.all(array == 0):
        return

    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- colour normalisation --------------------------------------------------
    if use_log10:
        from autoconf import conf as _conf

        try:
            log10_min = _conf.instance["visualize"]["general"]["general"][
                "log10_min_value"
            ]
        except Exception:
            log10_min = 1.0e-4

        clipped = np.clip(array, log10_min, None)
        norm = LogNorm(vmin=vmin or log10_min, vmax=vmax or clipped.max())
    elif vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(
        array,
        cmap=colormap,
        norm=norm,
        extent=extent,
        aspect=aspect,
        origin=origin,
    )

    plt.colorbar(im, ax=ax)

    # --- overlays --------------------------------------------------------------
    if array_overlay is not None:
        ax.imshow(
            array_overlay,
            cmap="Greys",
            alpha=0.5,
            extent=extent,
            aspect=aspect,
            origin=origin,
        )

    if mask is not None:
        ax.scatter(mask[:, 1], mask[:, 0], s=1, c="k")

    if grid is not None:
        ax.scatter(grid[:, 1], grid[:, 0], s=1, c="k")

    if positions is not None:
        colors = ["r", "g", "b", "m", "c", "y"]
        for i, pos in enumerate(positions):
            ax.scatter(pos[:, 1], pos[:, 0], s=20, c=colors[i % len(colors)], zorder=5)

    if lines is not None:
        for line in lines:
            if line is not None and len(line) > 0:
                ax.plot(line[:, 1], line[:, 0], linewidth=2)

    if vector_yx is not None:
        ax.quiver(
            vector_yx[:, 1],
            vector_yx[:, 0],
            vector_yx[:, 3],
            vector_yx[:, 2],
        )

    # --- labels / ticks --------------------------------------------------------
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)

    if extent is not None:
        apply_extent(ax, extent)

    # --- output ----------------------------------------------------------------
    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
            structure=structure,
        )
