"""
Standalone function for plotting 1D y-vs-x data.

Replaces ``MatPlot1D.plot_yx`` / ``MatWrap`` system.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from autoarray.plot.utils import apply_labels, conf_figsize, save_figure


def plot_yx(
    y,
    x=None,
    ax: Optional[plt.Axes] = None,
    # --- errors / extras --------------------------------------------------------
    y_errors: Optional[np.ndarray] = None,
    x_errors: Optional[np.ndarray] = None,
    y_extra: Optional[np.ndarray] = None,
    shaded_region: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    # --- cosmetics --------------------------------------------------------------
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    label: Optional[str] = None,
    color: str = "b",
    linestyle: str = "-",
    plot_axis_type: str = "linear",
    # --- figure control (used only when ax is None) -----------------------------
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    output_filename: str = "yx",
    output_format: str = "png",
) -> None:
    """
    Plot 1D y versus x data.

    Replaces ``MatPlot1D.plot_yx`` with direct matplotlib calls.

    Parameters
    ----------
    y
        1D numpy array of y values.
    x
        1D numpy array of x values.  When ``None`` integer indices are used.
    ax
        Existing ``Axes`` to draw onto.  ``None`` creates a new figure.
    y_errors, x_errors
        Per-point error values; trigger ``plt.errorbar``.
    y_extra
        Optional second y series to overlay.
    shaded_region
        Tuple ``(y1, y2)`` arrays; filled region drawn with alpha.
    title
        Figure title.
    xlabel, ylabel
        Axis labels.
    label
        Legend label for the main series.
    color
        Line / marker colour.
    linestyle
        Line style string.
    plot_axis_type
        One of ``"linear"``, ``"log"``, ``"loglog"``, ``"symlog"``.
    figsize
        Figure size in inches.
    output_path
        Directory for saving.  Empty / ``None`` calls ``plt.show()``.
    output_filename
        Base file name without extension.
    output_format
        File format, e.g. ``"png"``.
    """
    # --- autoarray extraction --------------------------------------------------
    if x is None and hasattr(y, "grid_radial"):
        x = y.grid_radial
    y = y.array if hasattr(y, "array") else np.asarray(y)
    if x is not None:
        x = x.array if hasattr(x, "array") else np.asarray(x)

    # guard: nothing to draw
    if y is None or len(y) == 0 or np.isnan(y).all():
        return

    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if x is None:
        x = np.arange(len(y))

    # --- main line / scatter ---------------------------------------------------
    if y_errors is not None or x_errors is not None:
        ax.errorbar(
            x,
            y,
            yerr=y_errors,
            xerr=x_errors,
            fmt="-o",
            color=color,
            label=label,
            markersize=3,
        )
    elif plot_axis_type == "scatter":
        ax.scatter(x, y, s=2, c=color, label=label)
    elif plot_axis_type in ("log", "semilogy"):
        ax.semilogy(x, y, color=color, linestyle=linestyle, label=label)
    elif plot_axis_type == "loglog":
        ax.loglog(x, y, color=color, linestyle=linestyle, label=label)
    else:
        ax.plot(x, y, color=color, linestyle=linestyle, label=label)

    if plot_axis_type == "symlog":
        ax.set_yscale("symlog")

    # --- extras ----------------------------------------------------------------
    if y_extra is not None:
        ax.plot(x, y_extra, color="r", linestyle="--", alpha=0.7)

    if shaded_region is not None:
        y1, y2 = shaded_region
        ax.fill_between(x, y1, y2, alpha=0.3)

    # --- labels ----------------------------------------------------------------
    apply_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    if label is not None:
        ax.legend(fontsize=12)

    # --- output ----------------------------------------------------------------
    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
        )
