"""
Shared utilities for the direct-matplotlib plot functions.
"""
import logging
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def conf_figsize(context: str = "figures") -> Tuple[int, int]:
    """
    Read figsize from ``visualize/general.yaml`` for the given context.

    Parameters
    ----------
    context
        Either ``"figures"`` (single-panel) or ``"subplots"`` (multi-panel).
    """
    try:
        from autoconf import conf

        return tuple(conf.instance["visualize"]["general"][context]["figsize"])
    except Exception:
        return (7, 7) if context == "figures" else (19, 16)


def save_figure(
    fig: plt.Figure,
    path: str,
    filename: str,
    format: str = "png",
    dpi: int = 300,
    structure=None,
) -> None:
    """
    Save *fig* to ``<path>/<filename>.<format>`` then close it.

    If *path* is an empty string or ``None``, ``plt.show()`` is called instead.
    After either action ``plt.close(fig)`` is always called to free memory.

    Parameters
    ----------
    fig
        The matplotlib figure to save.
    path
        Directory where the file is written.  Created if it does not exist.
    filename
        File name without extension.
    format
        File format passed to ``fig.savefig`` (e.g. ``"png"``, ``"pdf"``).
    dpi
        Resolution in dots per inch.
    structure
        Optional autoarray structure (e.g. ``Array2D``).  Required when
        *format* is ``"fits"`` — its ``output_to_fits`` method is used
        instead of ``fig.savefig``.
    """
    if path:
        os.makedirs(path, exist_ok=True)
        if format == "fits":
            if structure is not None and hasattr(structure, "output_to_fits"):
                structure.output_to_fits(
                    file_path=os.path.join(path, f"{filename}.fits"),
                    overwrite=True,
                )
            else:
                logger.warning(
                    f"save_figure: fits format requested for {filename} but no "
                    "compatible structure was provided; skipping."
                )
        else:
            try:
                fig.savefig(
                    os.path.join(path, f"{filename}.{format}"),
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
            except Exception as exc:
                logger.warning(
                    f"save_figure: could not save {filename}.{format}: {exc}"
                )
    else:
        plt.show()

    plt.close(fig)


def apply_extent(
    ax: plt.Axes,
    extent: Tuple[float, float, float, float],
    n_ticks: int = 3,
) -> None:
    """
    Apply axis limits and evenly spaced linear ticks to *ax*.

    Parameters
    ----------
    ax
        The matplotlib axes to configure.
    extent
        ``[xmin, xmax, ymin, ymax]`` limits.
    n_ticks
        Number of ticks on each axis.  ``3`` produces ``[-R, 0, R]`` for
        a symmetric extent, matching the reference ``plot_grid`` example.
    """
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(xmin, xmax, n_ticks))
    ax.set_yticks(np.linspace(ymin, ymax, n_ticks))
