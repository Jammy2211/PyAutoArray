"""
Shared utilities for the direct-matplotlib plot functions.
"""
import logging
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# autoarray → numpy conversion helpers (used by high-level plot functions)
# ---------------------------------------------------------------------------

def auto_mask_edge(array) -> Optional[np.ndarray]:
    """Return edge-pixel (y, x) coords from array.mask, or None."""
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except AttributeError:
        pass
    return None


def zoom_array(array):
    """Apply zoom_around_mask from config if requested."""
    try:
        from autoconf import conf
        zoom_around_mask = conf.instance["visualize"]["general"]["general"]["zoom_around_mask"]
    except Exception:
        zoom_around_mask = False

    if zoom_around_mask and hasattr(array, "mask") and not array.mask.is_all_false:
        from autoarray.mask.derive.zoom_2d import Zoom2D
        return Zoom2D(mask=array.mask).array_2d_from(array=array, buffer=1)
    return array


def numpy_grid(grid) -> Optional[np.ndarray]:
    """Convert a grid-like object to a plain (N,2) numpy array, or None."""
    if grid is None:
        return None
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def numpy_lines(lines) -> Optional[List[np.ndarray]]:
    """Convert lines (Grid2DIrregular or list) to list of (N,2) numpy arrays."""
    if lines is None:
        return None
    result = []
    try:
        for line in lines:
            try:
                arr = np.array(line.array if hasattr(line, "array") else line)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    result.append(arr)
            except Exception:
                pass
    except TypeError:
        pass
    return result or None


def numpy_positions(positions) -> Optional[List[np.ndarray]]:
    """Convert positions to list of (N,2) numpy arrays."""
    if positions is None:
        return None
    try:
        arr = np.array(positions.array if hasattr(positions, "array") else positions)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [arr]
    except Exception:
        pass
    if isinstance(positions, list):
        result = []
        for p in positions:
            try:
                result.append(np.array(p.array if hasattr(p, "array") else p))
            except Exception:
                pass
        return result or None
    return None


def symmetric_vmin_vmax(array):
    """Return ``(-abs_max, abs_max)`` for a symmetric (residual) colormap."""
    try:
        arr = array.native.array if hasattr(array, "native") else np.asarray(array)
        abs_max = float(np.nanmax(np.abs(arr)))
        return -abs_max, abs_max
    except Exception:
        return None, None


def symmetric_cmap_from(array, symmetric_value=None):
    """Return a matplotlib ``Normalize`` centred on zero for a symmetric colormap.

    Parameters
    ----------
    array
        The data array (autoarray or numpy).  Used to compute ``abs_max`` when
        *symmetric_value* is not provided.
    symmetric_value
        If given, fix the half-range to this value (``vmin=-symmetric_value``,
        ``vmax=+symmetric_value``).

    Returns
    -------
    matplotlib.colors.Normalize or None
    """
    import matplotlib.colors as colors

    if symmetric_value is not None:
        abs_max = float(symmetric_value)
    else:
        vmin, vmax = symmetric_vmin_vmax(array)
        if vmin is None:
            return None
        abs_max = max(abs(vmin), abs(vmax))

    return colors.Normalize(vmin=-abs_max, vmax=abs_max)


def set_with_color_values(ax, cmap, color_values, norm=None, fraction=0.047, pad=0.01):
    """Attach a colorbar to *ax* driven by *color_values* rather than a plotted artist.

    Useful for Delaunay mapper visualisation where ``ax.tripcolor`` already draws
    the mesh but we need a separate colorbar tied to specific solution values.

    Parameters
    ----------
    ax
        The matplotlib axes to attach the colorbar to.
    cmap
        A matplotlib colormap name or object.
    color_values
        The 1-D array of values that define the colorbar range.
    norm
        A ``matplotlib.colors.Normalize`` instance.  If ``None`` a default
        ``Normalize(vmin, vmax)`` is created from *color_values*.
    fraction, pad
        Passed directly to ``plt.colorbar``.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if norm is None:
        arr = np.asarray(color_values)
        norm = mcolors.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(color_values)
    return plt.colorbar(mappable=mappable, ax=ax, fraction=fraction, pad=pad)


def subplot_save(fig, output_path, output_filename, output_format):
    """Save a subplot figure or show it, then close."""
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        try:
            fig.savefig(
                os.path.join(output_path, f"{output_filename}.{output_format}"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
        except Exception as exc:
            logger.warning(f"subplot_save: could not save {output_filename}.{output_format}: {exc}")
    else:
        plt.show()
    plt.close(fig)


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
