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
    """Return the edge-pixel ``(y, x)`` coordinates of an autoarray mask.

    Used to overlay the mask boundary on ``plot_array`` images.  If *array*
    has no ``mask`` attribute, or the mask is fully unmasked, ``None`` is
    returned so no overlay is drawn.

    Parameters
    ----------
    array
        An autoarray ``Array2D`` (or any object with a ``.mask`` attribute
        that exposes ``.derive_grid.edge.array``).

    Returns
    -------
    numpy.ndarray or None
        Shape ``(N, 2)`` float array of ``(y, x)`` edge coordinates, or
        ``None`` when the array is unmasked or has no mask.
    """
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except AttributeError:
        pass
    return None


def zoom_array(array):
    """Crop *array* around its mask when ``zoom_around_mask`` is enabled in config.

    Reads ``visualize/general/general/zoom_around_mask`` from the autoconf
    configuration.  When the flag is ``True`` and *array* carries a non-trivial
    mask the array is cropped via ``Zoom2D`` so that downstream ``imshow``
    calls fill the axes without empty black borders.

    Parameters
    ----------
    array
        An autoarray ``Array2D`` (or any object).  Plain numpy arrays are
        returned unchanged.

    Returns
    -------
    array
        The (potentially cropped) array.  If the config flag is ``False``, or
        *array* has no mask / the mask is all-``False``, the input is returned
        unmodified.
    """
    try:
        from autoconf import conf

        zoom_around_mask = conf.instance["visualize"]["general"]["general"][
            "zoom_around_mask"
        ]
    except Exception:
        zoom_around_mask = False

    if zoom_around_mask and hasattr(array, "mask") and not array.mask.is_all_false:
        from autoarray.mask.derive.zoom_2d import Zoom2D

        return Zoom2D(mask=array.mask).array_2d_from(array=array, buffer=1)
    return array


def numpy_grid(grid) -> Optional[np.ndarray]:
    """Convert a grid-like object to a plain ``(N, 2)`` numpy array, or ``None``.

    Accepts autoarray ``Grid2D`` / ``Grid2DIrregular`` objects (via their
    ``.array`` attribute) as well as bare numpy arrays.  ``None`` inputs are
    passed through so callers can use this as a safe no-op.

    Parameters
    ----------
    grid
        An autoarray grid, a ``(N, 2)`` numpy array, or ``None``.

    Returns
    -------
    numpy.ndarray or None
        Plain ``(N, 2)`` float array with ``(y, x)`` columns, or ``None``.
    """
    if grid is None:
        return None
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def numpy_lines(lines) -> Optional[List[np.ndarray]]:
    """Convert a collection of lines to a list of ``(N, 2)`` numpy arrays.

    Accepts autoarray ``Grid2DIrregular`` objects or any iterable of
    ``(N, 2)`` array-like sequences.  Each element is converted to a plain
    numpy array; elements that cannot be converted are silently skipped.

    Parameters
    ----------
    lines
        An autoarray grid collection, a list of ``(N, 2)`` arrays, or ``None``.

    Returns
    -------
    list of numpy.ndarray or None
        List of ``(N, 2)`` float arrays (``y`` column 0, ``x`` column 1), or
        ``None`` when *lines* is ``None`` or no valid lines are found.
    """
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
    """Convert a positions object to a list of ``(N, 2)`` numpy arrays.

    Positions can be a single ``Grid2DIrregular`` (treated as one group),
    a plain ``(N, 2)`` array (treated as one group), or a list of such
    objects (each becomes one group, scatter-plotted in a distinct colour).

    Parameters
    ----------
    positions
        An autoarray ``Grid2DIrregular``, a ``(N, 2)`` numpy array, a list
        of the above, or ``None``.

    Returns
    -------
    list of numpy.ndarray or None
        Each element is a ``(N, 2)`` array of ``(y, x)`` coordinates
        representing one group of positions, or ``None`` when *positions*
        is ``None`` or cannot be converted.
    """
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
    """Return ``(-abs_max, abs_max)`` colour limits for a symmetric residual colormap.

    Computes the maximum absolute value of *array* and returns symmetric limits
    so that zero maps to the centre of the colormap.  Typically applied to
    residual maps and normalised residual maps.

    Parameters
    ----------
    array
        An autoarray ``Array2D`` (uses ``.native.array``) or a plain numpy
        array.

    Returns
    -------
    tuple of (float, float) or (None, None)
        ``(vmin, vmax)`` where ``vmin == -vmax == -abs_max``.  Returns
        ``(None, None)`` if the computation fails (e.g. all-NaN input).
    """
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
    """Save a subplot figure to disk, or display it, then close it.

    All ``subplot_*`` functions call this as their final step.  When
    *output_path* is non-empty the figure is written to
    ``<output_path>/<output_filename>.<output_format>``; otherwise
    ``plt.show()`` is called.  ``plt.close(fig)`` is always called to
    release memory.

    Parameters
    ----------
    fig
        The matplotlib ``Figure`` to save or show.
    output_path
        Directory to write the file.  Creates the directory if needed.
        ``None`` or an empty string causes ``plt.show()`` to be called.
    output_filename
        Base file name without extension.
    output_format
        File format string, e.g. ``"png"`` or ``"pdf"``.
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        try:
            fig.savefig(
                os.path.join(output_path, f"{output_filename}.{output_format}"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
        except Exception as exc:
            logger.warning(
                f"subplot_save: could not save {output_filename}.{output_format}: {exc}"
            )
    else:
        plt.show()
    plt.close(fig)


def conf_mat_plot_fontsize(section: str, default: int) -> int:
    """Read a font size from the ``mat_plot`` section of ``visualize/general.yaml``.

    Parameters
    ----------
    section
        Sub-key inside ``mat_plot``, e.g. ``"title"``, ``"xlabel"``,
        ``"ylabel"``, ``"xticks"``, or ``"yticks"``.
    default
        Value returned when the config key is absent or unreadable.

    Returns
    -------
    int
        The configured font size.
    """
    try:
        from autoconf import conf

        return int(
            conf.instance["visualize"]["general"]["mat_plot"][section]["fontsize"]
        )
    except Exception:
        return default


def _parse_figsize(raw) -> Tuple[int, int]:
    """Convert *raw* (a tuple/list or a string like ``"(7, 7)"``) to a 2-tuple."""
    if isinstance(raw, (tuple, list)):
        return tuple(raw)
    import ast

    return tuple(ast.literal_eval(str(raw)))


def conf_figsize(context: str = "figures") -> Tuple[int, int]:
    """
    Read figsize from ``visualize/general.yaml`` for the given context.

    For single-panel figures the value is taken from
    ``mat_plot/figure/figsize``; the *context* argument is kept for
    backward compatibility with subplot callers that pass ``"subplots"``.

    Parameters
    ----------
    context
        ``"figures"`` (single-panel) or ``"subplots"`` (multi-panel).
    """
    try:
        from autoconf import conf

        if context == "figures":
            raw = conf.instance["visualize"]["general"]["mat_plot"]["figure"]["figsize"]
            return _parse_figsize(raw)
        return tuple(conf.instance["visualize"]["general"][context]["figsize"])
    except Exception:
        return (7, 7) if context == "figures" else (19, 16)


def apply_labels(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply title, axis labels, and tick font sizes to *ax* from config.

    Reads font sizes from the ``mat_plot`` section of
    ``visualize/general.yaml`` so that users can override them globally
    without touching call sites.  Falls back to the values that the
    old ``MatWrap`` system used when the config is unavailable.

    Parameters
    ----------
    ax
        The matplotlib axes to configure.
    title
        Title string.
    xlabel
        X-axis label string.
    ylabel
        Y-axis label string.
    """
    title_fs = conf_mat_plot_fontsize("title", default=16)
    xlabel_fs = conf_mat_plot_fontsize("xlabel", default=14)
    ylabel_fs = conf_mat_plot_fontsize("ylabel", default=14)
    xticks_fs = conf_mat_plot_fontsize("xticks", default=12)
    yticks_fs = conf_mat_plot_fontsize("yticks", default=12)

    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.tick_params(axis="x", labelsize=xticks_fs)
    ax.tick_params(axis="y", labelsize=yticks_fs)


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
        File format(s) passed to ``fig.savefig``.  Either a single string
        (e.g. ``"png"``) or a list/tuple of strings (e.g. ``["png", "pdf"]``)
        to save in multiple formats in one call.
    dpi
        Resolution in dots per inch.
    structure
        Optional autoarray structure (e.g. ``Array2D``).  Required when
        *format* is ``"fits"`` — its ``output_to_fits`` method is used
        instead of ``fig.savefig``.
    """
    if path:
        os.makedirs(path, exist_ok=True)
        formats = format if isinstance(format, (list, tuple)) else [format]
        for fmt in formats:
            if fmt == "fits":
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
                        os.path.join(path, f"{filename}.{fmt}"),
                        dpi=dpi,
                        bbox_inches="tight",
                        pad_inches=0.1,
                    )
                except Exception as exc:
                    logger.warning(
                        f"save_figure: could not save {filename}.{fmt}: {exc}"
                    )
    else:
        plt.show()

    plt.close(fig)


def plot_visibilities_1d(vis, ax: plt.Axes, title: str = "") -> None:
    """Plot the real and imaginary components of a visibilities array as 1D line plots.

    Draws two overlapping lines — one for the real part and one for the
    imaginary part — with a legend.  Used by interferometer subplot functions
    to visualise raw or residual visibilities.

    Parameters
    ----------
    vis
        A ``Visibilities`` autoarray object (accessed via ``.slim``) or any
        array-like that can be cast to a complex numpy array.
    ax
        Matplotlib ``Axes`` to draw onto.
    title
        Axes title string.
    """
    try:
        y = np.array(vis.slim if hasattr(vis, "slim") else vis)
    except Exception:
        y = np.asarray(vis)
    ax.plot(y.real, label="Real", alpha=0.7)
    ax.plot(y.imag, label="Imaginary", alpha=0.7)
    ax.set_title(title)
    ax.legend(fontsize=8)


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
