import numpy as np
from typing import List, Optional, Union

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.plot.plots.utils import (
    auto_mask_edge,
    zoom_array,
    numpy_grid,
    numpy_lines,
    numpy_positions,
)


def plot_array_2d(
    array,
    output_path: Optional[str] = None,
    output_filename: str = "array",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    origin=None,
    border=None,
    grid=None,
    mesh_grid=None,
    positions=None,
    lines=None,
    patches=None,
    fill_region=None,
    array_overlay=None,
    title: str = "Array2D",
    ax=None,
):
    """
    Plot an ``Array2D`` (or plain 2D numpy array) with optional overlays.

    Handles extraction of the native 2D data, spatial extent, and mask edge
    from autoarray objects before delegating to ``plot_array``.

    Parameters
    ----------
    array
        An ``Array2D`` instance or a plain 2D numpy array.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format, e.g. ``"png"`` or ``"fits"``.
    colormap
        Matplotlib colormap name or object.  ``None`` uses the default.
    use_log10
        Apply log10 normalisation.
    origin, border, grid, mesh_grid
        Optional overlay coordinate arrays (autoarray or numpy).
    positions
        Positions to scatter (autoarray irregular grid or list of arrays).
    lines
        Lines to draw (autoarray irregular grid or list of arrays).
    patches
        Matplotlib patch objects.
    fill_region
        ``[y1, y2]`` arrays for ``ax.fill_between``.
    array_overlay
        A second ``Array2D`` rendered on top with partial alpha.
    title
        Figure title.
    ax
        Existing ``Axes`` to draw onto; if ``None`` a new figure is created.
    """
    if array is None or np.all(array == 0):
        return

    array = zoom_array(array)

    try:
        arr = array.native.array
        extent = array.geometry.extent
        mask = auto_mask_edge(array)
    except AttributeError:
        arr = np.asarray(array)
        extent = None
        mask = None

    overlay_arr = None
    if array_overlay is not None:
        try:
            overlay_arr = array_overlay.native.array
        except AttributeError:
            overlay_arr = np.asarray(array_overlay)

    plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=mask,
        border=numpy_grid(border),
        origin=numpy_grid(origin),
        grid=numpy_grid(grid),
        mesh_grid=numpy_grid(mesh_grid),
        positions=numpy_positions(positions),
        lines=numpy_lines(lines),
        array_overlay=overlay_arr,
        patches=patches,
        fill_region=fill_region,
        title=title,
        colormap=colormap,
        use_log10=use_log10,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        structure=array,
    )


def plot_grid_2d(
    grid,
    output_path: Optional[str] = None,
    output_filename: str = "grid",
    output_format: str = "png",
    color_array: Optional[np.ndarray] = None,
    plot_over_sampled_grid: bool = False,
    lines=None,
    indexes=None,
    title: str = "Grid2D",
    ax=None,
):
    """
    Scatter-plot a ``Grid2D`` (or plain (N,2) numpy array).

    Parameters
    ----------
    grid
        A ``Grid2D`` instance or plain ``(N, 2)`` numpy array.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    color_array
        1D array of values used to colour the scatter points.
    plot_over_sampled_grid
        If ``True`` and *grid* has an ``over_sampled`` attribute, plot that
        instead.
    lines
        Lines to overlay.
    indexes
        Index arrays to highlight in distinct colours.
    title
        Figure title.
    ax
        Existing ``Axes`` to draw onto.
    """
    if plot_over_sampled_grid and hasattr(grid, "over_sampled"):
        grid = grid.over_sampled

    plot_grid(
        grid=np.array(grid.array if hasattr(grid, "array") else grid),
        ax=ax,
        lines=numpy_lines(lines),
        color_array=color_array,
        indexes=indexes,
        title=title,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def plot_yx_1d(
    y,
    x=None,
    output_path: Optional[str] = None,
    output_filename: str = "yx",
    output_format: str = "png",
    shaded_region=None,
    plot_axis_type: str = "linear",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    ax=None,
):
    """
    1D line / scatter plot for ``Array1D`` or plain array data.

    Parameters
    ----------
    y
        ``Array1D`` instance or list / numpy array of y values.
    x
        ``Array1D``, ``Grid1D``, list, or numpy array of x values.
        Defaults to ``y.grid_radial`` when *y* is an ``Array1D``.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    shaded_region
        ``(y1, y2)`` tuple for ``ax.fill_between``.
    plot_axis_type
        Axis scale: ``"linear"``, ``"log"``, ``"loglog"``, ``"scatter"``, etc.
    title, xlabel, ylabel
        Text labels.
    ax
        Existing ``Axes`` to draw onto.
    """
    from autoarray.structures.arrays.uniform_1d import Array1D

    if isinstance(y, list):
        y = Array1D.no_mask(values=y, pixel_scales=1.0)
    if isinstance(x, list):
        x = Array1D.no_mask(values=x, pixel_scales=1.0)

    if x is None and hasattr(y, "grid_radial"):
        x = y.grid_radial

    y_arr = y.array if hasattr(y, "array") else np.array(y)
    x_arr = x.array if hasattr(x, "array") else np.array(x) if x is not None else None

    plot_yx(
        y=y_arr,
        x=x_arr,
        ax=ax,
        shaded_region=shaded_region,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_axis_type=plot_axis_type,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
