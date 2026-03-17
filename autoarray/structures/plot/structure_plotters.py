import numpy as np
from typing import List, Optional, Union

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D


# ---------------------------------------------------------------------------
# Shared helpers (no Visuals dependency)
# ---------------------------------------------------------------------------

def _auto_mask_edge(array) -> Optional[np.ndarray]:
    """Return edge-pixel (y, x) coords from array.mask, or None."""
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except AttributeError:
        pass
    return None


def _zoom_array(array):
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


def _output_for_mat_plot(mat_plot, is_for_subplot: bool, auto_filename: str):
    """Derive (output_path, output_filename, output_format) from a MatPlot object."""
    if is_for_subplot:
        return None, auto_filename, "png"

    output = mat_plot.output
    fmt_list = output.format_list
    fmt = fmt_list[0] if fmt_list else "show"

    filename = output.filename_from(auto_filename)

    if fmt == "show":
        return None, filename, "png"

    path = output.output_path_from(fmt)
    return path, filename, fmt


def _numpy_grid(grid) -> Optional[np.ndarray]:
    """Convert a grid-like object to a plain (N,2) numpy array, or None."""
    if grid is None:
        return None
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def _numpy_lines(lines) -> Optional[List[np.ndarray]]:
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


def _numpy_positions(positions) -> Optional[List[np.ndarray]]:
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


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

class Array2DPlotter(AbstractPlotter):
    def __init__(
        self,
        array: Array2D,
        mat_plot_2d: MatPlot2D = None,
        origin=None,
        border=None,
        grid=None,
        mesh_grid=None,
        positions=None,
        lines=None,
        vectors=None,
        patches=None,
        fill_region=None,
        array_overlay=None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)
        self.array = array
        self.origin = origin
        self.border = border
        self.grid = grid
        self.mesh_grid = mesh_grid
        self.positions = positions
        self.lines = lines
        self.vectors = vectors
        self.patches = patches
        self.fill_region = fill_region
        self.array_overlay = array_overlay

    def figure_2d(self):
        if self.array is None or np.all(self.array == 0):
            return

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(self.mat_plot_2d, is_sub, "array")

        array = _zoom_array(self.array)

        plot_array(
            array=array.native.array,
            ax=ax,
            extent=array.geometry.extent,
            mask=_auto_mask_edge(array),
            border=_numpy_grid(self.border),
            origin=_numpy_grid(self.origin),
            grid=_numpy_grid(self.grid),
            mesh_grid=_numpy_grid(self.mesh_grid),
            positions=_numpy_positions(self.positions),
            lines=_numpy_lines(self.lines),
            array_overlay=self.array_overlay.native.array if self.array_overlay is not None else None,
            patches=self.patches,
            fill_region=self.fill_region,
            title="Array2D",
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )


class Grid2DPlotter(AbstractPlotter):
    def __init__(
        self,
        grid: Grid2D,
        mat_plot_2d: MatPlot2D = None,
        lines=None,
        positions=None,
        indexes=None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)
        self.grid = grid
        self.lines = lines
        self.positions = positions
        self.indexes = indexes

    def figure_2d(
        self,
        color_array: np.ndarray = None,
        plot_grid_lines: bool = False,
        plot_over_sampled_grid: bool = False,
    ):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(self.mat_plot_2d, is_sub, "grid")

        grid_plot = self.grid.over_sampled if plot_over_sampled_grid else self.grid

        plot_grid(
            grid=np.array(grid_plot.array),
            ax=ax,
            lines=_numpy_lines(self.lines),
            color_array=color_array,
            indexes=self.indexes,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )


class YX1DPlotter(AbstractPlotter):
    def __init__(
        self,
        y: Union[Array1D, List],
        x: Optional[Union[Array1D, Grid1D, List]] = None,
        mat_plot_1d: MatPlot1D = None,
        shaded_region=None,
        vertical_line: Optional[float] = None,
        points=None,
        should_plot_grid: bool = False,
        should_plot_zero: bool = False,
        plot_axis_type: Optional[str] = None,
        plot_yx_dict=None,
        auto_labels=AutoLabels(),
    ):
        if isinstance(y, list):
            y = Array1D.no_mask(values=y, pixel_scales=1.0)
        if isinstance(x, list):
            x = Array1D.no_mask(values=x, pixel_scales=1.0)

        super().__init__(mat_plot_1d=mat_plot_1d)

        self.y = y
        self.x = y.grid_radial if x is None else x
        self.shaded_region = shaded_region
        self.vertical_line = vertical_line
        self.points = points
        self.should_plot_grid = should_plot_grid
        self.should_plot_zero = should_plot_zero
        self.plot_axis_type = plot_axis_type
        self.plot_yx_dict = plot_yx_dict or {}
        self.auto_labels = auto_labels

    def figure_1d(self):
        y_arr = self.y.array if hasattr(self.y, "array") else np.array(self.y)
        x_arr = self.x.array if hasattr(self.x, "array") else np.array(self.x)

        is_sub = self.mat_plot_1d.is_for_subplot
        ax = self.mat_plot_1d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_1d, is_sub, self.auto_labels.filename or "yx"
        )

        plot_yx(
            y=y_arr,
            x=x_arr,
            ax=ax,
            shaded_region=self.shaded_region,
            title=self.auto_labels.title or "",
            xlabel=self.auto_labels.xlabel or "",
            ylabel=self.auto_labels.ylabel or "",
            plot_axis_type=self.plot_axis_type or "linear",
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )
