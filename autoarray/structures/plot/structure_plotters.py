import numpy as np
from typing import List, Optional, Union

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.visuals.two_d import Visuals2D
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
# Helpers to extract plain numpy overlay data from Visuals2D/Visuals1D
# ---------------------------------------------------------------------------

def _lines_from_visuals(visuals_2d: Visuals2D) -> Optional[List[np.ndarray]]:
    """Return a list of (N, 2) numpy arrays from visuals_2d.lines."""
    if visuals_2d is None or visuals_2d.lines is None:
        return None
    lines = visuals_2d.lines
    result = []
    try:
        # Grid2DIrregular or list of array-like objects
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


def _positions_from_visuals(visuals_2d: Visuals2D) -> Optional[List[np.ndarray]]:
    """Return a list of (N, 2) numpy arrays from visuals_2d.positions."""
    if visuals_2d is None or visuals_2d.positions is None:
        return None
    positions = visuals_2d.positions
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
                arr = np.array(p.array if hasattr(p, "array") else p)
                result.append(arr)
            except Exception:
                pass
        return result or None
    return None


def _mask_edge_from(array: Array2D, visuals_2d: Optional[Visuals2D]) -> Optional[np.ndarray]:
    """Return edge-pixel coordinates to scatter as mask overlay."""
    if visuals_2d is not None and visuals_2d.mask is not None:
        try:
            return np.array(visuals_2d.mask.derive_grid.edge.array)
        except Exception:
            pass
    if array is not None and not array.mask.is_all_false:
        try:
            return np.array(array.mask.derive_grid.edge.array)
        except Exception:
            pass
    return None


def _grid_from_visuals(visuals_2d: Visuals2D) -> Optional[np.ndarray]:
    """Return grid scatter coordinates from visuals_2d.grid."""
    if visuals_2d is None or visuals_2d.grid is None:
        return None
    grid = visuals_2d.grid
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def _zoom_array(array):
    """
    Apply zoom_around_mask to *array* if the config requests it.

    Mirrors the behaviour of the old ``MatPlot2D.plot_array`` which read
    ``visualize/general.yaml::zoom_around_mask`` and, when True, trimmed the
    array to the bounding box of the unmasked region plus a 1-pixel buffer.
    Returns the (possibly trimmed) array unchanged when the config is False or
    the mask has no masked pixels.
    """
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
    """
    Derive (output_path, output_filename, output_format) from a MatPlot object.

    When in subplot mode, returns output_path=None so that plot_array does not
    save — the subplot is saved later by close_subplot_figure().
    """
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


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

class Array2DPlotter(AbstractPlotter):
    def __init__(
        self,
        array: Array2D,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        super().__init__(visuals_2d=visuals_2d, mat_plot_2d=mat_plot_2d)
        self.array = array

    def figure_2d(self):
        """Plot the array as a 2D image."""
        if self.array is None or np.all(self.array == 0):
            return

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None

        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub, "array"
        )

        array = _zoom_array(self.array)

        plot_array(
            array=array.native.array,
            ax=ax,
            extent=array.geometry.extent,
            mask=_mask_edge_from(array, self.visuals_2d),
            grid=_grid_from_visuals(self.visuals_2d),
            positions=_positions_from_visuals(self.visuals_2d),
            lines=_lines_from_visuals(self.visuals_2d),
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
        visuals_2d: Visuals2D = None,
    ):
        super().__init__(visuals_2d=visuals_2d, mat_plot_2d=mat_plot_2d)
        self.grid = grid

    def figure_2d(
        self,
        color_array: np.ndarray = None,
        plot_grid_lines: bool = False,
        plot_over_sampled_grid: bool = False,
    ):
        """Plot the grid as a 2D scatter."""
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None

        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub, "grid"
        )

        grid_plot = self.grid.over_sampled if plot_over_sampled_grid else self.grid

        plot_grid(
            grid=np.array(grid_plot.array),
            ax=ax,
            lines=_lines_from_visuals(self.visuals_2d),
            color_array=color_array,
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
        visuals_1d: Visuals1D = None,
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

        super().__init__(visuals_1d=visuals_1d, mat_plot_1d=mat_plot_1d)

        self.y = y
        self.x = y.grid_radial if x is None else x
        self.should_plot_grid = should_plot_grid
        self.should_plot_zero = should_plot_zero
        self.plot_axis_type = plot_axis_type
        self.plot_yx_dict = plot_yx_dict or {}
        self.auto_labels = auto_labels

    def figure_1d(self):
        """Plot the y and x values as a 1D line."""
        y_arr = self.y.array if hasattr(self.y, "array") else np.array(self.y)
        x_arr = self.x.array if hasattr(self.x, "array") else np.array(self.x)

        is_sub = self.mat_plot_1d.is_for_subplot
        ax = self.mat_plot_1d.setup_subplot() if is_sub else None

        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_1d, is_sub, self.auto_labels.filename or "yx"
        )

        shaded = None
        if self.visuals_1d is not None and self.visuals_1d.shaded_region is not None:
            shaded = self.visuals_1d.shaded_region

        plot_yx(
            y=y_arr,
            x=x_arr,
            ax=ax,
            shaded_region=shaded,
            title=self.auto_labels.title or "",
            xlabel=self.auto_labels.xlabel or "",
            ylabel=self.auto_labels.ylabel or "",
            plot_axis_type=self.plot_axis_type or "linear",
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )
