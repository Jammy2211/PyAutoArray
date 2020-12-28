from autoarray.mask import mask_2d
from autoarray.structures import arrays, grids, lines as l, vector_fields
from autoarray.plot.plotter import plotter as p, include as inc
from matplotlib import patches as ptch
import typing
from functools import wraps


class Visuals:
    def __init__(
        self,
        mask: mask_2d.Mask2D = None,
        lines: typing.List[l.Line] = None,
        positions: grids.GridIrregular = None,
        grid: grids.Grid = None,
        vector_field: vector_fields.VectorFieldIrregular = None,
        patches: typing.Union[ptch.Patch] = None,
        array_overlay: arrays.Array = None,
        origin: grids.Grid = None,
        border: grids.Grid = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        self.mask = mask
        self.lines = lines
        self.positions = positions
        self.grid = grid
        self.vector_field = vector_field
        self.patches = patches
        self.array_overlay = array_overlay
        self.origin = origin
        self.border = border
        self.parallel_overscan = parallel_overscan
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan

    def __add__(self, other):

        for attr, value in other.__dict__.items():
            if self.__dict__[attr] is None and other.__dict__[attr] is not None:
                self.__dict__[attr] = other.__dict__[attr]

        return self

    @property
    def plotter(self):
        return p.Plotter()

    @property
    def include(self):
        return inc.Include()

    def plot_via_plotter(self, plotter):

        if self.origin is not None:
            plotter.origin_scatter.scatter_grid(grid=self.origin)

        if self.mask is not None:
            plotter.mask_scatter.scatter_grid(
                grid=self.mask.geometry.edge_grid_sub_1.in_1d_binned
            )

        if self.border is not None:
            plotter.border_scatter.scatter_grid(grid=self.border)

        if self.grid is not None:
            plotter.grid_scatter.scatter_grid(grid=self.grid)

        if self.positions is not None:
            plotter.positions_scatter.scatter_grid_grouped(grid_grouped=self.positions)

        if self.vector_field is not None:
            plotter.vector_field_quiver.quiver_vector_field(
                vector_field=self.vector_field
            )

        if self.patches is not None:
            plotter.patch_overlay.overlay_patches(patches=self.patches)

        if self.lines is not None:
            plotter.line_plot.plot_grid_grouped(grid_grouped=self.lines)


def visuals_key_from_dictionary(dictionary):

    visuals_key = None

    for key, value in dictionary.items():
        if isinstance(value, Visuals):
            visuals_key = key

    return visuals_key


def set_plot_defaults(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        visuals_key = visuals_key_from_dictionary(dictionary=kwargs)

        if visuals_key is not None:
            visuals = kwargs[visuals_key]
        else:
            visuals = Visuals()
            visuals_key = "visuals"

        kwargs[visuals_key] = visuals

        include_key = inc.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = visuals.include
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = p.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = visuals.plotter
            plotter_key = "plotter"

        kwargs[plotter_key] = plotter

        print(include)

        return func(*args, **kwargs)

    return wrapper
