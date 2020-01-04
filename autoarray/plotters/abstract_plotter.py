from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np
from autoarray import exc

from autoarray import conf

def setting(section, name, python_type):
    return conf.instance.visualize.get(section, name, python_type)


def load_setting(value, name, python_type):
    return value if value is not None else setting(section="settings", name=name, python_type=python_type)


def load_include(value, name, python_type):
    return value if value is not None else setting(section="include", name=name, python_type=python_type)



class AbstractPlotter(object):

    def __init__(self,
    use_scaled_units=None,
    unit_conversion_factor=None,
    unit_label=None,
    figsize=None,
    aspect=None,
    cmap=None,
    norm=None,
    norm_min=None,
    norm_max=None,
    linthresh=None,
    linscale=None,
    cb_ticksize=None,
    cb_fraction=None,
    cb_pad=None,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=None,
    xlabelsize=None,
    ylabelsize=None,
    xyticksize=None,
    mask_pointsize=None,
    border_pointsize=None,
    point_pointsize=None,
    grid_pointsize=None,
     include_origin=None,
     include_mask=None,
     include_border=None,
     include_points=None,
     label_title=None, label_yunits=None, label_xunits=None, label_yticks=None, label_xticks=None,
                 output_path=None,
                 output_format="show",
                 output_filename=None
                 ):

        self.use_scaled_units = load_setting(value=use_scaled_units, name="use_scaled_units", python_type=bool)
        self.unit_conversion_factor = unit_conversion_factor
        self.unit_label = load_setting(value=unit_label, name="unit_label", python_type=str)
        self.figsize = load_setting(value=figsize, name="figsize", python_type=str)
        if isinstance(self.figsize, str):
            self.figsize = tuple(map(int, self.figsize[1:-1].split(',')))
        self.aspect = load_setting(value=aspect, name="aspect", python_type=str)
        self.cmap = load_setting(value=cmap, name="cmap", python_type=str)
        self.norm = load_setting(value=norm, name="norm", python_type=str)
        self.norm_min = load_setting(value=norm_min, name="norm_min", python_type=float)
        self.norm_max = load_setting(value=norm_max, name="norm_max", python_type=float)
        self.linthresh = load_setting(value=linthresh, name="linthresh", python_type=float)
        self.linscale = load_setting(value=linscale, name="linscale", python_type=float)

        self.cb_ticksize = load_setting(value=cb_ticksize, name="cb_ticksize", python_type=int)
        self.cb_fraction = load_setting(value=cb_fraction, name="cb_fraction", python_type=float)
        self.cb_pad = load_setting(value=cb_pad, name="cb_pad", python_type=float)
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.titlesize = load_setting(value=titlesize, name="titlesize", python_type=int)
        self.ylabelsize = load_setting(value=ylabelsize, name="ylabelsize", python_type=int)
        self.xlabelsize = load_setting(value=xlabelsize, name="xlabelsize", python_type=int)
        self.xyticksize = load_setting(value=xyticksize, name="xyticksize", python_type=int)
        self.mask_pointsize = load_setting(value=mask_pointsize, name="mask_pointsize", python_type=int)
        self.border_pointsize = load_setting(value=border_pointsize, name="border_pointsize", python_type=int)
        self.point_pointsize = load_setting(value=point_pointsize, name="point_pointsize", python_type=int)
        self.grid_pointsize = load_setting(value=grid_pointsize, name="grid_pointsize", python_type=int)

        self.include_origin = load_include(value=include_origin, name="origin", python_type=bool)
        self.include_mask = load_include(value=include_mask, name="mask", python_type=bool)
        self.include_border = load_include(value=include_border, name="border", python_type=bool)
        self.include_points = load_include(value=include_points, name="points", python_type=bool)

        self.label_title = label_title
        self.label_yunits = label_yunits
        self.label_xunits = label_xunits
        self.label_yticks = label_yticks
        self.label_xticks = label_xticks

        self.output_path = output_path
        self.output_format = output_format
        self.output_filename = output_filename

        self.as_subplot = False

    def setup_figure(self):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (rows, columns).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output_subplot_array* function.
        """
        if not self.as_subplot:
            fig = plt.figure(figsize=self.figsize)
            return fig

    def set_title(self):
        """Set the title and title size of the figure.

        Parameters
        -----------
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        """
        plt.title(label=self.label_title, fontsize=self.titlesize)

    @staticmethod
    def get_subplot_rows_columns_figsize(number_subplots):
        """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """
        if number_subplots <= 2:
            return 1, 2, (18, 8)
        elif number_subplots <= 4:
            return 2, 2, (13, 10)
        elif number_subplots <= 6:
            return 2, 3, (18, 12)
        elif number_subplots <= 9:
            return 3, 3, (25, 20)
        elif number_subplots <= 12:
            return 3, 4, (25, 20)
        elif number_subplots <= 16:
            return 4, 4, (25, 20)
        elif number_subplots <= 20:
            return 4, 5, (25, 20)
        else:
            return 6, 6, (25, 20)