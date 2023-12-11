import matplotlib

from autoconf import conf


def set_backend():
    """
    The matplotlib end used by default is the default matplotlib backend on a user's computer.

    The backend can be customized via the `config.visualize.general.ini` config file, if a user needs to overwrite
    the backend for visualization to work.

    This has been the case in order to circumvent compatibility issues with MACs.

    It is also common for high perforamcne computers (HPCs) to not support visualization and raise an error when
    a graphical backend (e.g. TKAgg) is used. Setting the backend to `Agg` addresses this.
    """
    backend = conf.get_matplotlib_backend()

    if backend not in "default":
        matplotlib.use(backend)

    try:
        hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
    except KeyError:
        hpc_mode = False

    if hpc_mode:
        matplotlib.use("Agg")


def remove_spaces_and_commas_from(colors):
    colors = [color.strip(",").strip(" ") for color in colors]
    colors = list(filter(None, colors))
    if len(colors) == 1:
        return colors[0]
    return colors


class AbstractMatWrap:
    def __init__(self, **kwargs):
        """
        An abstract base class for wrapping matplotlib plotting methods.

        Classes are used to wrap matplotlib so that the data structures in the `autoarray.structures` package can be
        plotted in standardized withs. This exploits how these structures have specific formats, units, properties etc.
        This allows us to make a simple API for plotting structures, for example to plot an `Array2D` structure:

        import autoarray as aa
        import autoarray.plot as aplt

        arr = aa.Array2D.no_mask(values=[[1.0, 1.0], [2.0, 2.0]], pixel_scales=2.0)
        aplt.Array2D(values=arr)

        The wrapped Mat objects make it simple to customize how matplotlib visualizes this data structure, for example
        we can customize the figure size and colormap using the `Figure` and `Cmap` objects.

        figure = aplt.Figure(figsize=(7,7), aspect="square")
        cmap = aplt.Cmap(cmap="jet", vmin=1.0, vmax=2.0)

        plotter = aplt.MatPlot2D(figure=figure, cmap=cmap)

        aplt.Array2D(values=arr, plotter=plotter)

        The `Plotter` object is detailed in the `autoarray.plot.plotter` package.

        The matplotlib wrapper objects in ths module also use configuration files to choose their default settings.
        For example, in `autoarray.config.visualize.mat_base.Figure.ini` you will note the section:

        figure:
        figsize=(7, 7)

        subplot:
        figsize=auto

        This specifies that when a data structure (like the `Array2D` above) is plotted, the figsize will always
        be  (7,7) when a single figure is plotted and it will be chosen automatically if a subplot is plotted. This
        allows one to customize the matplotlib settings of every plot in a project.
        """

        self.is_for_subplot = False
        self.kwargs = kwargs

    @property
    def config_dict(self):
        config_dict = conf.instance["visualize"][self.config_folder][
            self.__class__.__name__
        ][self.config_category]

        if "c" in config_dict:
            config_dict["c"] = remove_spaces_and_commas_from(colors=config_dict["c"])

        config_dict = {**config_dict, **self.kwargs}

        if "c" in config_dict:
            if config_dict["c"] is None:
                config_dict.pop("c")

        if "is_default" in config_dict:
            config_dict.pop("is_default")

        return config_dict

    @property
    def config_folder(self):
        return "mat_wrap"

    @property
    def config_category(self):
        if self.is_for_subplot:
            return "subplot"
        return "figure"
