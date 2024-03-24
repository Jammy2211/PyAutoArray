import numpy as np
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

    @property
    def log10_min_value(self):
        return conf.instance["visualize"]["general"]["general"]["log10_min_value"]

    @property
    def log10_max_value(self):
        return float(
            conf.instance["visualize"]["general"]["general"]["log10_max_value"]
        )

    def vmin_from(self, array: np.ndarray, use_log10: bool = False) -> float:
        """
        The vmin of a plot, for example the minimum value of the colormap and colorbar.

        If the vmin is manually input by the user, this value is used. Otherwise, the minimum value of the data being
        plotted is used, which is computed via nanmin to ensure that NaN entries in the data are ignored.

        If use_log10 is True, the minimum value of the colormap is the log10 of the minimum value of the data. To
        ensure negative values are not plotted, which often causes matplotlib errors, the minimum value of the colormap
        is rounded up to the log10_min_value attribute of the config file.

        Parameters
        ----------
        array
            The array of data which is to be plotted.
        use_log10
            If True, the minimum value of the colormap is the log10 of the minimum value of the data.

        Returns
        -------
        The minimum value of the colormap.
        """
        if self.config_dict["norm"] in "log":
            use_log10 = True

        if self.config_dict["vmin"] is None:
            vmin = np.nanmin(array)
        else:
            vmin = self.config_dict["vmin"]

        if use_log10 and (vmin < self.log10_min_value):
            vmin = self.log10_min_value

        return vmin

    def vmax_from(self, array: np.ndarray, use_log10: bool = False) -> float:
        """
        The vmax of a plot, for example the maximum value of the colormap and colorbar.

        If the vmax is manually input by the user, this value is used. Otherwise, the maximum value of the data being
        plotted is used, which is computed via nanmax to ensure that NaN entries in the data are ignored.

        If use_log10 is True, the maximum value of the colormap is the log10 of the maximum value of the data. To
        ensure values above the log10_max_value attribute of the config file are not plotted, this value is used
        as the maximum value of the colormap.

        Parameters
        ----------
        array
            The array of data which is to be plotted.
        use_log10
            If True, the maximum value of the colormap is the log10 of the maximum value of the data.

        Returns
        -------
        The maximum value of the colormap.
        """
        if self.config_dict["norm"] in "log":
            use_log10 = True

        if self.config_dict["vmax"] is None:
            vmax = np.nanmax(array)
        else:
            vmax = self.config_dict["vmax"]

        if use_log10 and (vmax > self.log10_max_value):
            vmax = self.log10_max_value

        return vmax
