import numpy as np

from autoconf import conf


def set_backend():
    """
    The matplotlib backend used by default is the default matplotlib backend on a user's computer.

    The backend can be customized via the `config.visualize.general.yaml` config file.
    """
    import matplotlib

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

        Each subclass wraps a specific matplotlib function and provides sensible defaults.
        Defaults can be overridden by passing keyword arguments to the constructor, or by
        editing the `mat_plot` section of `config/visualize/general.yaml` for the six
        user-configurable wrappers: Figure, YTicks, XTicks, Title, YLabel, XLabel.

        Example
        -------
        Customize a plotter::

            plotter = aplt.Array2DPlotter(
                array=array,
                output=aplt.Output(path="/path/to/output", format="png"),
                cmap=aplt.Cmap(cmap="hot"),
            )
        """
        self.kwargs = kwargs

    @property
    def defaults(self):
        """Hardcoded default kwargs for this wrapper. Subclasses override this."""
        return {}

    @property
    def config_dict(self):
        """Merge hardcoded defaults with any user-supplied kwargs."""
        config_dict = {**self.defaults, **self.kwargs}

        if "c" in config_dict:
            c = config_dict["c"]
            if isinstance(c, str) and "," in c:
                config_dict["c"] = remove_spaces_and_commas_from(c.split(","))

        if "c" in config_dict and config_dict["c"] is None:
            config_dict.pop("c")

        if "is_default" in config_dict:
            config_dict.pop("is_default")

        return config_dict

    @property
    def log10_min_value(self):
        return conf.instance["visualize"]["general"]["general"]["log10_min_value"]

    @property
    def log10_max_value(self):
        return float(
            conf.instance["visualize"]["general"]["general"]["log10_max_value"]
        )
