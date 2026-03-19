from enum import Enum
import gc
from typing import Union, Tuple

from autoconf import conf
from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Aspect(Enum):
    square = 1
    auto = 2
    equal = 3


class Figure(AbstractMatWrap):
    """
    Sets up the Matplotlib figure before plotting.

    This object wraps the following Matplotlib methods:

    - plt.figure: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
    - plt.close: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.close.html

    The figure size can be configured in `config/visualize/general.yaml` under `mat_plot.figure.figsize`,
    or overridden per-plot via ``Figure(figsize=(width, height))``.
    """

    @property
    def defaults(self):
        try:
            figsize = conf.instance["visualize"]["general"]["mat_plot"]["figure"]["figsize"]
            if isinstance(figsize, str):
                figsize = tuple(map(int, figsize[1:-1].split(",")))
        except Exception:
            figsize = (7, 7)
        return {"figsize": figsize, "aspect": "square"}

    @property
    def config_dict(self):
        config_dict = super().config_dict

        if config_dict.get("figsize") == "auto":
            config_dict["figsize"] = None
        elif isinstance(config_dict.get("figsize"), str):
            config_dict["figsize"] = tuple(
                map(int, config_dict["figsize"][1:-1].split(","))
            )

        return config_dict

    def aspect_for_subplot_from(self, extent):
        ratio = float((extent[1] - extent[0]) / (extent[3] - extent[2]))

        aspect = Aspect[self.config_dict["aspect"]]

        if aspect == Aspect.square:
            return ratio
        elif aspect == Aspect.auto:
            return 1.0 / ratio
        elif aspect == Aspect.equal:
            return 1.0

        raise ValueError(
            f"""
            The `aspect` variable used to set up the figure is {aspect}.

            This is not a valid value, which must be one of square / auto / equal.
            """
        )

    def aspect_from(self, shape_native: Union[Tuple[int, int]]) -> Union[float, str]:
        if isinstance(self.config_dict["aspect"], str):
            if self.config_dict["aspect"] in "square":
                return float(shape_native[1]) / float(shape_native[0])

        return self.config_dict["aspect"]

    def open(self):
        import matplotlib.pyplot as plt

        if not plt.fignum_exists(num=1):
            config_dict = self.config_dict
            config_dict.pop("aspect")
            fig = plt.figure(**config_dict)
            return fig, plt.gca()
        return None, None

    def close(self):
        import matplotlib.pyplot as plt

        plt.close()
        gc.collect()
