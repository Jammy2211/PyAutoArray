from typing import Optional

from autoconf import conf
from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class AbstractLabel(AbstractMatWrap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manual_label = self.kwargs.get("label")


class YLabel(AbstractLabel):
    @property
    def defaults(self):
        try:
            fontsize = conf.instance["visualize"]["general"]["mat_plot"]["ylabel"]["fontsize"]
        except Exception:
            fontsize = 16
        return {"fontsize": fontsize, "ylabel": ""}

    def set(self, auto_label: Optional[str] = None):
        import matplotlib.pyplot as plt

        config_dict = self.config_dict

        if self.manual_label is not None:
            config_dict.pop("ylabel", None)
            plt.ylabel(ylabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            config_dict.pop("ylabel", None)
            plt.ylabel(ylabel=auto_label, **config_dict)
        else:
            plt.ylabel(**config_dict)


class XLabel(AbstractLabel):
    @property
    def defaults(self):
        try:
            fontsize = conf.instance["visualize"]["general"]["mat_plot"]["xlabel"]["fontsize"]
        except Exception:
            fontsize = 16
        return {"fontsize": fontsize, "xlabel": ""}

    def set(self, auto_label: Optional[str] = None):
        import matplotlib.pyplot as plt

        config_dict = self.config_dict

        if self.manual_label is not None:
            config_dict.pop("xlabel", None)
            plt.xlabel(xlabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            config_dict.pop("xlabel", None)
            plt.xlabel(xlabel=auto_label, **config_dict)
        else:
            plt.xlabel(**config_dict)
