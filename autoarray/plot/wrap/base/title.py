from autoconf import conf
from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Title(AbstractMatWrap):
    def __init__(self, prefix: str = None, disable_log10_label: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.prefix = prefix
        self.disable_log10_label = disable_log10_label
        self.manual_label = self.kwargs.get("label")

    @property
    def defaults(self):
        try:
            fontsize = conf.instance["visualize"]["general"]["mat_plot"]["title"]["fontsize"]
        except Exception:
            fontsize = 24
        return {"fontsize": fontsize}

    def set(self, auto_title=None, use_log10: bool = False):
        import matplotlib.pyplot as plt

        config_dict = self.config_dict

        label = auto_title if self.manual_label is None else self.manual_label

        if self.prefix is not None:
            label = f"{self.prefix} {label}"

        if use_log10 and not self.disable_log10_label:
            label = f"{label} (log10)"

        if "label" in config_dict:
            config_dict.pop("label")

        plt.title(label=label, **config_dict)
