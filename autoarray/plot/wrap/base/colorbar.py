import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


class Colorbar:
    def __init__(
        self,
        fraction: float = 0.047,
        pad: float = 0.01,
        manual_tick_values: Optional[List[float]] = None,
        manual_tick_labels: Optional[List[str]] = None,
        **kwargs,
    ):
        self.fraction = fraction
        self.pad = pad
        self.manual_tick_values = manual_tick_values
        self.manual_tick_labels = manual_tick_labels

    def set(self, ax=None, norm=None):
        cb = plt.colorbar(ax=ax, fraction=self.fraction, pad=self.pad)
        if self.manual_tick_values is not None:
            cb.set_ticks(self.manual_tick_values)
        if self.manual_tick_labels is not None:
            cb.set_ticklabels(self.manual_tick_labels)
        return cb

    def set_with_color_values(self, cmap, color_values: np.ndarray, ax=None, norm=None):
        import matplotlib.cm as cm

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(color_values)

        cb = plt.colorbar(mappable=mappable, ax=ax, fraction=self.fraction, pad=self.pad)
        if self.manual_tick_values is not None:
            cb.set_ticks(self.manual_tick_values)
        if self.manual_tick_labels is not None:
            cb.set_ticklabels(self.manual_tick_labels)
        return cb
