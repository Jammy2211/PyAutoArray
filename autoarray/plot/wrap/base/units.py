import logging
from typing import Optional

from autoconf import conf

logger = logging.getLogger(__name__)


class Units:
    def __init__(
        self,
        use_scaled: Optional[bool] = None,
        use_raw: Optional[bool] = False,
        ticks_convert_factor: Optional[float] = None,
        ticks_label: Optional[str] = None,
        colorbar_convert_factor: Optional[float] = None,
        colorbar_label: Optional[str] = None,
        **kwargs
    ):
        """
        This object controls the units of a plotted figure, and performs multiple tasks when making the plot:

        1: Species the units of the plot (e.g. meters, kilometers) and contains a conversion factor which converts
        the plotted data from its current units (e.g. meters) to the units plotted (e.g. kilometeters). Pixel units
        can be used if `use_scaled=False`.

        2: Uses the conversion above to manually override the yticks and xticks of the figure, so it appears in the
        converted units.

        3: Sets the ylabel and xlabel to include a string containing the units.

        Parameters
        ----------
        use_scaled
            If True, plot the 2D data with y and x ticks corresponding to its scaled
            coordinates (its `pixel_scales` attribute is used as the `ticks_convert_factor`). If `False` plot them in
            pixel units.
        ticks_convert_factor
            If plotting the labels in scaled units, this factor multiplies the values that are used for the labels.
            This allows for additional unit conversions of the figure labels.
        """

        self.ticks_convert_factor = ticks_convert_factor
        self.ticks_label = ticks_label

        if use_scaled is not None:
            self.use_scaled = use_scaled
        else:
            try:
                self.use_scaled = conf.instance["visualize"]["general"]["units"][
                    "use_scaled"
                ]
            except KeyError:
                self.use_scaled = True

        self.use_raw = use_raw

        self.colorbar_convert_factor = colorbar_convert_factor
        self.colorbar_label = colorbar_label

        self.kwargs = kwargs
