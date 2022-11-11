from matplotlib import patches as ptch
from typing import Union

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D


class PatchOverlay(AbstractMatWrap2D):
    """
    Adds patches to a plotted figure using matplotlib `patches` objects.

    The coordinate system of each `Patch` uses that of the figure, which is typically set up using the plotted
    data structure. This makes it straight forward to add patches in specific locations.

    This object wraps methods described in below:

    https://matplotlib.org/3.3.2/api/collections_api.html
    """

    def overlay_patches(self, patches: Union[ptch.Patch]):
        """
        Overlay a list of patches on a figure, for example an `Ellipse`.
        `
        Parameters
        ----------
        patches : [Patch]
            The patches that are laid over the figure.
        """

        # patch_collection = PatchCollection(patches=patches, **self.config_dict)
        # plt.gcf().gca().add_collection(patch_collection)