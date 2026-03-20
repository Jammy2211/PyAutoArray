import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.wrap.base.colorbar import Colorbar


def _facecolors_from(values, simplices):
    facecolors = np.zeros(shape=simplices.shape[0])
    for i in range(simplices.shape[0]):
        facecolors[i] = np.sum(1.0 / 3.0 * values[simplices[i, :]])
    return facecolors


class DelaunayDrawer:
    def __init__(
        self,
        alpha: float = 0.7,
        edgecolor: str = "k",
        linewidth: float = 0.0,
        **kwargs,
    ):
        self.alpha = alpha
        self.edgecolor = edgecolor
        self.linewidth = linewidth

    def draw_delaunay_pixels(
        self,
        mapper,
        pixel_values: Optional[np.ndarray],
        cmap: Optional[Cmap],
        colorbar: Optional[Colorbar] = None,
        ax=None,
        use_log10: bool = False,
    ):
        if pixel_values is None:
            pixel_values = np.zeros(shape=mapper.source_plane_mesh_grid.shape[0])

        pixel_values = np.asarray(pixel_values)

        if ax is None:
            ax = plt.gca()

        if cmap is None:
            cmap = Cmap()

        source_pixelization_grid = mapper.source_plane_mesh_grid
        simplices = np.asarray(mapper.interpolator.delaunay.simplices)

        # Remove JAX-padded -1 values
        valid_mask = np.all(simplices >= 0, axis=1)
        simplices = simplices[valid_mask]

        facecolors = _facecolors_from(values=pixel_values, simplices=simplices)

        norm = cmap.norm_from(array=pixel_values, use_log10=use_log10)

        if use_log10:
            pixel_values = np.where(pixel_values < 1e-4, 1e-4, pixel_values)
            pixel_values = np.log10(pixel_values)

        vmin = cmap.vmin_from(array=pixel_values, use_log10=use_log10)
        vmax = cmap.vmax_from(array=pixel_values, use_log10=use_log10)

        color_values = np.clip(pixel_values, vmin, vmax)

        cmap_obj = plt.get_cmap(cmap.cmap) if not callable(cmap.cmap) else cmap.cmap

        if colorbar is not None:
            cb = colorbar.set_with_color_values(
                norm=norm,
                cmap=cmap_obj,
                color_values=color_values,
                ax=ax,
            )

        ax.tripcolor(
            source_pixelization_grid.array[:, 1],
            source_pixelization_grid.array[:, 0],
            simplices,
            facecolors=facecolors,
            edgecolors="None",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            alpha=self.alpha,
            linewidth=self.linewidth,
        )
