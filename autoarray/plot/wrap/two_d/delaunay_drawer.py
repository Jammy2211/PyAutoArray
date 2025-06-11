import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.plot.wrap.base.units import Units
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi

from autoarray.plot.wrap import base as wb


def facecolors_from(values, simplices):
    facecolors = np.zeros(shape=simplices.shape[0])
    for i in range(simplices.shape[0]):
        facecolors[i] = np.sum(1.0 / 3.0 * values[simplices[i, :]])

    return facecolors


class DelaunayDrawer(AbstractMatWrap2D):
    """
    Draws Voronoi pixels from a `MapperVoronoi` object (see `inversions.mapper`). This includes both drawing
    each Voronoi cell and coloring it according to a color value.

    The mapper contains the grid of (y,x) coordinate where the centre of each Voronoi cell is plotted.

    This object wraps methods described in below:

    https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill.html
    """

    def draw_delaunay_pixels(
        self,
        mapper: MapperVoronoi,
        pixel_values: Optional[np.ndarray],
        units: Units,
        cmap: Optional[wb.Cmap],
        colorbar: Optional[wb.Colorbar],
        colorbar_tickparams: Optional[wb.ColorbarTickParams] = None,
        ax=None,
        use_log10: bool = False,
    ):
        """
        Draws the Voronoi pixels of the input `mapper` using its `mesh_grid` which contains the (y,x)
        coordinate of the centre of every Voronoi cell. This uses the method `plt.fill`.

        Parameters
        ----------
        mapper
            A mapper object which contains the Voronoi mesh.
        pixel_values
            An array used to compute the color values that every Voronoi cell is plotted using.
        cmap
            The colormap used to plot each Voronoi cell.
        colorbar
            The `Colorbar` object in `mat_base` used to set the colorbar of the figure the Voronoi mesh is plotted on.
        colorbar_tickparams
            The `ColorbarTickParams` object in `mat_base` used to set the tick labels of the colorbar.
        ax
            The matplotlib axis the Voronoi mesh is plotted on.
        use_log10
            If `True`, the colorbar is plotted using a log10 scale.
        """

        if pixel_values is None:
            raise ValueError(
                "pixel_values input to DelaunayPlotter are None and thus cannot be plotted."
            )

        if ax is None:
            ax = plt.gca()

        source_pixelization_grid = mapper.mapper_grids.source_plane_mesh_grid

        simplices = mapper.delaunay.simplices

        facecolors = facecolors_from(values=pixel_values, simplices=simplices)

        norm = cmap.norm_from(array=pixel_values, use_log10=use_log10)

        if use_log10:
            pixel_values[pixel_values < 1e-4] = 1e-4
            pixel_values = np.log10(pixel_values)

        vmin = cmap.vmin_from(array=pixel_values, use_log10=use_log10)
        vmax = cmap.vmax_from(array=pixel_values, use_log10=use_log10)

        color_values = np.where(pixel_values > vmax, vmax, pixel_values)
        color_values = np.where(pixel_values < vmin, vmin, color_values)

        cmap = plt.get_cmap(cmap.cmap)

        if colorbar is not None:
            cb = colorbar.set_with_color_values(
                units=units,
                norm=norm,
                cmap=cmap,
                color_values=color_values,
                ax=ax,
                use_log10=use_log10,
            )

            if cb is not None and colorbar_tickparams is not None:
                colorbar_tickparams.set(cb=cb)

        ax.tripcolor(
            source_pixelization_grid[:, 1],
            source_pixelization_grid[:, 0],
            simplices,
            facecolors=facecolors,
            edgecolors="None",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **self.config_dict,
        )
