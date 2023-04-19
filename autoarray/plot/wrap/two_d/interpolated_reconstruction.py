import matplotlib.pyplot as plt
import numpy as np
from typing import Union

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.plot.wrap.base.units import Units
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay

from autoarray.plot.wrap import base as wb


class InterpolatedReconstruction(AbstractMatWrap2D):
    """
    Given a `Mapper` and a corresponding array of `pixel_values` (e.g. the reconstruction values of a Delaunay
    triangulation) plot the values using `plt.imshow()`.

    The `pixel_values` are an ndarray of values which correspond to the irregular pixels of the mesh (e.g. for
    a Delaunay triangulation they are the connecting corners of each triangle or Voronoi mesh). This cannot be plotted
    with `imshow()`, therefore this class first converts the `pixel_values` from this irregular grid to a uniform 2D
    array of square pixels via interpolation.

    The interpolation routine depends on the `Mapper`, with most mappers having their own built-in interpolation
    routine specific to that pixelization's mesh.

    This object wraps methods described in below:

    - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
    """

    def imshow_reconstruction(
        self,
        mapper: Union[MapperDelaunay, MapperVoronoiNoInterp, MapperVoronoi],
        pixel_values: np.ndarray,
        units: Units,
        cmap: wb.Cmap,
        colorbar: wb.Colorbar,
        colorbar_tickparams: wb.ColorbarTickParams = None,
        aspect=None,
        ax=None,
    ):
        """
        Given a `Mapper` and a corresponding array of `pixel_values` (e.g. the reconstruction values of a Delaunay
        triangulation) plot the values using `plt.imshow()`.

        The `pixel_values` are an ndarray of values which correspond to the irregular pixels of the mesh (e.g. for
        a Delaunay triangulation they are the connecting corners of each triangle or Voronoi mesh). This cannot be plotted
        with `imshow()`, therefore this class first converts the `pixel_values` from this irregular grid to a uniform 2D
        array of square pixels via interpolation.

        The interpolation routine depends on the `Mapper`, with most mappers having their own built-in interpolation
        routine specific to that pixelization's mesh.

        This object wraps methods described in below:

        - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

        Parameters
        ----------
        mapper
            An object which contains a 2D mesh (e.g. Voronoi mesh cells) and defines how to
            interpolate values from the pixelization's mesh.
        pixel_values
            The pixel values of the pixelization's mesh (e.g. a Voronoi mesh) which are interpolated to a uniform square
            array for plotting with `imshow()`.
        cmap
            The colormap used by `imshow()` to plot the pixelization's mesh values.
        colorbar
            The `Colorbar` object in `mat_base` used to set the colorbar of the figure the interpolated pixelization's mesh
            values (e.g. values interpolated from the Voronoi mesh) are plotted on.
        colorbar_tickparams
            Controls the tick parameters of the colorbar.
        """

        if pixel_values is None:
            return

        vmin = cmap.vmin_from(array=pixel_values)
        vmax = cmap.vmax_from(array=pixel_values)

        color_values = np.where(pixel_values > vmax, vmax, pixel_values)
        color_values = np.where(pixel_values < vmin, vmin, color_values)

        cmap = plt.get_cmap(cmap.cmap)

        if colorbar is not None:

            colorbar = colorbar.set_with_color_values(
                units=units, cmap=cmap, color_values=color_values, ax=ax
            )
            if colorbar is not None and colorbar_tickparams is not None:
                colorbar_tickparams.set(cb=colorbar)

        interpolation_array = mapper.interpolated_array_from(values=pixel_values)

        plt.imshow(
            X=interpolation_array.native,
            cmap=cmap,
            extent=mapper.source_plane_mesh_grid.geometry.extent_square,
            aspect=aspect,
        )
