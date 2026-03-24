"""
Standalone functions for plotting inversion / pixelization reconstructions.

Replaces the inversion-specific paths in ``MatPlot2D.plot_mapper``.
"""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from autoarray.plot.utils import apply_extent, conf_figsize, save_figure


def plot_inversion_reconstruction(
    pixel_values: np.ndarray,
    mapper,
    ax: Optional[plt.Axes] = None,
    # --- cosmetics --------------------------------------------------------------
    title: str = "Reconstruction",
    xlabel: str = 'x (")',
    ylabel: str = 'y (")',
    colormap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_log10: bool = False,
    zoom_to_brightest: bool = True,
    # --- overlays ---------------------------------------------------------------
    lines: Optional[List[np.ndarray]] = None,
    grid: Optional[np.ndarray] = None,
    # --- figure control (used only when ax is None) -----------------------------
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    output_filename: str = "reconstruction",
    output_format: str = "png",
) -> None:
    """
    Plot an inversion reconstruction using the appropriate mapper type.

    Chooses between rectangular (``imshow``/``pcolormesh``) and Delaunay
    (``tripcolor``) rendering based on the mapper's interpolator type.

    Parameters
    ----------
    pixel_values
        1D array of reconstructed flux values, one per source pixel.
    mapper
        Autoarray mapper object exposing ``interpolator``, ``mesh_geometry``,
        ``source_plane_mesh_grid``, etc.
    ax
        Existing ``Axes``.  ``None`` creates a new figure.
    title, xlabel, ylabel
        Text labels.
    colormap
        Matplotlib colormap name.
    vmin, vmax
        Explicit colour scale limits.
    use_log10
        Apply ``LogNorm``.
    zoom_to_brightest
        Pass through to ``mapper.extent_from``.
    lines
        Line overlays (e.g. critical curves).
    grid
        Scatter overlay (e.g. data-plane grid).
    figsize, output_path, output_filename, output_format
        Figure output controls.
    """
    from autoarray.inversion.mesh.interpolator.rectangular import (
        InterpolatorRectangular,
    )
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )
    from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
    from autoarray.inversion.mesh.interpolator.knn import InterpolatorKNearestNeighbor

    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- colour normalisation --------------------------------------------------
    if use_log10:
        norm = LogNorm(vmin=vmin or 1e-4, vmax=vmax)
    elif vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    extent = mapper.extent_from(values=pixel_values, zoom_to_brightest=zoom_to_brightest)

    if isinstance(mapper.interpolator, (InterpolatorRectangular, InterpolatorRectangularUniform)):
        _plot_rectangular(ax, pixel_values, mapper, norm, colormap, extent)
    elif isinstance(mapper.interpolator, (InterpolatorDelaunay, InterpolatorKNearestNeighbor)):
        _plot_delaunay(ax, pixel_values, mapper, norm, colormap)

    # --- overlays --------------------------------------------------------------
    if lines is not None:
        for line in lines:
            if line is not None and len(line) > 0:
                ax.plot(line[:, 1], line[:, 0], linewidth=2)

    if grid is not None:
        ax.scatter(grid[:, 1], grid[:, 0], s=1, c="w", alpha=0.5)

    apply_extent(ax, extent)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)

    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
        )


def _plot_rectangular(ax, pixel_values, mapper, norm, colormap, extent):
    """Render a rectangular mesh reconstruction with pcolormesh or imshow."""
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )
    import numpy as np

    shape_native = mapper.mesh_geometry.shape

    if pixel_values is None:
        pixel_values = np.zeros(shape_native[0] * shape_native[1])

    if isinstance(mapper.interpolator, InterpolatorRectangularUniform):
        from autoarray.structures.arrays.uniform_2d import Array2D
        from autoarray.structures.arrays import array_2d_util

        solution_array_2d = array_2d_util.array_2d_native_from(
            array_2d_slim=pixel_values,
            mask_2d=np.full(fill_value=False, shape=shape_native),
        )
        pix_array = Array2D.no_mask(
            values=solution_array_2d,
            pixel_scales=mapper.mesh_geometry.pixel_scales,
            origin=mapper.mesh_geometry.origin,
        )
        ax.imshow(
            pix_array.native.array,
            cmap=colormap,
            norm=norm,
            extent=pix_array.geometry.extent,
            aspect="auto",
            origin="upper",
        )
    else:
        y_edges, x_edges = mapper.mesh_geometry.edges_transformed.T
        Y, X = np.meshgrid(y_edges, x_edges, indexing="ij")
        im = ax.pcolormesh(
            X, Y,
            pixel_values.reshape(shape_native),
            shading="flat",
            norm=norm,
            cmap=colormap,
        )
        plt.colorbar(im, ax=ax)


def _plot_delaunay(ax, pixel_values, mapper, norm, colormap):
    """Render a Delaunay mesh reconstruction with tripcolor."""
    mesh_grid = mapper.source_plane_mesh_grid
    x = mesh_grid[:, 1]
    y = mesh_grid[:, 0]

    if hasattr(pixel_values, "array"):
        vals = pixel_values.array
    else:
        vals = pixel_values

    tc = ax.tripcolor(x, y, vals, cmap=colormap, norm=norm, shading="gouraud")
    plt.colorbar(tc, ax=ax)
