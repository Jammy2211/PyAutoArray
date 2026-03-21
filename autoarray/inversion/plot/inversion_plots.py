import logging
import numpy as np
from typing import Optional

import matplotlib.pyplot as plt
from autoconf import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.utils import (
    auto_mask_edge,
    numpy_grid,
    numpy_lines,
    numpy_positions,
    subplot_save,
)
from autoarray.inversion.plot.mapper_plots import plot_mapper
from autoarray.structures.arrays.uniform_2d import Array2D

logger = logging.getLogger(__name__)


def _plot_array(array, ax, title, colormap, use_log10, grid=None, positions=None, lines=None):
    try:
        arr = array.native.array
        extent = array.geometry.extent
        mask_overlay = auto_mask_edge(array)
    except AttributeError:
        arr = np.asarray(array)
        extent = None
        mask_overlay = None

    plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=mask_overlay,
        grid=numpy_grid(grid),
        positions=numpy_positions(positions),
        lines=numpy_lines(lines),
        title=title,
        colormap=colormap,
        use_log10=use_log10,
    )


def _plot_source(
    inversion,
    mapper,
    pixel_values,
    ax,
    title,
    filename,
    colormap,
    use_log10,
    zoom_to_brightest,
    mesh_grid,
    lines,
):
    """Plot source reconstruction via ``plot_mapper``."""
    try:
        plot_mapper(
            mapper=mapper,
            solution_vector=pixel_values,
            ax=ax,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=zoom_to_brightest,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (ValueError, TypeError, Exception) as exc:
        logger.info(f"Could not plot source {filename}: {exc}")


def subplot_of_mapper(
    inversion,
    mapper_index: int = 0,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_inversion",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
    grid=None,
    positions=None,
):
    """
    3×4 subplot showing all pixelization diagnostics for one mapper.

    Parameters
    ----------
    inversion
        An ``AbstractInversion`` instance.
    mapper_index
        Which mapper in the inversion to visualise.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename prefix (``_{mapper_index}`` is appended).
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    mesh_grid, lines, grid, positions
        Optional overlays.
    """
    mapper = inversion.cls_list_from(cls=Mapper)[mapper_index]
    effective_mesh_grid = mesh_grid

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes = axes.flatten()

    # panel 0: data subtracted
    try:
        array = inversion.data_subtracted_dict[mapper]
        _plot_array(array, axes[0], "Data Subtracted", colormap, use_log10, grid=grid, positions=positions, lines=lines)
    except (AttributeError, KeyError):
        pass

    # panel 1: reconstructed operated data
    try:
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities
        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        _plot_array(array, axes[1], "Reconstructed Image", colormap, use_log10, grid=grid, positions=positions, lines=lines)
    except (AttributeError, KeyError):
        pass

    # panel 2: reconstructed operated data (log10)
    try:
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities
        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        _plot_array(array, axes[2], "Reconstructed Image (log10)", colormap, True, grid=grid, positions=positions, lines=lines)
    except (AttributeError, KeyError):
        pass

    # panel 3: reconstructed operated data + mesh grid overlay
    try:
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities
        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        _plot_array(array, axes[3], "Mesh Pixel Grid Overlaid", colormap, use_log10,
                    grid=numpy_grid(mapper.image_plane_mesh_grid), positions=positions, lines=lines)
    except (AttributeError, KeyError):
        pass

    # reconstruction cmap vmax from config
    vmax_cmap = None
    try:
        factor = conf.instance["visualize"]["general"]["inversion"]["reconstruction_vmax_factor"]
        vmax_cmap = factor * np.max(inversion.reconstruction)
    except Exception:
        pass

    # panel 4: source reconstruction (zoomed)
    pixel_values = inversion.reconstruction_dict[mapper]
    _plot_source(inversion, mapper, pixel_values, axes[4], "Source Reconstruction", "reconstruction",
                 colormap, use_log10, True, effective_mesh_grid, lines)

    # panel 5: source reconstruction (unzoomed)
    _plot_source(inversion, mapper, pixel_values, axes[5], "Source Reconstruction (Unzoomed)", "reconstruction_unzoomed",
                 colormap, use_log10, False, effective_mesh_grid, lines)

    # panel 6: noise map (unzoomed)
    try:
        nm = inversion.reconstruction_noise_map_dict[mapper]
        _plot_source(inversion, mapper, nm, axes[6], "Noise-Map (Unzoomed)", "reconstruction_noise_map",
                     colormap, use_log10, False, effective_mesh_grid, lines)
    except (KeyError, TypeError):
        pass

    # panel 7: regularization weights (unzoomed)
    try:
        rw = inversion.regularization_weights_mapper_dict[mapper]
        _plot_source(inversion, mapper, rw, axes[7], "Regularization Weights (Unzoomed)", "regularization_weights",
                     colormap, use_log10, False, effective_mesh_grid, lines)
    except (IndexError, ValueError, KeyError, TypeError):
        pass

    # panel 8: sub pixels per image pixels
    try:
        sub_size = Array2D(
            values=mapper.over_sampler.sub_size,
            mask=inversion.dataset.mask,
        )
        _plot_array(sub_size, axes[8], "Sub Pixels Per Image Pixels", colormap, use_log10)
    except Exception:
        pass

    # panel 9: mesh pixels per image pixels
    try:
        mesh_arr = mapper.mesh_pixels_per_image_pixels
        _plot_array(mesh_arr, axes[9], "Mesh Pixels Per Image Pixels", colormap, use_log10)
    except Exception:
        pass

    # panel 10: image pixels per mesh pixel
    try:
        pw = mapper.data_weight_total_for_pix_from()
        _plot_source(inversion, mapper, pw, axes[10], "Image Pixels Per Source Pixel", "image_pixels_per_mesh_pixel",
                     colormap, use_log10, True, effective_mesh_grid, lines)
    except (TypeError, Exception):
        pass

    plt.tight_layout()
    subplot_save(fig, output_path, f"{output_filename}_{mapper_index}", output_format)


def subplot_mappings(
    inversion,
    pixelization_index: int = 0,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_mappings",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
    grid=None,
    positions=None,
):
    """
    2×2 subplot showing data, model image, reconstruction and unzoomed reconstruction.

    Parameters
    ----------
    inversion
        An ``AbstractInversion`` instance.
    pixelization_index
        Which mapper in the inversion to visualise.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename prefix (``_{pixelization_index}`` is appended).
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    mesh_grid, lines, grid, positions
        Optional overlays.
    """
    mapper = inversion.cls_list_from(cls=Mapper)[pixelization_index]

    try:
        total_pixels = conf.instance["visualize"]["general"]["inversion"]["total_mappings_pixels"]
    except Exception:
        total_pixels = 10

    pix_indexes = inversion.max_pixel_list_from(
        total_pixels=total_pixels,
        filter_neighbors=True,
        mapper_index=pixelization_index,
    )
    indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    # panel 0: data subtracted
    try:
        array = inversion.data_subtracted_dict[mapper]
        _plot_array(array, axes[0], "Data Subtracted", colormap, use_log10, grid=grid, positions=positions, lines=lines)
    except (AttributeError, KeyError):
        pass

    # panel 1: reconstructed operated data
    try:
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities
        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        _plot_array(array, axes[1], "Reconstructed Image", colormap, use_log10, grid=grid, positions=positions, lines=lines)
    except (AttributeError, KeyError):
        pass

    # panel 2: source reconstruction (zoomed)
    pixel_values = inversion.reconstruction_dict[mapper]
    _plot_source(inversion, mapper, pixel_values, axes[2], "Source Reconstruction", "reconstruction",
                 colormap, use_log10, True, mesh_grid, lines)

    # panel 3: source reconstruction (unzoomed)
    _plot_source(inversion, mapper, pixel_values, axes[3], "Source Reconstruction (Unzoomed)", "reconstruction_unzoomed",
                 colormap, use_log10, False, mesh_grid, lines)

    plt.tight_layout()
    subplot_save(fig, output_path, f"{output_filename}_{pixelization_index}", output_format)
