import csv
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from autoconf import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.array import plot_array
from autoarray.plot.utils import numpy_grid, numpy_lines, numpy_positions, subplot_save, hide_unused_axes
from autoarray.inversion.plot.mapper_plots import plot_mapper
from autoarray.structures.arrays.uniform_2d import Array2D

logger = logging.getLogger(__name__)


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

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes = axes.flatten()

    # panel 0: data subtracted
    try:
        plot_array(
            inversion.data_subtracted_dict[mapper],
            ax=axes[0],
            title="Data Subtracted",
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    # panels 1-3: reconstructed operated data (plain, log10, + mesh grid overlay)
    def _recon_array():
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities

        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        return array

    try:
        plot_array(
            _recon_array(),
            ax=axes[1],
            title="Reconstructed Image",
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
        plot_array(
            _recon_array(),
            ax=axes[2],
            title="Reconstructed Image (log10)",
            colormap=colormap,
            use_log10=True,
            grid=grid,
            positions=positions,
            lines=lines,
        )
        plot_array(
            _recon_array(),
            ax=axes[3],
            title="Mesh Pixel Grid Overlaid",
            colormap=colormap,
            use_log10=use_log10,
            grid=numpy_grid(mapper.image_plane_mesh_grid),
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    # panels 4-5: source reconstruction zoomed / unzoomed
    pixel_values = inversion.reconstruction_dict[mapper]
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[4],
        title="Source Reconstruction",
        colormap=colormap,
        use_log10=use_log10,
        zoom_to_brightest=True,
        mesh_grid=mesh_grid,
        lines=lines,
    )
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[5],
        title="Source Reconstruction (Unzoomed)",
        colormap=colormap,
        use_log10=use_log10,
        zoom_to_brightest=False,
        mesh_grid=mesh_grid,
        lines=lines,
    )

    # panel 6: noise map
    try:
        nm = inversion.reconstruction_noise_map_dict[mapper]
        plot_mapper(
            mapper,
            solution_vector=nm,
            ax=axes[6],
            title="Noise-Map (Unzoomed)",
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=False,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (KeyError, TypeError):
        pass

    # panel 7: regularization weights
    try:
        rw = inversion.regularization_weights_mapper_dict[mapper]
        plot_mapper(
            mapper,
            solution_vector=rw,
            ax=axes[7],
            title="Regularization Weights (Unzoomed)",
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=False,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (IndexError, ValueError, KeyError, TypeError):
        pass

    # panel 8: sub pixels per image pixels
    try:
        sub_size = Array2D(
            values=mapper.over_sampler.sub_size, mask=inversion.dataset.mask
        )
        plot_array(
            sub_size,
            ax=axes[8],
            title="Sub Pixels Per Image Pixels",
            colormap=colormap,
            use_log10=use_log10,
        )
    except Exception:
        pass

    # panel 9: mesh pixels per image pixels
    try:
        plot_array(
            mapper.mesh_pixels_per_image_pixels,
            ax=axes[9],
            title="Mesh Pixels Per Image Pixels",
            colormap=colormap,
            use_log10=use_log10,
        )
    except Exception:
        pass

    # panel 10: image pixels per mesh pixel
    try:
        pw = mapper.data_weight_total_for_pix_from()
        plot_mapper(
            mapper,
            solution_vector=pw,
            ax=axes[10],
            title="Image Pixels Per Source Pixel",
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=True,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (TypeError, Exception):
        pass

    hide_unused_axes(axes)
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
        total_pixels = conf.instance["visualize"]["general"]["inversion"][
            "total_mappings_pixels"
        ]
    except Exception:
        total_pixels = 10

    pix_indexes = inversion.max_pixel_list_from(
        total_pixels=total_pixels,
        filter_neighbors=True,
        mapper_index=pixelization_index,
    )
    mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    # panel 0: data subtracted
    try:
        plot_array(
            inversion.data_subtracted_dict[mapper],
            ax=axes[0],
            title="Data Subtracted",
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    # panel 1: reconstructed operated data
    try:
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities

        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        plot_array(
            array,
            ax=axes[1],
            title="Reconstructed Image",
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    pixel_values = inversion.reconstruction_dict[mapper]
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[2],
        title="Source Reconstruction",
        colormap=colormap,
        use_log10=use_log10,
        zoom_to_brightest=True,
        mesh_grid=mesh_grid,
        lines=lines,
    )
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[3],
        title="Source Reconstruction (Unzoomed)",
        colormap=colormap,
        use_log10=use_log10,
        zoom_to_brightest=False,
        mesh_grid=mesh_grid,
        lines=lines,
    )

    hide_unused_axes(axes)
    plt.tight_layout()
    subplot_save(
        fig, output_path, f"{output_filename}_{pixelization_index}", output_format
    )


def save_reconstruction_csv(
    inversion,
    output_path: Union[str, Path],
) -> None:
    """Write a CSV of each mapper's reconstruction and noise map to *output_path*.

    One file is written per mapper: ``source_plane_reconstruction_{i}.csv``,
    with columns ``y``, ``x``, ``reconstruction``, ``noise_map``.

    Parameters
    ----------
    inversion
        An ``AbstractInversion`` instance.
    output_path
        Directory in which to write the CSV files.
    """
    output_path = Path(output_path)
    mapper_list = inversion.cls_list_from(cls=Mapper)

    for i, mapper in enumerate(mapper_list):
        y = mapper.source_plane_mesh_grid[:, 0]
        x = mapper.source_plane_mesh_grid[:, 1]
        reconstruction = inversion.reconstruction_dict[mapper]
        noise_map = inversion.reconstruction_noise_map_dict[mapper]

        with open(output_path / f"source_plane_reconstruction_{i}.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y", "x", "reconstruction", "noise_map"])
            for j in range(len(x)):
                writer.writerow([float(y[j]), float(x[j]), float(reconstruction[j]), float(noise_map[j])])
