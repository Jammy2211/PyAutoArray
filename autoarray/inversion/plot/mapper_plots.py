import logging
import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.inversion import plot_inversion_reconstruction
from autoarray.plot.plots.utils import (
    auto_mask_edge,
    numpy_grid,
    numpy_lines,
    numpy_positions,
    subplot_save,
)

logger = logging.getLogger(__name__)


def plot_mapper(
    mapper,
    solution_vector=None,
    output_path: Optional[str] = None,
    output_filename: str = "mapper",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
    title: str = "Pixelization Mesh (Source-Plane)",
    zoom_to_brightest: bool = True,
    ax=None,
):
    """
    Plot a pixelization mapper reconstruction.

    Parameters
    ----------
    mapper
        A ``Mapper`` instance.
    solution_vector
        Per-pixel flux values.  ``None`` uses uniform colours.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    mesh_grid
        Mesh grid to overlay as scatter points.
    lines
        Lines to overlay.
    title
        Figure title.
    zoom_to_brightest
        Zoom the source plane to the brightest region.
    ax
        Existing ``Axes`` to draw onto.
    """
    try:
        plot_inversion_reconstruction(
            pixel_values=solution_vector,
            mapper=mapper,
            ax=ax,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=zoom_to_brightest,
            lines=numpy_lines(lines),
            grid=numpy_grid(mesh_grid),
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
        )
    except Exception as exc:
        logger.info(f"Could not plot the source-plane via the Mapper: {exc}")


def plot_mapper_image(
    image,
    output_path: Optional[str] = None,
    output_filename: str = "mapper_image",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    lines=None,
    title: str = "Image (Image-Plane)",
    ax=None,
):
    """
    Plot the image-plane image associated with a mapper.

    Parameters
    ----------
    image
        An ``Array2D`` instance or plain 2D numpy array.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    lines
        Lines to overlay.
    title
        Figure title.
    ax
        Existing ``Axes`` to draw onto.
    """
    try:
        arr = image.native.array
        extent = image.geometry.extent
    except AttributeError:
        arr = np.asarray(image)
        extent = None

    plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=auto_mask_edge(image) if hasattr(image, "mask") else None,
        lines=numpy_lines(lines),
        title=title,
        colormap=colormap,
        use_log10=use_log10,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def subplot_image_and_mapper(
    mapper,
    image,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_image_and_mapper",
    output_format: str = "png",
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
):
    """
    1×2 subplot: image-plane image (left) and pixelization mesh (right).

    Parameters
    ----------
    mapper
        A ``Mapper`` instance.
    image
        An ``Array2D`` instance to show in the image plane.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    mesh_grid
        Mesh grid to overlay on the reconstruction panel.
    lines
        Lines to overlay on both panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_mapper_image(image, colormap=colormap, use_log10=use_log10, lines=lines, ax=axes[0])
    plot_mapper(mapper, colormap=colormap, use_log10=use_log10, mesh_grid=mesh_grid, lines=lines, ax=axes[1])

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
