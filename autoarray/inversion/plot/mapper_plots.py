import logging
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.array import plot_array
from autoarray.plot.inversion import plot_inversion_reconstruction
from autoarray.plot.utils import numpy_grid, numpy_lines, subplot_save

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

    plot_array(
        image,
        ax=axes[0],
        title="Image (Image-Plane)",
        colormap=colormap,
        use_log10=use_log10,
        lines=lines,
    )
    plot_mapper(
        mapper,
        colormap=colormap,
        use_log10=use_log10,
        mesh_grid=mesh_grid,
        lines=lines,
        ax=axes[1],
    )

    plt.tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
