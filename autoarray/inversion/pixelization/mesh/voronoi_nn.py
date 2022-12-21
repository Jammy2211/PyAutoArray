from autoarray.inversion.pixelization.mesh.voronoi import VoronoiMagnification
from autoarray.inversion.pixelization.mesh.voronoi import VoronoiBrightnessImage


class VoronoiNNMagnification(VoronoiMagnification):
    """
    An irregular mesh of Voronoi pixels, which using natural neighbor interpolation are paired with a 2D grid of (y,x)
    coordinates. The Voronoi cell centers are derived in the image-plane by overlaying a uniform
    grid over the image.

    For a full description of how a mesh is paired with another grid,
    see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

    The Voronoi mesh represents pixels as an irregular 2D grid of Voronoi cells.

    A ``Pixelization`` using a ``Voronoi`` mesh has four grids associated with it:

    - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
      the source-plane).
    - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the centres of Voronoi
      cells in the source-plane).
    - ``source_plane_data_grid``: The observed data grid mapped to the source-plane (e.g. after gravitational lensing).
    - ``source_plane_mesh_grid``: The centre of each Voronoi cell in the source-plane
      (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

    Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with all Voronoi cells it falls within,
    using a natural neighbor interpolation scheme (https://en.wikipedia.org/wiki/Natural_neighbor_interpolation).

    The centers of the Voronoi cell pixels are derived in the image-plane by overlaying a uniform grid with the
    input ``shape`` over the masked image data's grid. All coordinates in this uniform grid which are contained
    within the mask are kept and mapped to the source-plane via gravitational lensing, where they form
    the Voronoi pixel centers.

    Parameters
    ----------
    shape
        The shape of the unmasked uniform grid in the image-plane which is laid over the masked image, in
        order to derive the image-plane (y,x) coordinates which act as the centres of the Voronoi pixels after
        being mapped to the source-plane via gravitational lensing.
    """
    @property
    def uses_interpolation(self):
        return True


class VoronoiNNBrightnessImage(VoronoiBrightnessImage):
    """
    An irregular mesh of Voronoi pixels, which using natural neighbor interpolation are paired with a 2D grid of (y,x)
    coordinates. The Voronoi cell centers are derived in the image-plane by applying a KMeans
    clustering algorithm to the image's weight map.

    For a full description of how a mesh is paired with another grid,
    see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

    The Voronoi mesh represents pixels as an irregular 2D grid of Voronoi cells.

    A ``Pixelization`` using a ``Voronoi`` mesh has four grids associated with it:

    - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
      the source-plane).
    - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the centres of Voronoi
      cells in the source-plane).
    - ``source_plane_data_grid``: The observed data grid mapped to the source-plane (e.g. after gravitational lensing).
    - ``source_plane_mesh_grid``: The centre of each Voronoi cell in the source-plane
      (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

    Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with all Voronoi cells it falls within,
    using a natural neighbor interpolation scheme (https://en.wikipedia.org/wiki/Natural_neighbor_interpolation).

    The centers of the Voronoi cell pixels are derived in the image plane, by applying a KMeans clustering algorithm
    to the masked image data's weight-map. The ``weight_floor`` and ``weight_power`` allow the KMeans algorithm to
    adapt the image-plane coordinates to the image's brightest or faintest values. The computed valies are
    mapped to the source-plane  via gravitational lensing, where they form the Voronoi cell pixel centers.

    Parameters
    ----------
    pixels
        The total number of pixels in the mesh, which is therefore also the number of (y,x) coordinates computed
        via the KMeans clustering algorithm in image-plane.
    weight_floor
        A parameter which reweights the data values the KMeans algorithm is applied too; as the floor increases
        more weight is applied to values with lower values thus allowing mesh pixels to be placed in these
        regions of the data.
    weight_power
        A parameter which reweights the data values the KMeans algorithm is applied too; as the power increases
        more weight is applied to values with higher values thus allowing mesh pixels to be placed in these
        regions of the data.
    """

    @property
    def uses_interpolation(self):
        return True

    @property
    def is_stochastic(self) -> bool:
        return True
