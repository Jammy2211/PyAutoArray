from typing import Callable, Optional

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization


class Pixelization:
    def __init__(
        self,
        mesh: AbstractMesh,
        regularization: Optional[AbstractRegularization] = None,
    ):
        """
        Pairs a 2D grid of (y,x) coordinates with a 2D mesh, which can be combined with a ``Regularization``
        scheme in order to reconstruct data via an ``Inversion``.

        **Image-Plane Example**

        For the simplest case, we may have a 2D dataset (e.g. an image) whose pixel centres correspond to a
        (y,x) grid of Cartesian coordinates, which are paired with a ``Pixelization``'s mesh.

        The visual below illustrates this, showing:

        - Left: Observed image of a galaxy
        - Centre: The (y,x) grid of coordinates corresponding to the centre of each pixel in the observed image. The
        centre of each pixel is shown by a magenta point.
        - Right: An overlaid ``Rectangular`` ``mesh``, where the square pixel boundaries of this mesh are shown by
        dashed black lines.

        The red points highlight a subset of points, and will be used through this documentation to illustrate certain
        behaviour of ``Pixelization``'s.

        .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_image_plane/data_image_plane.png?raw=true
          :width: 200
          :alt: Alternative text

        .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_image_plane/grid_image_plane.png?raw=true
          :width: 200
          :alt: Alternative text

          .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_image_plane/image_plane_mesh.png?raw=true
          :width: 200
          :alt: Alternative text

        Comparison of the central and right panels therefore show the core functionality of a ``Pixelization`` --
        it represents the mappings between a (y,x) grid of coordinates (in the example above the observed image's
        grid) and a (y,x) mesh (in the example above the ``Rectangular`` mesh shown by dashed black lines).


        **Image-Plane Example (Masked)**

        The mappings above are shown for 2D data which has not been masked.

        We shown below how a ``Pixelization`` treats an image which has had a 2.5" circular maskd applied to it:

        .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_masked_image_plane/data_image_plane.png?raw=true
          :width: 200
          :alt: Alternative text

        .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_masked_image_plane/grid_image_plane.png?raw=true
          :width: 200
          :alt: Alternative text

          .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_masked_image_plane/image_plane_mesh.png?raw=true
          :width: 200
          :alt: Alternative text

        The behaviour is analogous to the non-masked case, however only unmasked pixel's in the image's (y,x) grid
        of coordinates are paired with the mesh.


        **Source-Plane Example (Masked)**

        In **PyAutoGalaxy** the above two cases are representative of how ``Pixelization`` objects are used.

        In the strong lensing package **PyAutoLens**, gravitational lensing deflects the observed image's (y,x)
        grid of coordinates, such that the ``mesh`` is overlaid in the source-plane:

        .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_masked_source_plane/data_image_plane.png?raw=true
          :width: 200
          :alt: Alternative text

        .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_masked_source_plane/grid_image_plane.png?raw=true
          :width: 200
          :alt: Alternative text

          .. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/docs/api/images/pixelization_masked_source_plane/source_plane_mesh.png?raw=true
          :width: 200
          :alt: Alternative text

        The red points, highlighting through all visuals above, now show how after gravitational lensing the points
        change position from the image-plane to source-plane.


        **Pixelization Uses**

        The following objects / packages are used with ``Pixelizations``:

        - ``Mapper``s: Computes the mappings between the the data's (y,x) grid and the mesh's pixels.
        - ``Inversion``s: Use the ``Pixelization`` to reconstruct the data on the mesh via linear algebra.
        - ``Regularization``: Apply smoothing to the solutions computed using a ``Pixelization`` and ``Inversion``.

        In the example above, the ``mesh`` uses the ``Rectangular`` object, but other meshes are available (e.g.
        ``Delaunay``, ``Voronoi``).

        ** Source Code API**

        The ``Pixelization`` API uses the following terms for the grids shown above:

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh).
        - ``image_plane_mesh_grid``: The (y,x) centers of the mesh pixels in the image-plane.
        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane after gravitational lensing.
        - ``source_plane_mesh_grid``: The (y,x) centers of the mesh pixels mapped to the source-plane after
        gravitational lensing.

        If a transformation of coordinates is not applied (e.g. no gravitational lensing), the `image_plane`
        and `source_plane` grid are identical (this is the case in **PyAutoGalaxy**).

        Parameters
        ----------
        mesh
            The mesh object (e.g. Rectangular grid of pixels, Delaunay triangulation, Voronoi mesh) describing the
            pixels of the `Pixelization`.
        regularization
            The regularization object that can smooth ``Pixelization`` pixels with one another when it is used to
            reconstruct data via an `Inversion`.

        Examples
        --------
        import autogalaxy as ag

        grid_2d = al.Grid2D.uniform(
            shape_native=(50, 50),
            pixel_scales=0.1
        )

        mesh = al.mesh.Rectangular(shape=(10, 10))

        pixelization = al.Pixelization(mesh=mesh)

        Examples (Modeling)
        -------------------
        import autofit as af
        import autogalaxy as ag

        mesh = af.Model(ag.mesh.Rectangular)
        mesh.shape_0 = af.UniformPrior(lower_limit=10, upper_limit=20)
        mesh.shape_1 = af.UniformPrior(lower_limit=10, upper_limit=20)

        pixelization = af.Model(
            ag.Pixelization,
            mesh=mesh
            regularization=ag.reg.Constant
        )

        galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

        model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
        """
        self.mesh = mesh
        self.regularization = regularization

    @property
    def mapper_grids_from(self) -> Callable:
        return self.mesh.mapper_grids_from

    def __repr__(self):

        string = "{}\n{}".format(self.__class__.__name__, str(self.mesh))
        if self.regularization is not None:
            string += "{}\n{}".format(self.__class__.__name__, str(self.regularization))

        return string
