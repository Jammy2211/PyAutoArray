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
        A `Pixelization` groups a `Mesh` and `Regularization` into a single object, such that the regularization is
        applied to pixels of the paired mesh.

        However, the `Pixelization` object is not used explicitly in PyAutoArray.

        Its primary purpose is in PyAutoGalaxy / PyAutoLens, where it allows a `Mesh` and `Regularization` to be
        grouped together as a single model-component in a model-fit performed via PyAutoFit. This allows the user
        to be certain that the regularization is specifically applied to the mesh it is paired with.

        When this model-component is created, it is unpacked into a `Mesh` and `Regularization` when passed to the
        PyAutoArray `inversion` package.

        Parameters
        ----------
        mesh
            The mesh object (e.g. Delaunay triangulation, Voronoi mesh) describing the pixels of the `Pixelization`.
        regularization
            The regularization object that describes how pixelization pixels are smoothed with one another
            when the `Pixelizaiton` is used to reconstruct data via an `Inversion`.
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
