from typing import Optional

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization


class Pixelization:
    def __init__(
        self,
        mesh: AbstractMesh,
        regularization: Optional[AbstractRegularization] = None,
    ):

        self.mesh = mesh
        self.regularization = regularization

    def __repr__(self):

        string = "{}\n{}".format(self.__class__.__name__, str(self.mesh))
        if self.regularization is not None:
            string += "{}\n{}".format(self.__class__.__name__, str(self.regularization))

        return string
