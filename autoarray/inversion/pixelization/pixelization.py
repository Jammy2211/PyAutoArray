from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization


class Pixelization:
    def __init__(self, mesh: AbstractMesh, regularization: AbstractRegularization):

        self.mesh = mesh
        self.regularization = regularization
