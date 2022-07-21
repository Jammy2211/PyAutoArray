from autoarray.inversion.regularization.abstract import AbstractRegularization


class MockRegularization(AbstractRegularization):
    def __init__(self, regularization_matrix=None):

        super().__init__()

        self.regularization_matrix = regularization_matrix

    def regularization_matrix_via_pixel_neighbors_from(
        self, pixel_neighbors, pixel_neighbors_sizes
    ):
        return self.regularization_matrix

    def regularization_matrix_from(self, linear_obj):

        return self.regularization_matrix
