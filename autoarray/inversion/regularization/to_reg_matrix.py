import numpy as np


class LinearObjToRegMatrix:
    def __init__(self, regularization, linear_obj):

        self.regularization = regularization
        self.linear_obj = linear_obj

    @property
    def regularization_matrix(self):

        if self.regularization is None:
            return np.zeros((self.pixels, self.pixels))

        return self.regularization.regularization_matrix_from(mapper=self.linear_obj)

    @property
    def pixels(self):
        return self.linear_obj.pixels
