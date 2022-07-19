import numpy as np


class LinearObjReg:
    def __init__(self, linear_obj, regularization):

        self.linear_obj = linear_obj
        self.regularization = regularization

    @property
    def regularization_matrix(self):

        if self.regularization is None:
            return np.zeros((self.pixels, self.pixels))

        return self.regularization.regularization_matrix_from(mapper=self.linear_obj)

    @property
    def regularization_weights(self):

        if self.regularization is None:
            return np.zero((self.pixels,))

        return self.regularization.regularization_weights_from(mapper=self.linear_obj)

    @property
    def pixels(self):
        return self.linear_obj.pixels
