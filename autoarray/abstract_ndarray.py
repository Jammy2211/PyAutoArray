import numpy as np


class AbstractNDArray(np.ndarray):
    def __reduce__(self):

        pickled_state = super().__reduce__()

        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)

        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super().__setstate__(state[0:-1])
