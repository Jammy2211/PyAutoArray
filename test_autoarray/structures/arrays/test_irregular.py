from os import path

import numpy as np
import autoarray as aa

test_values_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__input_as_list__convert_correctly():
    values = aa.ArrayIrregular(values=[1.0, -1.0])

    assert type(values) == aa.ArrayIrregular
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_list == [1.0, -1.0]
