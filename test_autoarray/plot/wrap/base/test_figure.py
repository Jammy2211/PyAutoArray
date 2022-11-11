import autoarray.plot as aplt

from os import path

import matplotlib.pyplot as plt


def test__loads_values_from_config_if_not_manually_input():

    figure = aplt.Figure()

    assert figure.config_dict["figsize"] == (7, 7)
    assert figure.config_dict["aspect"] == "square"

    figure = aplt.Figure(aspect="auto")

    assert figure.config_dict["figsize"] == (7, 7)
    assert figure.config_dict["aspect"] == "auto"

    figure = aplt.Figure()
    figure.is_for_subplot = True

    assert figure.config_dict["figsize"] == None
    assert figure.config_dict["aspect"] == "square"

    figure = aplt.Figure(figsize=(6, 6))
    figure.is_for_subplot = True

    assert figure.config_dict["figsize"] == (6, 6)
    assert figure.config_dict["aspect"] == "square"


def test__aspect_from():

    figure = aplt.Figure(aspect="auto")

    aspect = figure.aspect_from(shape_native=(2, 2))

    assert aspect == "auto"

    figure = aplt.Figure(aspect="square")

    aspect = figure.aspect_from(shape_native=(2, 2))

    assert aspect == 1.0

    aspect = figure.aspect_from(shape_native=(4, 2))

    assert aspect == 0.5


def test__open_and_close__open_and_close_figures_correct():

    figure = aplt.Figure()

    figure.open()

    assert plt.fignum_exists(num=1) is True

    figure.close()

    assert plt.fignum_exists(num=1) is False
