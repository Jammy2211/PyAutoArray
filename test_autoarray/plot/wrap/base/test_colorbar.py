import autoarray.plot as aplt

import matplotlib.pyplot as plt
import numpy as np


def test__loads_values_from_config_if_not_manually_input():

    colorbar = aplt.Colorbar()

    assert colorbar.config_dict["fraction"] == 3.0
    assert colorbar.manual_tick_values == None
    assert colorbar.manual_tick_labels == None

    colorbar = aplt.Colorbar(
        manual_tick_values=(1.0, 2.0), manual_tick_labels=(3.0, 4.0)
    )

    assert colorbar.manual_tick_values == (1.0, 2.0)
    assert colorbar.manual_tick_labels == (3.0, 4.0)

    colorbar = aplt.Colorbar()
    colorbar.is_for_subplot = True

    assert colorbar.config_dict["fraction"] == 0.1

    colorbar = aplt.Colorbar(fraction=6.0)
    colorbar.is_for_subplot = True

    assert colorbar.config_dict["fraction"] == 6.0


def test__plot__works_for_reasonable_range_of_values():

    figure = aplt.Figure()

    fig, ax = figure.open()
    plt.imshow(np.ones((2, 2)))
    cb = aplt.Colorbar(fraction=1.0, pad=2.0)
    cb.set(ax=ax, units=None)
    figure.close()

    fig, ax = figure.open()
    plt.imshow(np.ones((2, 2)))
    cb = aplt.Colorbar(
        fraction=0.1,
        pad=0.5,
        manual_tick_values=[0.25, 0.5, 0.75],
        manual_tick_labels=[1.0, 2.0, 3.0],
    )
    cb.set(ax=ax, units=None)
    figure.close()

    fig, ax = figure.open()
    plt.imshow(np.ones((2, 2)))
    cb = aplt.Colorbar(fraction=0.1, pad=0.5)
    cb.set_with_color_values(
        cmap=aplt.Cmap().cmap, color_values=[1.0, 2.0, 3.0], ax=ax, units=None
    )
    figure.close()
