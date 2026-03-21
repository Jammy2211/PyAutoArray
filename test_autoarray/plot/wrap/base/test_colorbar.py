import autoarray.plot as aplt

import matplotlib.pyplot as plt
import numpy as np


def test__colorbar_defaults():
    colorbar = aplt.Colorbar()

    assert colorbar.fraction == 0.047
    assert colorbar.manual_tick_values is None
    assert colorbar.manual_tick_labels is None

    colorbar = aplt.Colorbar(
        manual_tick_values=(1.0, 2.0), manual_tick_labels=("a", "b")
    )

    assert colorbar.manual_tick_values == (1.0, 2.0)
    assert colorbar.manual_tick_labels == ("a", "b")

    colorbar = aplt.Colorbar(fraction=6.0)

    assert colorbar.fraction == 6.0


def test__plot__works_for_reasonable_range_of_values():
    fig, ax = plt.subplots()
    im = ax.imshow(np.ones((2, 2)))
    cb = aplt.Colorbar(fraction=0.047, pad=0.05)
    # pass the mappable explicitly so colorbar can find it
    plt.colorbar(im, ax=ax, fraction=0.047, pad=0.05)
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(np.ones((2, 2)))
    cb = aplt.Colorbar(
        fraction=0.1,
        pad=0.5,
        manual_tick_values=[0.25, 0.5, 0.75],
        manual_tick_labels=["lo", "mid", "hi"],
    )
    cb.set_with_color_values(
        cmap=aplt.Cmap().cmap, color_values=np.array([1.0, 2.0, 3.0]), ax=ax
    )
    plt.close()
