import autoarray.plot as aplt


def test__plot_y_vs_x__works_for_reasonable_values():
    fill_between = aplt.FillBetween()

    fill_between.fill_between_shaded_regions(
        x=[1, 2, 3], y1=[1.0, 2.0, 3.0], y2=[2.0, 3.0, 4.0]
    )
