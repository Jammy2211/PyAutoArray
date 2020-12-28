from autoarray.plot.plotter import visuals as vis


def test__add_visuals_together__replaces_nones():

    visuals_1 = vis.Visuals(mask=1)
    visuals_0 = vis.Visuals(border=10)

    visuals_0 += visuals_1

    assert visuals_0.mask == 1
    assert visuals_0.border == 10

    visuals_0 = vis.Visuals(mask=1)
    visuals_1 = vis.Visuals(mask=2)

    visuals_0 += visuals_1

    assert visuals_0.mask == 1
    assert visuals_0.border == None
