import autoarray.plot as aplt


class TestAbstractVisuals:
    def test__add_visuals_together__replaces_nones(self):

        visuals_1 = aplt.Visuals2D(mask=1)
        visuals_0 = aplt.Visuals2D(border=10)

        visuals = visuals_0 + visuals_1

        assert visuals.mask == 1
        assert visuals.border == 10
        assert visuals_1.mask == 1
        assert visuals_1.border == 10
        assert visuals_0.border == 10
        assert visuals_0.mask == None

        visuals_0 = aplt.Visuals2D(mask=1)
        visuals_1 = aplt.Visuals2D(mask=2)

        visuals = visuals_1 + visuals_0

        assert visuals.mask == 1
        assert visuals.border == None
        assert visuals_1.mask == 2
