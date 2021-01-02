import autoarray as aa
import autoarray.plot as aplt


class TestInclude2d:
    def test__loads_default_values_from_config_if_not_input(self):

        include = aplt.Include2D()

        assert include.origin == True
        assert include.mask == True
        assert include.border == True
        assert include.parallel_overscan == True
        assert include.serial_prescan == True
        assert include.serial_overscan == False

        include = aplt.Include2D(origin=False, border=False, serial_overscan=True)

        assert include.origin == False
        assert include.mask == True
        assert include.border == False
        assert include.parallel_overscan == True
        assert include.serial_prescan == True
        assert include.serial_overscan == True
