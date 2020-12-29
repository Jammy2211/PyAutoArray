import autoarray.plot as aplt
from os import path
from autoarray.plot.mat_wrap import mat_decorators

directory = path.dirname(path.realpath(__file__))


class TestDecorator:
    def test__kpc_per_scaled_extacted_from_object_if_available(self):

        kwargs = {"hi": 1}

        kpc_per_scaled = mat_decorators.kpc_per_scaled_of_object_from_kwargs(
            kwargs=kwargs
        )

        assert kpc_per_scaled == None

        class MockObj:
            def __init__(self, param1):

                self.param1 = param1

        obj = MockObj(param1=1)

        kwargs = {"hi": 1, "hello": obj}

        kpc_per_scaled = mat_decorators.kpc_per_scaled_of_object_from_kwargs(
            kwargs=kwargs
        )

        assert kpc_per_scaled == None

        class MockObj:
            def __init__(self, param1, kpc_per_scaled):

                self.param1 = param1
                self.kpc_per_scaled = kpc_per_scaled

        obj = MockObj(param1=1, kpc_per_scaled=2)

        kwargs = {"hi": 1, "hello": obj}

        kpc_per_scaled = mat_decorators.kpc_per_scaled_of_object_from_kwargs(
            kwargs=kwargs
        )

        assert kpc_per_scaled == 2
