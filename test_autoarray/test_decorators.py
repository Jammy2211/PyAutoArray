import autoarray as aa


class MockClass:
    def __init__(self, value, run_time_dict=None):
        self._value = value
        self.run_time_dict = run_time_dict

    @property
    @aa.profile_func
    def value(self):
        return self._value


def test__profile_decorator_times_decorated_function():
    cls = MockClass(value=1.0, run_time_dict={})
    cls.value

    assert "value_0" in cls.run_time_dict
