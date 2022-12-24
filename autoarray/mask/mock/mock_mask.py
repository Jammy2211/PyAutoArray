class MockIndexes2D:
    def __init__(self, native_for_slim=None):

        self.native_for_slim = native_for_slim


class MockMask:
    def __init__(self, native_for_slim=None):

        self.indexes = MockIndexes2D(native_for_slim=native_for_slim)
