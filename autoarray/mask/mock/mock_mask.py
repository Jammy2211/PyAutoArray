class MockIndexes2D:

    def __init__(self, native_index_for_slim_index=None):

        self.native_index_for_slim_index = native_index_for_slim_index


class MockMask:
    def __init__(self, native_index_for_slim_index=None):

        self.indexes = MockIndexes2D(native_index_for_slim_index=native_index_for_slim_index)
