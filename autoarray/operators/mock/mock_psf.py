class MockPSF:
    def __init__(self, operated_mapping_matrix=None):
        self.operated_mapping_matrix = operated_mapping_matrix

    def convolve_mapping_matrix(self, mapping_matrix, mask):
        return self.operated_mapping_matrix
