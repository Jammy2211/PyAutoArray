class MockPSF:
    def __init__(self, operated_mapping_matrix=None):
        self.operated_mapping_matrix = operated_mapping_matrix

    def convolved_mapping_matrix_from(self, mapping_matrix, mask):
        return self.operated_mapping_matrix
