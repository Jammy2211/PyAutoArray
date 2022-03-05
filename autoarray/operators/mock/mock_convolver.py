class MockConvolver:
    def __init__(self, blurred_mapping_matrix=None):
        self.blurred_mapping_matrix = blurred_mapping_matrix

    def convolve_mapping_matrix(self, mapping_matrix):
        return self.blurred_mapping_matrix
