class Preloads:
    def __init__(
        self,
        sparse_grids_of_planes=None,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix_sparse_preload=None,
        curvature_matrix_preload_counts=None,
    ):

        self.sparse_grids_of_planes = sparse_grids_of_planes
        self.mapper = mapper
        self.blurred_mapping_matrix = blurred_mapping_matrix
        self.curvature_matrix_sparse_preload = curvature_matrix_sparse_preload
        self.curvature_matrix_preload_counts = curvature_matrix_preload_counts
