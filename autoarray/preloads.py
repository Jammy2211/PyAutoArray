class Preloads:
    def __init__(
        self,
        sparse_grids_of_planes=None,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix=None,
    ):

        self.sparse_grids_of_planes = sparse_grids_of_planes
        self.mapper = mapper
        self.blurred_mapping_matrix = blurred_mapping_matrix
        self.curvature_matrix = curvature_matrix
