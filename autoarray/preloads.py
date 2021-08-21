class Preloads:
    def __init__(
        self,
        w_tilde=None,
        use_w_tilde=None,
        sparse_image_plane_grids_of_planes=None,
        relocated_grid=None,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix_sparse_preload=None,
        curvature_matrix_preload_counts=None,
        log_det_regularization_matrix_term=None,
    ):

        self.w_tilde = w_tilde
        self.use_w_tilde = use_w_tilde
        self.sparse_image_plane_grids_of_planes = sparse_image_plane_grids_of_planes
        self.relocated_grid = relocated_grid
        self.mapper = mapper
        self.blurred_mapping_matrix = blurred_mapping_matrix
        self.curvature_matrix_sparse_preload = curvature_matrix_sparse_preload
        self.curvature_matrix_preload_counts = curvature_matrix_preload_counts
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term
