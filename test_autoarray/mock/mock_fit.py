class MockFitInversion(object):
    def __init__(
        self,
        regularization_term,
        log_det_curvature_reg_matrix_term,
        log_det_regularization_matrix_term,
    ):

        self.regularization_term = regularization_term
        self.log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term
