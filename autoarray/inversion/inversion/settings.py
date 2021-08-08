class SettingsInversion:
    def __init__(
        self,
        use_linear_operators=False,
        tolerance=1e-8,
        maxiter=250,
        check_solution=True,
    ):

        self.use_linear_operators = use_linear_operators
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.check_solution = check_solution
