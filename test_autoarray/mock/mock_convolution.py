from autoarray.operators import convolver


class MockConvolver(convolver.Convolver):
    def __init__(self, mask, kernel):
        super(MockConvolver, self).__init__(mask=mask, kernel=kernel)

    def convolver_with_blurring_mask_added(self, blurring_mask):
        return MockConvolver(
            mask=self.mask, kernel=self.kernel, blurring_mask=blurring_mask
        )
