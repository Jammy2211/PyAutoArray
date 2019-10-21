from autoarray.operators import convolution


class MockConvolver(convolution.Convolver):
    def __init__(self, mask_2d, kernel_2d, blurring_mask=None):
        super(MockConvolver, self).__init__(mask_2d=mask_2d, kernel_2d=kernel_2d.in_2d)

    def convolver_with_blurring_mask_added(self, blurring_mask):
        return MockConvolver(
            mask_2d=self.mask, kernel_2d=self.kernel.in_2d, blurring_mask=blurring_mask
        )
