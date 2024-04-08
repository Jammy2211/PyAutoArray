from autoarray.numpy_wrapper import register_pytree_node_class

from autoarray.mask.mask_2d import Mask2D


class AbstractOverSampling:
    def over_sampler_from(self, mask: Mask2D) -> "AbstractOverSampler":
        raise NotImplementedError()


@register_pytree_node_class
class AbstractOverSampler:
    def tree_flatten(self):
        return (self.mask,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0])
