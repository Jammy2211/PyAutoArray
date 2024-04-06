from autoarray.numpy_wrapper import register_pytree_node_class


class AbstractOverSampling:
    pass


@register_pytree_node_class
class AbstractOverSampler:
    def tree_flatten(self):
        return (self.mask,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0])
