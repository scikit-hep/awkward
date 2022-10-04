import jax.numpy

from awkward._connect.jax.reducers import get_jax_reducer  # noqa: F401
from awkward._connect.jax.trees import register_pytree_class  # noqa: F401


def get_jax_ufunc(ufunc):
    return getattr(jax.numpy, ufunc.__name__)
