import jax.numpy

from awkward._connect.jax.reducers import get_jax_reducer  # noqa: F401
from awkward._connect.jax.trees import (  # noqa: F401
    AuxData,
    find_numpyarray_nodes,
    jax_flatten_highlevel,
    jax_unflatten_highlevel,
    replace_numpyarray_nodes,
)


def get_jax_ufunc(ufunc):
    return getattr(jax.numpy, ufunc.__name__)
