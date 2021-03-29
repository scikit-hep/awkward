# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import jax 
import types
import jax.tree_util 
import awkward as ak
import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

class AuxData(object):
    def __init__(self, layout):
        self.layout = layout

    def __eq__(self, other):
        if self.layout is not None and other.layout is not None:
            return self.layout.form == other.layout.form
        else:
            return self.layout == other.layout

def special_flatten(array):
    if isinstance(array, ak.Array):
        if array.tracers is not None:
            aux_data, children = AuxData(array.layout), array.tracers
        else:
            def create_databuffers(node, depth, databuffers):
                if isinstance(node, ak.layout.NumpyArray):
                    databuffers.append(node) 
            
            databuffers = []
            ak._util.recursive_walk(array.layout, create_databuffers, args=(databuffers,))

            array.layout.setidentities()
            aux_data = AuxData(array.layout)
            children = [jax.numpy.asarray(x) for x in databuffers]
        return children, aux_data
    else:
        raise ValueError("Can only differentiate Awkward Arrays, recieved array of type {0}".format(type(array)))

def special_unflatten(aux_data, children):
    if any(isinstance(x, jax.core.Tracer) for x in children):
        return ak.Array(aux_data.layout, tracers=children)
    elif all(child is None for child in children):
        return None
    else:  
        if aux_data.layout is None:
            assert len(children) == 1
            return np.ndarray.item(np.asarray(children[0]))

        def function(layout, num = 0):
            if isinstance(layout, ak.layout.NumpyArray):
                num = num + 1
                return lambda: ak.layout.NumpyArray(children[num - 1])
        
        arr = ak._util.recursively_apply(aux_data.layout, function, pass_depth=False)
        return ak.Array(arr)

jax.tree_util.register_pytree_node(ak.Array, special_flatten, special_unflatten)
