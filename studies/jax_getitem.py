import json
import jax 
import jax.tree_util 
import awkward as ak
import numpy as np
from numbers import Integral, Real
import pdb

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

class AuxData(object):
    def __init__(self, layout):
        def find_nparray_ptrs(node, depth, hash_ptrs):
            if isinstance(node, ak.layout.NumpyArray):
                hash_ptrs.append(node.ptr) 

        self.hash_ptrs = []
        ak._util.recursive_walk(layout, find_nparray_ptrs, args=(self.hash_ptrs,))
        self.layout = layout

    def __eq__(self, other):
        # AuxData is an object so that JAX can naively call __eq__ on it
        # def is_layout_equal(layout1, layout2):
            # TODO: Recurse through each object and check for equality. Bonus: Try doing it through recursive_walk
        return self.layout.form == other.layout.form
        #         if isinstance(layout1, (ak.layout.ListOffsetArray32,
        #                                 ak.layout.ListOffsetArray64,
        #                                 ak.layout.ListOffsetArrayU32)):
        #             if isinstance(layout2, (ak.layout.ListOffsetArray32,
        #                                     ak.layout.ListOffsetArray64,
        #                                     ak.layout.ListOffsetArrayU32)):
        #                 if np.array_equal(np.asarray(layout1.offsets), np.asarray(layout2.offsets)) and len(layout1) == len(layout2):
        #                     return is_layout_equal(layout1.content, layout2.content)

        #         elif isinstance(layout1, ak.layout.NumpyArray):
        #             if isinstance(layout2, ak.layout.NumpyArray):
        #                 return layout1.shape == layout2.shape and layout1.strides == layout2.strides and layout1.ndim == layout2.ndim 
        #     return False
        
        # return is_layout_equal(self.layout, other.layout)

class DifferentiableArray(ak.Array):
    def __init__(self, aux_data, tracers):
        self.aux_data = aux_data
        self.tracers = tracers
    
    @property
    def layout(self):
        return self.aux_data.layout

    @layout.setter
    def layout(self, layout):
        raise ValueError(
            "this operation cannot be performed in a JAX-compiled or JAX-differentiated function"
        )

    def __getitem__(self, where):
        out = self.layout[where]
        assert len(self.tracers) == len(self.aux_data.hash_ptrs)
        map_ptrs_to_tracers = dict(zip(self.aux_data.hash_ptrs, self.tracers))

        if np.isscalar(out):
            """
            TODO: Recurse here for scalars
            If where is a scalar, find ptr of self.layout, get the corresponding tracers from the dict and return tracer[where]
            If where is a iterable, and assuming the slicing doesn't change the pointer location, go one back in the slice, and get the scalar case
            """
            def recurse(array, recurse_where):
                if np.isscalar(recurse_where):
                    if isinstance(array, ak.layout.NumpyArray):
                        tracer = map_ptrs_to_tracers[array.ptr]
                        return tracer[recurse_where]
                elif isinstance(where, tuple):
                    return recurse(array[where[:-1]], where[len(where) - 1])
                    
            return recurse(self.layout, where)
        else:
            """
            Implement index tracer fetching with identities here
            """
            # if isinstance(out, ak.layout.ListOffsetArray32,
            #                    ak.layout.ListOffsetArray64,
            #                    ak.layout.ListOffsetArrayU32):

            children = []
                
            def fetch_indices_layout(layout):
                if isinstance(layout, ak.layout.NumpyArray):
                    return np.asarray(layout.identities)
                elif isinstance(layout, (ak.layout.ListOffsetArray32, ak.layout.ListOffsetArray64, ak.layout.ListOffsetArrayU32)):
                    return fetch_indices_layout(layout.content)
                else:
                    raise NotImplementedError

            if isinstance(out, ak.layout.NumpyArray):
                if out.ptr in map_ptrs_to_tracers:
                    tracer = map_ptrs_to_tracers[out.ptr]

                    indices = []
                    layout_indices = fetch_indices_layout(self.aux_data.layout)
                    for i in np.asarray(out.identities):
                        indices.append(np.where((layout_indices == i).all(axis=1))[0][0])

                    children.append(jax.numpy.take(tracer, indices))
                    out.setidentities()
                    aux_data = AuxData(out)
                    return DifferentiableArray(aux_data, children)
                else:
                    """
                    TODO: If the control reaches here, it means the slicing is copying data, handle that here
                    """
                    out_fieldloc = out.identities.fieldloc

                    def find_nparray_node(node, depth, fieldloc, nodenum, nodenum_index):
                        if isinstance(node, ak.layout.NumpyArray):
                            if node.identities.fieldloc == fieldloc:
                                nodenum_index = nodenum
                                return
                            else:
                                nodenum = nodenum + 1
                    nodenum_index = -1
                    ak._util.recursive_walk(self.aux_data.layout, find_nparray_node, args=(out_fieldloc, 0, nodenum_index))

                    tracer = self.tracers[nodenum_index]
                    indices = []
                    layout_indices = fetch_indices_layout(self.aux_data.layout)
                    for i in np.asarray(out.identities):
                        indices.append(np.where((layout_indices == i).all(axis=1))[0][0])
                    
                    children.append(jax.numpy.take(tracer, indices))
                    out.setidentities()
                    aux_data = AuxData(out)

                    return DifferentiableArray(aux_data, children)

            raise NotImplementedError

    def __setitem__(self, where, what):
        raise ValueError(
            "this operation cannot be performed in a JAX-compiled or JAX-differentiated function"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # optional sanity-check (i.e. sanity is optional)
        for x in inputs:
            if isinstance(x, DifferentiableArray):
                assert x.aux_data == self.aux_data
                assert len(x.tracers) == len(self.tracers)

        # ak.Array __add__, etc. map to the NumPy functions, switch to JAX
        for name, np_ufunc in np.core.umath.__dict__.items():
            if ufunc is np_ufunc:
                ufunc = getattr(jax.numpy, name)

        # need to apply the ufunc to the same argument list for each tracer separately
        nexttracers = []
        for i in range(len(self.tracers)):
            nextinputs = [
                x.tracers[i] if isinstance(x, DifferentiableArray) else x
                for x in inputs
            ]
            nexttracers.append(getattr(ufunc, method)(*nextinputs, **kwargs))

        # and return a new DifferentiableArray (keep it wrapped!)
        return DifferentiableArray(self.aux_data, nexttracers)

def special_flatten(array):
    if isinstance(array, DifferentiableArray):
        aux_data, children = array.aux_data, array.tracers
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

def special_unflatten(aux_data, children):
    if any(isinstance(x, jax.core.Tracer) for x in children):
        return DifferentiableArray(aux_data, children)
    elif all(child is None for child in children):
        return None
    else:  
        def function(layout, num = 0):
            if isinstance(layout, ak.layout.NumpyArray):
                num = num + 1
                return lambda: ak.layout.NumpyArray(children[num - 1])
        
        arr = ak._util.recursively_apply(aux_data.layout, function, pass_depth=False)

        return ak.Array(arr)

jax.tree_util.register_pytree_node(ak.Array, special_flatten, special_unflatten)
jax.tree_util.register_pytree_node(DifferentiableArray, special_flatten, special_unflatten)

###############################################################################
#  TESTING
###############################################################################

def func1_1(array):
    return 2 * array.y[2][0][0] + 10

def func1_2(array):
    return 2 * array.y[0][0][0] ** 2

def func2_1(array):
    return 2 * array.y[2][0] + 10

def func2_2(array):
    return 2 * array.y[0][0] ** 2

def func3_1(array):
    return 2 * array.y[2] + 10

def func3_2(array):
    return 2 * array.y[0] ** 2

def func4_1(array):
    return 2 * array.y[2] + 10

def func4_2(array):
    return 2 * array.y[0] ** 2

def func5_1(array):
    return 2 * array.y

def func5_2(array):
    return 2 * array.y ** 2

def func6_1(array):
    return 2 * array

def func6_2(array):
    return 2 * array ** 2

def func7_1(array):
    return array.y[2][0][0] ** 2 + array.y[2][0][1] ** 2

primal = ak.Array([
    [{"x": 1.1, "y": [1.0]}, {"x": 2.2, "y": [1.0, 2.2]}],
    [],
    [{"x": 3.3, "y": [1.0, 2.0, 3.0]}]
])

tangent = ak.Array([
    [{"x": 0.0, "y": [1.0]}, {"x": 2.0, "y": [1.5, 0.0]}],
    [],
    [{"x": 1.5, "y": [2.0, 0.5, 1.0]}]
])

primal_nparray = ak.Array([[1., 2., 3., 4., 5.]])
tangent_nparray = ak.Array([[0., 0., 1., 0., 0.]])

value_flat, value_tree = jax.tree_util.tree_flatten(primal_nparray)
# print(primal_nparray[0][:-1])
# print(jax.jvp(func1_1, (primal,), (tangent,)))
# print(jax.jvp(func1_2, (primal,), (tangent,)))
# print(jax.jvp(func2_1, (primal,), (tangent,)))
# print(jax.jvp(func2_2, (primal,), (tangent,)))
# print(jax.jvp(func3_1, (primal,), (tangent,)))
# print(jax.jvp(func3_2, (primal,), (tangent,)))
# print(jax.jvp(func4_1, (primal,), (tangent,)))
# print(jax.jvp(func4_2, (primal,), (tangent,)))
# print(jax.jvp(func5_2, (primal,), (tangent,)))
# print(jax.jvp(func5_2, (primal,), (tangent,)))
# pdb.set_trace()

# children, aux_data = special_flatten(primal_nparray)
# diffarr = DifferentiableArray(aux_data, children)
# val, func = jax.vjp(func6_2, diffarr)
# # print(diffarr)
# children, aux_data = special_flatten(val)
# diffarr = DifferentiableArray(aux_data, children)
# print(aux_data, children)
# # pdb.set_trace()
# print(func(diffarr))
# print(funceaval))
print(jax.jvp(func6_2, (primal_nparray,), (tangent_nparray,)))
# print(jax.jit(func6_2)(primal_nparray))
# print(jax.grad(func6_2)(primal_nparray))
# print(jax.jvp(func7_1, (primal,), (tangent,)))
