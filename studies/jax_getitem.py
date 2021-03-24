import jax 
import jax.tree_util 
import awkward as ak
import numpy as np
from numbers import Integral, Real

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

class AuxData(object):
    def __init__(self, layout):
        self.layout = layout

    def __eq__(self, other):
        if self.layout is not None:
            return self.layout.form == other.layout.form

def find_dataptrs(layout):
    def find_nparray_ptrs(node, depth, data_ptrs):
        if isinstance(node, ak.layout.NumpyArray):
            data_ptrs.append(node.ptr) 
    data_ptrs = []
    ak._util.recursive_walk(layout, find_nparray_ptrs, args=(data_ptrs,))

    return data_ptrs

class DifferentiableArray(ak.Array):
    def __init__(self, aux_data, tracers):
        self.aux_data = aux_data
        self.tracers = tracers
        if self.aux_data.layout is not None:
            self.data_ptrs = find_dataptrs(self.aux_data.layout)
            assert len(self.tracers) == len(self.data_ptrs)
            self.map_ptrs_to_tracers = dict(zip(self.data_ptrs, self.tracers))
        else:
            self.data_ptrs = None
            self.map_ptrs_to_tracers = None
    
    @property
    def layout(self):
        return self.aux_data.layout

    @layout.setter
    def layout(self, layout):
        raise ValueError(
            "this operation cannot be performed in a JAX-compiled or JAX-differentiated function"
        )

    def __getitem__(self, where):
        if self.layout is None:
            raise TypeError("Cannot slice a scalar")

        out = self.layout[where]
        
        def find_nparray_node_newptr(layout, outlayout):
            outlayout_fieldloc = outlayout.identities.fieldloc
            def find_nparray_node(node, depth, fieldloc, shape, nodenum, nodenum_index):
                if isinstance(node, ak.layout.NumpyArray):
                    if node.identities.fieldloc == fieldloc and np.asarray(node.identities).shape[1] == shape[1]:
                        nodenum_index = nodenum
                        return
                    else:
                        nodenum = nodenum + 1
            nodenum_index = -1
            ak._util.recursive_walk(layout, find_nparray_node, args=(outlayout_fieldloc, np.asarray(outlayout).shape, 0, nodenum_index))
            if nodenum_index == -1:
                raise ValueError("Couldn't find the node in new slice")
            return nodenum_index

        if not isinstance(out, ak.layout.Content):
            def recurse(array, recurse_where):
                if isinstance(recurse_where, Integral) or isinstance(recurse_where, str):
                    if isinstance(array.layout, ak.layout.NumpyArray):
                        if array.layout.ptr in self.map_ptrs_to_tracers:
                            tracer = self.map_ptrs_to_tracers[array.layout.ptr]
                        else:
                            tracer = array.tracers[find_nparray_node_newptr(self.layout, array.layout)]
                        return tracer[recurse_where]
                elif isinstance(where, tuple):
                    return recurse(array[where[:-1]], where[len(where) - 1])
                else:
                    raise ValueError("Can't slice the array with {0}".format(where))
                    
            child = [recurse(self, where)]
            aux_data = AuxData(None)
            return DifferentiableArray(aux_data, child)

        else:
            def fetch_indices_and_fieldloc_layout(layout):
                if isinstance(layout, ak.layout.NumpyArray):
                    return [((layout.identities.fieldloc, np.asarray(layout.identities).shape[1]), np.asarray(layout.identities))]
                elif isinstance(layout, ak._util.listtypes):
                    return fetch_indices_and_fieldloc_layout(layout.content)
                elif isinstance(layout, ak._util.indexedtypes):
                    return fetch_indices_and_fieldloc_layout(layout.project())
                elif isinstance(layout, ak._util.uniontypes):
                    raise ValueError("Can't differntiate an UnionArray type {0}".format(layout))
                elif isinstance(layout, ak._util.recordtypes):
                    indices = []
                    for content in layout.contents:
                        indices = indices + fetch_indices_and_fieldloc_layout(content) 
                    return indices
                elif isinstance(layout, ak._util.indexedtypes):
                    return fetch_indices_and_fieldloc_layout(layout.content)
                elif isinstance(layout, ak._util.indexedoptiontypes):
                    return fetch_indices_and_fieldloc_layout(layout.content)
                elif isinstance(layout, (ak.layout.BitMaskedArray, 
                                         ak.layout.ByteMaskedArray, 
                                         ak.layout.UnmaskedArray)):
                    return fetch_indices_and_fieldloc_layout(layout.content)
                else:
                    raise NotImplementedError
            
            def fetch_children_tracer(layout, preslice_identities, children = []):
                if isinstance(layout, ak.layout.NumpyArray):
                    def find_intersection_indices(preslice_identities, postslice_identities):
                        multiplier = np.append(np.cumprod((np.max(preslice_identities, axis=0) + 1)[::-1])[-2::-1], 1)
                        haystack = np.sum(preslice_identities * multiplier, axis=1)
                        needle = np.sum(postslice_identities * multiplier, axis=1)
                        return np.searchsorted(haystack, needle)
                    
                    def find_corresponding_identity(postslice_identities, preslice_identities):
                        for identity in preslice_identities:
                            if identity[0] == postslice_identities:
                                return identity[1]
                        raise ValueError("Couldn't find postslice identities in preslice identities")

                    if layout.ptr in self.map_ptrs_to_tracers:
                        tracer = self.map_ptrs_to_tracers[layout.ptr]
                        indices = find_intersection_indices(find_corresponding_identity((layout.identities.fieldloc, np.asarray(layout.identities).shape[1]), preslice_identities), np.asarray(layout.identities))
                        children.append(jax.numpy.take(tracer, indices))
                        return children
                    else:
                        tracer = self.tracers[find_nparray_node_newptr(self.layout, layout)]
                        indices = find_intersection_indices(find_corresponding_identity((layout.identities.fieldloc, np.asarray(layout.identities).shape[1]), preslice_identities), np.asarray(layout.identities))
                        children.append(jax.numpy.take(tracer, indices))
                        return children

                elif isinstance(layout, ak._util.listtypes):
                    return fetch_children_tracer(layout.content, preslice_identities)
                elif isinstance(layout, ak._util.uniontypes):
                    raise ValueError("Can't differntiate an UnionArray type {0}".format(layout))
                elif isinstance(layout, ak._util.recordtypes):
                    children = []
                    for content in layout.contents:
                        children = children + fetch_children_tracer(content, preslice_identities)
                    return children
                elif isinstance(layout, ak._util.indexedtypes):
                    return fetch_children_tracer(layout.content, preslice_identities)
                elif isinstance(layout, ak._util.indexedoptiontypes):
                    return fetch_children_tracer(layout.content, preslice_identities)
                elif isinstance(layout, (ak.layout.BitMaskedArray, 
                                         ak.layout.ByteMaskedArray, 
                                         ak.layout.UnmaskedArray)):
                    return fetch_children_tracer(layout.content, preslice_identities)
                else:
                    raise NotImplementedError("fetch_children_tracer not completely implemented yet for {0}".format(layout))
                
            children = fetch_children_tracer(out, fetch_indices_and_fieldloc_layout(self.aux_data.layout))
            out = out.deep_copy()
            out.setidentities()
            aux_data = AuxData(out)
            return DifferentiableArray(aux_data, children)

    def __setitem__(self, where, what):
        raise ValueError(
            "this operation cannot be performed in a JAX-compiled or JAX-differentiated function"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # optional sanity-check (i.e. sanity is optional)
        for x in inputs:
            if isinstance(x, DifferentiableArray):
                if self.layout is not None:
                    assert x.aux_data == self.aux_data
                    assert len(x.tracers) == len(self.tracers)
                else:
                    assert x.aux_data.layout == self.aux_data.layout
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
jax.tree_util.register_pytree_node(DifferentiableArray, special_flatten, special_unflatten)

###############################################################################
#  TESTING
###############################################################################

#### ak.layout.NumpyArray ####
test_numpyarray = ak.Array(np.arange(10, dtype=np.float64))
test_numpyarray_tangent = ak.Array(np.arange(10, dtype=np.float64))

def func_numpyarray_1(x):
    return x[4] ** 2

def func_numpyarray_2(x):
    return x[2:5] ** 2 + x[1:4] ** 2

def func_numpyarray_3(x):
    return x[::-1]

#### ak.layout.ListOffsetArray ####
test_listoffsetarray = ak.Array([[1., 2., 3.], [], [4., 5.]])
test_listoffsetarray_tangent = ak.Array([[0., 0., 0.], [], [0., 1.]])

def func_listoffsetarray_1(x):
    return x[2] * 2

def func_listoffsetarray_2(x):
    return x * x

def func_listoffsetarray_3(x):
    return x[0, 0] * x[2, 1]

def func_listoffsetarray_4(x):
    return x[::-1] ** 2

def func_listoffsetarray_5(x):
    return 2 * x[:-1]

def func_listoffsetarray_6(x):
    return x[0][0] * x[2][1]

#### ak.layout.RecordArray ####

test_recordarray = ak.Array([
    [{"x": 1.1, "y": [1.0]}, {"x": 2.2, "y": [1.0, 2.2]}],
    [],
    [{"x": 3.3, "y": [1.0, 2.0, 3.0]}]
])
test_recordarray_tangent = ak.Array([
    [{"x": 0.0, "y": [1.0]}, {"x": 2.0, "y": [1.5, 0.0]}],
    [],
    [{"x": 1.5, "y": [2.0, 0.5, 1.0]}]
])

def func_recordarray_1(array):
    return 2 * array.y[2][0][1] + 10
    
def func_recordarray_2(array):
    return 2 * array.y[0][0][0] ** 2

def func_recordarray_3(array):
    return 2 * array.y[2][0] + 10

def func_recordarray_4(array):
    return 2 * array.y[0][0] ** 2

def func_recordarray_5(array):
    return 2 * array.y[2] + 10

def func_recordarray_6(array):
    return 2 * array.y[0] ** 2

def func_recordarray_7(array):
    return 2 * array.y

def func_recordarray_8(array):
    return 2 * array.y ** 2

def func_recordarray_9(array):
    return 2 * array.y[2, 0, 1] + 10

def func_recordarray_10(array):
    return 2 * array.y[0, 0, 0] ** 2

def func_recordarray_11(array):
    return 2 * array.y[2, 0] + 10

def func_recordarray_12(array):
    return 2 * array.y[0, 0] ** 2

value_jvp, jvp_grad = jax.jvp(func_listoffsetarray_4, (test_listoffsetarray,), (test_listoffsetarray_tangent,))
jit_value = jax.jit(func_listoffsetarray_4)(test_listoffsetarray)
# value_vjp, vjp_func = jax.vjp(func_recordarray_12, test_recordarray)

# print(type(value_vjp))
# print(vjp_func(test_recordarray))
# value, grad = jax.value_and_grad(func_numpyarray_2)(test_nparray)

print("Value and Grad are {0} and {1}".format(value_jvp, jvp_grad))
print("JIT value is {0}".format(jit_value))
# print("VJP value and grad is {0} and {1}".format(value_vjp, vjp_func(test_nparray)))
# print("Value and grad are {0} and {1}".format(value, grad))
