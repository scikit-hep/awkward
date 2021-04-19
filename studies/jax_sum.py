import awkward as ak
import jax
import numpy as np
ak.jax.register()
jax.config.update("jax_platform_name", "cpu")

def sum_grad(array):
    def recurse(array, indices = np.zeros(len(array), dtype = np.int32)): 
        if isinstance(array, ak.layout.NumpyArray):

            def segment_sum_wrapper(arr, indices = np.zeros(len(arr), dtype=np.int32)): 
                # print(The indices)
                # indices = np.zeros(len(arr), dtype = np.int32)
                arr = jax.ops.segment_sum(arr, indices)
                return arr
            value, func = jax.vjp(segment_sum_wrapper, np.asarray(array), indices)
            return value, func
        
        elif isinstance(array, ak._util.listtypes):
            indices = array.offsets
            segment_sum_indices = []
            integer_tags = 0
            for i in range(len(indices) - 1):
                start = indices[i]
                stop = indices[i + 1]
                segment_sum_indices = segment_sum_indices + [integer_tags for _ in range(stop - start)]
                integer_tags = integer_tags + 1
            
            value, func = recurse(array.content, np.asarray(segment_sum_indices))
            _, aux_data = ak._connect._jax.jax_utils.special_flatten(ak.Array(array))
            children = []
            children.append(ak.from_jax(func(value)[0]))
            return ak._connect._jax.jax_utils.special_unflatten(aux_data, children)
        
    #    elif isinstance(array, ak._util.indexedtypes):
    #        return recurse(array.project())
    #    elif isinstance(array, ak._util.uniontypes):
    #        raise ValueError(
    #            "Can't differentiate an UnionArray type {0}".format(array)
    #        )
    #    elif isinstance(array, ak._util.recordtypes):
    #        indices = []
    #        children = []
    #        for content in array.contents:
    #            diff_arr = recurse(content)
    #            diff_children, _ = ak._connect._jax.jax_utils.special_flatten(ak.Array(diff_arr))
    #            children = children + diff_children
               
            
    #        return indices
    #    elif isinstance(outlayout, ak._util.indexedtypes):
    #        return fetch_indices_and_fieldloc_layout(outlayout.content)
    #    elif isinstance(outlayout, ak._util.indexedoptiontypes):
    #        return fetch_indices_and_fieldloc_layout(outlayout.content)
    #    elif isinstance(
    #        outlayout,
    #        (
    #            ak.layout.BitMaskedArray,
    #            ak.layout.ByteMaskedArray,
    #            ak.layout.UnmaskedArray,
    #        ),
    #    ):
    #        return fetch_indices_and_fieldloc_layout(outlayout.content)
    #    else:
    #        raise NotImplementedError
        
        
    
    return recurse(array.layout)

def sum_jax(arr):
    indices = np.zeros(len(arr), dtype = np.int32)
    arr1 = jax.ops.segment_sum(arr, indices)
    return arr1
        
arr = ak.Array(np.asarray([1., 2., 3., 4., 5.], dtype=np.float32))
arr1 = jax.numpy.array([1., 2., 3., 4., 5.])
arr2 = ak.Array([[1., 2., 3.], [4], [5., 6.]])

print(sum_grad(arr2))
# value, func = jax.vjp(, arr)

# print(value)
# print(func(value))
