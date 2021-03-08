import json
import jax 
import jax.tree_util 
import awkward as ak
import numpy as np
from numbers import Integral, Real

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

class AuxData(object):
    def __init__(self, form, length, indexes, datakeys, where_list):
        self.form = form
        self.length = length
        self.indexes = indexes
        self.datakeys = datakeys
        self.where_list = where_list

    def __eq__(self, other):
        # AuxData is an object so that JAX can naively call __eq__ on it
        return (
            self.form == other.form
            and self.length == other.length
            and self.indexes.keys() == other.indexes.keys()
            and all(
                # normally, array equality would be a problem for __eq__ (in an if-statement)
                np.array_equal(self.indexes[k], other.indexes[k])
                for k in self.indexes.keys()
            )
            and self.datakeys == other.datakeys
            and self.where_list == self.where_list
        )

class DifferentiableArray(ak.Array):
    def __init__(self, aux_data, tracers):
        self.aux_data = aux_data
        self.tracers = tracers

    @property
    def layout(self):
        buffers = dict(self.aux_data.indexes)
        for key, tracer in zip(self.aux_data.datakeys, self.tracers):
            if hasattr(tracer, "primal"):
                if isinstance(tracer.primal, jax.interpreters.xla._DeviceArray) or isinstance(tracer.primal, np.ndarray):
                    buffers[key] = tracer
        arr = ak.from_buffers(
            self.aux_data.form, self.aux_data.length, buffers, highlevel=False
        ) 
        return arr

    @layout.setter
    def layout(self, layout):
        raise ValueError(
            "this operation cannot be performed in a JAX-compiled or JAX-differentiated function"
        )

    def __getitem__(self, where):
        out = self.layout[where]

        if isinstance(out, ak.layout.Content):
            form, length, indexes = ak.to_buffers(
                out, form_key="getitem_node{id}", virtual="pass"
            )
            self.aux_data.where_list.append(where)
            
            aux_data = AuxData(self.aux_data.form, self.aux_data.length, self.aux_data.indexes, self.aux_data.datakeys, self.aux_data.where_list)
            return DifferentiableArray(aux_data, self.tracers)
        else: 
            # form, length, children = ak.to_buffers(self.layout)
            # child_buf_key = self.aux_data.datakeys.index(list(children.keys())[0])
            if isinstance(where, (Integral, np.integer)) and isinstance(self.layout, ak.layout.NumpyArray): 
                assert len(self.tracers) == 1
                return self.tracers[0][where]
            else:
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

def find_datanode(formjson, form_key):
    if isinstance(formjson, dict):
        if formjson.get("form_key") == form_key:
            return formjson
        for k, v in formjson.items():
            out = find_datanode(v, form_key)
            if out is not None:
                if out == formjson[k]:
                    formjson[k] = {
                        "class": "VirtualArray",
                        "form": out,
                        "has_length": True,
                        "has_identities": False,
                        "parameters": {},
                        "form_key": None,
                    }
                return out
        else:
            return None
    elif isinstance(formjson, list):
        for i, v in enumerate(formjson):
            out = find_datanode(v, form_key)
            if out is not None:
                if out == formjson[i]:
                    formjson[i] = {
                        "class": "VirtualArray",
                        "form": out,
                        "has_length": True,
                        "has_identities": False,
                        "parameters": {},
                        "form_key": None,
                    }
                return out
        else:
            return None
    else:
        return None

def special_flatten(array):
    if isinstance(array, DifferentiableArray):
        aux_data, children = array.aux_data, array.tracers
    else:
        # print(array)
        form, length, buffers = ak.to_buffers(array)
        formjson = json.loads(form.tojson())
        indexes = {k: v for k, v in buffers.items() if not k.endswith("-data")}
        datakeys = []
        for key in buffers:
            partition, form_key, role = key.split("-")
            if role == "data":
                nodejson = find_datanode(formjson, form_key)
                assert nodejson is not None
                node = ak.forms.Form.fromjson(json.dumps(nodejson))
                datakeys.append(key)
        nextform = ak.forms.Form.fromjson(json.dumps(formjson))
        aux_data = AuxData(nextform, length, indexes, datakeys, [])
        children = [jax.numpy.asarray(buffers[x], buffers[x].dtype) for x in datakeys]
    return children, aux_data

def special_unflatten(aux_data, children):
    if any(isinstance(x, jax.core.Tracer) for x in children):
        return DifferentiableArray(aux_data, children)
    else:  
        buffers = dict(aux_data.indexes)
        buffers.update(zip(aux_data.datakeys, children))

        arr = ak.from_buffers(aux_data.form, aux_data.length, buffers)
        print(aux_data.form)
        print(arr)
        print(aux_data.where_list)
        for where in aux_data.where_list:
            arr = arr.__getitem__(where)
        return arr

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
# print(jax.jvp(func6_2, (primal,), (tangent,)))
# print(jax.jvp(func6_2, (primal,), (tangent,)))
print(jax.jvp(func7_1, (primal,), (tangent,)))

