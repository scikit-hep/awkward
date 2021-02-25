import json

import jax
import jax.tree_util

import awkward as ak
import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

class AuxData(object):
    def __init__(self, form, length, indexes, datakeys):
        self.form = form
        self.length = length
        self.indexes = indexes
        self.datakeys = datakeys

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
        )

class DifferentiableArray(ak.Array):
    def __init__(self, aux_data, tracers):
        self.aux_data = aux_data
        self.tracers = tracers

    @property
    def layout(self):
        # print(self.tracers)
        buffers = dict(self.aux_data.indexes)
        for key, tracer in zip(self.aux_data.datakeys, self.tracers):
            if hasattr(tracer, "primal"):
                buffers[key] = tracer.primal
            elif hasattr(tracer, "val"):
                buffers[key] = tracer.val
            else:
                buffers[key] = ak.CannotMaterialize(tracer.shape)
        return ak.from_buffers(
            self.aux_data.form, self.aux_data.length, buffers, highlevel=False
        )

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
            aux_data = AuxData(form, length, indexes, self.aux_data.datakeys)
            return DifferentiableArray(aux_data, self.tracers)
        else:
            return out

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
        aux_data = AuxData(nextform, length, indexes, datakeys)
        children = [jax.numpy.asarray(buffers[x], buffers[x].dtype) for x in datakeys]
    return children, aux_data

def special_unflatten(aux_data, children):
    if any(isinstance(x, jax.core.Tracer) for x in children):
        return DifferentiableArray(aux_data, children)
    else:
        buffers = dict(aux_data.indexes)
        buffers.update(zip(aux_data.datakeys, children))
        return ak.from_buffers(aux_data.form, aux_data.length, buffers)

jax.tree_util.register_pytree_node(ak.Array, special_flatten, special_unflatten)
jax.tree_util.register_pytree_node(DifferentiableArray, special_flatten, special_unflatten)

###############################################################################
#  TESTING
###############################################################################

def func(array):
    return 2*array.y[0, 0, 0] + 10

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

primal_result, tangent_result = jax.jvp(func, (primal,), (tangent,))
print("resulting types", type(primal_result), type(tangent_result))
print(primal_result)
print(tangent_result)

jit_result = jax.jit(func)(primal)
print("resulting type", type(jit_result))
print(jit_result)
# def func(x):
#     return x[0] ** 2 
# tree = ak.Array([1., 2., 3., 4.])
# basis = ak.Array([0., 1., 0., 0.])
# print(jax.jvp(func, (tree,), (basis,)))