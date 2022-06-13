# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy
import ctypes
from awkward._v2.types.numpytype import primitive_to_dtype
import numpy



class ndarray(numpy.ndarray):
    """
    A wrapper class that inherits from numpy.ndarray and allows to attach the
    result pointer of the `Take` action in an `RDataFrame` event loop to the
    collection of values returned by that action. See
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for more
    information on subclassing numpy arrays.
    """
    def __new__(cls, numpy_array, result_ptr):
        """
        Dunder method invoked at the creation of an instance of this class. It
        creates a numpy array with an `RResultPtr` as an additional
        attribute.
        """
        obj = numpy.asarray(numpy_array).view(cls)
        obj.result_ptr = result_ptr
        return obj

    def __array_finalize__(self, obj):
        """
        Dunder method that fills in the instance default `result_ptr` value.
        """
        if obj is None: return
        self.result_ptr = getattr(obj, "result_ptr", None)

np = ak.nplike.NumpyMetadata.instance()
# numpy = ak.nplike.Numpy.instance()

cppyy.add_include_path("src/awkward/_v2/_connect")

compiler = ROOT.gInterpreter.Declare


done = compiler(
    """
#include "rdataframe/ak_array_builders.h"
"""
)
assert done is True


def from_rdataframe(data_frame, column, column_as_record=True):
    def _wrap_as_array(column, array, column_as_record):
        return (
            ak._v2.highlevel.Array({column: array})
            if column_as_record
            else ak._v2.highlevel.Array(array)
        )

    def _maybe_wrap(array, column_as_record):
        return (
            ak._v2._util.wrap(
                ak._v2.contents.RecordArray(
                    fields=[column],
                    contents=[array.layout],
                ),
                highlevel=True,
            )
            if column_as_record
            else array
        )

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)
    form_str = ROOT.awkward.type_to_form[column_type](0)
    print(form_str)

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)

    if form_str.startswith("{"):
        form = ak._v2.forms.from_json(form_str)

        if isinstance(form, ak._v2.forms.NumpyForm):
            dtype = primitive_to_dtype(form.primitive)
            array = ak.nplike.numpy.empty(
                data_frame.Count().GetValue(), form.primitive
            )
            ROOT.awkward.fill_array[column_type](
                result_ptrs, array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            )

            return _maybe_wrap(ak._v2.from_numpy(array), column_as_record)

        elif isinstance(form, ak._v2.forms.ListOffsetForm) and isinstance(
            form.content, ak._v2.forms.NumpyForm
        ):
            ptrs = ROOT.awkward.copy_offsets_and_flatten[column_type](result_ptrs)

            buffers = {}
            for i in range(len(ptrs) - 1):
                name = f"node{i}-offsets"
                buffers[name] = ptrs[i][0]

            name = f"node{len(ptrs) - 1}-data"
            buffers[name] = ptrs[len(ptrs) - 1][0]
            print(buffers)

            array = ak._v2.from_buffers(
                form,
                ptrs[0][1],
                buffers,
            )

            return _maybe_wrap(array, column_as_record)

        elif isinstance(form, ak._v2.forms.ListOffsetForm) and isinstance(
            form.content, ak._v2.forms.ListOffsetForm
        ):
            # pull in the Buffers (after which we can import from it)
            Buffers = cppyy.gbl.awkward.Buffers[column_type]
            #
            # pythonize the Buffers offsets_and_flatten function to take ownership on return
            Buffers.offsets_and_flatten.__creates__ = True
            cpp_buffers = Buffers(result_ptrs)
            
            ptrs = Buffers.offsets_and_flatten(cpp_buffers)
            print(ptrs)
            print(form_str)

            for i in range(len(ptrs) - 1):
                v = ak.nplike.numpy.frombuffer(ptrs[i].second, dtype=np.int64, count=ptrs[i].first)
                print(v)

            buffers = {}
            for i in range(len(ptrs) - 1):
                name = f"node{i}-offsets"
                #buffers[name] =
                arr = ak.nplike.numpy.frombuffer(ptrs[i].second, dtype=np.int64, count=ptrs[i].first)
                buffers[name] = ndarray(numpy_array=arr, result_ptr=ptrs[i].second)

            name = f"node{len(ptrs) - 1}-data"
            arr = ak.nplike.numpy.frombuffer(ptrs[len(ptrs) - 1].second, dtype = primitive_to_dtype(form.content.content.primitive), count=ptrs[len(ptrs) - 1].first)
            buffers[name] = ndarray(numpy_array=arr, result_ptr=ptrs[len(ptrs) - 1].second)
            print(buffers)

            array = ak._v2.from_buffers(
                form,
                ptrs[0].first - 1,
                buffers,
            )

            return _maybe_wrap(array, column_as_record)

        elif form_str == "awkward type":

            # Triggers event loop and execution of all actions booked in the associated RLoopManager.
            cpp_reference = result_ptrs.GetValue()

            return _wrap_as_array(column, cpp_reference, column_as_record)
        else:
            raise ak._v2._util.error(NotImplementedError)
