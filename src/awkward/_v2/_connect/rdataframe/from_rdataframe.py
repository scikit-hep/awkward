# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy
import ctypes
from awkward._v2.types.numpytype import primitive_to_dtype

cpp_type_of = {
    "float64": "double",
    "int64": "int64_t",
    "complex128": "std::complex<double>",
}

np = ak.nplike.NumpyMetadata.instance()

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

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)

    if form_str.startswith("{"):
        form = ak._v2.forms.from_json(form_str)

        if isinstance(form, ak._v2.forms.NumpyForm):
            dtype = primitive_to_dtype(form.primitive)
            array = ak.nplike.numpy.empty(data_frame.Count().GetValue(), form.primitive)
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

            array = ak._v2.from_buffers(
                form,
                ptrs[0][1],
                buffers,
            )

            return _maybe_wrap(array, column_as_record)

        elif isinstance(form, ak._v2.forms.ListOffsetForm) and isinstance(
            form.content, ak._v2.forms.ListOffsetForm
        ):
            dtype = ak._v2.types.numpytype.primitive_to_dtype(
                form.content.content.primitive
            )
            data_type = cpp_type_of[dtype.name]

            # pull in the CppBuffers (after which we can import from it)
            CppBuffers = cppyy.gbl.awkward.CppBuffers[column_type, data_type]
            cpp_buffers_self = CppBuffers(result_ptrs)

            # copy data from RDF and make nested offsets
            num_levels, offsets_length = CppBuffers.offsets_and_flatten(
                cpp_buffers_self
            )

            buffers = {}
            for level in range(num_levels):
                length = CppBuffers.offsets_length(cpp_buffers_self, level)
                offsets = ak.nplike.numpy.empty(length, np.int64)
                CppBuffers.copy_offsets(
                    cpp_buffers_self,
                    offsets.ctypes.data_as(ctypes.c_void_p),
                    length,
                    level,
                )
                buffers[f"node{level}-offsets"] = offsets

            data_length = CppBuffers.data_length(cpp_buffers_self)
            data = ak.nplike.numpy.empty(data_length, dtype=dtype)
            CppBuffers.copy_data(
                cpp_buffers_self, data.ctypes.data_as(ctypes.c_void_p), data_length
            )
            buffers[f"node{num_levels}-data"] = data

            array = ak._v2.from_buffers(
                form,
                offsets_length - 1,
                buffers,
            )

            return _maybe_wrap(array, column_as_record)

        else:
            raise ak._v2._util.error(NotImplementedError)

    elif form_str == "awkward type":

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        return _wrap_as_array(column, ak._v2.from_iter(cpp_reference), column_as_record)
    else:
        raise ak._v2._util.error(NotImplementedError)
