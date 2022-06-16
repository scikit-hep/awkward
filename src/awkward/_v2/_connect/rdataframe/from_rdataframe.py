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
    "uint8": "uint8_t",
}

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()

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
        list_depth = form.purelist_depth
        if list_depth > 3:
            raise ak._v2._util.error(NotImplementedError)

        def supported(form):
            if form.purelist_depth == 1:
                # special case for a list of strings form
                return isinstance(
                    form, (ak._v2.forms.ListOffsetForm, ak._v2.forms.NumpyForm)
                )
            else:
                return isinstance(form, ak._v2.forms.ListOffsetForm) and supported(
                    form.content
                )

        if not supported(form):
            raise ak._v2._util.error(NotImplementedError)

        def form_dtype(form):
            if form.purelist_depth == 1:
                # special case for a list of strings form
                return (
                    primitive_to_dtype(form.content.primitive)
                    if isinstance(form, ak._v2.forms.ListOffsetForm)
                    else primitive_to_dtype(form.primitive)
                )
            else:
                return form_dtype(form.content)

        dtype = form_dtype(form)
        buffers = {}
        depths = 0
        offsets_length = 0

        # pull in the CppBuffers (after which we can import from it)
        CppBuffers = cppyy.gbl.awkward.CppBuffers[column_type, cpp_type_of[dtype.name]]
        # pythonize the CppBuffers create_array function to take ownership on return
        CppBuffers.create_array.__creates__ = True
        cpp_buffers_self = CppBuffers(result_ptrs)

        if isinstance(form, ak._v2.forms.NumpyForm):
            length, data = CppBuffers.create_array(cpp_buffers_self)
            layout = ak._v2.contents.numpyarray.NumpyArray(
                numpy.frombuffer(data, dtype=dtype, count=length),
                parameters=form.parameters,
            )

            return _maybe_wrap(ak._v2.Array(layout), column_as_record)

        elif isinstance(form, ak._v2.forms.ListOffsetForm) and isinstance(
            form.content, ak._v2.forms.NumpyForm
        ):
            # list_depth == 2 or 1 if its the list of strings
            # copy data from RDF and make nested offsets
            depths, offsets_length = CppBuffers.offsets_and_flatten_2(cpp_buffers_self)

        elif list_depth == 3:
            depths, offsets_length = CppBuffers.offsets_and_flatten_3(cpp_buffers_self)
        else:
            depths, offsets_length = CppBuffers.offsets_and_flatten_4(cpp_buffers_self)

        for depth in range(depths):
            length = CppBuffers.offsets_length(cpp_buffers_self, depth)
            offsets = ak.nplike.numpy.empty(length, np.int64)
            CppBuffers.copy_offsets(
                cpp_buffers_self,
                offsets.ctypes.data_as(ctypes.c_void_p),
                length,
                depth,
            )
            buffers[f"node{depth}-offsets"] = offsets

        data_length = CppBuffers.data_length(cpp_buffers_self)
        data = ak.nplike.numpy.empty(data_length, dtype=dtype)
        CppBuffers.copy_data(
            cpp_buffers_self, data.ctypes.data_as(ctypes.c_void_p), data_length
        )
        buffers[f"node{depths}-data"] = data

        array = ak._v2.from_buffers(
            form,
            offsets_length - 1,
            buffers,
        )
        return _maybe_wrap(array, column_as_record)

    elif form_str == "awkward type":

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        return _wrap_as_array(column, ak._v2.from_iter(cpp_reference), column_as_record)
    else:
        raise ak._v2._util.error(NotImplementedError)
