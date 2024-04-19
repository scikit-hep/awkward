# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ctypes
import os
import textwrap

import cppyy
import ROOT

import awkward as ak
import awkward._connect.cling
import awkward._lookup
from awkward._backends.numpy import NumpyBackend
from awkward._layout import wrap_layout
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward.types.numpytype import primitive_to_dtype

cpp_type_of = {
    "bool": "bool",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
    "complex64": "std::complex<float>",
    "complex128": "std::complex<double>",
    "datetime64": "std::time_t",
    "timedelta64": "std::difftime",
}

np = NumpyMetadata.instance()
numpy = Numpy.instance()


cppyy.add_include_path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "include",
        )
    )
)
cppyy.add_include_path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.pardir,
            "header-only",
        )
    )
)


cppyy.include("rdataframe/jagged_builders.h")


def from_rdataframe(
    data_frame, columns, highlevel, behavior, with_name, offsets_type, keep_order
):
    if hasattr(data_frame, "proxied_node"):
        raise NotImplementedError("Distributed RDataFrame is not yet supported")

    def cpp_builder_type(depth, data_type):
        if depth == 1:
            return f"awkward::LayoutBuilder::Numpy<{data_type}>>"
        else:
            return f"awkward::LayoutBuilder::ListOffset<int64_t, {cpp_builder_type(depth - 1, data_type)}>"

    def cpp_fill_offsets_and_flatten(depth):
        if depth == 1:
            return textwrap.dedent(
                """
                for (auto const& it : vec1) {
                    builder1.append(it);
                }
                """
            )
        else:
            return textwrap.dedent(
                f"""
                for (auto const& vec{depth - 1} : vec{depth}) {{
                    auto& builder{depth - 1} = builder{depth}.begin_list();
                    {cpp_fill_offsets_and_flatten(depth - 1)}
                    builder{depth}.end_list();
                }}
                """
            )

    def cpp_fill_function(depth):
        if depth == 1:
            return textwrap.dedent(
                """
                template<class BUILDER, typename PRIMITIVE>
                void fill_from(BUILDER& builder, ROOT::RDF::RResultPtr<std::vector<PRIMITIVE>>& result) {
                    for (auto const& it : result) {
                        builder.append(it);
                    }
                }
                """
            )
        else:
            return textwrap.dedent(
                f"""
                template<class BUILDER, typename PRIMITIVE>
                void fill_offsets_and_flatten{depth}(BUILDER& builder{depth}, ROOT::RDF::RResultPtr<std::vector<PRIMITIVE>>& result) {{
                    for (auto const& vec{depth - 1} : result) {{
                      auto& builder{depth - 1} = builder{depth}.begin_list();
                      {cpp_fill_offsets_and_flatten(depth - 1)}
                      builder{depth}.end_list();
                    }}
                }}
                """
            )

    def form_dtype(form):
        if isinstance(form, ak.forms.NumpyForm) and form.inner_shape == ():
            return primitive_to_dtype(form.primitive)
        elif isinstance(form, ak.forms.ListOffsetForm):
            return form_dtype(form.content)

    # Register Take action for each column
    # 'Take' is a lazy action:
    column_types = {}
    result_ptrs = {}
    contents = {}
    index = {}
    awkward_type_cols = {}

    columns = (*columns, "rdfentry_")
    maybe_indexed = keep_order

    # Important note: This loop is separate from the next one
    # in order not to trigger the additional RDataFrame
    # Event loops
    for col in columns:
        column_types[col] = data_frame.GetColumnType(col)
        result_ptrs[col] = data_frame.Take[column_types[col]](col)

        if ROOT.awkward.is_awkward_type[column_types[col]]():
            maybe_indexed = True

    if not maybe_indexed:
        columns = columns[:-1]

    for col in columns:
        if ROOT.awkward.is_awkward_type[column_types[col]]():  # Retrieve Awkward arrays
            # ROOT::RDF::RResultPtr<T>::begin Returns an iterator to the beginning of
            # the contained object if this makes sense, throw a compilation error otherwise.
            #
            # Does not trigger event loop and execution of all actions booked in
            # the associated RLoopManager.
            lookup = result_ptrs[col].begin().lookup()
            generator = lookup[col].generator
            layout = generator.tolayout(lookup[col], 0, ())
            awkward_type_cols[col] = layout

        else:  # Convert the C++ vectors to Awkward arrays
            form_str = ROOT.awkward.type_to_form[column_types[col], offsets_type](0)

            if form_str == "unsupported type":
                raise TypeError(
                    f"{col!r} column's type {column_types[col]!r} is not supported."
                )

            form = ak.forms.from_json(form_str)

            list_depth = form.purelist_depth
            form_dtype_name = form_dtype(form).name
            data_type = cpp_type_of[form_dtype_name]

            # pull in the CppBuffers (after which we can import from it)
            CppBuffers = cppyy.gbl.awkward.CppBuffers
            cpp_buffers_self = CppBuffers()

            if isinstance(form, ak.forms.NumpyForm):
                NumpyBuilder = cppyy.gbl.awkward.LayoutBuilder.Numpy[data_type]
                builder = NumpyBuilder()
                builder_type = type(builder).__cpp_name__

                cpp_buffers_self.fill_from[builder_type, column_types[col]](
                    builder, result_ptrs[col]
                )

            elif isinstance(form, ak.forms.ListOffsetForm):
                if isinstance(form.content, ak.forms.NumpyForm):
                    # NOTE: list_depth == 2 or 1 if its the list of strings
                    list_depth = 2

                ListOffsetBuilder = cppyy.gbl.awkward.LayoutBuilder.ListOffset[
                    offsets_type,
                    cpp_builder_type(list_depth - 1, data_type),
                ]
                builder = ListOffsetBuilder()
                builder_type = type(builder).__cpp_name__

                if not hasattr(
                    cppyy.gbl.awkward, f"fill_offsets_and_flatten{list_depth}"
                ):
                    done = cppyy.cppdef(
                        "namespace awkward {" + cpp_fill_function(list_depth) + "}"
                    )
                    assert done is True

                fill_from_func = getattr(
                    cppyy.gbl.awkward, f"fill_offsets_and_flatten{list_depth}"
                )
                fill_from_func[builder_type, column_types[col]](
                    builder, result_ptrs[col]
                )
            else:
                raise AssertionError(f"unrecognized Form: {type(form)}")

            names_nbytes = cpp_buffers_self.names_nbytes[builder_type](builder)

            buffers = {}
            for item in names_nbytes:
                buffers[item.first] = numpy.empty(item.second, dtype=np.uint8)
                cpp_buffers_self.append(
                    item.first,
                    buffers[item.first].ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                )

            length = cpp_buffers_self.to_char_buffers[builder_type](builder)

            if col == "rdfentry_":
                index[col] = ak.from_buffers(
                    form,
                    length,
                    buffers,
                    byteorder=ak._util.native_byteorder,
                    highlevel=highlevel,
                    behavior=behavior,
                )
                index[col] = ak.index.Index64(
                    index[col].layout.to_backend_array(
                        allow_missing=True, backend=NumpyBackend.instance()
                    )
                )
            else:
                contents[col] = ak.from_buffers(
                    form,
                    length,
                    buffers,
                    byteorder=ak._util.native_byteorder,
                    highlevel=highlevel,
                    behavior=behavior,
                )

    for key, value in awkward_type_cols.items():
        if len(index["rdfentry_"]) < len(value):
            contents[key] = wrap_layout(
                ak.contents.IndexedArray(index["rdfentry_"], value),
                highlevel=highlevel,
                behavior=behavior,
            )
        else:
            contents[key] = value

    out = ak.zip(
        contents,
        depth_limit=1,
        highlevel=highlevel,
        behavior=behavior,
        with_name=with_name,
    )

    if keep_order:
        sorted = ak.index.Index64(index["rdfentry_"].data.argsort())
        out = wrap_layout(
            ak.contents.IndexedArray(sorted, out.layout),
            highlevel=highlevel,
            behavior=behavior,
        )

    return out
