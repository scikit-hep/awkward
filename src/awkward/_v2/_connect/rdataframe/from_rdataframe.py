# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

compiler = ROOT.gInterpreter.Declare

numpy = ak.nplike.Numpy.instance()

cppyy.cppdef(
    """
#include <type_traits>
#include <iterator>

namespace is_iterable_impl
{
    using std::begin;
    using std::end;

    template<class T>
    using check_specs = std::void_t<
        std::enable_if_t<std::is_same_v<
            decltype(begin(std::declval<T&>())), // has begin()
            decltype(end(std::declval<T&>()))    // has end()
        >>,                                      // ... begin() and end() are the same type ...
        decltype(*begin(std::declval<T&>()))     // ... which can be dereferenced
    >;

    template<class T, class = void>
    struct is_iterable
    : std::false_type
    {};

    template<class T>
    struct is_iterable<T, check_specs<T>>
    : std::true_type
    {};
}

template<class T>
using is_iterable = is_iterable_impl::is_iterable<T>;

template<class T>
constexpr bool is_iterable_v = is_iterable<T>::value;

template<typename T>
bool _is_arithmetic() {
    return std::is_arithmetic<T>::value;
};

template<typename T>
bool _is_iterable() {
    return is_iterable<T>::value;
};

template<typename T, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

"""
)


def from_rdataframe(data_frame, column, column_as_record=True):

    def _wrap_as_array(column, array, column_as_record):
        return (
            ak._v2.highlevel.Array({column: array})
            if column_as_record
            else ak._v2.highlevel.Array(array)
        )

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)
    print("Column type:", column_type)
    result_ptrs = data_frame_rnode.Take[column_type](column)
    print(result_ptrs)
    cpp_reference = result_ptrs.GetValue()
    print("cppyy typeid:", cppyy.typeid(cpp_reference))

    # check that its an std::vector - only if its type is not supported
    #
    # The conversion of STL vectors and TVec to numpy arrays happens
    # without copying the data.
    # The memory-adoption is achieved by the dictionary __array_interface__, which
    # is added dynamically to the Python objects by PyROOT.

    # array interface is added for STL vectors and RVecs of the following types:
    # float, double, int, unsigned int, long, unsigned long

    if hasattr(cpp_reference, "__array_interface__"):
        array = numpy.asarray(cpp_reference)

        return _wrap_as_array(column, array, column_as_record)

    if cppyy.typeid(
        cppyy.gbl.std.vector[column_type]()
    ) == cppyy.typeid(cpp_reference):

        # check if it's an integral or a floating point type
        if cppyy.gbl._is_arithmetic[cpp_reference.value_type]():

            array = numpy.asarray(cpp_reference)

            return _wrap_as_array(column, array, column_as_record)
        else:
            # check if it is an iterable
            if cppyy.gbl._is_iterable[cpp_reference.value_type]():
                # FIXME:
                array = [
                    numpy.asarray(cpp_reference[i]) for i in range(cpp_reference.size())
                ]
                return _wrap_as_array(column, array, column_as_record)

            raise ak._v2._util.error(NotImplementedError)

    else:
        raise ak._v2._util.error(NotImplementedError)
