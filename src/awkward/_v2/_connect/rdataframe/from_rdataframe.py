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
"""
)


def from_rdataframe(data_frame, column, column_as_record=True):

    column_type = data_frame.GetColumnType(column)
    result_ptrs = data_frame.Take[column_type](column)
    cpp_reference = result_ptrs.GetValue()

    # check that its an std::vector
    if cppyy.typeid(cppyy.gbl.std.vector[column_type]()) == cppyy.typeid(cpp_reference):

        # check if it's an integral or a floating point type
        if cppyy.gbl._is_arithmetic[cpp_reference.value_type]():

            array = numpy.asarray(cpp_reference)

            # FIXME: copy data?
            return (
                ak._v2.highlevel.Array({column: array})
                if column_as_record
                else ak._v2.highlevel.Array(array)
            )
        else:
            # check if it is iterable
            if cppyy.gbl._is_iterable[cpp_reference.value_type]():
                # print("Iterable!", cpp_reference.value_type)
                pass

            raise ak._v2._util.error(NotImplementedError)

    else:
        raise ak._v2._util.error(NotImplementedError)
