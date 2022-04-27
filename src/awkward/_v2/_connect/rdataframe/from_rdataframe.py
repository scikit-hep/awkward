# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

compiler = ROOT.gInterpreter.Declare
numpy = ak.nplike.Numpy.instance()

compiler(
    """
#include <iterator>
#include <stdlib.h>

template<typename T>
T* copy_ptr(const T* from_ptr, int64_t size)
{
    T* array = malloc(sizeof(T)*size);
    for (int64_t i = 0; i < size; i++) {
        array[i] = from_ptr[i];
    }
    return array;
}

template <typename T>
std::pair<std::vector<int64_t>, T>
offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& res_vec) {
    auto const& vals = res_vec.GetPtr();
    typedef typename T::value_type value_type;

    std::vector<int64_t> offsets;
    offsets.reserve(vals.size() + 1);
    offsets.emplace_back(0);
    int64_t length = 0;
    std::for_each(vals.begin(), vals.end(), [&] (auto const& n) {
        length += n.size();
        offsets.emplace_back(length);
    });

    std::vector<value_type> data;
    data.reserve(length);
    std::for_each(vals.begin(), vals.end(), [&] (auto const& n) {
        data.insert(data.end(), n.begin(), n.end());
    });
    return {offsets, data};
}

template <typename, typename = void>
constexpr bool is_iterable{};

template <typename T>
constexpr bool is_iterable<
    T,
    std::void_t< decltype(std::declval<T>().begin()),
                 decltype(std::declval<T>().end())
    >
> = true;

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {
};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {
};

template <typename T, typename std::enable_if<is_specialization<T, std::complex>::value, T>::type * = nullptr>
std::pair<std::string, std::pair<std::vector<int64_t>, T>> check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& res_vec) {
    return {std::string("complex"), {{}, {}}};
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, T>::type * = nullptr>
std::pair<std::string, std::pair<std::vector<int64_t>, T>> check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& res_vec) {
    return {std::string("primitive"), {{}, {}}};
}

template <typename T, typename std::enable_if<is_iterable<T>, T>::type * = nullptr>
std::pair<std::string, std::pair<std::vector<int64_t>, T>> check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& res_vec) {
    auto str = std::string(typeid(T).name());
    if (str.find("awkward") != string::npos) {
        return {std::string("awkward"), {{}, {}}};
    }
    else {
        typedef typename T::value_type value_type;
        if (is_iterable<value_type>) {
            cout << "FIXME: Fast copy is not implemented yet." << endl;
        } else if (std::is_arithmetic<value_type>::value) {
            return {std::string("iterable"), offsets_and_flatten(res_vec)};
        }
        return {std::string("iterable"), {{}, {}}};
    }
    return {"undefined", {{}, {}}};
}

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

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)
    ptrs_type, data_pair = ROOT.check_type_of[column_type](result_ptrs)

    if ptrs_type == "primitive" or ptrs_type == "complex":

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        content = ak._v2.contents.NumpyArray(numpy.asarray(cpp_reference))

        return (
            ak._v2._util.wrap(
                ak._v2.contents.RecordArray(
                    fields=[column],
                    contents=[content],
                ),
                highlevel=True,
            )
            if column_as_record
            else ak._v2._util.wrap(content, highlevel=True)
        )

    elif ptrs_type == "iterable":

        content = ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(data_pair.first),
            ak._v2.contents.NumpyArray(numpy.asarray(data_pair.second)),
        )

        return (
            ak._v2._util.wrap(
                ak._v2.contents.RecordArray(
                    fields=[column],
                    contents=[content],
                ),
                highlevel=True,
            )
            if column_as_record
            else ak._v2._util.wrap(
                content,
                highlevel=True,
            )
        )

    elif ptrs_type == "awkward":

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        return _wrap_as_array(column, cpp_reference, column_as_record)
    else:
        raise ak._v2._util.error(NotImplementedError)
