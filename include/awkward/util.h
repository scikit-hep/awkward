// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UTIL_H_
#define AWKWARD_UTIL_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  class Identities;
  template <typename T>
  class IndexOf;

  namespace util {
    /// @class array_deleter
    ///
    /// @brief Used as a `std::shared_ptr` deleter (second argument) to
    /// overload `delete ptr` with `delete[] ptr`.
    ///
    /// This is necessary for `std::shared_ptr` to contain array buffers.
    ///
    /// See also
    ///   - no_deleter, which does not free memory at all (for borrowed
    ///     references).
    ///   - pyobject_deleter, which reduces the reference count of a
    ///     Python object when there are no more C++ shared pointers
    ///     referencing it.
    template<typename T>
    class EXPORT_SYMBOL array_deleter {
    public:
      /// @brief Called by `std::shared_ptr` when its reference count reaches
      /// zero.
      void operator()(T const *p) {
        delete[] p;
      }
    };

    /// @class no_deleter
    ///
    /// @brief Used as a `std::shared_ptr` deleter (second argument) to
    /// overload `delete ptr` with nothing (no dereferencing).
    ///
    /// This could be used to pass borrowed references with the same
    /// C++ type as owned references.
    ///
    /// See also
    ///   - array_deleter, which frees array buffers, rather than objects.
    ///   - pyobject_deleter, which reduces the reference count of a
    ///     Python object when there are no more C++ shared pointers
    ///     referencing it.
    template<typename T>
    class EXPORT_SYMBOL no_deleter {
    public:
      /// @brief Called by `std::shared_ptr` when its reference count reaches
      /// zero.
      void operator()(T const *p) { }
    };

    /// @brief Puts quotation marks around a string and escapes the appropriate
    /// characters.
    ///
    /// @param x The string to quote.
    /// @param doublequote If `true`, apply double-quotes (`"`); if `false`,
    /// apply single-quotes (`'`).
    ///
    /// @note The implementation does not yet escape characters: it only adds
    /// strings. See issue
    /// [scikit-hep/awkward-1.0#186](https://github.com/scikit-hep/awkward-1.0/issues/186).
    std::string
      quote(const std::string& x, bool doublequote);

    /// @brief If the Error struct contains an error message (from a
    /// cpu-kernel through the C interface), raise that error as a C++
    /// exception.
    ///
    /// @param err The Error struct from a cpu-kernel.
    /// @param classname The name of this class to include in the error
    /// message.
    /// @param id The Identities to include in the error message.
    void
      handle_error(const struct Error& err,
                   const std::string& classname,
                   const Identities* id);

    /// @brief Converts an `offsets` index (from
    /// {@link ListOffsetArrayOf ListOffsetArray}, for instance) into a
    /// `starts` index by viewing it with the last element dropped.
    template <typename T>
    IndexOf<T>
      make_starts(const IndexOf<T>& offsets);

    /// @brief Converts an `offsets` index (from
    /// {@link ListOffsetArrayOf ListOffsetArray}, for instance) into a
    /// `stops` index by viewing it with the first element dropped.
    template <typename T>
    IndexOf<T>
      make_stops(const IndexOf<T>& offsets);

    using RecordLookup    = std::vector<std::string>;
    using RecordLookupPtr = std::shared_ptr<RecordLookup>;

    /// @brief Initializes a RecordLookup by assigning each element with
    /// a string representation of its field index position.
    ///
    /// For example, if `numfields = 3`, the return value is `["0", "1", "2"]`.
    RecordLookupPtr
      init_recordlookup(int64_t numfields);

    /// @brief Returns the field index associated with a key, given
    /// a RecordLookup and a number of fields.
    int64_t
      fieldindex(const RecordLookupPtr& recordlookup,
                 const std::string& key,
                 int64_t numfields);

    /// @brief Returns the key associated with a field index, given a
    /// RecordLookup and a number of fields.
    const std::string
      key(const RecordLookupPtr& recordlookup,
          int64_t fieldindex,
          int64_t numfields);

    /// @brief Returns `true` if a RecordLookup has a given `key`; `false`
    /// otherwise.
    bool
      haskey(const RecordLookupPtr& recordlookup,
             const std::string& key,
             int64_t numfields);

    /// @brief Returns a given RecordLookup as keys or generate anonymous ones
    /// form a number of fields.
    const std::vector<std::string>
      keys(const RecordLookupPtr& recordlookup, int64_t numfields);

    using Parameters = std::map<std::string, std::string>;

    /// @brief Returns `true` if the value associated with a `key` in
    /// `parameters` is equal to the specified `value`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    bool
      parameter_equals(const Parameters& parameters,
                       const std::string& key,
                       const std::string& value);

    /// @brief Returns `true` if all key-value pairs in `self` is equal to
    /// all key-value pairs in `other`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    bool
      parameters_equal(const Parameters& self, const Parameters& other);

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string; `false` otherwise.
    bool
      parameter_isstring(const Parameters& parameters, const std::string& key);

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string that matches `[A-Za-z_][A-Za-z_0-9]*`; `false` otherwise.
    bool
      parameter_isname(const Parameters& parameters, const std::string& key);

    /// @brief Returns the parameter associated with `key` as a string if
    /// #parameter_isstring; raises an error otherwise.
    const std::string
      parameter_asstring(const Parameters& parameters, const std::string& key);

    using TypeStrs = std::map<std::string, std::string>;

    /// @brief Extracts a custom type string from `typestrs` if required by
    /// one of the `parameters` or an empty string if there is no match.
    std::string
      gettypestr(const Parameters& parameters,
                 const TypeStrs& typestrs);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_identities32_from_listoffsetarray(
        int32_t* toptr,
        const int32_t* fromptr,
        const T* fromoffsets,
        int64_t fromptroffset,
        int64_t offsetsoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_identities64_from_listoffsetarray(
        int64_t* toptr,
        const int64_t* fromptr,
        const T* fromoffsets,
        int64_t fromptroffset,
        int64_t offsetsoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_identities32_from_listarray(
        bool* uniquecontents,
        int32_t* toptr,
        const int32_t* fromptr,
        const T* fromstarts,
        const T* fromstops,
        int64_t fromptroffset,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_identities64_from_listarray(
        bool* uniquecontents,
        int64_t* toptr,
        const int64_t* fromptr,
        const T* fromstarts,
        const T* fromstops,
        int64_t fromptroffset,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_identities32_from_indexedarray(
        bool* uniquecontents,
        int32_t* toptr,
        const int32_t* fromptr,
        const T* fromindex,
        int64_t fromptroffset,
        int64_t indexoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_identities64_from_indexedarray(
        bool* uniquecontents,
        int64_t* toptr,
        const int64_t* fromptr,
        const T* fromindex,
        int64_t fromptroffset,
        int64_t indexoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_identities32_from_unionarray(
        bool* uniquecontents,
        int32_t* toptr,
        const int32_t* fromptr,
        const T* fromtags,
        const I* fromindex,
        int64_t fromptroffset,
        int64_t tagsoffset,
        int64_t indexoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth,
        int64_t which);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_identities64_from_unionarray(
        bool* uniquecontents,
        int64_t* toptr,
        const int64_t* fromptr,
        const T* fromtags,
        const I* fromindex,
        int64_t fromptroffset,
        int64_t tagsoffset,
        int64_t indexoffset,
        int64_t tolength,
        int64_t fromlength,
        int64_t fromwidth,
        int64_t which);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_index_carry_64(
        T* toindex,
        const T* fromindex,
        const int64_t* carry,
        int64_t fromindexoffset,
        int64_t lenfromindex,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_index_carry_nocheck_64(
        T* toindex,
        const T* fromindex,
        const int64_t* carry,
        int64_t fromindexoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_at_64(
        int64_t* tocarry,
        const T* fromstarts,
        const T* fromstops,
        int64_t lenstarts,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t at);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_range_carrylength(
        int64_t* carrylength,
        const T* fromstarts,
        const T* fromstops,
        int64_t lenstarts,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t start,
        int64_t stop,
        int64_t step);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_range_64(
        T* tooffsets,
        int64_t* tocarry,
        const T* fromstarts,
        const T* fromstops,
        int64_t lenstarts,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t start,
        int64_t stop,
        int64_t step);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_range_counts_64(
        int64_t* total,
        const T* fromoffsets,
        int64_t lenstarts);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_range_spreadadvanced_64(
        int64_t* toadvanced,
        const int64_t* fromadvanced,
        const T* fromoffsets,
        int64_t lenstarts);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_array_64(
        int64_t* tocarry,
        int64_t* toadvanced,
        const T* fromstarts,
        const T* fromstops,
        const int64_t* fromarray,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t lenstarts,
        int64_t lenarray,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_next_array_advanced_64(
        int64_t* tocarry,
        int64_t* toadvanced,
        const T* fromstarts,
        const T* fromstops,
        const int64_t* fromarray,
        const int64_t* fromadvanced,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t lenstarts,
        int64_t lenarray,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_carry_64(
        T* tostarts,
        T* tostops,
        const T* fromstarts,
        const T* fromstops,
        const int64_t* fromcarry,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t lenstarts,
        int64_t lencarry);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_num_64(
        int64_t* tonum,
        const T* fromstarts,
        int64_t startsoffset,
        const T* fromstops,
        int64_t stopsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listoffsetarray_flatten_offsets_64(
        int64_t* tooffsets,
        const T* outeroffsets,
        int64_t outeroffsetsoffset,
        int64_t outeroffsetslen,
        const int64_t* inneroffsets,
        int64_t inneroffsetsoffset,
        int64_t inneroffsetslen);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_flatten_none2empty_64(
        int64_t* outoffsets,
        const T* outindex,
        int64_t outindexoffset,
        int64_t outindexlength,
        const int64_t* offsets,
        int64_t offsetsoffset,
        int64_t offsetslength);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_flatten_length_64(
        int64_t* total_length,
        const T* fromtags,
        int64_t fromtagsoffset,
        const I* fromindex,
        int64_t fromindexoffset,
        int64_t length,
        int64_t** offsetsraws,
        int64_t* offsetsoffsets);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_flatten_combine_64(
        int8_t* totags,
        int64_t* toindex,
        int64_t* tooffsets,
        const T* fromtags,
        int64_t fromtagsoffset,
        const I* fromindex,
        int64_t fromindexoffset,
        int64_t length,
        int64_t** offsetsraws,
        int64_t* offsetsoffsets);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_flatten_nextcarry_64(
        int64_t* tocarry,
        const T* fromindex,
        int64_t indexoffset,
        int64_t lenindex,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_numnull(
        int64_t* numnull,
        const T* fromindex,
        int64_t indexoffset,
        int64_t lenindex);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_getitem_nextcarry_outindex_64(
        int64_t* tocarry,
        T* toindex,
        const T* fromindex,
        int64_t indexoffset,
        int64_t lenindex,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_getitem_nextcarry_outindex_mask_64(
        int64_t* tocarry,
        int64_t* toindex,
        const T* fromindex,
        int64_t indexoffset,
        int64_t lenindex,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_getitem_nextcarry_64(
        int64_t* tocarry,
        const T* fromindex,
        int64_t indexoffset,
        int64_t lenindex,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_getitem_carry_64(
        T* toindex,
        const T* fromindex,
        const int64_t* fromcarry,
        int64_t indexoffset,
        int64_t lenindex,
        int64_t lencarry);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_overlay_mask8_to64(
        int64_t* toindex,
        const int8_t* mask,
        int64_t maskoffset,
        const T* fromindex,
        int64_t indexoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_mask8(
        int8_t* tomask,
        const T* fromindex,
        int64_t indexoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_simplify32_to64(
        int64_t* toindex,
        const T* outerindex,
        int64_t outeroffset,
        int64_t outerlength,
        const int32_t* innerindex,
        int64_t inneroffset,
        int64_t innerlength);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_simplifyU32_to64(
        int64_t* toindex,
        const T* outerindex,
        int64_t outeroffset,
        int64_t outerlength,
        const uint32_t* innerindex,
        int64_t inneroffset,
        int64_t innerlength);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_simplify64_to64(
        int64_t* toindex,
        const T* outerindex,
        int64_t outeroffset,
        int64_t outerlength,
        const int64_t* innerindex,
        int64_t inneroffset,
        int64_t innerlength);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_regular_index(
        I* toindex,
        const T* fromtags,
        int64_t tagsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_project_64(
        int64_t* lenout,
        int64_t* tocarry,
        const T* fromtags,
        int64_t tagsoffset,
        const I* fromindex,
        int64_t indexoffset,
        int64_t length,
        int64_t which);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_compact_offsets64(
        int64_t* tooffsets,
        const T* fromstarts,
        const T* fromstops,
        int64_t startsoffset,
        int64_t stopsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listoffsetarray_compact_offsets64(
        int64_t* tooffsets,
        const T* fromoffsets,
        int64_t offsetsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_broadcast_tooffsets64(
        int64_t* tocarry,
        const int64_t* fromoffsets,
        int64_t offsetsoffset,
        int64_t offsetslength,
        const T* fromstarts,
        int64_t startsoffset,
        const T* fromstops,
        int64_t stopsoffset,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listoffsetarray_toRegularArray(
        int64_t* size,
        const T* fromoffsets,
        int64_t offsetsoffset,
        int64_t offsetslength);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_simplify8_32_to8_64(
        int8_t* totags,
        int64_t* toindex,
        const T* outertags,
        int64_t outertagsoffset,
        const I* outerindex,
        int64_t outerindexoffset,
        const int8_t* innertags,
        int64_t innertagsoffset,
        const int32_t* innerindex,
        int64_t innerindexoffset,
        int64_t towhich,
        int64_t innerwhich,
        int64_t outerwhich,
        int64_t length,
        int64_t base);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_simplify8_U32_to8_64(
        int8_t* totags,
        int64_t* toindex,
        const T* outertags,
        int64_t outertagsoffset,
        const I* outerindex,
        int64_t outerindexoffset,
        const int8_t* innertags,
        int64_t innertagsoffset,
        const uint32_t* innerindex,
        int64_t innerindexoffset,
        int64_t towhich,
        int64_t innerwhich,
        int64_t outerwhich,
        int64_t length,
        int64_t base);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_simplify8_64_to8_64(
        int8_t* totags,
        int64_t* toindex,
        const T* outertags,
        int64_t outertagsoffset,
        const I* outerindex,
        int64_t outerindexoffset,
        const int8_t* innertags,
        int64_t innertagsoffset,
        const int64_t* innerindex,
        int64_t innerindexoffset,
        int64_t towhich,
        int64_t innerwhich,
        int64_t outerwhich,
        int64_t length,
        int64_t base);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_simplify_one_to8_64(
        int8_t* totags,
        int64_t* toindex,
        const T* fromtags,
        int64_t fromtagsoffset,
        const I* fromindex,
        int64_t fromindexoffset,
        int64_t towhich,
        int64_t fromwhich,
        int64_t length,
        int64_t base);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_jagged_expand_64(
        int64_t* multistarts,
        int64_t* multistops,
        const int64_t* singleoffsets,
        int64_t* tocarry,
        const T* fromstarts,
        int64_t fromstartsoffset,
        const T* fromstops,
        int64_t fromstopsoffset,
        int64_t jaggedsize,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_jagged_apply_64(
        int64_t* tooffsets,
        int64_t* tocarry,
        const int64_t* slicestarts,
        int64_t slicestartsoffset,
        const int64_t* slicestops,
        int64_t slicestopsoffset,
        int64_t sliceouterlen,
        const int64_t* sliceindex,
        int64_t sliceindexoffset,
        int64_t sliceinnerlen,
        const T* fromstarts,
        int64_t fromstartsoffset,
        const T* fromstops,
        int64_t fromstopsoffset,
        int64_t contentlen);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_getitem_jagged_descend_64(
        int64_t* tooffsets,
        const int64_t* slicestarts,
        int64_t slicestartsoffset,
        const int64_t* slicestops,
        int64_t slicestopsoffset,
        int64_t sliceouterlen,
        const T* fromstarts,
        int64_t fromstartsoffset,
        const T* fromstops,
        int64_t fromstopsoffset);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_reduce_next_64(
        int64_t* nextcarry,
        int64_t* nextparents,
        int64_t* outindex,
        const T* index,
        int64_t indexoffset,
        int64_t* parents,
        int64_t parentsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_UnionArray_fillna_64(
        int64_t* toindex,
        const T* fromindex,
        int64_t offset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_ListArray_min_range(
        int64_t* tomin,
        const T* fromstarts,
        const T* fromstops,
        int64_t lenstarts,
        int64_t startsoffset,
        int64_t stopsoffset);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_ListArray_rpad_axis1_64(
        int64_t* toindex,
        const T* fromstarts,
        const T* fromstops,
        T* tostarts,
        T* tostops,
        int64_t target,
        int64_t length,
        int64_t startsoffset,
        int64_t stopsoffset);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_ListArray_rpad_and_clip_length_axis1(
        int64_t* tolength,
        const T* fromstarts,
        const T* fromstops,
        int64_t target,
        int64_t lenstarts,
        int64_t startsoffset,
        int64_t stopsoffset);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_ListOffsetArray_rpad_length_axis1(
        T* tooffsets,
        const T* fromoffsets,
        int64_t offsetsoffset,
        int64_t fromlength,
        int64_t target,
        int64_t* tolength);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_ListOffsetArray_rpad_axis1_64(
        int64_t* toindex,
        const T* fromoffsets,
        int64_t offsetsoffset,
        int64_t fromlength,
        int64_t target);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_ListOffsetArray_rpad_and_clip_axis1_64(
        int64_t* toindex,
        const T* fromoffsets,
        int64_t offsetsoffset,
        int64_t length,
        int64_t target);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_validity(
        const T* starts,
        int64_t startsoffset,
        const T* stops,
        int64_t stopsoffset,
        int64_t length,
        int64_t lencontent);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_indexedarray_validity(
        const T* index,
        int64_t indexoffset,
        int64_t length,
        int64_t lencontent,
        bool isoption);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T, typename I>
    ERROR
      awkward_unionarray_validity(
        const T* tags,
        int64_t tagsoffset,
        const I* index,
        int64_t indexoffset,
        int64_t length,
        int64_t numcontents,
        const int64_t* lencontents);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_localindex_64(
        int64_t* toindex,
        const T* offsets,
        int64_t offsetsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_combinations_length_64(
        int64_t* totallen,
        int64_t* tooffsets,
        int64_t n,
        bool replacement,
        const T* starts,
        int64_t startsoffset,
        const T* stops,
        int64_t stopsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template <typename T>
    ERROR
      awkward_listarray_combinations_64(
        int64_t** tocarry,
        int64_t n,
        bool replacement,
        const T* starts,
        int64_t startsoffset,
        const T* stops,
        int64_t stopsoffset,
        int64_t length);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template<typename T>
    T
      awkward_index_getitem_at_nowrap(
        const T* ptr,
        int64_t offset,
        int64_t at);

    /// @brief Wraps several cpu-kernels from the C interface with a template
    /// to make it easier and more type-safe to call.
    template<typename T>
    void
      awkward_index_setitem_at_nowrap(
        T* ptr,
        int64_t offset,
        int64_t at,
        T value);

    template <typename T>
    ERROR
      awkward_numpyarray_argsort(
        int64_t* toptr,
        const T* fromptr,
        int64_t length,
        const int64_t* offsets,
        int64_t offsetslength,
        bool ascending,
        bool stable);

    template <typename T>
    ERROR
      awkward_numpyarray_sort(
        T* toptr,
        const T* fromptr,
        int64_t length,
        const int64_t* offsets,
        int64_t offsetslength,
        const int64_t* starts,
        const int64_t* parents,
        int64_t parentsoffset,
        int64_t parentslength,
        bool ascending,
        bool stable);
  }
}

#endif // AWKWARD_UTIL_H_
