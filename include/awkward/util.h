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
    template<typename T>
    class EXPORT_SYMBOL array_deleter {
    public:
      void operator()(T const *p) {
        delete[] p;
      }
    };

    template<typename T>
    class EXPORT_SYMBOL no_deleter {
    public:
      void operator()(T const *p) { }
    };

    std::string quote(const std::string& x, bool doublequote);

    void handle_error(const struct Error& err, const std::string& classname, const Identities* id);

    template <typename T>
    IndexOf<T> make_starts(const IndexOf<T>& offsets);
    template <typename T>
    IndexOf<T> make_stops(const IndexOf<T>& offsets);

    typedef std::vector<std::string> RecordLookup;
    std::shared_ptr<RecordLookup> init_recordlookup(int64_t numfields);
    int64_t fieldindex(const std::shared_ptr<RecordLookup>& recordlookup, const std::string& key, int64_t numfields);
    const std::string key(const std::shared_ptr<RecordLookup>& recordlookup, int64_t fieldindex, int64_t numfields);
    bool haskey(const std::shared_ptr<RecordLookup>& recordlookup, const std::string& key, int64_t numfields);
    const std::vector<std::string> keys(const std::shared_ptr<RecordLookup>& recordlookup, int64_t numfields);

    typedef std::map<std::string, std::string> Parameters;
    bool parameter_equals(const Parameters& parameters, const std::string& key, const std::string& value);
    bool parameters_equal(const Parameters& self, const Parameters& other);

    template <typename T>
    ERROR awkward_identities32_from_listoffsetarray(int32_t* toptr, const int32_t* fromptr, const T* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    ERROR awkward_identities64_from_listoffsetarray(int64_t* toptr, const int64_t* fromptr, const T* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    ERROR awkward_identities32_from_listarray(bool* uniquecontents, int32_t* toptr, const int32_t* fromptr, const T* fromstarts, const T* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    ERROR awkward_identities64_from_listarray(bool* uniquecontents, int64_t* toptr, const int64_t* fromptr, const T* fromstarts, const T* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    ERROR awkward_identities32_from_indexedarray(bool* uniquecontents, int32_t* toptr, const int32_t* fromptr, const T* fromindex, int64_t fromptroffset, int64_t indexoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    ERROR awkward_identities64_from_indexedarray(bool* uniquecontents, int64_t* toptr, const int64_t* fromptr, const T* fromindex, int64_t fromptroffset, int64_t indexoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T, typename I>
    ERROR awkward_identities32_from_unionarray(bool* uniquecontents, int32_t* toptr, const int32_t* fromptr, const T* fromtags, const I* fromindex, int64_t fromptroffset, int64_t tagsoffset, int64_t indexoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth, int64_t which);
    template <typename T, typename I>
    ERROR awkward_identities64_from_unionarray(bool* uniquecontents, int64_t* toptr, const int64_t* fromptr, const T* fromtags, const I* fromindex, int64_t fromptroffset, int64_t tagsoffset, int64_t indexoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth, int64_t which);
    template <typename T>
    ERROR awkward_index_carry_64(T* toindex, const T* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length);
    template <typename T>
    ERROR awkward_index_carry_nocheck_64(T* toindex, const T* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length);
    template <typename T>
    ERROR awkward_listarray_getitem_next_at_64(int64_t* tocarry, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at);
    template <typename T>
    ERROR awkward_listarray_getitem_next_range_carrylength(int64_t* carrylength, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
    template <typename T>
    ERROR awkward_listarray_getitem_next_range_64(T* tooffsets, int64_t* tocarry, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
    template <typename T>
    ERROR awkward_listarray_getitem_next_range_counts_64(int64_t* total, const T* fromoffsets, int64_t lenstarts);
    template <typename T>
    ERROR awkward_listarray_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const T* fromoffsets, int64_t lenstarts);
    template <typename T>
    ERROR awkward_listarray_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const T* fromstarts, const T* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
    template <typename T>
    ERROR awkward_listarray_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const T* fromstarts, const T* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
    template <typename T>
    ERROR awkward_listarray_getitem_carry_64(T* tostarts, T* tostops, const T* fromstarts, const T* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry);
    template <typename T>
    ERROR awkward_listarray_count(T* tocount, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
    template <typename T>
    ERROR awkward_listarray_count_64(int64_t* tocount, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
    template <typename T>
    ERROR awkward_listoffsetarray_count(T* tocount, const T* fromoffsets, int64_t lenoffsets);
    template <typename T>
    ERROR awkward_listoffsetarray_count_64(int64_t* tocount, const T* fromoffsets, int64_t lenoffsets);
    template <typename T>
    ERROR awkward_indexedarray_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const T* fromindex, int64_t lenindex, int64_t indexoffset);
    template <typename T>
    ERROR awkward_listarray_flatten_length(int64_t* tolen, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
    template <typename T>
    ERROR awkward_listarray_flatten_64(int64_t* tocarry, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
    template <typename T>
    ERROR awkward_listarray_flatten_scale_64(T* tostarts, T* tostops, const int64_t* scale, const T* fromstarts, const T* fromstops,  int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
    template <typename T>
    ERROR awkward_indexedarray_flatten_nextcarry_64(int64_t* tocarry, const T* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
    template <typename T>
    ERROR awkward_indexedarray_numnull(int64_t* numnull, const T* fromindex, int64_t indexoffset, int64_t lenindex);
    template <typename T>
    ERROR awkward_indexedarray_getitem_nextcarry_outindex_64(int64_t* tocarry, T* toindex, const T* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
    template <typename T>
    ERROR awkward_indexedarray_getitem_nextcarry_outindex_mask_64(int64_t* tocarry, int64_t* toindex, const T* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
    template <typename T>
    ERROR awkward_indexedarray_getitem_nextcarry_64(int64_t* tocarry, const T* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
    template <typename T>
    ERROR awkward_indexedarray_getitem_carry_64(T* toindex, const T* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry);
    template <typename T>
    ERROR awkward_indexedarray_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const T* fromindex, int64_t indexoffset, int64_t length);
    template <typename T>
    ERROR awkward_indexedarray_mask8(int8_t* tomask, const T* fromindex, int64_t indexoffset, int64_t length);
    template <typename T>
    ERROR awkward_indexedarray_simplify32_to64(int64_t* toindex, const T* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength);
    template <typename T>
    ERROR awkward_indexedarray_simplifyU32_to64(int64_t* toindex, const T* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength);
    template <typename T>
    ERROR awkward_indexedarray_simplify64_to64(int64_t* toindex, const T* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength);
    template <typename T, typename I>
    ERROR awkward_unionarray_regular_index(I* toindex, const T* fromtags, int64_t tagsoffset, int64_t length);
    template <typename T, typename I>
    ERROR awkward_unionarray_project_64(int64_t* lenout, int64_t* tocarry, const T* fromtags, int64_t tagsoffset, const I* fromindex, int64_t indexoffset, int64_t length, int64_t which);
    template <typename T>
    ERROR awkward_listarray_compact_offsets64(int64_t* tooffsets, const T* fromstarts, const T* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length);
    template <typename T>
    ERROR awkward_listoffsetarray_compact_offsets64(int64_t* tooffsets, const T* fromoffsets, int64_t offsetsoffset, int64_t length);
    template <typename T>
    ERROR awkward_listarray_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const T* fromstarts, int64_t startsoffset, const T* fromstops, int64_t stopsoffset, int64_t lencontent);
    template <typename T>
    ERROR awkward_listoffsetarray_toRegularArray(int64_t* size, const T* fromoffsets, int64_t offsetsoffset, int64_t offsetslength);
    template <typename T, typename I>
    ERROR awkward_unionarray_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const T* outertags, int64_t outertagsoffset, const I* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
    template <typename T, typename I>
    ERROR awkward_unionarray_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const T* outertags, int64_t outertagsoffset, const I* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
    template <typename T, typename I>
    ERROR awkward_unionarray_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const T* outertags, int64_t outertagsoffset, const I* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
    template <typename T, typename I>
    ERROR awkward_unionarray_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const T* fromtags, int64_t fromtagsoffset, const I* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base);

    template <typename T>
    Error awkward_listarray_getitem_jagged_expand_64(int64_t* multistarts, int64_t* multistops, const int64_t* singleoffsets, int64_t* tocarry, const T* fromstarts, int64_t fromstartsoffset, const T* fromstops, int64_t fromstopsoffset, int64_t jaggedsize, int64_t length);

    template <typename T>
    ERROR awkward_listarray_getitem_jagged_apply_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const int64_t* sliceindex, int64_t sliceindexoffset, int64_t sliceinnerlen, const T* fromstarts, int64_t fromstartsoffset, const T* fromstops, int64_t fromstopsoffset, int64_t contentlen);

    template <typename T>
    ERROR awkward_listarray_getitem_jagged_descend_64(int64_t* tooffsets, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const T* fromstarts, int64_t fromstartsoffset, const T* fromstops, int64_t fromstopsoffset);

    template <typename T>
    ERROR awkward_indexedarray_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, const T* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length);

  }
}

#endif // AWKWARD_UTIL_H_
