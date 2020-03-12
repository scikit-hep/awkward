// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDARRAY_H_
#define AWKWARD_RECORDARRAY_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Identities.h"
#include "awkward/Content.h"

namespace awkward {
  class EXPORT_SYMBOL RecordArray: public Content, public std::enable_shared_from_this<RecordArray> {
  public:
    RecordArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::vector<std::shared_ptr<Content>>& contents, const std::shared_ptr<util::RecordLookup>& recordlookup, int64_t length);
    RecordArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::vector<std::shared_ptr<Content>>& contents, const std::shared_ptr<util::RecordLookup>& recordlookup);

    const std::vector<std::shared_ptr<Content>> contents() const;
    const std::shared_ptr<util::RecordLookup> recordlookup() const;
    bool istuple() const;
    const std::shared_ptr<Content> setitem_field(int64_t where, const std::shared_ptr<Content>& what) const;
    const std::shared_ptr<Content> setitem_field(const std::string& where, const std::shared_ptr<Content>& what) const;

    const std::string classname() const override;
    void setidentities() override;
    void setidentities(const std::shared_ptr<Identities>& identities) override;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> type(const std::map<std::string, std::string>& typestrs) const override;
    void tojson_part(ToJson& builder) const override;
    void nbytes_part(std::map<size_t, int64_t>& largest) const override;
    int64_t length() const override;
    const std::shared_ptr<Content> shallow_copy() const override;
    const std::shared_ptr<Content> deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const override;
    void check_for_iteration() const override;
    const std::shared_ptr<Content> getitem_nothing() const override;
    const std::shared_ptr<Content> getitem_at(int64_t at) const override;
    const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const override;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Content> getitem_field(const std::string& key) const override;
    const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const override;
    const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> carry(const Index64& carry) const override;
    const std::string purelist_parameter(const std::string& key) const override;
    bool purelist_isregular() const override;
    int64_t purelist_depth() const override;
    const std::pair<int64_t, int64_t> minmax_depth() const override;
    const std::pair<bool, int64_t> branch_depth() const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    // operations
    const std::string validityerror(const std::string& path) const override;
    const Index64 count64() const override;
    const std::shared_ptr<Content> count(int64_t axis) const override;
    const std::shared_ptr<Content> flatten(int64_t axis) const override;
    bool mergeable(const std::shared_ptr<Content>& other, bool mergebool) const override;
    const std::shared_ptr<Content> merge(const std::shared_ptr<Content>& other) const override;
    const std::shared_ptr<SliceItem> asslice() const override;
    const std::shared_ptr<Content> fillna(const std::shared_ptr<Content>& value) const override;
    const std::shared_ptr<Content> rpad(int64_t length, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const override;

    const std::shared_ptr<Content> field(int64_t fieldindex) const;
    const std::shared_ptr<Content> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Content>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Content>>> fielditems() const;
    const std::shared_ptr<RecordArray> astuple() const;

    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const override;

  protected:
    template <typename S>
    const std::shared_ptr<Content> getitem_next_jagged_generic(const Index64& slicestarts, const Index64& slicestops, const S& slicecontent, const Slice& tail) const;

  private:
    const std::vector<std::shared_ptr<Content>> contents_;
    const std::shared_ptr<util::RecordLookup> recordlookup_;
    int64_t length_;
  };
}

#endif // AWKWARD_RECORDARRAY_H_
