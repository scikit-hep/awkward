// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDARRAY_H_
#define AWKWARD_RECORDARRAY_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Identity.h"
#include "awkward/Content.h"

namespace awkward {
  class RecordArray: public Content {
  public:
    RecordArray(const std::shared_ptr<Identity>& id, const std::shared_ptr<Type>& type, const std::vector<std::shared_ptr<Content>>& contents, const std::shared_ptr<util::RecordLookup>& recordlookup);
    RecordArray(const std::shared_ptr<Identity>& id, const std::shared_ptr<Type>& type, const std::vector<std::shared_ptr<Content>>& contents);
    RecordArray(const std::shared_ptr<Identity>& id, const std::shared_ptr<Type>& type, int64_t length, bool istuple);

    const std::vector<std::shared_ptr<Content>> contents() const;
    const std::shared_ptr<util::RecordLookup> recordlookup() const;
    bool istuple() const;

    const std::string classname() const override;
    void setid() override;
    void setid(const std::shared_ptr<Identity>& id) override;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> type() const override;
    const std::shared_ptr<Content> astype(const std::shared_ptr<Type>& type) const override;
    void tojson_part(ToJson& builder) const override;
    int64_t length() const override;
    const std::shared_ptr<Content> shallow_copy() const override;
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
    const std::pair<int64_t, int64_t> minmax_depth() const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    const std::shared_ptr<Content> field(int64_t fieldindex) const;
    const std::shared_ptr<Content> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Content>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Content>>> fielditems() const;
    const RecordArray astuple() const;

    void append(const std::shared_ptr<Content>& content, const std::string& key);
    void append(const std::shared_ptr<Content>& content);

  protected:
    void checktype() const override;

    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override;

  private:
    std::vector<std::shared_ptr<Content>> contents_;
    std::shared_ptr<util::RecordLookup> recordlookup_;
    int64_t length_;
  };
}

#endif // AWKWARD_RECORDARRAY_H_
