// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_EMPTYARRAY_H_
#define AWKWARD_EMPTYARRAY_H_

#include <cassert>
#include <string>
#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  class EmptyArray: public Content {
  public:
    EmptyArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters);

    const std::string classname() const override;
    void setidentities() override;
    void setidentities(const std::shared_ptr<Identities>& identities) override;
    const std::shared_ptr<Type> type() const override;
    const std::shared_ptr<Content> astype(const std::shared_ptr<Type>& type) const override;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
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
    const std::shared_ptr<Content> carry(const Index64& carry) const override;
    const std::pair<int64_t, int64_t> minmax_depth() const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;
    const Index64 toindex64() const override;

    // operations
    const std::shared_ptr<Content> count(int64_t axis) const override;
    const std::shared_ptr<Content> flatten(int64_t axis) const override;

  protected:
    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override;
  };
}

#endif // AWKWARD_EMPTYARRAY_H_
