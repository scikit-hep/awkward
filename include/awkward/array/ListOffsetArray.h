// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTOFFSETARRAY_H_
#define AWKWARD_LISTOFFSETARRAY_H_

#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Index.h"
#include "awkward/Identity.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T>
  class ListOffsetArrayOf: public Content {
  public:
    ListOffsetArrayOf<T>(const std::shared_ptr<Identity>& id, const util::Parameters& parameters, const IndexOf<T>& offsets, const std::shared_ptr<Content>& content);
    const IndexOf<T> offsets() const;
    const std::shared_ptr<Content> content() const;

    const std::string classname() const override;
    void setid() override;
    void setid(const std::shared_ptr<Identity>& id) override;
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

  protected:
    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;

  private:
    const IndexOf<T> offsets_;
    const std::shared_ptr<Content> content_;
  };

  typedef ListOffsetArrayOf<int32_t>  ListOffsetArray32;
  typedef ListOffsetArrayOf<uint32_t> ListOffsetArrayU32;
  typedef ListOffsetArrayOf<int64_t>  ListOffsetArray64;
}

#endif // AWKWARD_LISTOFFSETARRAY_H_
