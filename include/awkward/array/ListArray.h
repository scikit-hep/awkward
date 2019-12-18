// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTARRAY_H_
#define AWKWARD_LISTARRAY_H_

#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Index.h"
#include "awkward/Identity.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T>
  class ListArrayOf: public Content {
  public:
    ListArrayOf<T>(const std::shared_ptr<Identity>& id, const std::shared_ptr<Type>& type, const IndexOf<T>& starts, const IndexOf<T>& stops, const std::shared_ptr<Content>& content)
        : Content(id, type)
        , starts_(starts)
        , stops_(stops)
        , content_(content) {
      if (type_.get() != nullptr) {
        checktype();
      }
    }

    const IndexOf<T> starts() const { return starts_; }
    const IndexOf<T> stops() const { return stops_; }
    const std::shared_ptr<Content> content() const { return content_; }

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
    const std::vector<std::string> keyaliases(int64_t fieldindex) const override;
    const std::vector<std::string> keyaliases(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

  protected:
    void checktype() const override;

    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;

  private:
    const IndexOf<T> starts_;
    const IndexOf<T> stops_;
    const std::shared_ptr<Content> content_;
  };

  typedef ListArrayOf<int32_t>  ListArray32;
  typedef ListArrayOf<uint32_t> ListArrayU32;
  typedef ListArrayOf<int64_t>  ListArray64;
}

#endif // AWKWARD_LISTARRAY_H_
