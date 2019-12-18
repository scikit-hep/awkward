// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORD_H_
#define AWKWARD_RECORD_H_

#include "awkward/array/RecordArray.h"

namespace awkward {
  class Record: public Content {
  public:
    Record(const RecordArray& array, int64_t at)
        : Content(Identity::none(), Type::none())
        , array_(array)
        , at_(at) {
      if (type_.get() != nullptr) {
        checktype();
      }
    }

    const std::shared_ptr<Content> array() const { return array_.shallow_copy(); }
    int64_t at() const { return at_; }
    const std::vector<std::shared_ptr<Content>> contents() const {
      std::vector<std::shared_ptr<Content>> out;
      for (auto item : array_.contents()) {
        out.push_back(item.get()->getitem_at_nowrap(at_));
      }
      return out;
    }
    const std::shared_ptr<RecordArray::Lookup> lookup() const { return array_.lookup(); }
    const std::shared_ptr<RecordArray::ReverseLookup> reverselookup() const { return array_.reverselookup(); }
    bool istuple() const { return lookup().get() == nullptr; }

    bool isscalar() const override;
    const std::string classname() const override;
    const std::shared_ptr<Identity> id() const override;
    void setid() override;
    void setid(const std::shared_ptr<Identity>& id) override;
    bool isbare() const override;
    bool istypeptr(Type* pointer) const override;
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

    const std::shared_ptr<Content> field(int64_t fieldindex) const;
    const std::shared_ptr<Content> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Content>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Content>>> fielditems() const;
    const Record astuple() const;

  protected:
    void checktype() const override;

    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override;

  private:
    RecordArray array_;
    int64_t at_;
  };
}

#endif // AWKWARD_RECORD_H_
