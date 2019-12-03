// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORD_H_
#define AWKWARD_RECORD_H_

#include "awkward/array/RecordArray.h"

namespace awkward {
  class Record: public Content {
  public:
    Record(const RecordArray& array, int64_t at)
        : array_(array)
        , at_(at) { }

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

    virtual bool isscalar() const;
    virtual const std::string classname() const;
    virtual const std::shared_ptr<Identity> id() const;
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    virtual void tojson_part(ToJson& builder) const;
    virtual const std::shared_ptr<Type> type_part() const;
    virtual int64_t length() const;
    virtual const std::shared_ptr<Content> shallow_copy() const;
    virtual void check_for_iteration() const;
    virtual const std::shared_ptr<Content> getitem_nothing() const;
    virtual const std::shared_ptr<Content> getitem_at(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem_field(const std::string& key) const;
    virtual const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const;
    virtual const std::shared_ptr<Content> carry(const Index64& carry) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;

    int64_t numfields() const;
    int64_t index(const std::string& key) const;
    const std::string key(int64_t index) const;
    bool has(const std::string& key) const;
    const std::vector<std::string> aliases(int64_t index) const;
    const std::vector<std::string> aliases(const std::string& key) const;
    const std::shared_ptr<Content> field(int64_t index) const;
    const std::shared_ptr<Content> field(const std::string& key) const;
    const std::vector<std::string> keys() const;
    const std::vector<std::shared_ptr<Content>> values() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Content>>> items() const;
    const Record astuple() const;

  protected:
    virtual const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const;

  private:
    const RecordArray array_;
    int64_t at_;
  };
}

#endif // AWKWARD_RECORD_H_
