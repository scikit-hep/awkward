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
    typedef std::unordered_map<std::string, size_t> Lookup;
    typedef std::vector<std::string> ReverseLookup;

    RecordArray(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const std::vector<std::shared_ptr<Content>>& contents, const std::shared_ptr<Lookup>& lookup, const std::shared_ptr<ReverseLookup>& reverselookup)
        : Content(id, type)
        , contents_(contents)
        , lookup_(lookup)
        , reverselookup_(reverselookup)
        , length_(0) {
      if (reverselookup_.get() == nullptr  &&  lookup_.get() == nullptr) { }
      else if (reverselookup_.get() != nullptr  &&  lookup_.get() != nullptr) { }
      else {
        throw std::runtime_error("either 'lookup' and 'reverselookup' should both be None or neither should be");
      }
      if (contents_.size() == 0) {
        throw std::runtime_error("this constructor can only be used with non-empty contents");
      }
      if (type_.get() != nullptr) {
        checktype();
      }
    }
    RecordArray(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const std::vector<std::shared_ptr<Content>>& contents)
        : Content(id, type)
        , contents_(contents)
        , lookup_(nullptr)
        , reverselookup_(nullptr)
        , length_(0) {
      if (contents_.size() == 0) {
        throw std::runtime_error("this constructor can only be used with non-empty contents");
      }
      if (type_.get() != nullptr) {
        checktype();
      }
    }
    RecordArray(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, int64_t length, bool istuple)
        : Content(id, type)
        , contents_()
        , lookup_(istuple ? nullptr : new Lookup)
        , reverselookup_(istuple ? nullptr : new ReverseLookup)
        , length_(length) {
      if (type_.get() != nullptr) {
        checktype();
      }
    }

    const std::vector<std::shared_ptr<Content>> contents() const { return contents_; }
    const std::shared_ptr<Lookup> lookup() const { return lookup_; }
    const std::shared_ptr<ReverseLookup> reverselookup() const { return reverselookup_; }
    bool istuple() const { return lookup_.get() == nullptr; }

    virtual const std::string classname() const;
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> astype(const std::shared_ptr<Type> type) const;
    virtual void tojson_part(ToJson& builder) const;
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
    virtual const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> carry(const Index64& carry) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;
    virtual int64_t numfields() const;
    virtual int64_t fieldindex(const std::string& key) const;
    virtual const std::string key(int64_t fieldindex) const;
    virtual bool haskey(const std::string& key) const;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const;
    virtual const std::vector<std::string> keys() const;

    const std::shared_ptr<Content> field(int64_t fieldindex) const;
    const std::shared_ptr<Content> field(const std::string& key) const;
    const std::vector<std::shared_ptr<Content>> fields() const;
    const std::vector<std::pair<std::string, std::shared_ptr<Content>>> fielditems() const;
    const RecordArray astuple() const;

    void append(const std::shared_ptr<Content>& content, const std::string& key);
    void append(const std::shared_ptr<Content>& content);
    void setkey(int64_t fieldindex, const std::string& key);

  protected:
    virtual void checktype() const;

    virtual const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const;

  private:
    std::vector<std::shared_ptr<Content>> contents_;
    std::shared_ptr<Lookup> lookup_;
    std::shared_ptr<ReverseLookup> reverselookup_;
    int64_t length_;
  };
}

#endif // AWKWARD_RECORDARRAY_H_
