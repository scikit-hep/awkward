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

    RecordArray(const std::shared_ptr<Identity> id, const std::vector<std::shared_ptr<Content>>& contents, const std::shared_ptr<Lookup>& lookup, const std::shared_ptr<ReverseLookup>& reverselookup)
        : id_(id)
        , contents_(contents)
        , lookup_(lookup)
        , reverselookup_(reverselookup) { }
    RecordArray(const std::shared_ptr<Identity> id, const std::vector<std::shared_ptr<Content>>& contents)
        : id_(id)
        , contents_(contents)
        , lookup_(nullptr)
        , reverselookup_(nullptr) { }
    RecordArray(const std::shared_ptr<Identity> id)
        : id_(id)
        , contents_()
        , lookup_(nullptr)
        , reverselookup_(nullptr) { }

    const std::vector<std::shared_ptr<Content>> contents() const { return contents_; }
    const std::shared_ptr<Lookup> lookup() const { return lookup_; }
    const std::shared_ptr<ReverseLookup> reverselookup() const { return reverselookup_; }

    virtual const std::string classname() const;
    virtual const std::shared_ptr<Identity> id() const { return id_; }
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
    virtual const std::shared_ptr<Content> carry(const Index64& carry) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;

    int64_t numfields() const;
    const std::shared_ptr<Content> field(int64_t i) const;
    const std::shared_ptr<Content> field(const std::string& fieldname) const;
    void append(const std::shared_ptr<Content>& content, const std::string& key);
    void append(const std::shared_ptr<Content>& content);
    void setkey(int64_t i, const std::string& key);

  protected:
    virtual const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const;

  private:
    std::shared_ptr<Identity> id_;
    std::vector<std::shared_ptr<Content>> contents_;
    std::shared_ptr<Lookup> lookup_;
    std::shared_ptr<ReverseLookup> reverselookup_;
  };
}

#endif // AWKWARD_RECORDARRAY_H_
