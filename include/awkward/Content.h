// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Identity.h"
#include "awkward/Slice.h"

namespace awkward {
  class Content {
  public:
    virtual ~Content() { }

    virtual const std::string classname() const = 0;
    virtual const std::shared_ptr<Identity> id() const = 0;
    virtual void setid() = 0;
    virtual void setid(const std::shared_ptr<Identity> id) = 0;
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const = 0;
    virtual int64_t length() const = 0;
    virtual const std::shared_ptr<Content> shallow_copy() const = 0;
    virtual void checksafe() const = 0;
    virtual const std::shared_ptr<Content> getitem_at(int64_t at) const = 0;
    virtual const std::shared_ptr<Content> getitem_at_unsafe(int64_t at) const = 0;
    virtual const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const = 0;
    virtual const std::shared_ptr<Content> getitem_range_unsafe(int64_t start, int64_t stop) const = 0;
    virtual const std::shared_ptr<Content> getitem(const Slice& where) const;
    virtual const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const = 0;
    virtual const std::shared_ptr<Content> carry(const Index64& carry) const = 0;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const = 0;

    const std::string tostring() const;
    const std::shared_ptr<Content> getitem_ellipsis(const Slice& tail, const Index64& advanced) const;
    const std::shared_ptr<Content> getitem_newaxis(const Slice& tail, const Index64& advanced) const;
  };
}

#endif // AWKWARD_CONTENT_H_
