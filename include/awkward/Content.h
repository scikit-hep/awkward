// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include <utility>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Slice.h"
#include "awkward/Identity.h"

namespace awkward {
  class Content {
  public:
    virtual const std::shared_ptr<Identity> id() const = 0;
    virtual void setid() = 0;
    virtual void setid(const std::shared_ptr<Identity> id) = 0;
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const = 0;
    virtual int64_t length() const = 0;
    virtual const std::shared_ptr<Content> shallow_copy() const = 0;
    virtual const std::shared_ptr<Content> get(int64_t at) const = 0;
    virtual const std::shared_ptr<Content> slice(int64_t start, int64_t stop) const = 0;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const = 0;
    virtual const std::shared_ptr<Content> getitem(Slice& slice) const = 0;
    virtual const std::shared_ptr<Content> getitem_next(std::shared_ptr<SliceItem> head, Slice& tail, std::shared_ptr<Index> carry) const = 0;

    const std::string tostring() const;
  };
}

#endif // AWKWARD_CONTENT_H_
