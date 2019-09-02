// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Identity.h"

namespace awkward {
  class Content {
  public:
    virtual const std::shared_ptr<Identity> id() const = 0;
    virtual void setid() = 0;
    virtual void setid(const std::shared_ptr<Identity> id) = 0;
    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const = 0;
    virtual int64_t length() const = 0;
    virtual std::shared_ptr<Content> shallow_copy() const = 0;
    virtual std::shared_ptr<Content> get(int64_t at) const = 0;
    virtual std::shared_ptr<Content> slice(int64_t start, int64_t stop) const = 0;

    const std::string repr() const;
  };
}

#endif // AWKWARD_CONTENT_H_
