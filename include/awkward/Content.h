// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include "awkward/util.h"

namespace awkward {
  class Content {
  public:
    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const = 0;
    virtual IndexType length() const = 0;
    virtual std::shared_ptr<Content> shallow_copy() const = 0;
    virtual std::shared_ptr<Content> get(AtType at) const = 0;
    virtual std::shared_ptr<Content> slice(AtType start, AtType stop) const = 0;
  };
}

#endif // AWKWARD_CONTENT_H_
