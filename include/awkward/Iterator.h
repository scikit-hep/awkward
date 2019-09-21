// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ITERATOR_H_
#define AWKWARD_ITERATOR_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"

namespace awkward {
  class Iterator {
  public:
    Iterator(const std::shared_ptr<Content> content)
        : content_(content)
        , where_(0) { }

    const std::shared_ptr<Content> content() const { return content_; }
    const int64_t where() const { return where_; }

    const bool isdone() const { return where_ >= content_.get()->length(); }
    const std::shared_ptr<Content> next() { return content_.get()->get(where_++); }

    const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    const std::string tostring() const;

  private:
    const std::shared_ptr<Content> content_;
    int64_t where_;
  };
}

#endif // AWKWARD_ITERATOR_H_
