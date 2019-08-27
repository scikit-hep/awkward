// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTOFFSETARRAYCONTENT_H_
#define AWKWARD_LISTOFFSETARRAYCONTENT_H_

#include <sstream>
#include <memory>

#include "awkward/util.h"
#include "awkward/Index.h"
#include "awkward/Content.h"

namespace awkward {
  class ListOffsetArray: public Content {
  public:
    ListOffsetArray(const Index offsets, const std::shared_ptr<Content> content)
        : id_(nullptr)
        , offsets_(offsets)
        , content_(content) { }

    const Index offsets() const { return offsets_; }
    const std::shared_ptr<Content> content() const { return content_.get()->shallow_copy(); }

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid(const std::shared_ptr<Identity> id) { id_ = id; };
    virtual void setid();
    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    virtual IndexType length() const;
    virtual std::shared_ptr<Content> shallow_copy() const;
    virtual std::shared_ptr<Content> get(IndexType at) const;
    virtual std::shared_ptr<Content> slice(IndexType start, IndexType stop) const;

  private:
    std::shared_ptr<Identity> id_;
    const Index offsets_;
    const std::shared_ptr<Content> content_;
  };
}

#endif // AWKWARD_LISTOFFSETARRAYCONTENT_H_
