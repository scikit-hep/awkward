// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NUMPYARRAY_H_
#define AWKWARD_NUMPYARRAY_H_

#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <memory>
#include <stdexcept>

#include "awkward/cpu-kernels/util.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  class NumpyArray: public Content {
  public:
    NumpyArray(const std::shared_ptr<Identity> id, const std::shared_ptr<void> ptr, const std::vector<ssize_t> shape, const std::vector<ssize_t> strides, ssize_t byteoffset, ssize_t itemsize, const std::string format)
        : id_(id)
        , ptr_(ptr)
        , shape_(shape)
        , strides_(strides)
        , byteoffset_(byteoffset)
        , itemsize_(itemsize)
        , format_(format) {
          assert(shape_.size() == strides_.size());
        }

    const std::shared_ptr<void> ptr() const { return ptr_; }
    const std::vector<ssize_t> shape() const { return shape_; }
    const std::vector<ssize_t> strides() const { return strides_; }
    ssize_t byteoffset() const { return byteoffset_; }
    ssize_t itemsize() const { return itemsize_; }
    const std::string format() const { return format_; }

    ssize_t ndim() const;
    bool isscalar() const;
    bool isempty() const;
    bool iscompact() const;
    void* byteptr() const;
    ssize_t bytelength() const;
    uint8_t getbyte(ssize_t at) const;

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    virtual int64_t length() const;
    virtual const std::shared_ptr<Content> shallow_copy() const;
    virtual const std::shared_ptr<Content> get(int64_t at) const;
    virtual const std::shared_ptr<Content> slice(int64_t start, int64_t stop) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;

    const std::shared_ptr<Content> getitem(Slice& slice);
    const std::shared_ptr<Content> getitem_next(std::shared_ptr<SliceItem> head, Slice& tail, std::shared_ptr<Index> carry);

  private:
    std::shared_ptr<Identity> id_;
    const std::shared_ptr<void> ptr_;
    const std::vector<ssize_t> shape_;
    const std::vector<ssize_t> strides_;
    const ssize_t byteoffset_;
    const ssize_t itemsize_;
    const std::string format_;
  };
}

#endif // AWKWARD_NUMPYARRAY_H_
