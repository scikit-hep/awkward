// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NUMPYARRAY_H_
#define AWKWARD_NUMPYARRAY_H_

#include <cassert>
#include <string>
#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
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
    void* byteptr() const;
    ssize_t bytelength() const;
    uint8_t getbyte(ssize_t at) const;

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    virtual int64_t length() const;
    virtual const std::shared_ptr<Content> shallow_copy() const;
    virtual const std::shared_ptr<Content> getitem_at(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem(const Slice& slice) const;
    virtual const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry, const Index64& advanced) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;

    bool iscontiguous() const;
    void become_contiguous();
    const NumpyArray contiguous() const;
    const NumpyArray contiguous_next(Index64 bytepos) const;
    const NumpyArray getitem_bystrides(const std::shared_ptr<SliceItem>& head, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride) const;

  private:
    std::shared_ptr<Identity> id_;
    std::shared_ptr<void> ptr_;
    std::vector<ssize_t> shape_;
    std::vector<ssize_t> strides_;
    ssize_t byteoffset_;
    const ssize_t itemsize_;
    const std::string format_;
  };
}

#endif // AWKWARD_NUMPYARRAY_H_
