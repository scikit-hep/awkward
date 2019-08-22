// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NUMPYARRAYINDEX_H_
#define AWKWARD_NUMPYARRAYINDEX_H_

#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <memory>
#include <stdexcept>

#include "awkward/util.h"
#include "awkward/Content.h"

#include <iostream>

namespace awkward {
  class NumpyArray: public Content {
  public:
    NumpyArray(const std::shared_ptr<byte> ptr, const std::vector<ssize_t> shape, const std::vector<ssize_t> strides, ssize_t byteoffset, ssize_t itemsize, const std::string format)
        : ptr_(ptr)
        , shape_(shape)
        , strides_(strides)
        , byteoffset_(byteoffset)
        , itemsize_(itemsize)
        , format_(format) {
          assert(shape_.size() == strides_.size());
        }

    ~NumpyArray() {
      std::cout << "NumpyArray destructor" << std::endl;
    }

    const std::shared_ptr<byte> ptr() const { return ptr_; }
    const std::vector<ssize_t> shape() const { return shape_; }
    const std::vector<ssize_t> strides() const { return strides_; }
    ssize_t byteoffset() const { return byteoffset_; }
    ssize_t itemsize() const { return itemsize_; }
    const std::string format() const { return format_; }

    ssize_t ndim() const;
    ssize_t length() const;
    bool isscalar() const;
    bool isempty() const;
    bool iscompact() const;
    void* byteptr() const;
    ssize_t bytelength() const;
    byte getbyte(ssize_t at) const;

    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    virtual std::shared_ptr<Content> get(AtType at) const;
    virtual std::shared_ptr<Content> slice(AtType start, AtType stop) const;

  private:
    const std::shared_ptr<byte> ptr_;
    const std::vector<ssize_t> shape_;
    const std::vector<ssize_t> strides_;
    const ssize_t byteoffset_;
    const ssize_t itemsize_;
    const std::string format_;
  };
}

#endif // AWKWARD_NUMPYARRAYINDEX_H_
