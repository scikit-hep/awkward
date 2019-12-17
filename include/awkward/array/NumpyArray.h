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
  private:
    static const std::shared_ptr<Type> unwrap_type(const std::shared_ptr<Type>& type, const std::vector<ssize_t>& shape);

  public:
    NumpyArray(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const std::shared_ptr<void> ptr, const std::vector<ssize_t> shape, const std::vector<ssize_t> strides, ssize_t byteoffset, ssize_t itemsize, const std::string format)
        : Content(id, unwrap_type(type, shape))
        , ptr_(ptr)
        , shape_(shape)
        , strides_(strides)
        , byteoffset_(byteoffset)
        , itemsize_(itemsize)
        , format_(format) {
      if (shape_.size() != strides_.size()) {
        throw std::runtime_error("len(shape) must be equal to len(strides)");
      }
      if (type_.get() != nullptr) {
        checktype();
      }
    }

    const std::shared_ptr<void> ptr() const { return ptr_; }
    const std::vector<ssize_t> shape() const { return shape_; }
    const std::vector<ssize_t> strides() const { return strides_; }
    ssize_t byteoffset() const { return byteoffset_; }
    ssize_t itemsize() const { return itemsize_; }
    const std::string format() const { return format_; }

    ssize_t ndim() const;
    bool isempty() const;
    void* byteptr() const;
    void* byteptr(ssize_t at) const;
    ssize_t bytelength() const;
    uint8_t getbyte(ssize_t at) const;

    virtual bool isscalar() const;
    virtual const std::string classname() const;
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual bool istypeptr(Type* pointer) const;
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> astype(const std::shared_ptr<Type> type) const;
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    virtual void tojson_part(ToJson& builder) const;
    virtual int64_t length() const;
    virtual const std::shared_ptr<Content> shallow_copy() const;
    virtual void check_for_iteration() const;
    virtual const std::shared_ptr<Content> getitem_nothing() const;
    virtual const std::shared_ptr<Content> getitem_at(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem_field(const std::string& key) const;
    virtual const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const;
    virtual const std::shared_ptr<Content> getitem(const Slice& where) const;
    virtual const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> carry(const Index64& carry) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;
    virtual int64_t numfields() const;
    virtual int64_t fieldindex(const std::string& key) const;
    virtual const std::string key(int64_t fieldindex) const;
    virtual bool haskey(const std::string& key) const;
    virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const;
    virtual const std::vector<std::string> keyaliases(const std::string& key) const;
    virtual const std::vector<std::string> keys() const;

    bool iscontiguous() const;
    void become_contiguous();
    const NumpyArray contiguous() const;

  protected:
    virtual void checktype() const;

    virtual const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("NumpyArray has its own getitem_next system");
    }
    virtual const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("NumpyArray has its own getitem_next system");
    }
    virtual const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("NumpyArray has its own getitem_next system");
    }
    virtual const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("NumpyArray has its own getitem_next system");
    }
    virtual const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("NumpyArray has its own getitem_next system");
    }

    const NumpyArray contiguous_next(Index64 bytepos) const;
    const NumpyArray getitem_bystrides(const std::shared_ptr<SliceItem>& head, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceAt& at, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceRange& range, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceEllipsis& ellipsis, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceNewAxis& newaxis, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceAt& at, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceRange& range, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceEllipsis& ellipsis, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceNewAxis& newaxis, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceArray64& array, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;

  void tojson_boolean(ToJson& builder) const;
  template <typename T>
  void tojson_integer(ToJson& builder) const;
  template <typename T>
  void tojson_real(ToJson& builder) const;
  void tojson_string(ToJson& builder) const;

  private:
    std::shared_ptr<void> ptr_;
    std::vector<ssize_t> shape_;
    std::vector<ssize_t> strides_;
    ssize_t byteoffset_;
    const ssize_t itemsize_;
    const std::string format_;
  };
}

#endif // AWKWARD_NUMPYARRAY_H_
