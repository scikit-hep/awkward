// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/RegularType.h"

#include "awkward/array/NumpyArray.h"

namespace awkward {
  ssize_t NumpyArray::ndim() const {
    return shape_.size();
  }

  bool NumpyArray::isscalar() const {
    return ndim() == 0;
  }

  bool NumpyArray::isempty() const {
    for (auto x : shape_) {
      if (x == 0) {
        return true;
      }
    }
    return false;  // false for isscalar(), too
  }

  void* NumpyArray::byteptr() const {
    return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_);
  }

  void* NumpyArray::byteptr(ssize_t at) const {
    return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
  }

  ssize_t NumpyArray::bytelength() const {
    if (isscalar()) {
      return itemsize_;
    }
    else {
      return shape_[0]*strides_[0];
    }
  }

  uint8_t NumpyArray::getbyte(ssize_t at) const {
    return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
  }

  const std::string NumpyArray::classname() const {
    return "NumpyArray";
  }

  void NumpyArray::setid(const std::shared_ptr<Identity> id) {
    if (id.get() != nullptr  &&  length() != id.get()->length()) {
      util::handle_error(failure("content and its id must have the same length", kSliceNone, kSliceNone), classname(), id_.get());
    }
    id_ = id;
  }

  void NumpyArray::setid() {
    assert(!isscalar());
    if (length() <= kMaxInt32) {
      Identity32* rawid = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
      std::shared_ptr<Identity> newid(rawid);
      struct Error err = awkward_new_identity32(rawid->ptr().get(), length());
      util::handle_error(err, classname(), id_.get());
      setid(newid);
    }
    else {
      Identity64* rawid = new Identity64(Identity::newref(), Identity::FieldLoc(), 1, length());
      std::shared_ptr<Identity> newid(rawid);
      struct Error err = awkward_new_identity64(rawid->ptr().get(), length());
      util::handle_error(err, classname(), id_.get());
      setid(newid);
    }
  }

  template <typename T>
  void tostring_as(std::stringstream& out, T* ptr, int64_t length) {
    if (length <= 10) {
      for (int64_t i = 0;  i < length;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << ptr[i];
      }
    }
    else {
      for (int64_t i = 0;  i < 5;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << ptr[i];
      }
      out << " ... ";
      for (int64_t i = length - 5;  i < length;  i++) {
        if (i != length - 5) {
          out << " ";
        }
        out << ptr[i];
      }
    }
  }

  const std::string NumpyArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    assert(!isscalar());
    std::stringstream out;
    out << indent << pre << "<" << classname() << " format=\"" << format_ << "\" shape=\"";
    for (ssize_t i = 0;  i < ndim();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << shape_[i];
    }
    out << "\" ";
    if (!iscontiguous()) {
      out << "strides=\"";
      for (ssize_t i = 0;  i < ndim();  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << strides_[i];
      }
      out << "\" ";
    }
    out << "data=\"";
#ifdef _MSC_VER
    if (ndim() == 1  &&  format_.compare("l") == 0) {
#else
    if (ndim() == 1  &&  format_.compare("i") == 0) {
#endif
      tostring_as<int32_t>(out, reinterpret_cast<int32_t*>(byteptr()), length());
    }
#ifdef _MSC_VER
    else if (ndim() == 1  &&  format_.compare("q") == 0) {
#else
    else if (ndim() == 1  &&  format_.compare("l") == 0) {
#endif
      tostring_as<int64_t>(out, reinterpret_cast<int64_t*>(byteptr()), length());
    }
    else if (ndim() == 1  &&  format_.compare("f") == 0) {
      tostring_as<float>(out, reinterpret_cast<float*>(byteptr()), length());
    }
    else if (ndim() == 1  &&  format_.compare("d") == 0) {
      tostring_as<double>(out, reinterpret_cast<double*>(byteptr()), length());
    }
    else {
      ssize_t len = bytelength();
      if (len <= 32) {
        for (ssize_t i = 0;  i < len;  i++) {
          if (i != 0  &&  i % 4 == 0) {
            out << " ";
          }
          out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
        }
      }
      else {
        for (ssize_t i = 0;  i < 16;  i++) {
          if (i != 0  &&  i % 4 == 0) {
            out << " ";
          }
          out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
        }
        out << " ... ";
        for (ssize_t i = len - 16;  i < len;  i++) {
          if (i != len - 16  &&  i % 4 == 0) {
            out << " ";
          }
          out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
        }
      }
    }
    out << "\" at=\"0x";
    out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get());
    if (id_.get() == nullptr) {
      out << "\"/>" << post;
    }
    else {
      out << "\">\n";
      out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
      out << indent << "</" << classname() << ">" << post;
    }
    return out.str();
  }

  void tojson_boolean(ToJson& builder, bool* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.boolean(array[i]);
    }
  }

  template <typename T>
  void tojson_integer(ToJson& builder, T* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.integer(array[i]);
    }
  }

  template <typename T>
  void tojson_real(ToJson& builder, T* array, int64_t length) {
    for (int i = 0;  i < length;  i++) {
      builder.real(array[i]);
    }
  }

  void NumpyArray::tojson_part(ToJson& builder) const {
    assert(!isscalar());
    if (ndim() == 1) {
      if (format_.compare("d") == 0) {
        tojson_real(builder, reinterpret_cast<double*>(byteptr()), length());
      }
      else if (format_.compare("f") == 0) {
        tojson_real(builder, reinterpret_cast<float*>(byteptr()), length());
      }
#ifdef _MSC_VER
      else if (format_.compare("q") == 0) {
#else
      else if (format_.compare("l") == 0) {
#endif
        tojson_integer(builder, reinterpret_cast<int64_t*>(byteptr()), length());
      }
#ifdef _MSC_VER
      else if (format_.compare("Q") == 0) {
#else
      else if (format_.compare("L") == 0) {
#endif
        tojson_integer(builder, reinterpret_cast<uint64_t*>(byteptr()), length());
      }
#ifdef _MSC_VER
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
        tojson_integer(builder, reinterpret_cast<int32_t*>(byteptr()), length());
      }
#ifdef _MSC_VER
      else if (format_.compare("L") == 0) {
#else
      else if (format_.compare("I") == 0) {
#endif
        tojson_integer(builder, reinterpret_cast<uint32_t*>(byteptr()), length());
      }
      else if (format_.compare("h") == 0) {
        tojson_real(builder, reinterpret_cast<int16_t*>(byteptr()), length());
      }
      else if (format_.compare("H") == 0) {
        tojson_real(builder, reinterpret_cast<uint16_t*>(byteptr()), length());
      }
      else if (format_.compare("b") == 0) {
        tojson_real(builder, reinterpret_cast<int8_t*>(byteptr()), length());
      }
      else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
        tojson_real(builder, reinterpret_cast<uint8_t*>(byteptr()), length());
      }
      else {
        throw std::invalid_argument(std::string("cannot convert Numpy format \"") + format_ + std::string("\" into JSON"));
      }
    }
    else {
      for (int64_t i = 0;  i < length();  i++) {
        builder.beginlist();
        getitem_at_unsafe(i).get()->tojson_part(builder);
        builder.endlist();
      }
    }
  }

  const std::shared_ptr<Type> NumpyArray::type_part() const {
    if (ndim() == 1) {
      if (format_.compare("d") == 0) {
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::float64));
      }
      else if (format_.compare("f") == 0) {
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::float32));
      }
#ifdef _MSC_VER
      else if (format_.compare("q") == 0) {
#else
      else if (format_.compare("l") == 0) {
#endif
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::int64));
      }
#ifdef _MSC_VER
      else if (format_.compare("Q") == 0) {
#else
      else if (format_.compare("L") == 0) {
#endif
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::uint64));
      }
#ifdef _MSC_VER
      else if (format_.compare("l") == 0) {
#else
      else if (format_.compare("i") == 0) {
#endif
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::int32));
      }
#ifdef _MSC_VER
      else if (format_.compare("L") == 0) {
#else
      else if (format_.compare("I") == 0) {
#endif
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::uint32));
      }
      else if (format_.compare("h") == 0) {
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::int16));
      }
      else if (format_.compare("H") == 0) {
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::uint16));
      }
      else if (format_.compare("b") == 0) {
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::int8));
      }
      else if (format_.compare("B") == 0  ||  format_.compare("c") == 0) {
        return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::uint8));
      }
      else {
        throw std::invalid_argument(std::string("Numpy format \"") + format_ + std::string("\" cannot be expressed as a PrimitiveType"));
      }
    }
    else {
      NumpyArray tmp(id_, ptr_, std::vector<ssize_t>({ 1 }), std::vector<ssize_t>({ itemsize_ }), byteoffset_, itemsize_, format_);
      std::shared_ptr<Type> out = tmp.type_part();
      for (ssize_t i = shape_.size() - 1;  i > 0;  i--) {
        out = std::shared_ptr<Type>(new RegularType(out, (int64_t)shape_[i]));
      }
      return out;
    }
  }

  int64_t NumpyArray::length() const {
    if (isscalar()) {
      return -1;
    }
    else {
      return (int64_t)shape_[0];
    }
  }

  const std::shared_ptr<Content> NumpyArray::shallow_copy() const {
    return std::shared_ptr<Content>(new NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_));
  }

  void NumpyArray::checksafe() const {
    if (id_.get() != nullptr  &&  id_.get()->length() < shape_[0]) {
      util::handle_error(failure("len(id) < len(array)", kSliceNone, kSliceNone), id_.get()->classname(), nullptr);
    }
  }

  const std::shared_ptr<Content> NumpyArray::getitem_at(int64_t at) const {
    assert(!isscalar());
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += shape_[0];
    }
    if (regular_at < 0  ||  regular_at >= shape_[0]) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
    }
    return getitem_at_unsafe(regular_at);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_at_unsafe(int64_t at) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
    const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
    const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
    std::shared_ptr<Identity> id;
    if (id_.get() != nullptr) {
      if (at >= id_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, at), id_.get()->classname(), nullptr);
      }
      id = id_.get()->getitem_range_unsafe(at, at + 1);
    }
    return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides, byteoffset, itemsize_, format_));
  }

  const std::shared_ptr<Content> NumpyArray::getitem_range(int64_t start, int64_t stop) const {
    assert(!isscalar());
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), shape_[0]);
    return getitem_range_unsafe(regular_start, regular_stop);
  }

  const std::shared_ptr<Content> NumpyArray::getitem_range_unsafe(int64_t start, int64_t stop) const {
    ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
    std::vector<ssize_t> shape;
    shape.push_back((ssize_t)(stop - start));
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    std::shared_ptr<Identity> id;
    if (id_.get() != nullptr) {
      if (stop > id_.get()->length()) {
        util::handle_error(failure("index out of range", kSliceNone, stop), id_.get()->classname(), nullptr);
      }
      id = id_.get()->getitem_range_unsafe(start, stop);
    }
    return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides_, byteoffset, itemsize_, format_));
  }

  const std::shared_ptr<Content> NumpyArray::getitem(const Slice& where) const {
    assert(!isscalar());

    if (!where.isadvanced()  &&  id_.get() == nullptr) {
      std::vector<ssize_t> nextshape = { 1 };
      nextshape.insert(nextshape.end(), shape_.begin(), shape_.end());
      std::vector<ssize_t> nextstrides = { shape_[0]*strides_[0] };
      nextstrides.insert(nextstrides.end(), strides_.begin(), strides_.end());
      NumpyArray next(id_, ptr_, nextshape, nextstrides, byteoffset_, itemsize_, format_);

      std::shared_ptr<SliceItem> nexthead = where.head();
      Slice nexttail = where.tail();
      NumpyArray out = next.getitem_bystrides(nexthead, nexttail, 1);

      std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides(out.strides_.begin() + 1, out.strides_.end());
      return std::shared_ptr<Content>(new NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_));
    }

    else {
      NumpyArray safe = contiguous();   // maybe become_contiguous() to change in-place?

      std::vector<ssize_t> nextshape = { 1 };
      nextshape.insert(nextshape.end(), safe.shape_.begin(), safe.shape_.end());
      std::vector<ssize_t> nextstrides = { safe.shape_[0]*safe.strides_[0] };
      nextstrides.insert(nextstrides.end(), safe.strides_.begin(), safe.strides_.end());
      NumpyArray next(safe.id_, safe.ptr_, nextshape, nextstrides, safe.byteoffset_, itemsize_, format_);

      std::shared_ptr<SliceItem> nexthead = where.head();
      Slice nexttail = where.tail();
      Index64 nextcarry(1);
      nextcarry.ptr().get()[0] = 0;
      Index64 nextadvanced(0);
      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, 1, next.strides_[0], true);

      std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides(out.strides_.begin() + 1, out.strides_.end());
      return std::shared_ptr<Content>(new NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_));
    }
  }

  const std::shared_ptr<Content> NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    assert(!isscalar());
    Index64 carry(shape_[0]);
    struct Error err = awkward_carry_arange_64(carry.ptr().get(), shape_[0]);
    util::handle_error(err, classname(), id_.get());
    return getitem_next(head, tail, carry, advanced, shape_[0], strides_[0], false).shallow_copy();
  }

  const std::shared_ptr<Content> NumpyArray::carry(const Index64& carry) const {
    assert(!isscalar());

    std::shared_ptr<void> ptr(new uint8_t[(size_t)(carry.length()*strides_[0])], awkward::util::array_deleter<uint8_t>());
    struct Error err = awkward_numpyarray_getitem_next_null_64(
      reinterpret_cast<uint8_t*>(ptr.get()),
      reinterpret_cast<uint8_t*>(ptr_.get()),
      carry.length(),
      strides_[0],
      byteoffset_,
      carry.ptr().get());
    util::handle_error(err, classname(), id_.get());

    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }

    std::vector<ssize_t> shape = { (ssize_t)carry.length() };
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    return std::shared_ptr<Content>(new NumpyArray(id, ptr, shape, strides_, 0, itemsize_, format_));
  }

  const std::pair<int64_t, int64_t> NumpyArray::minmax_depth() const {
    return std::pair<int64_t, int64_t>((int64_t)shape_.size(), (int64_t)shape_.size());
  }

  const std::vector<ssize_t> flatten_shape(const std::vector<ssize_t> shape) {
    if (shape.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      std::vector<ssize_t> out = { shape[0]*shape[1] };
      out.insert(out.end(), shape.begin() + 2, shape.end());
      return out;
    }
  }

  const std::vector<ssize_t> flatten_strides(const std::vector<ssize_t> strides) {
    if (strides.size() == 1) {
      return std::vector<ssize_t>();
    }
    else {
      return std::vector<ssize_t>(strides.begin() + 1, strides.end());
    }
  }

  bool NumpyArray::iscontiguous() const {
    ssize_t x = itemsize_;
    for (ssize_t i = ndim() - 1;  i >= 0;  i--) {
      if (x != strides_[i]) return false;
      x *= shape_[i];
    }
    return true;  // true for isscalar(), too
  }

  void NumpyArray::become_contiguous() {
    if (!iscontiguous()) {
      NumpyArray x = contiguous();
      id_ = x.id_;
      ptr_ = x.ptr_;
      shape_ = x.shape_;
      strides_ = x.strides_;
      byteoffset_ = x.byteoffset_;
    }
  }

  const NumpyArray NumpyArray::contiguous() const {
    if (iscontiguous()) {
      return NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
    }
    else {
      Index64 bytepos(shape_[0]);
      struct Error err = awkward_numpyarray_contiguous_init_64(bytepos.ptr().get(), shape_[0], strides_[0]);
      util::handle_error(err, classname(), id_.get());
      return contiguous_next(bytepos);
    }
  }

  const NumpyArray NumpyArray::contiguous_next(Index64 bytepos) const {
    if (iscontiguous()) {
      std::shared_ptr<void> ptr(new uint8_t[(size_t)(bytepos.length()*strides_[0])], awkward::util::array_deleter<uint8_t>());
      struct Error err = awkward_numpyarray_contiguous_copy_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        bytepos.length(),
        strides_[0],
        byteoffset_,
        bytepos.ptr().get());
      util::handle_error(err, classname(), id_.get());
      return NumpyArray(id_, ptr, shape_, strides_, 0, itemsize_, format_);
    }

    else if (shape_.size() == 1) {
      std::shared_ptr<void> ptr(new uint8_t[(size_t)(bytepos.length()*itemsize_)], awkward::util::array_deleter<uint8_t>());
      struct Error err = awkward_numpyarray_contiguous_copy_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        bytepos.length(),
        itemsize_,
        byteoffset_,
        bytepos.ptr().get());
      util::handle_error(err, classname(), id_.get());
      std::vector<ssize_t> strides = { itemsize_ };
      return NumpyArray(id_, ptr, shape_, strides, 0, itemsize_, format_);
    }

    else {
      NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);

      Index64 nextbytepos(bytepos.length()*shape_[1]);
      struct Error err = awkward_numpyarray_contiguous_next_64(
        nextbytepos.ptr().get(),
        bytepos.ptr().get(),
        bytepos.length(),
        (int64_t)shape_[1],
        (int64_t)strides_[1]);
      util::handle_error(err, classname(), id_.get());

      NumpyArray out = next.contiguous_next(nextbytepos);
      std::vector<ssize_t> outstrides = { shape_[1]*out.strides_[0] };
      outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
      return NumpyArray(out.id_, out.ptr_, shape_, outstrides, out.byteoffset_, itemsize_, format_);
    }
  }

  const NumpyArray NumpyArray::getitem_bystrides(const std::shared_ptr<SliceItem>& head, const Slice& tail, int64_t length) const {
    if (head.get() == nullptr) {
      return NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      if (ndim() < 2) {
        util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
      }

      int64_t i = at->at();
      if (i < 0) i += shape_[1];
      if (i < 0  ||  i >= shape_[1]) {
        util::handle_error(failure("index out of range", kSliceNone, at->at()), classname(), id_.get());
      }

      ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)i)*strides_[1];
      NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length);

      std::vector<ssize_t> outshape = { (ssize_t)length };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      return NumpyArray(out.id_, out.ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      if (ndim() < 2) {
        util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
      }

      int64_t start = range->start();
      int64_t stop = range->stop();
      int64_t step = range->step();
      if (step == Slice::none()) {
        step = 1;
      }
      awkward_regularize_rangeslice(&start, &stop, step > 0, range->hasstart(), range->hasstop(), (int64_t)shape_[1]);

      int64_t numer = abs(start - stop);
      int64_t denom = abs(step);
      int64_t d = numer / denom;
      int64_t m = numer % denom;
      int64_t lenhead = d + (m != 0 ? 1 : 0);

      ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)start)*strides_[1];
      NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length*lenhead);

      std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides = { strides_[0], strides_[1]*((ssize_t)step) };
      outstrides.insert(outstrides.end(), out.strides_.begin() + 1, out.strides_.end());
      return NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }

    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      std::pair<int64_t, int64_t> minmax = minmax_depth();
      assert(minmax.first == minmax.second);
      int64_t mindepth = minmax.first;

      if (tail.length() == 0  ||  mindepth - 1 == tail.dimlength()) {
        std::shared_ptr<SliceItem> nexthead = tail.head();
        Slice nexttail = tail.tail();
        return getitem_bystrides(nexthead, nexttail, length);
      }
      else {
        std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
        std::vector<std::shared_ptr<SliceItem>> items = { std::shared_ptr<SliceItem>(new SliceEllipsis()) };
        items.insert(items.end(), tailitems.begin(), tailitems.end());

        std::shared_ptr<SliceItem> nexthead(new SliceRange(Slice::none(), Slice::none(), 1));
        Slice nexttail(items);
        return getitem_bystrides(nexthead, nexttail, length);
      }
    }

    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      NumpyArray out = getitem_bystrides(nexthead, nexttail, length);

      std::vector<ssize_t> outshape = { (ssize_t)length, 1 };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides = { out.strides_[0] };
      outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
      return NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  const NumpyArray NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const {
    if (head.get() == nullptr) {
      std::shared_ptr<void> ptr(new uint8_t[(size_t)(carry.length()*stride)], awkward::util::array_deleter<uint8_t>());
      struct Error err = awkward_numpyarray_getitem_next_null_64(
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(ptr_.get()),
        carry.length(),
        stride,
        byteoffset_,
        carry.ptr().get());
      util::handle_error(err, classname(), id_.get());

      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->getitem_carry_64(carry);
      }

      std::vector<ssize_t> shape = { (ssize_t)carry.length() };
      shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
      std::vector<ssize_t> strides = { (ssize_t)stride };
      strides.insert(strides.end(), strides_.begin() + 1, strides_.end());
      return NumpyArray(id, ptr, shape, strides, 0, itemsize_, format_);
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      if (ndim() < 2) {
        util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
      }

      NumpyArray next(first ? id_ : std::shared_ptr<Identity>(nullptr), ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();

      // if we had any array slices, this int would become an array
      assert(advanced.length() == 0);

      int64_t regular_at = at->at();
      if (regular_at < 0) {
        regular_at += shape_[1];
      }
      if (!(0 <= regular_at  &&  regular_at < shape_[1])) {
        util::handle_error(failure("index out of range", kSliceNone, at->at()), classname(), id_.get());
      }

      Index64 nextcarry(carry.length());
      struct Error err = awkward_numpyarray_getitem_next_at_64(
        nextcarry.ptr().get(),
        carry.ptr().get(),
        carry.length(),
        shape_[1],   // because this is contiguous
        regular_at);
      util::handle_error(err, classname(), id_.get());

      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length, next.strides_[0], false);

      std::vector<ssize_t> outshape = { (ssize_t)length };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      return NumpyArray(out.id_, out.ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      if (ndim() < 2) {
        util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
      }

      int64_t start = range->start();
      int64_t stop = range->stop();
      int64_t step = range->step();
      if (step == Slice::none()) {
        step = 1;
      }
      awkward_regularize_rangeslice(&start, &stop, step > 0, range->hasstart(), range->hasstop(), (int64_t)shape_[1]);

      int64_t numer = abs(start - stop);
      int64_t denom = abs(step);
      int64_t d = numer / denom;
      int64_t m = numer % denom;
      int64_t lenhead = d + (m != 0 ? 1 : 0);

      NumpyArray next(first ? id_ : std::shared_ptr<Identity>(nullptr), ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();

      if (advanced.length() == 0) {
        Index64 nextcarry(carry.length()*lenhead);
        struct Error err = awkward_numpyarray_getitem_next_range_64(
          nextcarry.ptr().get(),
          carry.ptr().get(),
          carry.length(),
          lenhead,
          shape_[1],   // because this is contiguous
          start,
          step);
        util::handle_error(err, classname(), id_.get());

        NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length*lenhead, next.strides_[0], false);
        std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
        outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
        std::vector<ssize_t> outstrides = { (ssize_t)lenhead*out.strides_[0] };
        outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
        return NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
      }

      else {
        Index64 nextcarry(carry.length()*lenhead);
        Index64 nextadvanced(carry.length()*lenhead);
        struct Error err = awkward_numpyarray_getitem_next_range_advanced_64(
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          carry.ptr().get(),
          advanced.ptr().get(),
          carry.length(),
          lenhead,
          shape_[1],   // because this is contiguous
          start,
          step);
        util::handle_error(err, classname(), id_.get());

        NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*lenhead, next.strides_[0], false);
        std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)lenhead };
        outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
        std::vector<ssize_t> outstrides = { (ssize_t)lenhead*out.strides_[0] };
        outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
        return NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
      }
    }

    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      std::pair<int64_t, int64_t> minmax = minmax_depth();
      assert(minmax.first == minmax.second);
      int64_t mindepth = minmax.first;

      if (tail.length() == 0  ||  mindepth - 1 == tail.dimlength()) {
        std::shared_ptr<SliceItem> nexthead = tail.head();
        Slice nexttail = tail.tail();
        return getitem_next(nexthead, nexttail, carry, advanced, length, stride, false);
      }
      else {
        std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
        std::vector<std::shared_ptr<SliceItem>> items = { std::shared_ptr<SliceItem>(new SliceEllipsis()) };
        items.insert(items.end(), tailitems.begin(), tailitems.end());
        std::shared_ptr<SliceItem> nexthead(new SliceRange(Slice::none(), Slice::none(), 1));
        Slice nexttail(items);
        return getitem_next(nexthead, nexttail, carry, advanced, length, stride, false);
      }
    }

    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      NumpyArray out = getitem_next(nexthead, nexttail, carry, advanced, length, stride, false);

      std::vector<ssize_t> outshape = { (ssize_t)length, 1 };
      outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
      std::vector<ssize_t> outstrides = { out.strides_[0] };
      outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
      return NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
    }

    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      if (ndim() < 2) {
        util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
      }

      NumpyArray next(first ? id_ : std::shared_ptr<Identity>(nullptr), ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();

      Index64 flathead = array->ravel();
      struct Error err = awkward_regularize_arrayslice_64(
        flathead.ptr().get(),
        flathead.length(),
        shape_[1]);
      util::handle_error(err, classname(), id_.get());

      if (advanced.length() == 0) {
        Index64 nextcarry(carry.length()*flathead.length());
        Index64 nextadvanced(carry.length()*flathead.length());
        struct Error err = awkward_numpyarray_getitem_next_array_64(
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          carry.ptr().get(),
          flathead.ptr().get(),
          carry.length(),
          flathead.length(),
          shape_[1]);   // because this is contiguous
        util::handle_error(err, classname(), id_.get());

        NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*flathead.length(), next.strides_[0], false);

        std::vector<ssize_t> outshape = { (ssize_t)length };
        std::vector<int64_t> arrayshape = array->shape();
        for (auto x = arrayshape.begin();  x != arrayshape.end();  ++x) {
          outshape.push_back((ssize_t)(*x));
        }
        outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());

        std::vector<ssize_t> outstrides(out.strides_.begin(), out.strides_.end());
        for (auto x = arrayshape.rbegin();  x != arrayshape.rend();  ++x) {
          outstrides.insert(outstrides.begin(), ((ssize_t)(*x))*outstrides[0]);
        }
        return NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
      }

      else {
        Index64 nextcarry(carry.length());
        struct Error err = awkward_numpyarray_getitem_next_array_advanced_64(
          nextcarry.ptr().get(),
          carry.ptr().get(),
          advanced.ptr().get(),
          flathead.ptr().get(),
          carry.length(),
          shape_[1]);   // because this is contiguous
        util::handle_error(err, classname(), id_.get());

        NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length*array->length(), next.strides_[0], false);

        std::vector<ssize_t> outshape = { (ssize_t)length };
        outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
        return NumpyArray(out.id_, out.ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
      }
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }
}
