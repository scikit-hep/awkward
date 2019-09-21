// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/NumpyArray.h"

using namespace awkward;

ssize_t NumpyArray::ndim() const {
  return shape_.size();
}

bool NumpyArray::isscalar() const {
  return ndim() == 0;
}

bool NumpyArray::isempty() const {
  for (auto x : shape_) {
    if (x == 0) return true;
  }
  return false;  // false for isscalar(), too
}

void* NumpyArray::byteptr() const {
  return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_);
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

void NumpyArray::setid(const std::shared_ptr<Identity> id) {
  id_ = id;
}

void NumpyArray::setid() {
  assert(!isscalar());
  Identity32* id32 = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
  std::shared_ptr<Identity> newid(id32);
  Error err = awkward_identity_new32(length(), id32->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
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
  out << indent << pre << "<NumpyArray format=\"" << format_ << "\" shape=\"";
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
    out << indent << "</NumpyArray>" << post;
  }
  return out.str();
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

const std::shared_ptr<Content> NumpyArray::get(int64_t at) const {
  return getitem(Slice(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceAt(at)) }), true));
}

const std::shared_ptr<Content> NumpyArray::slice(int64_t start, int64_t stop) const {
  return getitem(Slice(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceRange(start, stop, 1)) }), true));

  // FIXME: id should be propagated through the new getitem
  // assert(!isscalar());
  // ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
  // std::vector<ssize_t> shape;
  // shape.push_back((ssize_t)(stop - start));
  // shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
  // std::shared_ptr<Identity> id(nullptr);
  // if (id_.get() != nullptr) {
  //   id = id_.get()->slice(start, stop);
  // }
  // return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides_, byteoffset, itemsize_, format_));
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

    int64_t* rawptr = bytepos.ptr().get();
    int64_t stride = strides_[0];
    for (int64_t i = 0;  i < shape_[0];  i++) {
      rawptr[i] = i*stride;
    }

    return contiguous_next(bytepos);
  }
}

const NumpyArray NumpyArray::contiguous_next(Index64 bytepos) const {
  if (iscontiguous()) {
    int64_t len = bytepos.length();
    int64_t stride = strides_[0];
    std::shared_ptr<void> ptr(new uint8_t[(size_t)(len*stride)], awkward::util::array_deleter<uint8_t>());

    int64_t offset = byteoffset_;
    int64_t* pos = bytepos.ptr().get();
    uint8_t* fromptr = reinterpret_cast<uint8_t*>(ptr_.get());
    uint8_t* toptr = reinterpret_cast<uint8_t*>(ptr.get());
    for (int64_t i = 0;  i < len;  i++) {
      memcpy(&toptr[i*stride], &fromptr[offset + pos[i]], stride);
    }

    return NumpyArray(id_, ptr, shape_, strides_, 0, itemsize_, format_);
  }

  else if (shape_.size() == 1) {
    int64_t len = bytepos.length();
    int64_t stride = itemsize_;
    std::shared_ptr<void> ptr(new uint8_t[(size_t)(len*stride)], awkward::util::array_deleter<uint8_t>());

    int64_t offset = byteoffset_;
    int64_t* pos = bytepos.ptr().get();
    uint8_t* fromptr = reinterpret_cast<uint8_t*>(ptr_.get());
    uint8_t* toptr = reinterpret_cast<uint8_t*>(ptr.get());
    for (int64_t i = 0;  i < len;  i++) {
      memcpy(&toptr[i*stride], &fromptr[offset + pos[i]], stride);
    }

    std::vector<ssize_t> strides = { itemsize_ };
    return NumpyArray(id_, ptr, shape_, strides, 0, itemsize_, format_);
  }

  else {
    NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);

    int64_t len = bytepos.length();
    int64_t shape1 = (int64_t)shape_[1];
    int64_t strides1 = (int64_t)strides_[1];
    Index64 nextbytepos(len*shape1);

    int64_t* frompos = bytepos.ptr().get();
    int64_t* topos = nextbytepos.ptr().get();
    for (int64_t i = 0;  i < len;  i++) {
      for (int64_t j = 0;  j < shape1;  j++) {
        topos[i*shape1 + j] = frompos[i] + j*strides1;
      }
    }

    NumpyArray out = next.contiguous_next(nextbytepos);
    std::vector<ssize_t> outstrides = { shape_[1]*out.strides_[0] };
    outstrides.insert(outstrides.end(), out.strides_.begin(), out.strides_.end());
    return NumpyArray(out.id_, out.ptr_, shape_, outstrides, out.byteoffset_, itemsize_, format_);
  }
}

const std::shared_ptr<Content> NumpyArray::getitem(const Slice& where) const {
  assert(!isscalar());

  if (!where.isadvanced()) {
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
    NumpyArray safe = contiguous();   // maybe become_contiguous()

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
    NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, 1, next.strides_[0]);

    std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides(out.strides_.begin() + 1, out.strides_.end());
    return std::shared_ptr<Content>(new NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_));
  }
}

void set_range(int64_t& start, int64_t& stop, bool posstep, bool hasstart, bool hasstop, int64_t length) {
  if (posstep) {
    if (!hasstart)          start = 0;
    else if (start < 0)     start += length;
    if (start < 0)          start = 0;
    if (start > length)     start = length;

    if (!hasstop)           stop = length;
    else if (stop < 0)      stop += length;
    if (stop < 0)           stop = 0;
    if (stop > length)      stop = length;
    if (stop < start)       stop = start;
  }

  else {
    if (!hasstart)          start = length - 1;
    else if (start < 0)     start += length;
    if (start < -1)         start = -1;
    if (start > length - 1) start = length - 1;

    if (!hasstop)           stop = -1;
    else if (stop < 0)      stop += length;
    if (stop < -1)          stop = -1;
    if (stop > length - 1)  stop = length - 1;
    if (stop > start)       stop = start;
  }
}

const NumpyArray NumpyArray::getitem_bystrides(const std::shared_ptr<SliceItem>& head, const Slice& tail, int64_t length) const {
  if (head.get() == nullptr) {
    return NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
  }

  if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
    if (ndim() < 2) {
      throw std::invalid_argument("too many indexes for array");
    }

    int64_t i = at->at();
    if (i < 0) i += shape_[1];
    if (i < 0  ||  i >= shape_[1]) {
      throw std::invalid_argument("index out of range");
    }

    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)i)*strides_[1];
    NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    return NumpyArray(id_, ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
  }

  else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
    if (ndim() < 2) {
      throw std::invalid_argument("too many indexes for array");
    }

    int64_t start = range->start();
    int64_t stop = range->stop();
    int64_t step = range->step();
    set_range(start, stop, step > 0, range->hasstart(), range->hasstop(), (int64_t)shape_[1]);

    int64_t numer = abs(start - stop);
    int64_t denom = abs(step);
    int64_t d = numer / denom;
    int64_t m = numer % denom;
    int64_t headlen = d + (m != 0 ? 1 : 0);

    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)start)*strides_[1];
    NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length*headlen);

    std::vector<ssize_t> outshape = { (ssize_t)length, (ssize_t)headlen };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides = { strides_[0], strides_[1]*((ssize_t)step) };
    outstrides.insert(outstrides.end(), out.strides_.begin() + 1, out.strides_.end());
    return NumpyArray(id_, ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
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
      Slice nexttail(items, true);
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
    return NumpyArray(id_, ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_);
  }

  else {
    throw std::runtime_error("unrecognized slice item type");
  }
}

const NumpyArray NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, Index64& carry, Index64& advanced, int64_t length, int64_t stride) const {
  if (head.get() == nullptr) {
    int64_t len = carry.length();
    std::shared_ptr<void> ptr(new uint8_t[(size_t)(len*stride)], awkward::util::array_deleter<uint8_t>());

    int64_t offset = byteoffset_;
    int64_t* pos = carry.ptr().get();
    uint8_t* fromptr = reinterpret_cast<uint8_t*>(ptr_.get());
    uint8_t* toptr = reinterpret_cast<uint8_t*>(ptr.get());
    for (int64_t i = 0;  i < len;  i++) {
      memcpy(&toptr[i*stride], &fromptr[offset + pos[i]*stride], stride);
    }

    std::vector<ssize_t> shape = { (ssize_t)len };
    shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
    std::vector<ssize_t> strides = { (ssize_t)stride };
    strides.insert(strides.end(), strides_.begin() + 1, strides_.end());
    return NumpyArray(id_, ptr, shape, strides, 0, itemsize_, format_);
  }

  if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
    if (ndim() < 2) {
      throw std::invalid_argument("too many indexes for array");
    }
    throw std::runtime_error("getitem_next range");
  }

  else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
    throw std::runtime_error("getitem_next ellipsis");
  }

  else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
    throw std::runtime_error("getitem_next newaxis");
  }

  else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
    if (ndim() < 2) {
      throw std::invalid_argument("too many indexes for array");
    }

    // FIXME: handle empty array separately

    NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), byteoffset_, itemsize_, format_);
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();

    Index64 flathead = array->ravel();

    int64_t lencarry = carry.length();
    int64_t lenflathead = flathead.length();
    int64_t* flatheadptr = flathead.ptr().get();
    int64_t skip = shape_[1];
    for (int64_t i = 0;  i < lenflathead;  i++) {
      if (flatheadptr[i] < 0) {
        flatheadptr[i] += skip;
      }
      if (flatheadptr[i] < 0  ||  flatheadptr[i] >= skip) {
        throw std::invalid_argument("index out of range");
      }
    }

    if (advanced.length() == 0) {
      Index64 nextcarry(lencarry*lenflathead);
      Index64 nextadvanced(lencarry*lenflathead);
      int64_t* carryptr = carry.ptr().get();
      int64_t* nextcarryptr = nextcarry.ptr().get();
      int64_t* nextadvancedptr = nextadvanced.ptr().get();
      for (int64_t i = 0;  i < lencarry;  i++) {
        for (int64_t j = 0;  j < lenflathead;  j++) {
          nextcarryptr[i*lenflathead + j] = skip*carryptr[i] + flatheadptr[j];
          nextadvancedptr[i*lenflathead + j] = j;
        }
      }

      NumpyArray out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*lenflathead, next.strides_[0]);

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
      throw std::runtime_error("FIXME");
    }
  }

  else {
    throw std::runtime_error("unrecognized slice item type");
  }
}
