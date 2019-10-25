// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/getitem.h"
#include "awkward/Content.h"
#include "awkward/Identity.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/fillable/GrowableBuffer.h"

#include "awkward/io/root.h"

namespace awkward {
  void FromROOT_nestedvector_fill(std::vector<GrowableBuffer<int64_t>>& levels, GrowableBuffer<int64_t>& carry, int64_t& bytepos, const NumpyArray& rawdata, int64_t whichlevel, int64_t itemsize) {
    if (whichlevel == levels.size()) {
      carry.append(bytepos);
      bytepos += itemsize;
    }
    else {
      int32_t length = *reinterpret_cast<int32_t*>(rawdata.byteptr(bytepos));
      bytepos += sizeof(int32_t);
      for (int32_t i = 0;  i < length;  i++) {
        FromROOT_nestedvector_fill(levels, carry, bytepos, rawdata, whichlevel + 1, itemsize);
      }
      int64_t previous = levels[whichlevel].getitem_at_unsafe(levels[whichlevel].length() - 1);
      levels[whichlevel].append(previous + length);
    }
  }

  const std::shared_ptr<Content> FromROOT_nestedvector(const Index32& byteoffsets, const NumpyArray& rawdata, int64_t depth, int64_t itemsize, std::string format, const FillableOptions& options) {
    assert(depth > 0);
    assert(rawdata.ndim() == 1);

    Index64 level0(byteoffsets.length());
    level0.setitem_at_unsafe(0, 0);

    std::vector<GrowableBuffer<int64_t>> levels;
    for (int64_t i = 0;  i < depth;  i++) {
      levels.push_back(GrowableBuffer<int64_t>(options));
      levels[(size_t)i].append(0);
    }

    GrowableBuffer<int64_t> carry(options);

    for (int64_t i = 0;  i < byteoffsets.length();  i++) {
      int64_t bytepos = byteoffsets.getitem_at_unsafe(i);
      FromROOT_nestedvector_fill(levels, carry, bytepos, rawdata, 0, itemsize);
      level0.setitem_at_unsafe(i + 1, levels[0].length());
    }

    std::shared_ptr<void> ptr(new uint8_t[(size_t)(carry.length()*itemsize)], awkward::util::array_deleter<uint8_t>());
    Error err = awkward_numpyarray_getitem_next_null_64(
      reinterpret_cast<uint8_t*>(ptr.get()),
      reinterpret_cast<uint8_t*>(rawdata.ptr().get()),
      carry.length(),
      itemsize,
      rawdata.byteoffset(),
      carry.ptr().get());
    util::handle_error(err, rawdata.classname(), rawdata.id().get());
    std::vector<ssize_t> shape = { (ssize_t)carry.length() };
    std::vector<ssize_t> strides = { (ssize_t)itemsize };

    std::shared_ptr<Content> out(new NumpyArray(Identity::none(), ptr, shape, strides, 0, itemsize, format));
    for (int64_t i = depth - 1;  i >= 0;  i--) {
      out = std::shared_ptr<Content>(new ListOffsetArray64(Identity::none(), levels[(size_t)i].toindex(), out));
    }
    return out;
  }

}
