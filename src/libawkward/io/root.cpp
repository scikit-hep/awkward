// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/io/root.cpp", line)

#include <cstring>

#include "awkward/Content.h"
#include "awkward/Identities.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/builder/GrowableBuffer.h"

#include "awkward/io/root.h"

namespace awkward {
  void
  FromROOT_nestedvector_fill(std::vector<GrowableBuffer<int64_t>>& levels,
                             GrowableBuffer<int64_t>& bytepos_tocopy,
                             int64_t& bytepos,
                             const NumpyArray& rawdata,
                             int64_t whichlevel,
                             int64_t itemsize) {
    if (whichlevel == (int64_t)levels.size()) {
      bytepos_tocopy.append(bytepos);
      bytepos += itemsize;
    }

    else {
      uint32_t bigendian = *reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(rawdata.data()) + bytepos);

      // FIXME: check native endianness
      uint32_t length =
        ((bigendian >> 24) & 0xff)     |  // move byte 3 to byte 0
        ((bigendian <<  8) & 0xff0000) |  // move byte 1 to byte 2
        ((bigendian >>  8) & 0xff00)   |  // move byte 2 to byte 1
        ((bigendian << 24) & 0xff000000); // byte 0 to byte 3

      bytepos += sizeof(int32_t);
      for (uint32_t i = 0;  i < length;  i++) {
        FromROOT_nestedvector_fill(levels,
                                   bytepos_tocopy,
                                   bytepos,
                                   rawdata,
                                   whichlevel + 1,
                                   itemsize);
      }
      int64_t previous = levels[(unsigned int)whichlevel].getitem_at_nowrap(
        levels[(unsigned int)whichlevel].length() - 1);
      levels[(unsigned int)whichlevel].append(previous + length);
    }
  }

  const ContentPtr
  FromROOT_nestedvector(const Index64& byteoffsets,
                        const NumpyArray& rawdata,
                        int64_t depth,
                        int64_t itemsize,
                        std::string format,
                        const ArrayBuilderOptions& options) {
    if (depth <= 0) {
      throw std::runtime_error(
        std::string("FromROOT_nestedvector: depth <= 0") + FILENAME(__LINE__));
    }
    if (rawdata.ndim() != 1) {
      throw std::runtime_error(
        std::string("FromROOT_nestedvector: rawdata.ndim() != 1") + FILENAME(__LINE__));
    }

    Index64 level0(byteoffsets.length());
    level0.setitem_at_nowrap(0, 0);

    std::vector<GrowableBuffer<int64_t>> levels;
    for (int64_t i = 0;  i < depth;  i++) {
      levels.push_back(GrowableBuffer<int64_t>(options));
      levels[(size_t)i].append(0);
    }

    GrowableBuffer<int64_t> bytepos_tocopy(options);

    for (int64_t i = 0;  i < byteoffsets.length() - 1;  i++) {
      int64_t bytepos = byteoffsets.getitem_at_nowrap(i);
      FromROOT_nestedvector_fill(levels,
                                 bytepos_tocopy,
                                 bytepos,
                                 rawdata,
                                 0,
                                 itemsize);
      level0.setitem_at_nowrap(i + 1, levels[0].length());
    }

    std::shared_ptr<void> ptr(
      new uint8_t[(size_t)(bytepos_tocopy.length()*itemsize)],
      kernel::array_deleter<uint8_t>());
    ssize_t offset = rawdata.byteoffset();
    uint8_t* toptr = reinterpret_cast<uint8_t*>(ptr.get());
    uint8_t* fromptr = reinterpret_cast<uint8_t*>(rawdata.ptr().get());
    for (int64_t i = 0;  i < bytepos_tocopy.length();  i++) {
      ssize_t bytepos = (ssize_t)bytepos_tocopy.getitem_at_nowrap(i);
      std::memcpy(&toptr[(ssize_t)(i*itemsize)],
                  &fromptr[offset + bytepos],
                  (size_t)itemsize);
    }

    util::dtype dtype = util::format_to_dtype(format, itemsize);

    std::vector<ssize_t> shape = { (ssize_t)bytepos_tocopy.length() };
    std::vector<ssize_t> strides = { (ssize_t)itemsize };
    ContentPtr out = std::make_shared<NumpyArray>(Identities::none(),
                                                  util::Parameters(),
                                                  ptr,
                                                  shape,
                                                  strides,
                                                  0,
                                                  (ssize_t)itemsize,
                                                  format,
                                                  dtype,
                                                  kernel::lib::cpu);

    for (int64_t i = depth - 1;  i >= 0;  i--) {
      Index64 index(levels[(size_t)i].ptr(), 0, levels[(size_t)i].length(), kernel::lib::cpu);
      out = std::make_shared<ListOffsetArray64>(Identities::none(),
                                                util::Parameters(),
                                                index,
                                                out);
    }
    return out;
  }

}
