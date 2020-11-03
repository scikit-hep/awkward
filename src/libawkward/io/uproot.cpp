// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/io/root.cpp", line)

#include <cstring>

#include "awkward/Index.h"
#include "awkward/Content.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/GrowableBuffer.h"

#include "awkward/io/uproot.h"

namespace awkward {
  template <typename T>
  T byteswap(T big);

  template <>
  int32_t byteswap<int32_t>(int32_t big) {
    uint64_t biggie = (uint64_t)big;
    return (int32_t)((biggie >> 24) & 0x000000ff) |
                    ((biggie >>  8) & 0x0000ff00) |
                    ((biggie <<  8) & 0x00ff0000) |
                    ((biggie << 24) & 0xff000000);
  }

  template <>
  int64_t byteswap<int64_t>(int64_t big) {
    uint64_t biggie = (uint64_t)big;
    return (int64_t)((biggie >> 56) & 0x00000000000000ff) |
                    ((biggie >> 40) & 0x000000000000ff00) |
                    ((biggie >> 24) & 0x0000000000ff0000) |
                    ((biggie >>  8) & 0x00000000ff000000) |
                    ((biggie <<  8) & 0x000000ff00000000) |
                    ((biggie << 24) & 0x0000ff0000000000) |
                    ((biggie << 40) & 0x00ff000000000000) |
                    ((biggie << 56) & 0xff00000000000000);
  }

  template <>
  double byteswap<double>(double big) {
    int64_t little = byteswap<int64_t>(*reinterpret_cast<int64_t*>(&big));
    return *reinterpret_cast<double*>(&little);
  }

  template <typename T, int ITEMSIZE>
  const ContentPtr
  uproot_issue_90_impl(const NumpyArray& data,
                       const Index32& byte_offsets,
                       util::dtype dtype) {
    uint8_t* data_ptr = reinterpret_cast<uint8_t*>(data.data());
    int32_t* byte_offsets_ptr = byte_offsets.data();

    ArrayBuilderOptions options(1024, 1.5);

    Index64 offsets1(byte_offsets.length());
    int64_t* offsets1_ptr = offsets1.data();
    GrowableBuffer<int64_t> offsets2 = GrowableBuffer<int64_t>::empty(options);
    GrowableBuffer<T> content = GrowableBuffer<T>::empty(options);

    offsets1_ptr[0] = 0;
    offsets2.append(0);
    int64_t last_offsets2 = 0;
    for (int64_t entry = 0;  entry < byte_offsets.length() - 1;  entry++) {
      int32_t pos = byte_offsets_ptr[entry];

      int32_t numbytes = byteswap(*reinterpret_cast<int32_t*>(&data_ptr[pos]));
      numbytes &= ~((int32_t)0x40000000);
      numbytes += 4;
      pos += sizeof(int32_t) + 2;  // skip 2-byte version

      int64_t count = 0;
      while (pos < byte_offsets_ptr[entry] + numbytes) {
        int32_t length = byteswap(*reinterpret_cast<int32_t*>(&data_ptr[pos]));
        pos += sizeof(int32_t);

        for (int64_t j = 0;  j < length;  j++) {
          content.append(byteswap(*reinterpret_cast<T*>(&data_ptr[pos])));
          pos += sizeof(T);
        }

        last_offsets2 += length;
        offsets2.append(last_offsets2);

        count++;
      }
      offsets1_ptr[entry + 1] = offsets1_ptr[entry] + count;
    }

    std::vector<ssize_t> shape = { (ssize_t)content.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(T) };
    ContentPtr outcontent = std::make_shared<NumpyArray>(Identities::none(),
                                                         util::Parameters(),
                                                         content.ptr(),
                                                         shape,
                                                         strides,
                                                         0,
                                                         sizeof(T),
                                                         util::dtype_to_format(dtype),
                                                         dtype,
                                                         kernel::lib::cpu);

    Index64 outoffsets2(offsets2.ptr(), 0, offsets2.length(), kernel::lib::cpu);
    ContentPtr outlist2 = std::make_shared<ListOffsetArray64>(Identities::none(),
                                                              util::Parameters(),
                                                              outoffsets2,
                                                              outcontent);
    return std::make_shared<ListOffsetArray64>(Identities::none(),
                                               util::Parameters(),
                                               offsets1,
                                               outlist2);
  }

  const ContentPtr
  uproot_issue_90(const Form& form,
                  const NumpyArray& data,
                  const Index32& byte_offsets) {
    if (const ListOffsetForm* raw1 = dynamic_cast<const ListOffsetForm*>(&form)) {
      FormPtr content1 = raw1->content();
      if (const ListOffsetForm* raw2 = dynamic_cast<const ListOffsetForm*>(content1.get())) {
        FormPtr content2 = raw2->content();
        if (const NumpyForm* raw3 = dynamic_cast<const NumpyForm*>(content2.get())) {
          if (raw3->dtype() == util::dtype::int32) {
            return uproot_issue_90_impl<int32_t, sizeof(int32_t)>(data,
                                                                  byte_offsets,
                                                                  util::dtype::int32);
          }
          else if (raw3->dtype() == util::dtype::float64) {
            return uproot_issue_90_impl<double, sizeof(double)>(data,
                                                                byte_offsets,
                                                                util::dtype::float64);
          }
        }
      }
    }

    throw std::invalid_argument(
        std::string("uproot_issue_90 only handles two types") + FILENAME(__LINE__)
    );
  }
}
