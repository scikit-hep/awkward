// BSD 3-Clause License; see
// https://github.com/scikit-hep/awkward/blob/main/LICENSE

// Regression tests for:
//   * BitMasked use-after-free / wrong-byte bugs (the mask reference member),
//   * BitMasked<valid_when=false> inversion,
//   * BitMasked masks spanning more than one GrowableBuffer panel,
//   * GrowableBuffer::Panel::concatenate_to_from multi-panel + non-zero `from`
//     with sizeof(PRIMITIVE) > 1,
//   * Indexed::extend_index / IndexedOption::extend_valid zero-size underflow.

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "awkward/GrowableBuffer.h"
#include "awkward/LayoutBuilder.h"

template <typename PRIMITIVE, typename BUILDER>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<PRIMITIVE>;

template <bool VALID_WHEN, bool LSB_ORDER, typename BUILDER>
using BitMaskedBuilder =
    awkward::LayoutBuilder::BitMasked<VALID_WHEN, LSB_ORDER, BUILDER>;

template <typename PRIMITIVE, typename BUILDER>
using IndexedBuilder = awkward::LayoutBuilder::Indexed<PRIMITIVE, BUILDER>;

template <typename PRIMITIVE, typename BUILDER>
using IndexedOptionBuilder =
    awkward::LayoutBuilder::IndexedOption<PRIMITIVE, BUILDER>;

template <typename PRIMITIVE>
using NumpyLeaf = awkward::LayoutBuilder::Numpy<PRIMITIVE>;

// Allocates a zero-filled buffer for each node name in the nbytes map.
// `storage` owns the memory and must outlive `buffers`.
void
allocate(const std::map<std::string, size_t>& names_nbytes,
         std::map<std::string, std::vector<uint8_t>>& storage,
         std::map<std::string, void*>& buffers) {
  for (const auto& it : names_nbytes) {
    storage[it.first] = std::vector<uint8_t>(it.second, uint8_t(0));
    buffers[it.first] = reinterpret_cast<void*>(storage[it.first].data());
  }
}

// --- BitMasked: reuse after clear(), valid_when, > 8 elements ---------------

template <bool VALID_WHEN>
void
test_bitmasked_roundtrip() {
  using Builder = BitMaskedBuilder<VALID_WHEN, true, NumpyLeaf<double>>;
  Builder builder;

  // 10 elements: valid pattern matching the existing test
  //   [valid, invalid, valid, valid, valid, invalid, invalid, valid, valid, valid]
  bool valid[10] = {true, false, true, true, true, false, false, true, true, true};
  for (size_t i = 0; i < 10; i++) {
    if (valid[i]) {
      auto& sub = builder.append_valid();
      sub.append((double)i);
    } else {
      auto& sub = builder.append_invalid();
      sub.append(-1000.0);
    }
  }
  assert(builder.length() == 10);

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes;
  builder.buffer_nbytes(names_nbytes);
  // ceil(10 / 8) == 2 mask bytes
  assert(names_nbytes["node0-mask"] == 2);

  std::map<std::string, std::vector<uint8_t>> storage;
  std::map<std::string, void*> buffers;
  allocate(names_nbytes, storage, buffers);
  builder.to_buffers(buffers);

  // LSB order: a bit is set when the element is valid (or invalid if
  // !valid_when). byte0 covers elements 0..7 (all meaningful); byte1 covers
  // elements 8..9 -- only the low 2 bits are meaningful, the padding bits are
  // don't-care and depend on the inversion.
  uint8_t b0 = 0, b1 = 0;
  for (size_t i = 0; i < 8; i++) {
    bool bit = VALID_WHEN ? valid[i] : !valid[i];
    if (bit) b0 |= (uint8_t)(1 << i);
  }
  for (size_t i = 8; i < 10; i++) {
    bool bit = VALID_WHEN ? valid[i] : !valid[i];
    if (bit) b1 |= (uint8_t)(1 << (i - 8));
  }
  uint8_t trailing_mask = (uint8_t)((1 << (10 - 8)) - 1);  // low 2 bits
  uint8_t* mask = reinterpret_cast<uint8_t*>(buffers["node0-mask"]);
  assert(mask[0] == b0);
  assert((uint8_t)(mask[1] & trailing_mask) == b1);

  // Now reuse the builder after clear() -- this used to be a heap UAF.
  builder.clear();
  assert(builder.length() == 0);

  for (size_t i = 0; i < 10; i++) {
    if (valid[i]) {
      auto& sub = builder.append_valid();
      sub.append((double)i);
    } else {
      auto& sub = builder.append_invalid();
      sub.append(-1000.0);
    }
  }
  assert(builder.length() == 10);
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes2;
  builder.buffer_nbytes(names_nbytes2);
  assert(names_nbytes2["node0-mask"] == 2);
  std::map<std::string, std::vector<uint8_t>> storage2;
  std::map<std::string, void*> buffers2;
  allocate(names_nbytes2, storage2, buffers2);
  builder.to_buffers(buffers2);
  uint8_t* mask2 = reinterpret_cast<uint8_t*>(buffers2["node0-mask"]);
  assert(mask2[0] == b0);
  assert((uint8_t)(mask2[1] & trailing_mask) == b1);
}

// --- BitMasked: mask spanning more than one panel (> 8192 elements) ---------

void
test_bitmasked_multipanel() {
  // Default options reserve 1024 bytes per panel, so > 8192 elements forces
  // the mask buffer to span multiple panels.
  using Builder = BitMaskedBuilder<true, true, NumpyLeaf<int64_t>>;
  Builder builder;

  const size_t n = 20000;
  for (size_t i = 0; i < n; i++) {
    if (i % 3 == 0) {
      builder.append_invalid().append(-1);
    } else {
      builder.append_valid().append((int64_t)i);
    }
  }
  assert(builder.length() == n);

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes;
  builder.buffer_nbytes(names_nbytes);
  size_t expected_bytes = (n + 7) / 8;
  assert(names_nbytes["node0-mask"] == expected_bytes);

  std::map<std::string, std::vector<uint8_t>> storage;
  std::map<std::string, void*> buffers;
  allocate(names_nbytes, storage, buffers);
  builder.to_buffers(buffers);

  uint8_t* mask = reinterpret_cast<uint8_t*>(buffers["node0-mask"]);
  for (size_t i = 0; i < n; i++) {
    bool expected = (i % 3 != 0);  // valid_when == true
    bool got = (mask[i / 8] & (uint8_t)(1 << (i % 8))) != 0;
    assert(got == expected);
  }
}

// --- GrowableBuffer: multi-panel concatenate_to_from with non-zero `from` ---

void
test_growablebuffer_concatenate_from() {
  // int32_t => sizeof(PRIMITIVE) == 4, exercising the byte/element mismatch.
  awkward::BuilderOptions options(8, 1.5);  // small panels -> several panels
  awkward::GrowableBuffer<int32_t> buffer(options);

  const size_t n = 100;
  for (size_t i = 0; i < n; i++) {
    buffer.append((int32_t)(i + 1));
  }
  assert(buffer.length() == n);

  // concatenate_from with from=2 should skip the first two elements of the
  // FIRST panel only and copy every remaining element contiguously.
  const size_t from = 2;
  std::vector<int32_t> out(n, int32_t(-999));
  buffer.concatenate_from(out.data(), 0, from);

  for (size_t i = 0; i < n - from; i++) {
    assert(out[i] == (int32_t)(i + 1 + from));
  }
}

// --- Indexed / IndexedOption: zero-size extend underflow --------------------

void
test_indexed_zero_extend() {
  IndexedBuilder<int64_t, NumpyLeaf<double>> builder;
  // extend_index(0) on empty content used to underflow stop-1 to SIZE_MAX,
  // poisoning max_index_ and making is_valid() falsely fail.
  builder.extend_index(0);
  assert(builder.length() == 0);
  std::string error;
  assert(builder.is_valid(error) == true);

  // Subsequent real appends must still validate.
  auto& sub = builder.append_index();
  sub.append(1.1);
  assert(builder.length() == 1);
  assert(builder.is_valid(error) == true);
}

void
test_indexedoption_zero_extend() {
  IndexedOptionBuilder<int64_t, NumpyLeaf<double>> builder;
  builder.extend_valid(0);
  assert(builder.length() == 0);
  std::string error;
  assert(builder.is_valid(error) == true);

  auto& sub = builder.append_valid();
  sub.append(2.2);
  assert(builder.length() == 1);
  assert(builder.is_valid(error) == true);
}

int
main(int /* argc */, char** /* argv */) {
  test_bitmasked_roundtrip<true>();
  test_bitmasked_roundtrip<false>();
  test_bitmasked_multipanel();
  test_growablebuffer_concatenate_from();
  test_indexed_zero_extend();
  test_indexedoption_zero_extend();
  return 0;
}
