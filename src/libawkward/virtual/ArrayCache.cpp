// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <atomic>

#include "awkward/virtual/ArrayCache.h"

namespace awkward {
  std::atomic<int64_t> numkeys{0};

  const std::string
  ArrayCache::newkey() {
    std::string out = std::string("ak") + std::to_string(numkeys);
    numkeys++;
    return out;
  }

  // Note: if you're creating a pure C++ cache (and it's not ridiculously
  // large), define it in
  // include/awkward/virtual/ArrayCache.h and implement it in this file.
}
