// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <string>
#include <map>
#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  template <typename T>
  class IndexOf;

  class EXPORT_SYMBOL Index {
    virtual const std::shared_ptr<Index> shallow_copy() const = 0;
    virtual IndexOf<int64_t> to64() const = 0;
  };

  template <typename T>
  class EXPORT_SYMBOL IndexOf: public Index {
  public:
    IndexOf<T>(int64_t length);
    IndexOf<T>(const std::shared_ptr<T>& ptr, int64_t offset, int64_t length);

    const std::shared_ptr<T> ptr() const;
    int64_t offset() const;
    int64_t length() const;

    const std::string classname() const;
    const std::string tostring() const;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const;
    T getitem_at(int64_t at) const;
    T getitem_at_nowrap(int64_t at) const;
    void setitem_at_nowrap(int64_t at, T value) const;
    IndexOf<T> getitem_range(int64_t start, int64_t stop) const;
    IndexOf<T> getitem_range_nowrap(int64_t start, int64_t stop) const;
    void nbytes_part(std::map<size_t, int64_t>& largest) const;
    const std::shared_ptr<Index> shallow_copy() const override;
    IndexOf<int64_t> to64() const override;

    const IndexOf<T> deep_copy() const;

  private:
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
  };

  using Index8   = IndexOf<int8_t>;
  using IndexU8  = IndexOf<uint8_t>;
  using Index32  = IndexOf<int32_t>;
  using IndexU32 = IndexOf<uint32_t>;
  using Index64  = IndexOf<int64_t>;
}

#endif // AWKWARD_INDEX_H_
