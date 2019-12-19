// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <cmath>
#include <cstring>
#include <cassert>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/Index.h"

namespace awkward {
  template <typename T>
  class GrowableBuffer {
  public:
    static GrowableBuffer<T> empty(const FillableOptions& options);
    static GrowableBuffer<T> empty(const FillableOptions& options, int64_t minreserve);
    static GrowableBuffer<T> full(const FillableOptions& options, T value, int64_t length);
    static GrowableBuffer<T> arange(const FillableOptions& options, int64_t length);

    GrowableBuffer(const FillableOptions& options, std::shared_ptr<T> ptr, int64_t length, int64_t reserved);
    GrowableBuffer(const FillableOptions& options);
    const std::shared_ptr<T> ptr() const;
    int64_t length() const;
    void set_length(int64_t newlength);
    int64_t reserved() const;
    void set_reserved(int64_t minreserved);
    void clear();
    void append(T datum);
    T getitem_at_nowrap(int64_t at) const;

  private:
    const FillableOptions options_;
    std::shared_ptr<T> ptr_;
    int64_t length_;
    int64_t reserved_;
  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_
